"""
AWS Bedrock Service for LLM interactions and agent deployment.

Handles communication with Amazon Bedrock for text generation,
knowledge base queries, and agent orchestration.
"""

import json
from typing import Any, AsyncGenerator

import boto3
import structlog
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_agent.config import Settings, get_settings
from rag_agent.models import QueryResponse, SourceReference, VectorSearchResult

logger = structlog.get_logger(__name__)


class BedrockService:
    """
    Service for AWS Bedrock LLM operations and agent management.
    
    Provides text generation, RAG queries via knowledge bases, and
    agent-based reasoning capabilities.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize Bedrock Service."""
        self.settings = settings or get_settings()
        self._runtime_client = None
        self._agent_client = None
        self._kb_client = None

    def _get_boto_kwargs(self) -> dict:
        """Get boto3 client kwargs, only including credentials if explicitly set."""
        kwargs = {"region_name": self.settings.aws.region}
        if self.settings.aws.access_key_id and self.settings.aws.secret_access_key:
            kwargs["aws_access_key_id"] = self.settings.aws.access_key_id
            kwargs["aws_secret_access_key"] = self.settings.aws.secret_access_key
        return kwargs

    @property
    def runtime_client(self):
        """Lazy initialization of Bedrock Runtime client."""
        if self._runtime_client is None:
            self._runtime_client = boto3.client("bedrock-runtime", **self._get_boto_kwargs())
        return self._runtime_client

    @property
    def agent_client(self):
        """Lazy initialization of Bedrock Agent Runtime client."""
        if self._agent_client is None:
            self._agent_client = boto3.client("bedrock-agent-runtime", **self._get_boto_kwargs())
        return self._agent_client

    @property
    def kb_client(self):
        """Lazy initialization of Bedrock Agent client for KB management."""
        if self._kb_client is None:
            self._kb_client = boto3.client("bedrock-agent", **self._get_boto_kwargs())
        return self._kb_client

    def _get_model_provider(self, model_id: str) -> str:
        """
        Determine the model provider from the model ID.
        
        Args:
            model_id: Bedrock model identifier
            
        Returns:
            Provider name: 'qwen', 'meta', 'amazon', 'mistral', 'cohere', 'ai21'
        """
        model_lower = model_id.lower()
        if "qwen" in model_lower:
            return "qwen"
        if "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "llama" in model_lower or "meta" in model_lower:
            return "meta"
        elif "titan" in model_lower or "amazon" in model_lower:
            return "amazon"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        elif "cohere" in model_lower or "command" in model_lower:
            return "cohere"
        elif "ai21" in model_lower or "jamba" in model_lower or "jurassic" in model_lower:
            return "ai21"
        else:
            return "generic"

    def _build_context_string(self, context: list[VectorSearchResult] | None) -> str:
        """Build a context string from vector search results."""
        if not context:
            return ""
        context_str = "\n\nContext:\n"
        for i, result in enumerate(context, 1):
            context_str += f"[Source {i}] (Score: {result.score:.3f})\n{result.content}\n\n"
        return context_str

    def _build_request_body(
        self,
        prompt: str,
        model_id: str,
        system_prompt: str | None = None,
        context: list[VectorSearchResult] | None = None,
    ) -> dict[str, Any]:
        """
        Build the request body for any supported Bedrock model.
        
        Args:
            prompt: User prompt
            model_id: Bedrock model identifier
            system_prompt: Optional system instructions
            context: Optional RAG context
            
        Returns:
            Request body formatted for the specific model
        """
        provider = self._get_model_provider(model_id)
        context_str = self._build_context_string(context)
        full_prompt = context_str + prompt if context_str else prompt
        
        if provider == "anthropic":
            # Claude models use Messages API
            messages = [{"role": "user", "content": full_prompt}]
            request_body = {
                "anthropic_version": self.settings.bedrock.anthropic_version,
                "max_tokens": self.settings.bedrock.max_tokens,
                "temperature": self.settings.bedrock.temperature,
                "top_p": self.settings.bedrock.top_p,
                "top_k": self.settings.bedrock.top_k,
                "messages": messages,
            }
            if system_prompt:
                request_body["system"] = system_prompt
            if self.settings.bedrock.stop_sequences:
                request_body["stop_sequences"] = self.settings.bedrock.stop_sequences
                
        elif provider == "meta":
            # Llama models
            formatted_prompt = full_prompt
            if system_prompt:
                formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{full_prompt} [/INST]"
            else:
                formatted_prompt = f"<s>[INST] {full_prompt} [/INST]"
            request_body = {
                "prompt": formatted_prompt,
                "max_gen_len": self.settings.bedrock.max_tokens,
                "temperature": self.settings.bedrock.temperature,
                "top_p": self.settings.bedrock.top_p,
            }
            
        elif provider == "amazon":
            # Amazon Titan models
            text_config = {
                "maxTokenCount": self.settings.bedrock.max_tokens,
                "temperature": self.settings.bedrock.temperature,
                "topP": self.settings.bedrock.top_p,
            }
            if self.settings.bedrock.stop_sequences:
                text_config["stopSequences"] = self.settings.bedrock.stop_sequences
            formatted_prompt = f"{system_prompt}\n\n{full_prompt}" if system_prompt else full_prompt
            request_body = {
                "inputText": formatted_prompt,
                "textGenerationConfig": text_config,
            }
            
        elif provider == "mistral":
            # Mistral/Mixtral models
            formatted_prompt = full_prompt
            if system_prompt:
                formatted_prompt = f"<s>[INST] {system_prompt}\n\n{full_prompt} [/INST]"
            else:
                formatted_prompt = f"<s>[INST] {full_prompt} [/INST]"
            request_body = {
                "prompt": formatted_prompt,
                "max_tokens": self.settings.bedrock.max_tokens,
                "temperature": self.settings.bedrock.temperature,
                "top_p": self.settings.bedrock.top_p,
                "top_k": self.settings.bedrock.top_k,
            }
            if self.settings.bedrock.stop_sequences:
                request_body["stop"] = self.settings.bedrock.stop_sequences
                
        elif provider == "cohere":
            # Cohere Command models
            request_body = {
                "message": full_prompt,
                "max_tokens": self.settings.bedrock.max_tokens,
                "temperature": self.settings.bedrock.temperature,
                "p": self.settings.bedrock.top_p,
                "k": self.settings.bedrock.top_k,
            }
            if system_prompt:
                request_body["preamble"] = system_prompt
            if self.settings.bedrock.stop_sequences:
                request_body["stop_sequences"] = self.settings.bedrock.stop_sequences
                
        elif provider == "ai21":
            # AI21 Jamba/Jurassic models
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": full_prompt})
            request_body = {
                "messages": messages,
                "max_tokens": self.settings.bedrock.max_tokens,
                "temperature": self.settings.bedrock.temperature,
                "top_p": self.settings.bedrock.top_p,
            }
            if self.settings.bedrock.stop_sequences:
                request_body["stop"] = self.settings.bedrock.stop_sequences
                
        else:
            # Generic fallback
            formatted_prompt = f"{system_prompt}\n\n{full_prompt}" if system_prompt else full_prompt
            request_body = {
                "prompt": formatted_prompt,
                "max_tokens": self.settings.bedrock.max_tokens,
                "temperature": self.settings.bedrock.temperature,
            }
        
        return request_body

    def _parse_response(self, response_body: dict[str, Any], model_id: str) -> dict[str, Any]:
        """
        Parse the response from any supported Bedrock model.
        
        Args:
            response_body: Raw response from Bedrock
            model_id: Bedrock model identifier
            
        Returns:
            Parsed response with content and token usage
        """
        provider = self._get_model_provider(model_id)
        
        if (provider == "qwen"):
            content = response_body.get("response", "")
            token_usage = {
                "input_tokens": response_body.get("input_tokens", 0),
                "output_tokens": response_body.get("output_tokens", 0),
            }
            return {"content": content, "token_usage": token_usage}
        elif (provider == "anthropic"):
            content = response_body.get("content", [{}])[0].get("text", "")
            usage = response_body.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        elif provider == "meta":
            content = response_body.get("generation", "")
            token_usage = {
                "input_tokens": response_body.get("prompt_token_count", 0),
                "output_tokens": response_body.get("generation_token_count", 0),
            }
        elif provider == "amazon":
            results = response_body.get("results", [{}])
            content = results[0].get("outputText", "") if results else ""
            token_usage = {
                "input_tokens": response_body.get("inputTextTokenCount", 0),
                "output_tokens": results[0].get("tokenCount", 0) if results else 0,
            }
        elif provider == "mistral":
            outputs = response_body.get("outputs", [{}])
            content = outputs[0].get("text", "") if outputs else ""
            token_usage = {}
        elif provider == "cohere":
            content = response_body.get("text", "")
            token_usage = {
                "input_tokens": response_body.get("meta", {}).get("billed_units", {}).get("input_tokens", 0),
                "output_tokens": response_body.get("meta", {}).get("billed_units", {}).get("output_tokens", 0),
            }
        elif provider == "ai21":
            choices = response_body.get("choices", [{}])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = response_body.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        else:
            content = response_body.get("completion", response_body.get("text", response_body.get("generation", "")))
            token_usage = {}
        
        return {"content": content, "token_usage": token_usage}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_id: str | None = None,
        context: list[VectorSearchResult] | None = None,
    ) -> dict[str, Any]:
        """
        Generate text using a Bedrock LLM.
        
        Supports multiple model providers:
        - Anthropic (Claude)
        - Meta (Llama)
        - Amazon (Titan)
        - Mistral (Mistral/Mixtral)
        - Cohere (Command)
        - AI21 (Jamba/Jurassic)
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            model_id: Optional model ID override
            context: Optional RAG context from vector search
            
        Returns:
            Generated text response with metadata
        """
        model_id = model_id or self.settings.bedrock.llm_model_id
        provider = self._get_model_provider(model_id)
        
        # Build request body for the specific model
        request_body = self._build_request_body(prompt, model_id, system_prompt, context)
        
        try:
            response = self.runtime_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )
            
            response_body = json.loads(response["body"].read())
            
            # Parse response for the specific model
            parsed = self._parse_response(response_body, model_id)
            
            logger.info(
                "Text generated successfully",
                model_id=model_id,
                provider=provider,
                prompt_length=len(prompt),
                response_length=len(parsed["content"]),
            )
            
            return {
                "content": parsed["content"],
                "model_id": model_id,
                "provider": provider,
                "token_usage": parsed["token_usage"],
            }
            
        except ClientError as e:
            logger.error("Failed to generate text", model_id=model_id, provider=provider, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def query_knowledge_base(
        self,
        query: str,
        knowledge_base_id: str | None = None,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        Query a Bedrock Knowledge Base for relevant context.
        
        Args:
            query: Search query
            knowledge_base_id: Optional KB ID override
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant search results
        """
        kb_id = knowledge_base_id or self.settings.bedrock.knowledge_base_id
        if not kb_id:
            raise ValueError("Knowledge base ID not configured")
        
        top_k = top_k or self.settings.vector.top_k
        
        try:
            response = self.agent_client.retrieve(
                knowledgeBaseId=kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": top_k,
                    }
                },
            )
            
            results = []
            for result in response.get("retrievalResults", []):
                content = result.get("content", {}).get("text", "")
                score = result.get("score", 0.0)
                location = result.get("location", {})
                metadata = result.get("metadata", {})
                
                results.append(
                    VectorSearchResult(
                        document_id=metadata.get("document_id", location.get("s3Location", {}).get("uri", "")),
                        chunk_id=metadata.get("chunk_id", ""),
                        score=score,
                        content=content,
                        metadata={**metadata, "location": location},
                    )
                )
            
            logger.info(
                "Knowledge base query completed",
                query_length=len(query),
                results_count=len(results),
            )
            return results
            
        except ClientError as e:
            logger.error("Failed to query knowledge base", kb_id=kb_id, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def retrieve_and_generate(
        self,
        query: str,
        knowledge_base_id: str | None = None,
        model_id: str | None = None,
    ) -> QueryResponse:
        """
        Perform RAG: retrieve context and generate response.
        
        Uses Bedrock's built-in RetrieveAndGenerate API for optimized RAG.
        
        Args:
            query: User query
            knowledge_base_id: Optional KB ID override
            model_id: Optional model ID override
            
        Returns:
            Query response with answer and sources
        """
        import time
        start_time = time.time()
        
        kb_id = knowledge_base_id or self.settings.bedrock.knowledge_base_id
        model_id = model_id or self.settings.bedrock.llm_model_id
        
        if not kb_id:
            raise ValueError("Knowledge base ID not configured")
        
        try:
            response = self.agent_client.retrieve_and_generate(
                input={"text": query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": kb_id,
                        "modelArn": f"arn:aws:bedrock:{self.settings.aws.region}::foundation-model/{model_id}",
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": self.settings.vector.top_k,
                            }
                        },
                    },
                },
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            answer = response.get("output", {}).get("text", "")
            citations = response.get("citations", [])
            
            # Build source references
            sources = []
            for citation in citations:
                for ref in citation.get("retrievedReferences", []):
                    content = ref.get("content", {}).get("text", "")
                    location = ref.get("location", {})
                    metadata = ref.get("metadata", {})
                    
                    sources.append(
                        SourceReference(
                            document_id=metadata.get("document_id", ""),
                            title=metadata.get("title", location.get("s3Location", {}).get("uri", "Unknown")),
                            source=location.get("s3Location", {}).get("uri", ""),
                            chunk_id=metadata.get("chunk_id", ""),
                            relevance_score=0.0,  # Not provided by API
                            snippet=content[:500] if content else "",
                        )
                    )
            
            logger.info(
                "RAG query completed",
                query_length=len(query),
                answer_length=len(answer),
                sources_count=len(sources),
                latency_ms=latency_ms,
            )
            
            return QueryResponse(
                query=query,
                answer=answer,
                sources=sources,
                model_id=model_id,
                latency_ms=latency_ms,
                token_usage={},
            )
            
        except ClientError as e:
            logger.error("RAG query failed", kb_id=kb_id, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def invoke_agent(
        self,
        prompt: str,
        agent_id: str | None = None,
        agent_alias_id: str = "TSTALIASID",
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Invoke a Bedrock Agent for multi-step reasoning.
        
        Args:
            prompt: User input
            agent_id: Optional agent ID override
            agent_alias_id: Agent alias ID (default: test alias)
            session_id: Optional session ID for conversation continuity
            
        Yields:
            Streamed response chunks
        """
        agent_id = agent_id or self.settings.bedrock.agent_id
        if not agent_id:
            raise ValueError("Agent ID not configured")
        
        session_id = session_id or str(__import__("uuid").uuid4())
        
        try:
            response = self.agent_client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
            )
            
            # Stream the response
            event_stream = response.get("completion", [])
            for event in event_stream:
                if "chunk" in event:
                    chunk_data = event["chunk"]
                    if "bytes" in chunk_data:
                        yield chunk_data["bytes"].decode("utf-8")
                        
        except ClientError as e:
            logger.error("Agent invocation failed", agent_id=agent_id, error=str(e))
            raise

    async def list_foundation_models(self) -> list[dict[str, Any]]:
        """
        List available foundation models in Bedrock.
        
        Returns:
            List of available models with their metadata
        """
        client = boto3.client("bedrock", **self._get_boto_kwargs())
        
        try:
            response = client.list_foundation_models()
            models = response.get("modelSummaries", [])
            
            logger.info("Foundation models listed", count=len(models))
            return models
            
        except ClientError as e:
            logger.error("Failed to list foundation models", error=str(e))
            raise

