"""
RAG Agent Implementation using Strands Agents SDK.

Main agent class that orchestrates RAG operations using Strands Agents
with AWS Bedrock for LLM and knowledge retrieval.
"""

import time
from typing import Any, AsyncGenerator
from uuid import uuid4

import boto3
import structlog
from strands import Agent
from strands.models import BedrockModel

from rag_agent.config import Settings, get_settings
from rag_agent.models import (
    Conversation,
    ConversationMessage,
    QueryRequest,
    QueryResponse,
    SourceReference,
    VectorSearchResult,
)
from rag_agent.services.bedrock import BedrockService
from rag_agent.services.dynamodb import DynamoDBService
from rag_agent.services.embeddings import EmbeddingService
from rag_agent.services.s3_vectors import S3VectorService
from rag_agent.tools import get_all_tools, get_core_tools
from rag_agent.tools.base import ToolContext

logger = structlog.get_logger(__name__)


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent using Strands Agents SDK.
    
    Implements an agentic RAG system with:
    - Strands Agents SDK for tool orchestration
    - AWS Bedrock for LLM operations
    - S3/DynamoDB for storage
    - Automatic tool selection and execution
    
    The agent can:
    - Search documents using semantic similarity
    - Retrieve and summarize documents
    - Query Bedrock Knowledge Bases
    - Fetch web content when needed
    """

    def __init__(
        self,
        settings: Settings | None = None,
        include_web_tools: bool = True,
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            settings: Application settings
            include_web_tools: Whether to include web search tools (default: True)
        """
        self.settings = settings or get_settings()
        
        # Initialize AWS services
        self.bedrock = BedrockService(self.settings)
        self.embeddings = EmbeddingService(self.settings)
        self.vectors = S3VectorService(self.settings)
        self.dynamodb = DynamoDBService(self.settings)
        
        # Create boto3 session for Strands
        self._boto_session = self._create_boto_session()
        
        # Initialize Strands Bedrock model
        self._model = self._create_bedrock_model()
        
        # Get tools
        self._tools = get_all_tools() if include_web_tools else get_core_tools()
        
        # Filter out knowledge_base_search if KB not configured
        if not self.settings.bedrock.knowledge_base_id:
            from rag_agent.tools import knowledge_base_search
            self._tools = [t for t in self._tools if t != knowledge_base_search]
        
        # Create Strands Agent
        self._agent = self._create_agent()
        
        # Conversation memory
        self._conversations: dict[str, Conversation] = {}
        
        logger.info(
            "RAG Agent initialized with Strands SDK",
            tools_count=len(self._tools),
            model=self.settings.bedrock.llm_model_id,
        )

    def _create_boto_session(self) -> boto3.Session:
        """Create a boto3 session with configured credentials."""
        kwargs = {"region_name": self.settings.aws.region}
        if self.settings.aws.access_key_id and self.settings.aws.secret_access_key:
            kwargs["aws_access_key_id"] = self.settings.aws.access_key_id
            kwargs["aws_secret_access_key"] = self.settings.aws.secret_access_key
        return boto3.Session(**kwargs)

    def _create_bedrock_model(self) -> BedrockModel:
        """Create the Strands BedrockModel instance."""
        return BedrockModel(
            model_id=self.settings.bedrock.llm_model_id,
            boto_session=self._boto_session,
            temperature=self.settings.bedrock.temperature,
            top_p=self.settings.bedrock.top_p,
            max_tokens=self.settings.bedrock.max_tokens,
        )

    def _create_agent(self) -> Agent:
        """Create the Strands Agent with tools and model."""
        system_prompt = self._build_system_prompt()
        
        return Agent(
            model=self._model,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are a helpful AI assistant with access to a knowledge base and various tools.
Your goal is to provide accurate, helpful responses based on the available information.

## Guidelines
1. Always search the knowledge base first for relevant information using vector_search
2. Cite your sources when providing information from the knowledge base
3. If the knowledge base doesn't have relevant information, use web_search if available
4. Be honest when you don't have enough information to answer
5. Keep responses concise but comprehensive

## Response Format
When answering questions:
- Start with a direct answer when possible
- Provide supporting details from sources
- Include source references for factual claims
- Acknowledge uncertainty when appropriate
- IMPORTANT: Do NOT simply repeat or echo back tool output verbatim. Instead, format and present the information in a natural, conversational way. The tool output is already visible to the user, so your response should add value by summarizing, organizing, or explaining the results.

## Tool Usage
- Use vector_search to find relevant documents by semantic similarity
- Use get_document to retrieve full document details
- Use list_documents to see available documents
- Use summarize_document to get quick document overviews
- Use knowledge_base_search for Bedrock Knowledge Base queries (if configured)
- Use web_search for current/external information (if available)
- Use fetch_url to read specific webpage content

When tools return results, format them nicely rather than just repeating the raw output.
"""

    async def _retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant document chunks
        """
        top_k = top_k or self.settings.vector.top_k
        
        # Generate query embedding
        query_embedding = await self.embeddings.generate_query_embedding(query)
        
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []
        
        # Search for similar documents
        results = await self.vectors.search_vectors(
            query_vector=query_embedding,
            top_k=top_k,
        )
        
        logger.info(
            "Context retrieved",
            query=query[:50],
            results_count=len(results),
        )
        
        return results

    def __call__(self, message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        This is the simplest interface - just call the agent like a function.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        # Set up tool context with services
        with ToolContext(
            embedding_service=self.embeddings,
            vector_service=self.vectors,
            dynamodb_service=self.dynamodb,
            bedrock_service=self.bedrock,
        ):
            response = self._agent(message)
            return str(response)

    async def query(
        self,
        request: QueryRequest,
        conversation_id: str | None = None,
    ) -> QueryResponse:
        """
        Process a RAG query.
        
        Retrieves relevant context and generates a response using the agent.
        
        Args:
            request: Query request with parameters
            conversation_id: Optional conversation ID for context
            
        Returns:
            Query response with answer and sources
        """
        start_time = time.time()
        
        logger.info("Processing query", query=request.query[:100])
        
        # Retrieve context for sources
        context = await self._retrieve_context(
            query=request.query,
            top_k=request.top_k,
        )
        
        # Filter by similarity threshold
        context = [
            r for r in context
            if r.score >= request.similarity_threshold
        ]
        
        # Get conversation history if available
        history_context = ""
        if conversation_id and conversation_id in self._conversations:
            conv = self._conversations[conversation_id]
            recent_messages = conv.messages[-4:]  # Last 2 exchanges
            history_context = "\n".join([
                f"{msg.role}: {msg.content}"
                for msg in recent_messages
            ])
        
        # Build enhanced prompt with history
        enhanced_prompt = request.query
        if history_context:
            enhanced_prompt = f"Previous conversation:\n{history_context}\n\nCurrent question: {request.query}"
        
        # Use the agent to generate response
        with ToolContext(
            embedding_service=self.embeddings,
            vector_service=self.vectors,
            dynamodb_service=self.dynamodb,
            bedrock_service=self.bedrock,
        ):
            response_text = str(self._agent(enhanced_prompt))
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Build source references
        sources = []
        if request.include_sources:
            for result in context:
                sources.append(
                    SourceReference(
                        document_id=result.document_id,
                        title=result.metadata.get("title", "Unknown"),
                        source=result.metadata.get("source", result.document_id),
                        chunk_id=result.chunk_id,
                        relevance_score=result.score,
                        snippet=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    )
                )
        
        # Update conversation if tracking
        if conversation_id:
            await self._update_conversation(
                conversation_id,
                request.query,
                response_text,
            )
        
        logger.info(
            "Query processed",
            latency_ms=latency_ms,
            sources_count=len(sources),
        )
        
        return QueryResponse(
            query=request.query,
            answer=response_text,
            sources=sources,
            model_id=self.settings.bedrock.llm_model_id,
            latency_ms=latency_ms,
            token_usage={},
        )

    async def chat(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Send a chat message and get a response.
        
        Simpler interface for conversational interactions.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID (created if not provided)
            
        Returns:
            Tuple of (response, conversation_id)
        """
        conversation_id = conversation_id or str(uuid4())
        
        request = QueryRequest(
            query=message,
            include_sources=False,
        )
        
        response = await self.query(request, conversation_id)
        
        return response.answer, conversation_id

    async def _update_conversation(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Update conversation history."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = Conversation()
        
        conv = self._conversations[conversation_id]
        conv.messages.append(ConversationMessage(role="user", content=user_message))
        conv.messages.append(ConversationMessage(role="assistant", content=assistant_message))
        
        # Persist to DynamoDB
        try:
            await self.dynamodb.save_conversation(conv)
        except Exception as e:
            logger.warning("Failed to persist conversation", error=str(e))

    def execute(self, query: str) -> str:
        """
        Execute a query using the agent with tools.
        
        The Strands agent automatically handles tool selection and execution.
        
        Args:
            query: User query
            
        Returns:
            Agent response after tool execution
        """
        with ToolContext(
            embedding_service=self.embeddings,
            vector_service=self.vectors,
            dynamodb_service=self.dynamodb,
            bedrock_service=self.bedrock,
        ):
            return str(self._agent(query))

    async def stream_response(
        self,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response for a query.
        
        Note: Strands streaming support depends on the model.
        Falls back to non-streaming if not available.
        
        Args:
            query: User query
            
        Yields:
            Response chunks
        """
        # For now, yield the full response
        # Strands streaming API can be integrated when available
        with ToolContext(
            embedding_service=self.embeddings,
            vector_service=self.vectors,
            dynamodb_service=self.dynamodb,
            bedrock_service=self.bedrock,
        ):
            response = str(self._agent(query))
            yield response

    def list_tools(self) -> list[str]:
        """Return list of available tool names."""
        return [t.__name__ for t in self._tools]

    @property
    def model_id(self) -> str:
        """Return the configured model ID."""
        return self.settings.bedrock.llm_model_id
