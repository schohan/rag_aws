"""
RAG Agent Implementation.

Main agent class that orchestrates RAG operations using Google ADK patterns
with AWS Bedrock services for LLM and knowledge retrieval.
"""

import time
from typing import Any, AsyncGenerator
from uuid import uuid4

import structlog

from rag_agent.config import Settings, get_settings
from rag_agent.models import (
    AgentAction,
    AgentState,
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
from rag_agent.tools.base import Tool, ToolRegistry, ToolResult

logger = structlog.get_logger(__name__)


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent.
    
    Implements an agentic RAG system following Google ADK patterns:
    - Tool-based architecture for extensibility
    - Multi-step reasoning with observation loops
    - Context-aware response generation
    - Conversation memory management
    
    Uses AWS Bedrock for LLM operations and S3/DynamoDB for storage.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        tools: list[Tool] | None = None,
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            settings: Application settings
            tools: Optional list of tools to register
        """
        self.settings = settings or get_settings()
        
        # Initialize services
        self.bedrock = BedrockService(self.settings)
        self.embeddings = EmbeddingService(self.settings)
        self.vectors = S3VectorService(self.settings)
        self.dynamodb = DynamoDBService(self.settings)
        
        # Initialize tool registry
        self.tools = ToolRegistry()
        self._register_default_tools()
        
        # Register additional tools
        if tools:
            for tool in tools:
                self.tools.register(tool)
        
        # Conversation memory
        self._conversations: dict[str, Conversation] = {}
        
        logger.info(
            "RAG Agent initialized",
            tools_count=len(self.tools),
            model=self.settings.bedrock.llm_model_id,
        )

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        from rag_agent.tools.search import VectorSearchTool, KnowledgeBaseSearchTool
        from rag_agent.tools.document import (
            DocumentRetrievalTool,
            ListDocumentsTool,
            DocumentSummaryTool,
        )
        from rag_agent.tools.web import WebSearchTool, URLContentTool
        
        # Core RAG tools
        self.tools.register(VectorSearchTool(self.embeddings, self.vectors))
        self.tools.register(DocumentRetrievalTool(self.dynamodb, self.vectors))
        self.tools.register(ListDocumentsTool(self.dynamodb))
        
        # Optional tools
        if self.settings.bedrock.knowledge_base_id:
            self.tools.register(KnowledgeBaseSearchTool(self.bedrock))
        
        self.tools.register(DocumentSummaryTool(self.dynamodb, self.vectors))
        self.tools.register(WebSearchTool())
        self.tools.register(URLContentTool())

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in [self.tools.get(name) for name in self.tools.list_tools()]
            if tool
        ])
        
        return f"""You are a helpful AI assistant with access to a knowledge base and various tools.
Your goal is to provide accurate, helpful responses based on the available information.

## Available Tools
{tool_descriptions}

## Guidelines
1. Always search the knowledge base first for relevant information
2. Cite your sources when providing information from the knowledge base
3. If the knowledge base doesn't have relevant information, use web search
4. Be honest when you don't have enough information to answer
5. Keep responses concise but comprehensive

## Response Format
When answering questions:
- Start with a direct answer when possible
- Provide supporting details from sources
- Include source references for factual claims
- Acknowledge uncertainty when appropriate
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

    async def query(
        self,
        request: QueryRequest,
        conversation_id: str | None = None,
    ) -> QueryResponse:
        """
        Process a RAG query.
        
        Retrieves relevant context and generates a response using the LLM.
        
        Args:
            request: Query request with parameters
            conversation_id: Optional conversation ID for context
            
        Returns:
            Query response with answer and sources
        """
        start_time = time.time()
        
        logger.info("Processing query", query=request.query[:100])
        
        # Retrieve context
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
        
        # Build prompt with context
        context_str = ""
        if context:
            context_str = "## Relevant Context\n\n"
            for i, result in enumerate(context, 1):
                context_str += f"### Source {i} (Relevance: {result.score:.2%})\n"
                context_str += f"{result.content}\n\n"
        
        prompt = f"""{context_str}

## Conversation History
{history_context if history_context else "No previous conversation."}

## Current Question
{request.query}

Please provide a helpful, accurate response based on the context above."""
        
        # Generate response
        response = await self.bedrock.generate_text(
            prompt=prompt,
            system_prompt=self._build_system_prompt(),
            context=context if not context_str else None,  # Already included in prompt
        )
        
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
                response.get("content", ""),
            )
        
        logger.info(
            "Query processed",
            latency_ms=latency_ms,
            sources_count=len(sources),
        )
        
        return QueryResponse(
            query=request.query,
            answer=response.get("content", ""),
            sources=sources,
            model_id=response.get("model_id", self.settings.bedrock.llm_model_id),
            latency_ms=latency_ms,
            token_usage=response.get("token_usage", {}),
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

    async def execute_with_tools(
        self,
        query: str,
        max_steps: int = 5,
    ) -> AgentState:
        """
        Execute a query using the agent's tools.
        
        Implements a ReAct-style loop where the agent can:
        1. Think about what to do
        2. Use a tool
        3. Observe the result
        4. Repeat or provide final answer
        
        Args:
            query: User query
            max_steps: Maximum number of tool-use steps
            
        Returns:
            Final agent state with actions and answer
        """
        state = AgentState(query=query)
        
        # Build tool schemas for the LLM
        tool_schemas = self.tools.get_function_schemas()
        
        for step in range(max_steps):
            logger.info(f"Agent step {step + 1}/{max_steps}")
            
            # Build prompt with current state
            actions_str = ""
            if state.actions:
                actions_str = "\n## Previous Actions\n"
                for action in state.actions:
                    actions_str += f"- Tool: {action.tool_name}\n"
                    actions_str += f"  Input: {action.tool_input}\n"
                    actions_str += f"  Result: {action.observation}\n\n"
            
            prompt = f"""## User Query
{query}

{actions_str}

Based on the above, decide what to do next:
1. If you have enough information, provide a final answer
2. If you need more information, use one of the available tools

Respond in this format:
THOUGHT: <your reasoning>
ACTION: <tool_name or "final_answer">
ACTION_INPUT: <tool input as JSON or your final answer>
"""
            
            response = await self.bedrock.generate_text(
                prompt=prompt,
                system_prompt=self._build_system_prompt(),
            )
            
            content = response.get("content", "")
            
            # Parse the response
            thought, action, action_input = self._parse_agent_response(content)
            
            if action == "final_answer":
                state.final_answer = action_input
                state.is_complete = True
                break
            
            # Execute the tool
            if action and action in self.tools:
                try:
                    import json
                    tool_input = json.loads(action_input) if isinstance(action_input, str) else action_input
                    result = await self.tools.execute(action, **tool_input)
                    observation = str(result.data) if result.data else result.error
                except Exception as e:
                    observation = f"Error: {str(e)}"
                
                state.actions.append(
                    AgentAction(
                        action_type="tool_use",
                        tool_name=action,
                        tool_input=tool_input if isinstance(tool_input, dict) else {"raw": action_input},
                        observation=observation,
                    )
                )
            else:
                # Invalid tool, record the error
                state.actions.append(
                    AgentAction(
                        action_type="error",
                        tool_name=action,
                        observation=f"Tool '{action}' not found. Available: {self.tools.list_tools()}",
                    )
                )
        
        if not state.is_complete:
            state.final_answer = "I was unable to complete the task within the allowed steps."
            state.is_complete = True
        
        return state

    def _parse_agent_response(
        self,
        response: str,
    ) -> tuple[str | None, str | None, str | None]:
        """
        Parse the agent's response to extract thought, action, and input.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Tuple of (thought, action, action_input)
        """
        import re
        
        thought = None
        action = None
        action_input = None
        
        # Extract THOUGHT
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract ACTION
        action_match = re.search(r"ACTION:\s*(\w+)", response)
        if action_match:
            action = action_match.group(1).strip()
        
        # Extract ACTION_INPUT
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?=THOUGHT:|ACTION:|$)", response, re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()
        
        return thought, action, action_input

    async def stream_response(
        self,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response for a query.
        
        Yields response chunks as they're generated.
        
        Args:
            query: User query
            
        Yields:
            Response chunks
        """
        # Retrieve context first
        context = await self._retrieve_context(query)
        
        prompt = f"""Context:
{chr(10).join([r.content for r in context[:3]])}

Question: {query}

Please provide a helpful response based on the context above."""
        
        # Use Bedrock agent for streaming if available
        if self.settings.bedrock.agent_id:
            async for chunk in self.bedrock.invoke_agent(prompt):
                yield chunk
        else:
            # Fall back to non-streaming
            response = await self.bedrock.generate_text(
                prompt=prompt,
                system_prompt=self._build_system_prompt(),
            )
            yield response.get("content", "")

    def register_tool(self, tool: Tool) -> None:
        """
        Register a new tool with the agent.
        
        Args:
            tool: Tool to register
        """
        self.tools.register(tool)
        logger.info("Tool registered", tool_name=tool.name)

    def list_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return self.tools.list_tools()

