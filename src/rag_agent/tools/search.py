"""
Vector Search Tool for RAG operations.

Provides semantic search capabilities using S3 Vectors or Bedrock Knowledge Bases.
"""

from typing import Any

import structlog

from rag_agent.config import get_settings
from rag_agent.services.embeddings import EmbeddingService
from rag_agent.services.s3_vectors import S3VectorService
from rag_agent.services.bedrock import BedrockService
from rag_agent.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

logger = structlog.get_logger(__name__)


class VectorSearchTool(Tool):
    """
    Tool for performing semantic vector search.
    
    Searches the vector store for documents similar to the query
    using embedding-based similarity.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_service: S3VectorService | None = None,
    ):
        """Initialize the vector search tool."""
        self.settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_service = vector_service or S3VectorService()

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="vector_search",
            description=(
                "Search for relevant documents in the knowledge base using semantic similarity. "
                "Use this tool when you need to find information related to a specific topic or question."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    description="The search query to find relevant documents",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    description="Number of results to return (default: 5)",
                    type="integer",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="filter_document_id",
                    description="Optional document ID to filter results",
                    type="string",
                    required=False,
                ),
            ],
            returns="List of relevant document chunks with similarity scores",
            examples=[
                {
                    "query": "What is the company's refund policy?",
                    "top_k": 3,
                },
                {
                    "query": "How do I reset my password?",
                },
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the vector search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filter_document_id: Optional document filter
            
        Returns:
            ToolResult with search results
        """
        query = kwargs.get("query")
        if not query:
            return ToolResult.error("Query parameter is required")

        top_k = kwargs.get("top_k", self.settings.vector.top_k)
        filter_document_id = kwargs.get("filter_document_id")

        try:
            # Generate query embedding
            logger.info("Generating query embedding", query=query[:100])
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            if not query_embedding:
                return ToolResult.error("Failed to generate query embedding")

            # Build filters if specified
            filters = None
            if filter_document_id:
                filters = {"document_id": filter_document_id}

            # Perform vector search
            logger.info("Performing vector search", top_k=top_k)
            results = await self.vector_service.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters,
            )

            # Format results for agent consumption
            formatted_results = [
                {
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "content": r.content,
                    "metadata": r.metadata,
                }
                for r in results
            ]

            logger.info("Vector search completed", results_count=len(formatted_results))

            return ToolResult.success(
                data=formatted_results,
                query=query,
                results_count=len(formatted_results),
            )

        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return ToolResult.error(f"Vector search failed: {str(e)}")


class KnowledgeBaseSearchTool(Tool):
    """
    Tool for searching Bedrock Knowledge Bases.
    
    Uses Amazon Bedrock's managed knowledge base for retrieval.
    """

    def __init__(self, bedrock_service: BedrockService | None = None):
        """Initialize the knowledge base search tool."""
        self.settings = get_settings()
        self.bedrock_service = bedrock_service or BedrockService()

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="knowledge_base_search",
            description=(
                "Search the managed knowledge base for relevant information. "
                "Use this tool when you need authoritative answers from the document repository."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    description="The question or topic to search for",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    description="Number of results to return (default: 5)",
                    type="integer",
                    required=False,
                    default=5,
                ),
            ],
            returns="Relevant passages from the knowledge base with source information",
            examples=[
                {
                    "query": "What are the system requirements?",
                    "top_k": 5,
                },
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the knowledge base search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            ToolResult with search results
        """
        query = kwargs.get("query")
        if not query:
            return ToolResult.error("Query parameter is required")

        top_k = kwargs.get("top_k", self.settings.vector.top_k)

        if not self.settings.bedrock.knowledge_base_id:
            return ToolResult.error("Knowledge base ID not configured")

        try:
            logger.info("Querying knowledge base", query=query[:100])
            results = await self.bedrock_service.query_knowledge_base(
                query=query,
                top_k=top_k,
            )

            formatted_results = [
                {
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "content": r.content,
                    "metadata": r.metadata,
                }
                for r in results
            ]

            logger.info("Knowledge base search completed", results_count=len(formatted_results))

            return ToolResult.success(
                data=formatted_results,
                query=query,
                results_count=len(formatted_results),
            )

        except Exception as e:
            logger.error("Knowledge base search failed", error=str(e))
            return ToolResult.error(f"Knowledge base search failed: {str(e)}")

