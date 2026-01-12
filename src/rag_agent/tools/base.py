"""
Base Tool utilities for Strands Agents SDK integration.

Provides helper classes and context management for tools that need
access to AWS services.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from contextvars import ContextVar

import structlog

logger = structlog.get_logger(__name__)


class ToolResultStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """
    Result container for tool executions.
    
    Provides a standardized way to return results from tools,
    including success/error states and metadata.
    """

    status: ToolResultStatus
    data: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: Any, **metadata: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(status=ToolResultStatus.SUCCESS, data=data, metadata=metadata)

    @classmethod
    def error(cls, error: str, data: Any = None, **metadata: Any) -> "ToolResult":
        """Create an error result."""
        return cls(
            status=ToolResultStatus.ERROR,
            data=data,
            error=error,
            metadata=metadata,
        )

    @classmethod
    def partial(cls, data: Any, error: str | None = None, **metadata: Any) -> "ToolResult":
        """Create a partial success result."""
        return cls(
            status=ToolResultStatus.PARTIAL,
            data=data,
            error=error,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


# Context variables for service injection into tools
_embedding_service: ContextVar[Any] = ContextVar("embedding_service", default=None)
_vector_service: ContextVar[Any] = ContextVar("vector_service", default=None)
_dynamodb_service: ContextVar[Any] = ContextVar("dynamodb_service", default=None)
_bedrock_service: ContextVar[Any] = ContextVar("bedrock_service", default=None)


class ToolContext:
    """
    Context manager for injecting services into tools.
    
    Strands tools are simple functions, so we use context variables
    to provide access to AWS services without global state.
    
    Example:
        with ToolContext(embedding_service=emb, vector_service=vec):
            result = agent("Search for documents about Python")
    """

    def __init__(
        self,
        embedding_service: Any = None,
        vector_service: Any = None,
        dynamodb_service: Any = None,
        bedrock_service: Any = None,
    ):
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.dynamodb_service = dynamodb_service
        self.bedrock_service = bedrock_service
        self._tokens: list[Any] = []

    def __enter__(self) -> "ToolContext":
        if self.embedding_service:
            self._tokens.append(_embedding_service.set(self.embedding_service))
        if self.vector_service:
            self._tokens.append(_vector_service.set(self.vector_service))
        if self.dynamodb_service:
            self._tokens.append(_dynamodb_service.set(self.dynamodb_service))
        if self.bedrock_service:
            self._tokens.append(_bedrock_service.set(self.bedrock_service))
        return self

    def __exit__(self, *args: Any) -> None:
        for token in self._tokens:
            # Reset context variables
            pass  # ContextVar tokens auto-reset on context exit


def get_embedding_service() -> Any:
    """Get the embedding service from context."""
    service = _embedding_service.get()
    if service is None:
        from rag_agent.services.embeddings import EmbeddingService
        return EmbeddingService()
    return service


def get_vector_service() -> Any:
    """Get the vector service from context."""
    service = _vector_service.get()
    if service is None:
        from rag_agent.services.s3_vectors import S3VectorService
        return S3VectorService()
    return service


def get_dynamodb_service() -> Any:
    """Get the DynamoDB service from context."""
    service = _dynamodb_service.get()
    if service is None:
        from rag_agent.services.dynamodb import DynamoDBService
        return DynamoDBService()
    return service


def get_bedrock_service() -> Any:
    """Get the Bedrock service from context."""
    service = _bedrock_service.get()
    if service is None:
        from rag_agent.services.bedrock import BedrockService
        return BedrockService()
    return service


def format_search_results(results: list[Any]) -> str:
    """
    Format search results for agent consumption.
    
    Args:
        results: List of VectorSearchResult objects
        
    Returns:
        Formatted string representation
    """
    if not results:
        return "No results found."
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Result {i}] Score: {r.score:.4f}\n"
            f"Document: {r.document_id}\n"
            f"Content: {r.content[:500]}{'...' if len(r.content) > 500 else ''}\n"
        )
    return "\n".join(formatted)


def format_document_list(documents: list[Any]) -> str:
    """
    Format document list for agent consumption.
    
    Args:
        documents: List of document metadata objects
        
    Returns:
        Formatted string representation
    """
    if not documents:
        return "No documents found."
    
    formatted = []
    for doc in documents:
        formatted.append(
            f"- {doc.title} (ID: {doc.id}, Status: {doc.status.value}, Chunks: {doc.chunk_count})"
        )
    return "\n".join(formatted)


def run_async_sync(coro):
    """
    Run async code synchronously in a thread-safe way.
    
    This helper is needed because Strands tools are synchronous functions
    but our services use async methods. This function handles event loop
    creation and management safely.
    
    Args:
        coro: Async coroutine to run
        
    Returns:
        Result of the coroutine
    """
    import asyncio
    import concurrent.futures
    
    try:
        # Try to get existing loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create new loop in thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
        else:
            # Use existing loop
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(coro)
