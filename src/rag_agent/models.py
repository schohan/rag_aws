"""
Data models for the RAG Agent application.

Defines Pydantic models for documents, embeddings, queries, and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int | None = None


class DocumentChunk(BaseModel):
    """A chunk of a document with its embedding."""

    metadata: ChunkMetadata
    content: str
    embedding: list[float] | None = None


class Document(BaseModel):
    """Document model for ingestion and storage."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    content: str
    source: str
    content_type: str = "text/plain"
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: datetime | None = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class DocumentMetadata(BaseModel):
    """Lightweight document metadata for DynamoDB storage."""

    id: str
    title: str
    source: str
    content_type: str
    status: DocumentStatus
    chunk_count: int = 0
    token_count: int | None = None
    created_at: str
    updated_at: str
    indexed_at: str | None = None
    s3_key: str | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""

    document_id: str
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    include_sources: bool = True
    filters: dict[str, Any] = Field(default_factory=dict)


class SourceReference(BaseModel):
    """Reference to a source document."""

    document_id: str
    title: str
    source: str
    chunk_id: str
    relevance_score: float
    snippet: str


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    query: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    model_id: str
    latency_ms: float
    token_usage: dict[str, int] = Field(default_factory=dict)


class AgentAction(BaseModel):
    """An action taken by the agent."""

    action_type: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """Current state of the agent during execution."""

    session_id: UUID = Field(default_factory=uuid4)
    query: str
    actions: list[AgentAction] = Field(default_factory=list)
    context: list[VectorSearchResult] = Field(default_factory=list)
    final_answer: str | None = None
    is_complete: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """A message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A conversation session with the agent."""

    id: UUID = Field(default_factory=uuid4)
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

