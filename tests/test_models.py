"""
Tests for data models.
"""

import pytest
from datetime import datetime
from uuid import UUID

from rag_agent.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    ChunkMetadata,
    QueryRequest,
    QueryResponse,
    VectorSearchResult,
    AgentState,
    AgentAction,
)


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            title="Test Doc",
            content="Test content",
            source="test.txt",
        )
        
        assert doc.title == "Test Doc"
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert doc.status == DocumentStatus.PENDING
        assert isinstance(doc.id, UUID)
        assert isinstance(doc.created_at, datetime)

    def test_document_with_metadata(self):
        """Test document with custom metadata."""
        doc = Document(
            title="Test",
            content="Content",
            source="source",
            metadata={"key": "value"},
        )
        
        assert doc.metadata == {"key": "value"}

    def test_document_chunk_creation(self):
        """Test creating a document chunk."""
        from uuid import uuid4
        
        doc_id = uuid4()
        chunk_metadata = ChunkMetadata(
            document_id=doc_id,
            chunk_index=0,
            start_char=0,
            end_char=100,
            token_count=25,
        )
        
        chunk = DocumentChunk(
            metadata=chunk_metadata,
            content="Test chunk content",
        )
        
        assert chunk.content == "Test chunk content"
        assert chunk.metadata.document_id == doc_id
        assert chunk.metadata.chunk_index == 0


class TestQueryModels:
    """Tests for query-related models."""

    def test_query_request_defaults(self):
        """Test QueryRequest with defaults."""
        request = QueryRequest(query="Test question")
        
        assert request.query == "Test question"
        assert request.top_k == 5
        assert request.similarity_threshold == 0.7
        assert request.include_sources is True

    def test_query_request_validation(self):
        """Test QueryRequest validation."""
        with pytest.raises(ValueError):
            QueryRequest(query="")  # Empty query
        
        with pytest.raises(ValueError):
            QueryRequest(query="Test", similarity_threshold=1.5)  # Invalid threshold

    def test_query_response(self):
        """Test QueryResponse creation."""
        response = QueryResponse(
            query="Test",
            answer="Test answer",
            model_id="test-model",
            latency_ms=100.5,
        )
        
        assert response.answer == "Test answer"
        assert response.latency_ms == 100.5
        assert response.sources == []


class TestVectorSearchResult:
    """Tests for VectorSearchResult model."""

    def test_vector_search_result(self):
        """Test creating a search result."""
        result = VectorSearchResult(
            document_id="doc-123",
            chunk_id="chunk-456",
            score=0.95,
            content="Matching content",
            metadata={"source": "test"},
        )
        
        assert result.document_id == "doc-123"
        assert result.score == 0.95
        assert result.metadata["source"] == "test"


class TestAgentModels:
    """Tests for agent-related models."""

    def test_agent_state_creation(self):
        """Test creating agent state."""
        state = AgentState(query="Test query")
        
        assert state.query == "Test query"
        assert state.actions == []
        assert state.is_complete is False
        assert state.final_answer is None

    def test_agent_action(self):
        """Test agent action model."""
        action = AgentAction(
            action_type="tool_use",
            tool_name="vector_search",
            tool_input={"query": "test"},
            observation="Found 3 results",
        )
        
        assert action.tool_name == "vector_search"
        assert action.observation == "Found 3 results"
        assert isinstance(action.timestamp, datetime)

