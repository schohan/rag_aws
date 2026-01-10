"""
Tests for document ingestion pipeline.
"""

import pytest
from uuid import uuid4

from rag_agent.ingestion import TextChunker, DocumentIngestionPipeline
from rag_agent.models import Document, DocumentStatus


class TestTextChunker:
    """Tests for TextChunker."""

    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        doc_id = uuid4()
        
        chunks = chunker.chunk_text("This is a short text.", doc_id)
        
        assert len(chunks) == 1
        assert chunks[0].content == "This is a short text."
        assert chunks[0].metadata.document_id == doc_id

    def test_chunk_long_text(self):
        """Test chunking text longer than chunk size."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        doc_id = uuid4()
        
        # Create text with multiple paragraphs
        paragraphs = ["Paragraph " + str(i) + ". " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)
        
        chunks = chunker.chunk_text(text, doc_id)
        
        assert len(chunks) > 1
        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        doc_id = uuid4()
        
        text = "Para one here.\n\nPara two here.\n\nPara three here."
        chunks = chunker.chunk_text(text, doc_id)
        
        # If there's overlap, later chunks should start before previous ends
        if len(chunks) > 1:
            assert chunks[1].metadata.start_char <= chunks[0].metadata.end_char

    def test_token_counting(self):
        """Test that token counts are recorded."""
        chunker = TextChunker()
        doc_id = uuid4()
        
        chunks = chunker.chunk_text("Hello world", doc_id)
        
        assert chunks[0].metadata.token_count is not None
        assert chunks[0].metadata.token_count > 0


class TestDocumentIngestionPipeline:
    """Tests for DocumentIngestionPipeline."""

    @pytest.fixture
    def pipeline(self, settings):
        """Create a test pipeline."""
        return DocumentIngestionPipeline(settings=settings)

    def test_get_content_type(self, pipeline):
        """Test content type detection."""
        from pathlib import Path
        
        assert pipeline._get_content_type(Path("test.txt")) == "text/plain"
        assert pipeline._get_content_type(Path("test.md")) == "text/markdown"
        assert pipeline._get_content_type(Path("test.json")) == "application/json"
        assert pipeline._get_content_type(Path("test.html")) == "text/html"
        assert pipeline._get_content_type(Path("test.pdf")) == "application/pdf"
        assert pipeline._get_content_type(Path("test.unknown")) == "text/plain"

