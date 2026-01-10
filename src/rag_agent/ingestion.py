"""
Document Ingestion Pipeline.

Handles document processing, chunking, embedding generation,
and storage for the RAG knowledge base.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

import structlog
import tiktoken

from rag_agent.config import Settings, get_settings
from rag_agent.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    ChunkMetadata,
)
from rag_agent.services.dynamodb import DynamoDBService
from rag_agent.services.embeddings import EmbeddingService
from rag_agent.services.s3_vectors import S3VectorService

logger = structlog.get_logger(__name__)


class TextChunker:
    """
    Text chunking utility for document processing.
    
    Splits text into overlapping chunks suitable for embedding
    and retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            encoding_name: Tiktoken encoding name for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def chunk_text(
        self,
        text: str,
        document_id: UUID,
    ) -> list[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full document text
            document_id: Parent document ID
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = ChunkMetadata(
                    chunk_id=uuid4(),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    token_count=self.count_tokens(current_chunk),
                )
                chunks.append(DocumentChunk(metadata=chunk_metadata, content=current_chunk))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_start = current_start + overlap_start
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_metadata = ChunkMetadata(
                chunk_id=uuid4(),
                document_id=document_id,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                token_count=self.count_tokens(current_chunk),
            )
            chunks.append(DocumentChunk(metadata=chunk_metadata, content=current_chunk))
        
        logger.info(
            "Text chunked",
            document_id=str(document_id),
            total_chars=len(text),
            chunk_count=len(chunks),
        )
        
        return chunks


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents into the RAG knowledge base.
    
    Handles the full ingestion workflow:
    1. Document parsing
    2. Text chunking
    3. Embedding generation
    4. Vector storage
    5. Metadata storage
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize the ingestion pipeline."""
        self.settings = settings or get_settings()
        
        self.chunker = TextChunker(
            chunk_size=self.settings.chunking.size,
            chunk_overlap=self.settings.chunking.overlap,
        )
        self.embeddings = EmbeddingService(self.settings)
        self.vectors = S3VectorService(self.settings)
        self.dynamodb = DynamoDBService(self.settings)

    async def ingest_text(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Ingest a text document.
        
        Args:
            content: Document text content
            title: Document title
            source: Source identifier (URL, file path, etc.)
            metadata: Optional additional metadata
            
        Returns:
            Processed document with chunks and embeddings
        """
        document = Document(
            title=title,
            content=content,
            source=source,
            content_type="text/plain",
            metadata=metadata or {},
        )
        
        return await self._process_document(document)

    async def ingest_file(
        self,
        file_path: Path | str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Ingest a file document.
        
        Args:
            file_path: Path to the file
            title: Optional title (defaults to filename)
            metadata: Optional additional metadata
            
        Returns:
            Processed document
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine content type
        content_type = self._get_content_type(file_path)
        
        # Read and parse content
        content = await self._read_file(file_path, content_type)
        
        document = Document(
            title=title or file_path.stem,
            content=content,
            source=str(file_path),
            content_type=content_type,
            metadata=metadata or {},
        )
        
        return await self._process_document(document)

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type from file extension."""
        extension = file_path.suffix.lower()
        content_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".html": "text/html",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return content_types.get(extension, "text/plain")

    async def _read_file(self, file_path: Path, content_type: str) -> str:
        """Read and parse file content."""
        if content_type == "text/plain" or content_type == "text/markdown":
            return file_path.read_text(encoding="utf-8")
        
        elif content_type == "application/json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        
        elif content_type == "text/html":
            # Basic HTML text extraction
            import re
            html = file_path.read_text(encoding="utf-8")
            # Remove script and style elements
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Clean up whitespace
            return re.sub(r"\s+", " ", text).strip()
        
        elif content_type == "application/pdf":
            # PDF extraction would require additional library like pypdf
            raise NotImplementedError("PDF parsing not implemented. Install pypdf and implement.")
        
        else:
            # Try to read as text
            return file_path.read_text(encoding="utf-8")

    async def _process_document(self, document: Document) -> Document:
        """
        Process a document through the ingestion pipeline.
        
        Args:
            document: Document to process
            
        Returns:
            Processed document with status updated
        """
        logger.info("Processing document", document_id=str(document.id), title=document.title)
        
        try:
            # Update status to processing
            document.status = DocumentStatus.PROCESSING
            await self._save_metadata(document)
            
            # Chunk the document
            document.chunks = self.chunker.chunk_text(document.content, document.id)
            
            if not document.chunks:
                logger.warning("No chunks generated", document_id=str(document.id))
                document.status = DocumentStatus.FAILED
                await self._save_metadata(document)
                return document
            
            # Generate embeddings for each chunk
            logger.info(
                "Generating embeddings",
                document_id=str(document.id),
                chunk_count=len(document.chunks),
            )
            
            chunk_texts = [chunk.content for chunk in document.chunks]
            embeddings = await self.embeddings.generate_embeddings_batch(chunk_texts)
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(document.chunks, embeddings):
                chunk.embedding = embedding if embedding else None
            
            # Store vectors
            logger.info("Storing vectors", document_id=str(document.id))
            await self.vectors.store_vectors(document.chunks, document.id)
            
            # Upload document content to S3
            s3_key = await self.vectors.upload_document(
                document_id=str(document.id),
                content=document.content.encode("utf-8"),
                content_type=document.content_type,
            )
            
            # Update status to indexed
            document.status = DocumentStatus.INDEXED
            document.indexed_at = datetime.utcnow()
            await self._save_metadata(document, s3_key=s3_key)
            
            logger.info(
                "Document indexed successfully",
                document_id=str(document.id),
                chunk_count=len(document.chunks),
            )
            
            return document
            
        except Exception as e:
            logger.error(
                "Document processing failed",
                document_id=str(document.id),
                error=str(e),
            )
            document.status = DocumentStatus.FAILED
            await self._save_metadata(document)
            raise

    async def _save_metadata(
        self,
        document: Document,
        s3_key: str | None = None,
    ) -> None:
        """Save document metadata to DynamoDB."""
        metadata = DocumentMetadata(
            id=str(document.id),
            title=document.title,
            source=document.source,
            content_type=document.content_type,
            status=document.status,
            chunk_count=len(document.chunks),
            token_count=sum(c.metadata.token_count or 0 for c in document.chunks),
            created_at=document.created_at.isoformat(),
            updated_at=document.updated_at.isoformat(),
            indexed_at=document.indexed_at.isoformat() if document.indexed_at else None,
            s3_key=s3_key,
            extra_metadata=document.metadata,
        )
        
        await self.dynamodb.save_document_metadata(metadata)

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        logger.info("Deleting document", document_id=document_id)
        
        try:
            # Get document metadata to find vector IDs
            metadata = await self.dynamodb.get_document_metadata(document_id)
            
            if not metadata:
                logger.warning("Document not found", document_id=document_id)
                return False
            
            # Delete from DynamoDB
            await self.dynamodb.delete_document(document_id)
            
            logger.info("Document deleted", document_id=document_id)
            return True
            
        except Exception as e:
            logger.error("Document deletion failed", document_id=document_id, error=str(e))
            raise

    async def reindex_document(self, document_id: str) -> Document | None:
        """
        Reindex an existing document.
        
        Args:
            document_id: Document ID to reindex
            
        Returns:
            Reindexed document or None if not found
        """
        logger.info("Reindexing document", document_id=document_id)
        
        # Get document content from S3
        content = await self.vectors.get_document(document_id)
        
        if not content:
            logger.warning("Document content not found", document_id=document_id)
            return None
        
        # Get metadata
        metadata = await self.dynamodb.get_document_metadata(document_id)
        
        if not metadata:
            logger.warning("Document metadata not found", document_id=document_id)
            return None
        
        # Create document and reprocess
        document = Document(
            id=UUID(document_id),
            title=metadata.title,
            content=content.decode("utf-8"),
            source=metadata.source,
            content_type=metadata.content_type,
            metadata=metadata.extra_metadata,
        )
        
        return await self._process_document(document)

