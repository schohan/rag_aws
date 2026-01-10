"""
Document Retrieval and Management Tools.

Tools for retrieving, listing, and managing documents in the RAG system.
"""

from typing import Any

import structlog

from rag_agent.config import get_settings
from rag_agent.services.dynamodb import DynamoDBService
from rag_agent.services.s3_vectors import S3VectorService
from rag_agent.models import DocumentStatus
from rag_agent.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

logger = structlog.get_logger(__name__)


class DocumentRetrievalTool(Tool):
    """
    Tool for retrieving document metadata and content.
    
    Fetches document information from DynamoDB and optionally
    retrieves full content from S3.
    """

    def __init__(
        self,
        dynamodb_service: DynamoDBService | None = None,
        s3_service: S3VectorService | None = None,
    ):
        """Initialize the document retrieval tool."""
        self.settings = get_settings()
        self.dynamodb_service = dynamodb_service or DynamoDBService()
        self.s3_service = s3_service or S3VectorService()

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="get_document",
            description=(
                "Retrieve detailed information about a specific document. "
                "Use this when you need the full content or metadata of a document."
            ),
            parameters=[
                ToolParameter(
                    name="document_id",
                    description="The unique identifier of the document",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="include_content",
                    description="Whether to include the full document content (default: false)",
                    type="boolean",
                    required=False,
                    default=False,
                ),
            ],
            returns="Document metadata and optionally full content",
            examples=[
                {"document_id": "doc-123-456"},
                {"document_id": "doc-789", "include_content": True},
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the document retrieval.
        
        Args:
            document_id: Document identifier
            include_content: Whether to include full content
            
        Returns:
            ToolResult with document information
        """
        document_id = kwargs.get("document_id")
        if not document_id:
            return ToolResult.error("Document ID is required")

        include_content = kwargs.get("include_content", False)

        try:
            # Get document metadata
            logger.info("Retrieving document metadata", document_id=document_id)
            metadata = await self.dynamodb_service.get_document_metadata(document_id)

            if not metadata:
                return ToolResult.error(f"Document not found: {document_id}")

            result = {
                "id": metadata.id,
                "title": metadata.title,
                "source": metadata.source,
                "content_type": metadata.content_type,
                "status": metadata.status.value,
                "chunk_count": metadata.chunk_count,
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
                "indexed_at": metadata.indexed_at,
                "extra_metadata": metadata.extra_metadata,
            }

            # Optionally get full content
            if include_content and metadata.s3_key:
                logger.info("Retrieving document content", document_id=document_id)
                content = await self.s3_service.get_document(document_id)
                if content:
                    result["content"] = content.decode("utf-8")

            logger.info("Document retrieved successfully", document_id=document_id)
            return ToolResult.success(data=result)

        except Exception as e:
            logger.error("Document retrieval failed", document_id=document_id, error=str(e))
            return ToolResult.error(f"Failed to retrieve document: {str(e)}")


class ListDocumentsTool(Tool):
    """
    Tool for listing documents in the knowledge base.
    
    Supports filtering by status and pagination.
    """

    def __init__(self, dynamodb_service: DynamoDBService | None = None):
        """Initialize the list documents tool."""
        self.settings = get_settings()
        self.dynamodb_service = dynamodb_service or DynamoDBService()

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="list_documents",
            description=(
                "List documents in the knowledge base. "
                "Use this to find available documents or check their processing status."
            ),
            parameters=[
                ToolParameter(
                    name="status",
                    description="Filter by document status",
                    type="string",
                    required=False,
                    enum=["pending", "processing", "indexed", "failed"],
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum number of documents to return (default: 20)",
                    type="integer",
                    required=False,
                    default=20,
                ),
            ],
            returns="List of document summaries",
            examples=[
                {},
                {"status": "indexed", "limit": 10},
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the document listing.
        
        Args:
            status: Optional status filter
            limit: Maximum results
            
        Returns:
            ToolResult with document list
        """
        status_str = kwargs.get("status")
        limit = kwargs.get("limit", 20)

        try:
            # Default to indexed documents if no status specified
            status = DocumentStatus(status_str) if status_str else DocumentStatus.INDEXED

            logger.info("Listing documents", status=status.value, limit=limit)
            documents = await self.dynamodb_service.list_documents_by_status(
                status=status,
                limit=limit,
            )

            result = [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "status": doc.status.value,
                    "chunk_count": doc.chunk_count,
                    "created_at": doc.created_at,
                }
                for doc in documents
            ]

            logger.info("Documents listed", count=len(result))
            return ToolResult.success(
                data=result,
                total_count=len(result),
                status_filter=status.value,
            )

        except Exception as e:
            logger.error("Document listing failed", error=str(e))
            return ToolResult.error(f"Failed to list documents: {str(e)}")


class DocumentSummaryTool(Tool):
    """
    Tool for getting a summary of a document.
    
    Uses the LLM to generate a concise summary of document content.
    """

    def __init__(
        self,
        dynamodb_service: DynamoDBService | None = None,
        s3_service: S3VectorService | None = None,
    ):
        """Initialize the document summary tool."""
        from rag_agent.services.bedrock import BedrockService

        self.settings = get_settings()
        self.dynamodb_service = dynamodb_service or DynamoDBService()
        self.s3_service = s3_service or S3VectorService()
        self.bedrock_service = BedrockService()

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="summarize_document",
            description=(
                "Generate a summary of a specific document. "
                "Use this when you need a quick overview of a document's content."
            ),
            parameters=[
                ToolParameter(
                    name="document_id",
                    description="The unique identifier of the document to summarize",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="max_length",
                    description="Approximate maximum length of the summary in words (default: 200)",
                    type="integer",
                    required=False,
                    default=200,
                ),
            ],
            returns="A concise summary of the document",
            examples=[
                {"document_id": "doc-123"},
                {"document_id": "doc-456", "max_length": 100},
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the document summarization.
        
        Args:
            document_id: Document identifier
            max_length: Maximum summary length in words
            
        Returns:
            ToolResult with document summary
        """
        document_id = kwargs.get("document_id")
        if not document_id:
            return ToolResult.error("Document ID is required")

        max_length = kwargs.get("max_length", 200)

        try:
            # Get document content
            logger.info("Retrieving document for summarization", document_id=document_id)
            content = await self.s3_service.get_document(document_id)

            if not content:
                return ToolResult.error(f"Document content not found: {document_id}")

            content_str = content.decode("utf-8")

            # Get metadata for context
            metadata = await self.dynamodb_service.get_document_metadata(document_id)

            # Generate summary using LLM
            logger.info("Generating document summary", document_id=document_id)
            prompt = f"""Please provide a concise summary of the following document in approximately {max_length} words.

Document Title: {metadata.title if metadata else 'Unknown'}
Source: {metadata.source if metadata else 'Unknown'}

Content:
{content_str[:10000]}  # Limit content to avoid token limits

Summary:"""

            response = await self.bedrock_service.generate_text(
                prompt=prompt,
                system_prompt="You are a helpful assistant that creates clear, accurate document summaries.",
            )

            summary = response.get("content", "")

            logger.info("Summary generated", document_id=document_id, summary_length=len(summary))

            return ToolResult.success(
                data={
                    "document_id": document_id,
                    "title": metadata.title if metadata else "Unknown",
                    "summary": summary,
                },
            )

        except Exception as e:
            logger.error("Summarization failed", document_id=document_id, error=str(e))
            return ToolResult.error(f"Failed to summarize document: {str(e)}")

