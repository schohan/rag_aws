"""
Document Retrieval and Management Tools.

Tools for retrieving, listing, and managing documents in the RAG system.
Uses Strands Agents SDK @tool decorator for tool definition.
"""

import structlog
from strands import tool

from rag_agent.config import get_settings
from rag_agent.models import DocumentStatus
from rag_agent.tools.base import (
    get_dynamodb_service,
    get_vector_service,
    get_bedrock_service,
    format_document_list,
    run_async_sync,
)

logger = structlog.get_logger(__name__)


@tool
def get_document(
    document_id: str,
    include_content: bool = False,
) -> str:
    """
    Retrieve detailed information about a specific document.
    
    Use this when you need the full content or metadata of a document
    from the knowledge base.
    
    Args:
        document_id: The unique identifier of the document
        include_content: Whether to include the full document content (default: false)
        
    Returns:
        Document metadata and optionally full content
    """
    dynamodb_service = get_dynamodb_service()
    s3_service = get_vector_service()
    
    try:
        logger.info("Retrieving document metadata", document_id=document_id)
        
        # Run async code synchronously
        metadata = run_async_sync(dynamodb_service.get_document_metadata(document_id))
        
        if not metadata:
            return f"Error: Document not found: {document_id}"
        
        result_lines = [
            f"Document ID: {metadata.id}",
            f"Title: {metadata.title}",
            f"Source: {metadata.source}",
            f"Content Type: {metadata.content_type}",
            f"Status: {metadata.status.value}",
            f"Chunk Count: {metadata.chunk_count}",
            f"Created: {metadata.created_at}",
            f"Updated: {metadata.updated_at}",
        ]
        
        if metadata.indexed_at:
            result_lines.append(f"Indexed: {metadata.indexed_at}")
        
        if metadata.extra_metadata:
            result_lines.append(f"Extra Metadata: {metadata.extra_metadata}")
        
        # Optionally get full content
        if include_content and metadata.s3_key:
            logger.info("Retrieving document content", document_id=document_id)
            content = run_async_sync(s3_service.get_document(document_id))
            
            if content:
                content_str = content.decode("utf-8")
                result_lines.append(f"\nContent:\n{content_str[:5000]}")
                if len(content_str) > 5000:
                    result_lines.append("... (content truncated)")
        
        logger.info("Document retrieved successfully", document_id=document_id)
        return "\n".join(result_lines)
        
    except Exception as e:
        logger.error("Document retrieval failed", document_id=document_id, error=str(e))
        return f"Error: Failed to retrieve document - {str(e)}"


@tool
def list_documents(
    status: str = "indexed",
    limit: int = 20,
) -> str:
    """
    List documents in the knowledge base.
    
    Use this to find available documents or check their processing status.
    
    Args:
        status: Filter by document status. Options: "pending", "processing", "indexed", "failed"
        limit: Maximum number of documents to return (default: 20)
        
    Returns:
        List of document summaries with their IDs, titles, and status
    """
    import asyncio
    
    dynamodb_service = get_dynamodb_service()
    
    try:
        # Parse status
        try:
            doc_status = DocumentStatus(status)
        except ValueError:
            doc_status = DocumentStatus.INDEXED
        
        logger.info("Listing documents", status=doc_status.value, limit=limit)
        
        # Run async code synchronously
        documents = run_async_sync(
            dynamodb_service.list_documents_by_status(status=doc_status, limit=limit)
        )
        
        logger.info("Documents listed", count=len(documents))
        
        if not documents:
            return f"No documents found with status: {doc_status.value}"
        
        header = f"Found {len(documents)} document(s) with status '{doc_status.value}':\n"
        return header + format_document_list(documents)
        
    except Exception as e:
        logger.error("Document listing failed", error=str(e))
        return f"Error: Failed to list documents - {str(e)}"


@tool
def summarize_document(
    document_id: str,
    max_length: int = 200,
) -> str:
    """
    Generate a summary of a specific document.
    
    Use this when you need a quick overview of a document's content.
    The summary is generated using the LLM based on the document content.
    
    Args:
        document_id: The unique identifier of the document to summarize
        max_length: Approximate maximum length of the summary in words (default: 200)
        
    Returns:
        A concise summary of the document
    """
    dynamodb_service = get_dynamodb_service()
    s3_service = get_vector_service()
    bedrock_service = get_bedrock_service()
    
    try:
        logger.info("Retrieving document for summarization", document_id=document_id)
        
        # Get document content
        content = run_async_sync(s3_service.get_document(document_id))
        
        if not content:
            return f"Error: Document content not found: {document_id}"
        
        content_str = content.decode("utf-8")
        
        # Get metadata for context
        metadata = run_async_sync(dynamodb_service.get_document_metadata(document_id))
        
        # Generate summary using LLM
        logger.info("Generating document summary", document_id=document_id)
        prompt = f"""Please provide a concise summary of the following document in approximately {max_length} words.

Document Title: {metadata.title if metadata else 'Unknown'}
Source: {metadata.source if metadata else 'Unknown'}

Content:
{content_str[:10000]}

Summary:"""
        
        response = run_async_sync(
            bedrock_service.generate_text(
                prompt=prompt,
                system_prompt="You are a helpful assistant that creates clear, accurate document summaries.",
            )
        )
        
        summary = response.get("content", "")
        
        logger.info("Summary generated", document_id=document_id, summary_length=len(summary))
        
        title = metadata.title if metadata else "Unknown"
        return f"Summary of '{title}' (ID: {document_id}):\n\n{summary}"
        
    except Exception as e:
        logger.error("Summarization failed", document_id=document_id, error=str(e))
        return f"Error: Failed to summarize document - {str(e)}"
