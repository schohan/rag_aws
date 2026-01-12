"""
Vector Search Tools for RAG operations.

Provides semantic search capabilities using S3 Vectors or Bedrock Knowledge Bases.
Uses Strands Agents SDK @tool decorator for tool definition.
"""

import structlog
from strands import tool

from rag_agent.config import get_settings
from rag_agent.tools.base import (
    get_embedding_service,
    get_vector_service,
    get_bedrock_service,
    format_search_results,
    run_async_sync,
)

logger = structlog.get_logger(__name__)


@tool
def vector_search(
    query: str,
    top_k: int = 5,
    filter_document_id: str | None = None,
) -> str:
    """
    Search for relevant documents in the knowledge base using semantic similarity.
    
    Use this tool when you need to find information related to a specific topic or question.
    It performs embedding-based similarity search across all indexed documents.
    
    Args:
        query: The search query to find relevant documents
        top_k: Number of results to return (default: 5)
        filter_document_id: Optional document ID to filter results to a specific document
        
    Returns:
        Formatted list of relevant document chunks with similarity scores
    """
    settings = get_settings()
    embedding_service = get_embedding_service()
    vector_service = get_vector_service()
    
    top_k = top_k or settings.vector.top_k
    
    try:
        logger.info("Generating query embedding", query=query[:100])
        
        # Run async code synchronously (Strands tools are sync)
        query_embedding = run_async_sync(embedding_service.generate_query_embedding(query))
        
        if not query_embedding:
            return "Error: Failed to generate query embedding"
        
        # Build filters if specified
        filters = None
        if filter_document_id:
            filters = {"document_id": filter_document_id}
        
        logger.info("Performing vector search", top_k=top_k)
        
        # Perform vector search
        results = run_async_sync(
            vector_service.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        )
        
        logger.info(
            "Vector search completed",
            results_count=len(results),
            query=query[:100],
        )
        
        if not results:
            return (
                "No results found matching the query. "
                "This could mean:\n"
                "- No documents have been indexed yet\n"
                "- The query doesn't match any indexed content\n"
                "- Try rephrasing your query or checking if documents were successfully ingested"
            )
        
        return format_search_results(results)
        
    except Exception as e:
        logger.error("Vector search failed", error=str(e))
        return f"Error: Vector search failed - {str(e)}"


@tool
def knowledge_base_search(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search the managed Bedrock Knowledge Base for relevant information.
    
    Use this tool when you need authoritative answers from the document repository.
    This uses Amazon Bedrock's managed knowledge base for optimized retrieval.
    
    Args:
        query: The question or topic to search for
        top_k: Number of results to return (default: 5)
        
    Returns:
        Relevant passages from the knowledge base with source information
    """
    settings = get_settings()
    bedrock_service = get_bedrock_service()
    
    if not settings.bedrock.knowledge_base_id:
        return "Error: Knowledge base ID not configured"
    
    top_k = top_k or settings.vector.top_k
    
    try:
        logger.info("Querying knowledge base", query=query[:100])
        
        # Run async code synchronously
        results = run_async_sync(bedrock_service.query_knowledge_base(query=query, top_k=top_k))
        
        logger.info("Knowledge base search completed", results_count=len(results))
        
        return format_search_results(results)
        
    except Exception as e:
        logger.error("Knowledge base search failed", error=str(e))
        return f"Error: Knowledge base search failed - {str(e)}"
