"""
Agent Tools module.

Defines tools that the RAG agent can use during execution,
using Strands Agents SDK @tool decorator pattern.
"""

from rag_agent.tools.base import ToolResult, ToolContext
from rag_agent.tools.search import vector_search, knowledge_base_search
from rag_agent.tools.document import get_document, list_documents, summarize_document
from rag_agent.tools.web import web_search, fetch_url

__all__ = [
    # Base utilities
    "ToolResult",
    "ToolContext",
    # Search tools
    "vector_search",
    "knowledge_base_search",
    # Document tools
    "get_document",
    "list_documents",
    "summarize_document",
    # Web tools
    "web_search",
    "fetch_url",
]


def get_all_tools() -> list:
    """
    Get all available RAG tools.
    
    Returns:
        List of tool functions for use with Strands Agent
    """
    return [
        vector_search,
        knowledge_base_search,
        get_document,
        list_documents,
        summarize_document,
        web_search,
        fetch_url,
    ]


def get_core_tools() -> list:
    """
    Get core RAG tools (without web tools).
    
    Returns:
        List of core tool functions
    """
    return [
        vector_search,
        get_document,
        list_documents,
        summarize_document,
    ]
