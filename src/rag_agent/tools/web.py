"""
Web Tools for external knowledge retrieval.

Provides web search and URL content fetching capabilities.
Uses Strands Agents SDK @tool decorator for tool definition.
"""

import re
import structlog
from strands import tool

from rag_agent.config import get_settings
from rag_agent.tools.base import run_async_sync

logger = structlog.get_logger(__name__)


@tool
def web_search(
    query: str,
    num_results: int = 5,
) -> str:
    """
    Search the web for current information.
    
    Use this when you need up-to-date information or when the knowledge base
    doesn't contain relevant data. This is useful for real-time information,
    recent events, or topics not covered in the document repository.
    
    Note: This requires a search API to be configured. Without configuration,
    it returns placeholder results.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        List of web search results with titles, snippets, and URLs
    """
    settings = get_settings()
    
    try:
        logger.info("Performing web search", query=query[:100], num_results=num_results)
        
        # Stub implementation - in production, integrate with a real search API
        # Example integrations: SerpAPI, Google Custom Search, Bing Search API
        #
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(
        #         "https://serpapi.com/search",
        #         params={"q": query, "api_key": api_key, "num": num_results},
        #     ) as response:
        #         data = await response.json()
        #         results = data.get("organic_results", [])
        
        # For now, return a placeholder indicating the tool is available
        logger.warning("Web search API not configured, returning placeholder")
        
        return (
            f"Web search for '{query}':\n\n"
            "Note: Web search API is not configured. To enable real web search, "
            "integrate with a search API (SerpAPI, Google Custom Search, etc.).\n\n"
            "Placeholder result:\n"
            f"- Title: Search result for '{query}'\n"
            "  URL: https://example.com\n"
            "  Snippet: Configure a search API for real results."
        )
        
    except Exception as e:
        logger.error("Web search failed", query=query, error=str(e))
        return f"Error: Web search failed - {str(e)}"


@tool
def fetch_url(
    url: str,
    max_length: int = 5000,
) -> str:
    """
    Fetch and extract text content from a URL.
    
    Use this when you need to read the content of a specific webpage.
    The tool fetches the page and extracts readable text content,
    stripping HTML tags and scripts.
    
    Args:
        url: The URL to fetch content from
        max_length: Maximum content length to return in characters (default: 5000)
        
    Returns:
        Extracted text content from the URL
    """
    import aiohttp
    
    try:
        logger.info("Fetching URL content", url=url)
        
        async def fetch():
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={"User-Agent": "RAG-Agent/1.0"},
                ) as response:
                    if response.status != 200:
                        return None, f"HTTP {response.status}"
                    return await response.text(), None
        
        # Run async code synchronously
        content, error = run_async_sync(fetch())
        
        if error:
            return f"Error: Failed to fetch URL - {error}"
        
        if not content:
            return "Error: No content received from URL"
        
        # Basic HTML stripping
        # Remove script and style elements
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)
        # Clean up whitespace
        content = re.sub(r"\s+", " ", content).strip()
        
        # Truncate if necessary
        truncated = False
        if len(content) > max_length:
            content = content[:max_length]
            truncated = True
        
        logger.info("URL content fetched", url=url, content_length=len(content))
        
        result = f"Content from {url}:\n\n{content}"
        if truncated:
            result += "\n\n... (content truncated)"
        
        return result
        
    except Exception as e:
        logger.error("URL content extraction failed", url=url, error=str(e))
        return f"Error: Failed to fetch URL - {str(e)}"
