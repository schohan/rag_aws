"""
Web Search Tool for external knowledge retrieval.

Provides web search capabilities for questions that may require
up-to-date information beyond the knowledge base.
"""

from typing import Any

import aiohttp
import structlog

from rag_agent.config import get_settings
from rag_agent.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

logger = structlog.get_logger(__name__)


class WebSearchTool(Tool):
    """
    Tool for searching the web for external information.
    
    Can be used when the knowledge base doesn't contain relevant
    information or when real-time data is needed.
    
    Note: This is a stub implementation. In production, integrate with
    a real search API (Google, Bing, SerpAPI, etc.)
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the web search tool."""
        self.settings = get_settings()
        self.api_key = api_key

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="web_search",
            description=(
                "Search the web for current information. "
                "Use this when you need up-to-date information or when the "
                "knowledge base doesn't contain relevant data."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    description="The search query",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    description="Number of results to return (default: 5)",
                    type="integer",
                    required=False,
                    default=5,
                ),
            ],
            returns="List of web search results with titles, snippets, and URLs",
            examples=[
                {"query": "latest AI developments 2024"},
                {"query": "Python best practices", "num_results": 3},
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the web search.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            ToolResult with search results
        """
        query = kwargs.get("query")
        if not query:
            return ToolResult.error("Query parameter is required")

        num_results = kwargs.get("num_results", 5)

        try:
            logger.info("Performing web search", query=query[:100], num_results=num_results)

            # Stub implementation - in production, integrate with a real search API
            # Example with SerpAPI:
            # async with aiohttp.ClientSession() as session:
            #     async with session.get(
            #         "https://serpapi.com/search",
            #         params={
            #             "q": query,
            #             "api_key": self.api_key,
            #             "num": num_results,
            #         },
            #     ) as response:
            #         data = await response.json()
            #         results = data.get("organic_results", [])

            # For now, return a placeholder indicating the tool is available
            results = [
                {
                    "title": f"Search result for: {query}",
                    "snippet": "This is a placeholder. Configure a search API for real results.",
                    "url": "https://example.com",
                    "position": 1,
                }
            ]

            logger.info("Web search completed", results_count=len(results))

            return ToolResult.partial(
                data=results,
                error="Web search API not configured. Using placeholder results.",
                query=query,
            )

        except Exception as e:
            logger.error("Web search failed", query=query, error=str(e))
            return ToolResult.error(f"Web search failed: {str(e)}")


class URLContentTool(Tool):
    """
    Tool for fetching and extracting content from a URL.
    
    Retrieves webpage content for analysis or incorporation
    into responses.
    """

    def __init__(self):
        """Initialize the URL content tool."""
        self.settings = get_settings()

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="fetch_url",
            description=(
                "Fetch and extract text content from a URL. "
                "Use this when you need to read the content of a specific webpage."
            ),
            parameters=[
                ToolParameter(
                    name="url",
                    description="The URL to fetch content from",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="max_length",
                    description="Maximum content length to return in characters (default: 5000)",
                    type="integer",
                    required=False,
                    default=5000,
                ),
            ],
            returns="Extracted text content from the URL",
            examples=[
                {"url": "https://example.com/article"},
                {"url": "https://docs.example.com", "max_length": 10000},
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the URL content fetch.
        
        Args:
            url: URL to fetch
            max_length: Maximum content length
            
        Returns:
            ToolResult with extracted content
        """
        url = kwargs.get("url")
        if not url:
            return ToolResult.error("URL parameter is required")

        max_length = kwargs.get("max_length", 5000)

        try:
            logger.info("Fetching URL content", url=url)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={"User-Agent": "RAG-Agent/1.0"},
                ) as response:
                    if response.status != 200:
                        return ToolResult.error(
                            f"Failed to fetch URL: HTTP {response.status}"
                        )

                    content = await response.text()

            # Basic HTML stripping (in production, use a proper HTML parser like BeautifulSoup)
            import re

            # Remove script and style elements
            content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
            # Remove HTML tags
            content = re.sub(r"<[^>]+>", " ", content)
            # Clean up whitespace
            content = re.sub(r"\s+", " ", content).strip()

            # Truncate if necessary
            if len(content) > max_length:
                content = content[:max_length] + "..."

            logger.info("URL content fetched", url=url, content_length=len(content))

            return ToolResult.success(
                data={
                    "url": url,
                    "content": content,
                    "content_length": len(content),
                },
            )

        except aiohttp.ClientError as e:
            logger.error("Failed to fetch URL", url=url, error=str(e))
            return ToolResult.error(f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            logger.error("URL content extraction failed", url=url, error=str(e))
            return ToolResult.error(f"Failed to extract content: {str(e)}")

