"""
Tests for agent tools.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from rag_agent.tools.base import (
    Tool,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolResultStatus,
    ToolRegistry,
)


class MockTool(Tool):
    """A mock tool for testing."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mock_tool",
            description="A mock tool for testing",
            parameters=[
                ToolParameter(
                    name="input",
                    description="Test input",
                    type="string",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult.success(data=f"Processed: {kwargs.get('input')}")


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ToolResult.success(data={"key": "value"}, extra="metadata")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.metadata["extra"] == "metadata"

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult.error("Something went wrong")
        
        assert result.status == ToolResultStatus.ERROR
        assert result.error == "Something went wrong"

    def test_partial_result(self):
        """Test creating a partial result."""
        result = ToolResult.partial(data=["partial"], error="Not all found")
        
        assert result.status == ToolResultStatus.PARTIAL
        assert result.data == ["partial"]
        assert result.error == "Not all found"


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_to_function_schema(self):
        """Test converting to OpenAI function schema."""
        definition = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="query",
                    description="Search query",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Max results",
                    type="integer",
                    required=False,
                ),
            ],
        )
        
        schema = definition.to_function_schema()
        
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "query" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]
        assert "limit" not in schema["parameters"]["required"]


class TestTool:
    """Tests for Tool base class."""

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test executing a tool."""
        tool = MockTool()
        result = await tool.execute(input="test")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.data == "Processed: test"

    @pytest.mark.asyncio
    async def test_tool_callable(self):
        """Test tool is callable."""
        tool = MockTool()
        result = await tool(input="test")
        
        assert result.status == ToolResultStatus.SUCCESS

    def test_validate_params_success(self):
        """Test parameter validation success."""
        tool = MockTool()
        is_valid, error = tool.validate_params(input="test")
        
        assert is_valid is True
        assert error is None

    def test_validate_params_missing_required(self):
        """Test parameter validation with missing required param."""
        tool = MockTool()
        is_valid, error = tool.validate_params()
        
        assert is_valid is False
        assert "input" in error


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        
        registry.register(tool)
        
        assert "mock_tool" in registry
        assert len(registry) == 1

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        
        retrieved = registry.get("mock_tool")
        
        assert retrieved is tool

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        
        assert registry.get("nonexistent") is None

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        
        removed = registry.unregister("mock_tool")
        
        assert removed is True
        assert "mock_tool" not in registry

    def test_list_tools(self):
        """Test listing tool names."""
        registry = ToolRegistry()
        registry.register(MockTool())
        
        tools = registry.list_tools()
        
        assert "mock_tool" in tools

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool through the registry."""
        registry = ToolRegistry()
        registry.register(MockTool())
        
        result = await registry.execute("mock_tool", input="test")
        
        assert result.status == ToolResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist."""
        registry = ToolRegistry()
        
        result = await registry.execute("nonexistent")
        
        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error

    def test_get_function_schemas(self):
        """Test getting function schemas for all tools."""
        registry = ToolRegistry()
        registry.register(MockTool())
        
        schemas = registry.get_function_schemas()
        
        assert len(schemas) == 1
        assert schemas[0]["name"] == "mock_tool"

