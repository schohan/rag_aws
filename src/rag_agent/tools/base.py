"""
Base Tool definitions following Google ADK patterns.

Provides abstract base classes and registries for agent tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, Generic
from enum import Enum

from pydantic import BaseModel, Field


class ToolResultStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """Result of a tool execution."""

    status: ToolResultStatus
    data: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: Any, **metadata: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(status=ToolResultStatus.SUCCESS, data=data, metadata=metadata)

    @classmethod
    def error(cls, error: str, data: Any = None, **metadata: Any) -> "ToolResult":
        """Create an error result."""
        return cls(
            status=ToolResultStatus.ERROR,
            data=data,
            error=error,
            metadata=metadata,
        )

    @classmethod
    def partial(cls, data: Any, error: str | None = None, **metadata: Any) -> "ToolResult":
        """Create a partial success result."""
        return cls(
            status=ToolResultStatus.PARTIAL,
            data=data,
            error=error,
            metadata=metadata,
        )


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    """Schema definition for a tool."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="What the tool does")
    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Tool parameters",
    )
    returns: str = Field(
        default="ToolResult",
        description="Return type description",
    )
    examples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Usage examples",
    )

    def to_function_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            param_schema = {"type": param.type, "description": param.description}
            if param.enum:
                param_schema["enum"] = param.enum
            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class Tool(ABC):
    """
    Abstract base class for agent tools.
    
    Follows Google ADK patterns for tool definition and execution.
    Tools are callable components that the agent can use to perform
    specific actions during reasoning.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition schema."""
        pass

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self.definition.name

    @property
    def description(self) -> str:
        """Return the tool description."""
        return self.definition.description

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with execution outcome
        """
        pass

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Make the tool callable."""
        return await self.execute(**kwargs)

    def validate_params(self, **kwargs: Any) -> tuple[bool, str | None]:
        """
        Validate parameters against the tool definition.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in self.definition.parameters:
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter: {param.name}"

            if param.name in kwargs and param.enum:
                if kwargs[param.name] not in param.enum:
                    return False, f"Invalid value for {param.name}. Must be one of: {param.enum}"

        return True, None


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides tool discovery, registration, and retrieval for the agent.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def get_all_definitions(self) -> list[ToolDefinition]:
        """Return definitions for all registered tools."""
        return [tool.definition for tool in self._tools.values()]

    def get_function_schemas(self) -> list[dict[str, Any]]:
        """Return function schemas for all tools (for LLM function calling)."""
        return [tool.definition.to_function_schema() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            ToolResult from execution
        """
        tool = self.get(name)
        if not tool:
            return ToolResult.error(f"Tool not found: {name}")

        # Validate parameters
        is_valid, error = tool.validate_params(**kwargs)
        if not is_valid:
            return ToolResult.error(error or "Invalid parameters")

        return await tool.execute(**kwargs)

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
) -> Callable[[Callable], Tool]:
    """
    Decorator to create a tool from a function.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: Optional parameter definitions
        
    Returns:
        Decorator that creates a Tool from a function
    """

    def decorator(func: Callable) -> Tool:
        class FunctionTool(Tool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters or [],
                )

            async def execute(self, **kwargs: Any) -> ToolResult:
                try:
                    result = await func(**kwargs)
                    return ToolResult.success(result)
                except Exception as e:
                    return ToolResult.error(str(e))

        return FunctionTool()

    return decorator

