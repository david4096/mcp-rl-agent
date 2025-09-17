"""Action discovery and execution framework for MCP tools."""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
import structlog

from ..interfaces import MCPTool, MCPAction, MCPResult
from .client import MCPClientManager
from ..config.models import MCPServerConfig

logger = structlog.get_logger(__name__)


@dataclass
class ActionContext:
    """Context information for action execution."""
    conversation_id: str
    step: int
    previous_actions: List[MCPAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionFilter:
    """Filter for selecting relevant actions based on context."""

    def __init__(self):
        self.filters: List[Callable[[MCPTool, ActionContext], bool]] = []

    def add_filter(self, filter_func: Callable[[MCPTool, ActionContext], bool]) -> None:
        """Add a filter function."""
        self.filters.append(filter_func)

    def filter_tools(self, tools: List[MCPTool], context: ActionContext) -> List[MCPTool]:
        """Apply all filters to the tools list."""
        filtered_tools = tools
        for filter_func in self.filters:
            filtered_tools = [tool for tool in filtered_tools if filter_func(tool, context)]
        return filtered_tools

    @staticmethod
    def no_repetition_filter(max_repetitions: int = 3) -> Callable[[MCPTool, ActionContext], bool]:
        """Filter to prevent excessive repetition of the same tool."""
        def filter_func(tool: MCPTool, context: ActionContext) -> bool:
            recent_actions = context.previous_actions[-10:]  # Look at last 10 actions
            tool_usage_count = sum(1 for action in recent_actions if action.tool.name == tool.name)
            return tool_usage_count < max_repetitions
        return filter_func

    @staticmethod
    def server_availability_filter() -> Callable[[MCPTool, ActionContext], bool]:
        """Filter to only include tools from available servers."""
        def filter_func(tool: MCPTool, context: ActionContext) -> bool:
            # This would check if the server is available
            # For now, always return True
            return True
        return filter_func


class ActionValidator:
    """Validates action arguments before execution."""

    @staticmethod
    def validate_action(action: MCPAction) -> tuple[bool, Optional[str]]:
        """Validate an action's arguments against its tool schema."""
        try:
            schema = action.tool.parameters
            if not schema:
                return True, None

            # Basic validation - check required fields
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in action.arguments:
                    return False, f"Missing required field: {field}"

            # Type validation for properties
            properties = schema.get("properties", {})
            for field, value in action.arguments.items():
                if field in properties:
                    field_schema = properties[field]
                    if not ActionValidator._validate_field_type(value, field_schema):
                        expected_type = field_schema.get("type", "unknown")
                        return False, f"Invalid type for field '{field}': expected {expected_type}"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def _validate_field_type(value: Any, field_schema: Dict[str, Any]) -> bool:
        """Validate a field's value against its schema type."""
        expected_type = field_schema.get("type")

        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        else:
            # Unknown type or no type specified
            return True


class ActionExecutor:
    """Executes actions with retry logic, timeout handling, and error recovery."""

    def __init__(self, client_manager: MCPClientManager, max_retries: int = 3, timeout: float = 30.0):
        self.client_manager = client_manager
        self.max_retries = max_retries
        self.timeout = timeout

    async def execute_action(self, action: MCPAction) -> MCPResult:
        """Execute an action with retry logic."""
        logger.debug("Executing action", action_id=action.action_id, tool=action.tool.name)

        # Validate action first
        is_valid, error_msg = ActionValidator.validate_action(action)
        if not is_valid:
            return MCPResult(
                action_id=action.action_id,
                success=False,
                result=None,
                error=f"Validation failed: {error_msg}"
            )

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.client_manager.execute_action(action),
                    timeout=self.timeout
                )

                if result.success:
                    logger.debug("Action executed successfully", action_id=action.action_id, attempt=attempt + 1)
                    return result
                else:
                    last_error = result.error
                    logger.warning("Action failed", action_id=action.action_id, attempt=attempt + 1, error=result.error)

            except asyncio.TimeoutError:
                last_error = f"Action timed out after {self.timeout} seconds"
                logger.warning("Action timed out", action_id=action.action_id, attempt=attempt + 1, timeout=self.timeout)

            except Exception as e:
                last_error = str(e)
                logger.warning("Action execution error", action_id=action.action_id, attempt=attempt + 1, error=str(e))

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

        # All retries failed
        return MCPResult(
            action_id=action.action_id,
            success=False,
            result=None,
            error=f"Action failed after {self.max_retries} attempts. Last error: {last_error}"
        )


class ActionManager:
    """Main class for managing action discovery and execution."""

    def __init__(self, client_manager: MCPClientManager):
        self.client_manager = client_manager
        self.action_filter = ActionFilter()
        self.action_executor = ActionExecutor(client_manager)
        self._setup_default_filters()

    def _setup_default_filters(self) -> None:
        """Set up default action filters."""
        self.action_filter.add_filter(ActionFilter.no_repetition_filter())
        self.action_filter.add_filter(ActionFilter.server_availability_filter())

    async def discover_actions(self, context: ActionContext) -> List[MCPTool]:
        """Discover available actions given the current context."""
        logger.debug("Discovering actions", conversation_id=context.conversation_id, step=context.step)

        # Get all available tools
        all_tools = self.client_manager.get_all_tools()

        # Apply filters
        filtered_tools = self.action_filter.filter_tools(all_tools, context)

        logger.info("Discovered actions",
                   total_tools=len(all_tools),
                   filtered_tools=len(filtered_tools),
                   conversation_id=context.conversation_id)

        return filtered_tools

    async def execute_action(self, tool: MCPTool, arguments: Dict[str, Any], context: ActionContext) -> MCPResult:
        """Create and execute an action."""
        action_id = str(uuid.uuid4())
        action = MCPAction(
            tool=tool,
            arguments=arguments,
            action_id=action_id
        )

        logger.info("Executing action",
                   action_id=action_id,
                   tool=tool.name,
                   conversation_id=context.conversation_id)

        result = await self.action_executor.execute_action(action)

        # Update context with executed action
        context.previous_actions.append(action)

        return result

    def get_action_space_size(self) -> int:
        """Get the current action space size."""
        return len(self.client_manager.get_all_tools())

    def get_tools_by_server(self, server_id: str) -> List[MCPTool]:
        """Get tools for a specific server."""
        return self.client_manager.get_server_tools(server_id)

    def add_filter(self, filter_func: Callable[[MCPTool, ActionContext], bool]) -> None:
        """Add a custom filter function."""
        self.action_filter.add_filter(filter_func)

    async def refresh_tools(self) -> None:
        """Refresh tool discovery from all connected servers."""
        logger.info("Refreshing tool discovery")

        # This would trigger re-discovery on all servers
        # For now, we assume tools are static after connection
        # In a full implementation, we'd re-query each server

        all_tools = self.client_manager.get_all_tools()
        logger.info("Tool refresh complete", total_tools=len(all_tools))