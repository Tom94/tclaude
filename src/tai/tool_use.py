# tai -- Terminal AI
#
# Copyright (C) 2025 Thomas Müller <contact@tom94.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import asyncio
import importlib
import inspect
from typing import Callable

from .common import History
from .json import JSON, get, get_or, get_or_default


def get_available_tools() -> dict[str, Callable[[object], object]]:
    """
    Dynamically import tools.py and extract all callable functions.
    Returns a dictionary mapping function names to their callable objects.
    """
    try:
        tools_module = importlib.import_module(".tools", package=__package__)
        available_tools: dict[str, Callable[[object], object]] = {}

        for name, obj in inspect.getmembers(tools_module):
            if inspect.isfunction(obj) and not name.startswith("_"):
                available_tools[name] = obj

        return available_tools
    except ImportError:
        return {}


def get_tool_definitions() -> list[JSON]:
    """
    Generate tool definitions for Claude based on functions in tools.py.
    """
    available_tools = get_available_tools()
    tool_definitions: list[JSON] = []

    for name, func in available_tools.items():
        # Get function signature and docstring
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # Parse parameters
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            param_desc = f"Parameter {param_name}"

            # Try to extract type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation is str:
                    param_type = "string"
                elif param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif hasattr(param.annotation, "__origin__"):
                    # Handle generic types like List[str]
                    if param.annotation.__origin__ is list:
                        param_type = "array"
                    elif param.annotation.__origin__ is dict:
                        param_type = "object"

            properties[param_name] = {"type": param_type, "description": param_desc}

            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        tool_def: JSON = {
            "name": name,
            "description": doc,
            "input_schema": {"type": "object", "properties": properties, "required": required},
        }

        tool_definitions.append(tool_def)

    return tool_definitions


tool_definitions = get_tool_definitions()


async def use_tools(messages: History) -> dict[str, JSON]:
    """
    Use the tools specified in the messages to perform actions.
    This function is called when the model indicates that it wants to use a tool.
    """
    available_tools = get_available_tools()
    tool_results: list[dict[str, JSON]] = []
    tool_tasks: list[asyncio.Task[JSON] | None] = []

    # Find the last assistant message with tool use
    last_message = messages[-1] if messages else None
    if not last_message or last_message.get("role") != "assistant":
        return {"role": "user", "content": tool_results}

    # Process each content block that contains tool use. Tools are run in parallel.
    for content_block in get_or_default(last_message, "content", list[JSON]):
        if get(content_block, "type", str) == "tool_use":
            tool_name = get(content_block, "name", str)
            tool_input = get_or(content_block, "input", {})
            tool_use_id = get(content_block, "id", str)

            if tool_name and tool_name in available_tools:
                # Call the tool function with the provided input
                tool_use_task: asyncio.Task[JSON] = asyncio.create_task(available_tools[tool_name](**tool_input))
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id})
                tool_tasks.append(tool_use_task)
            else:
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": f"Tool {tool_name} not found", "is_error": True})
                tool_tasks.append(None)

    # Wait for all async tool calls to complete
    for tool_result, task in zip(tool_results, tool_tasks):
        if task is None:
            continue

        try:
            tool_result["content"] = str(await task)
        except (KeyboardInterrupt, asyncio.CancelledError, Exception) as e:
            if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
                tool_result["content"] = "Tool execution was cancelled."
            else:
                tool_result["content"] = f"Error executing tool: {str(e)}"
            tool_result["is_error"] = True

    return {"role": "user", "content": tool_results}
