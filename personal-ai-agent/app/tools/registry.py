"""Tool registry.

A tool is a plain Python function paired with a JSON schema describing it to
the LLM. New tools only need a ``@register_tool`` decorator.
"""

from __future__ import annotations

from typing import Any, Callable

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(name: str, schema: dict) -> Callable[[Callable], Callable]:
    """Register a callable as a tool exposed to the LLM."""

    def decorator(func: Callable) -> Callable:
        TOOL_REGISTRY[name] = {"func": func, "schema": schema}
        return func

    return decorator


def tool_schemas() -> list[dict]:
    """Return the list of tool schemas ready to be passed to the LLM."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def execute_tool(name: str, input_data: dict) -> Any:
    """Look up and execute a registered tool."""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' is not registered")
    return TOOL_REGISTRY[name]["func"](**input_data)


def list_tools() -> list[str]:
    return list(TOOL_REGISTRY.keys())