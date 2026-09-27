"""Tool definitions.

A tool is a controlled function the model may *request*. The runtime
decides whether it actually runs (see runtime/policy.py). Each Tool
carries the JSON schema sent to the API plus the Python handler that
executes it.

Handlers receive (tool_input: dict, ctx: ToolContext) and return a
string, which becomes the tool_result the model sees.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ToolContext:
    """Everything a handler might need that isn't in tool_input."""
    corpus_db: sqlite3.Connection | None = None
    analytics_db: sqlite3.Connection | None = None
    state: Any = None  # TaskState; typed loosely to avoid a circular import


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, ToolContext], str]
    read_only: bool = True

    def api_schema(self) -> dict:
        """The shape the Anthropic API expects in the `tools` list."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, tool_input: dict, ctx: ToolContext) -> str:
        return self.handler(tool_input, ctx)


class ToolRegistry:
    """The set of tools that exist, looked up by name.

    Registering a tool does not grant permission to call it — that is the
    policy engine's job (runtime/policy.py). This is only the catalogue.
    """

    def __init__(self, tools: list[Tool] | None = None):
        # Keyed by name because every lookup originates with the model, which
        # refers to a tool by its name string and nothing else.
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool, replacing any earlier tool of the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """The tool with this name, or None if there isn't one.

        Returns None rather than raising, because the name comes from the
        model and the model can ask for a tool that doesn't exist. The agent
        loop turns that None into a readable error it can react to.
        """
        return self._tools.get(name)

    def api_schemas(self, names: list[str]) -> list[dict]:
        """API-shaped schemas for the named tools, skipping unknown names.

        The names come from a workflow YAML's tool allowlist, so a typo there
        offers the model one fewer tool rather than crashing the run.
        """
        schemas = []
        for name in names:
            tool = self.get(name)
            if tool is not None:
                schemas.append(tool.api_schema())
        return schemas

    def names(self) -> list[str]:
        """Every registered tool name, alphabetically."""
        return sorted(self._tools.keys())
