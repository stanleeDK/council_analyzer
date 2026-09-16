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
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def api_schemas(self, names: list[str]) -> list[dict]:
        return [self._tools[n].api_schema() for n in names if n in self._tools]

    def names(self) -> list[str]:
        return sorted(self._tools)
