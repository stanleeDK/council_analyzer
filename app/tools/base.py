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
    """Looks up Tool objects by name.

    Internally this is just a dictionary: {tool_name: Tool object}. Every
    method below is a small, named operation on that one dictionary.
    """

    def __init__(self, tools: list[Tool] | None = None):
        # self._tools is the dictionary that holds everything. It starts
        # empty; if a list of tools was passed in, we add each one below.
        self._tools: dict[str, Tool] = {}

        # tools is optional (defaults to None), so guard against that before
        # looping over it.
        if tools is None:
            tools = []

        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add one tool to the registry.

        If a tool with this name is already registered, it gets replaced -
        dictionary assignment always overwrites an existing key.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name.

        Returns None if the name isn't registered, rather than raising an
        error. This matters because the caller (the model) can ask for a
        tool that doesn't exist, and we want that to be a normal, checkable
        result - not a crash.
        """
        if name in self._tools:
            return self._tools[name]
        return None

    def api_schemas(self, names: list[str]) -> list[dict]:
        """Build the list of tool schemas to send to the Anthropic API.

        Takes a list of tool names (e.g. what one agent is allowed to use)
        and returns each one's api_schema(). Any name that isn't actually
        registered is quietly skipped, rather than raising an error.
        """
        schemas: list[dict] = []
        for name in names:
            tool = self.get(name)
            if tool is None:
                continue  # not a real tool name - skip it
            schemas.append(tool.api_schema())
        return schemas

    def names(self) -> list[str]:
        """All registered tool names, alphabetically."""
        all_names = list(self._tools.keys())
        return sorted(all_names)
