"""Human-in-the-loop approval gates.

When policy says a tool call needs approval, the runtime stops and asks.
Autonomy is a product decision, not a property of the model - so the
approver is pluggable: interactive at a terminal, always-yes for
unattended runs, always-no for tests.
"""
from __future__ import annotations

import json
from typing import Protocol


class Approver(Protocol):
    def approve(self, agent: str, tool: str, tool_input: dict, reason: str = "") -> bool:
        ...


class CLIApprover:
    """Prompts at the terminal. Anything but 'y' is a rejection."""

    def approve(self, agent: str, tool: str, tool_input: dict, reason: str = "") -> bool:
        print("\n" + "-" * 60)
        print("APPROVAL REQUIRED")
        print(f"  Agent:  {agent}")
        print(f"  Tool:   {tool}")
        print(f"  Input:  {json.dumps(tool_input, indent=2)[:800]}")
        if reason:
            print(f"  Reason: {reason}")
        print("-" * 60)
        answer = input("Approve this action? [y/N] ").strip().lower()
        return answer == "y"


class AutoApprover:
    """Approves everything - for unattended runs. Policy still applies."""

    def approve(self, agent: str, tool: str, tool_input: dict, reason: str = "") -> bool:
        return True


class DenyingApprover:
    """Rejects everything - used in tests to prove the gate actually gates."""

    def approve(self, agent: str, tool: str, tool_input: dict, reason: str = "") -> bool:
        return False
