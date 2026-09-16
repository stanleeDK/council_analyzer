"""Policy engine.

The model proposes a tool call; this decides whether it actually runs.
Three outcomes: allowed, allowed-but-needs-human-approval, denied.

Policy is data (loaded from a workflow YAML), not code, so a workflow can
grant the researcher document search while denying it SQL, without
touching the agent implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

APPROVAL_NEVER = "never"
APPROVAL_ALWAYS = "always"


@dataclass
class Decision:
    allowed: bool
    requires_approval: bool = False
    reason: str = ""


@dataclass
class PolicyEngine:
    # agent name -> tool names that agent may request
    agent_tools: dict[str, list[str]] = field(default_factory=dict)
    # tool name -> "never" | "always"
    approvals: dict[str, str] = field(default_factory=dict)

    def decide(self, agent: str, tool: str) -> Decision:
        allowed_tools = self.agent_tools.get(agent, [])
        if tool not in allowed_tools:
            return Decision(
                allowed=False,
                reason=f"agent '{agent}' is not permitted to call '{tool}' "
                       f"(allowed: {', '.join(allowed_tools) or 'none'})",
            )
        needs = self.approvals.get(tool, APPROVAL_NEVER) == APPROVAL_ALWAYS
        return Decision(allowed=True, requires_approval=needs)

    def tools_for(self, agent: str) -> list[str]:
        return list(self.agent_tools.get(agent, []))


def always_deny() -> PolicyEngine:
    """Default-deny policy - useful as a safe base in tests."""
    return PolicyEngine(agent_tools={}, approvals={})
