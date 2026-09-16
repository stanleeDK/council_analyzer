"""Agent-loop tests using a fake Anthropic client.

These exercise the loop's control flow - tool dispatch, policy denial,
approval rejection, budget cutoff, iteration cap - without touching the
network. The fakes mimic the shapes the loop actually reads off a
response: .content blocks with .type/.text/.name/.input/.id,
.stop_reason, and .usage.
"""
import sqlite3

import pytest

from app.observability.traces import Tracer, get_trace_db
from app.runtime.agent import Agent, AgentSpec
from app.runtime.approval import AutoApprover, DenyingApprover
from app.runtime.budget import Budget
from app.runtime.policy import PolicyEngine
from app.tools.base import Tool, ToolContext, ToolRegistry


# -- fakes ----------------------------------------------------------------

class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, name, input, id="tu_1"):
        self.name = name
        self.input = input
        self.id = id


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = FakeUsage()
        self.stop_details = None


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return FakeResponse([FakeTextBlock("done")])
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


# -- fixtures -------------------------------------------------------------

@pytest.fixture
def tracer(tmp_path):
    db = get_trace_db(tmp_path / "traces.db")
    t = Tracer(db=db, run_id="test_run", workflow="test", echo=False)
    t.start_run("test objective")
    return t


def echo_tool(calls: list) -> Tool:
    def handler(tool_input, ctx):
        calls.append(tool_input)
        return f"echoed: {tool_input.get('value', '')}"

    return Tool(name="echo", description="echoes", handler=handler,
                input_schema={"type": "object", "properties": {"value": {"type": "string"}}})


def build_agent(responses, tracer, policy=None, budget=None, approver=None, tool=None,
                max_iterations=6):
    tool = tool or echo_tool([])
    return Agent(
        spec=AgentSpec(name="researcher", model="claude-sonnet-5", system="sys"),
        client=FakeClient(responses),
        registry=ToolRegistry([tool]),
        policy=policy or PolicyEngine(agent_tools={"researcher": ["echo"]}),
        budget=budget or Budget(),
        tracer=tracer,
        ctx=ToolContext(),
        approver=approver or AutoApprover(),
        max_iterations=max_iterations,
    )


# -- tests ----------------------------------------------------------------

def test_returns_text_when_model_stops_without_tools(tracer):
    agent = build_agent([FakeResponse([FakeTextBlock("final answer")])], tracer)
    result = agent.run("question")
    assert result.text == "final answer"
    assert result.stopped_because == "end_turn"
    assert result.tool_calls == 0


def test_executes_tool_then_continues_loop(tracer):
    calls = []
    responses = [
        FakeResponse([FakeToolUseBlock("echo", {"value": "hello"})], stop_reason="tool_use"),
        FakeResponse([FakeTextBlock("answer using tool output")]),
    ]
    agent = build_agent(responses, tracer, tool=echo_tool(calls))
    result = agent.run("question")

    assert calls == [{"value": "hello"}]          # the handler actually ran
    assert result.text == "answer using tool output"
    assert result.tool_calls == 1


def test_tool_result_is_fed_back_to_the_model(tracer):
    responses = [
        FakeResponse([FakeToolUseBlock("echo", {"value": "xyz"})], stop_reason="tool_use"),
        FakeResponse([FakeTextBlock("done")]),
    ]
    agent = build_agent(responses, tracer)
    agent.run("question")

    second_call_messages = agent.client.messages.calls[1]["messages"]
    tool_results = second_call_messages[-1]["content"]
    assert tool_results[0]["type"] == "tool_result"
    assert tool_results[0]["tool_use_id"] == "tu_1"
    assert "echoed: xyz" in tool_results[0]["content"]


def test_policy_denial_stops_tool_from_running(tracer):
    calls = []
    responses = [
        FakeResponse([FakeToolUseBlock("echo", {"value": "hi"})], stop_reason="tool_use"),
        FakeResponse([FakeTextBlock("understood, I cannot use that")]),
    ]
    agent = build_agent(responses, tracer, tool=echo_tool(calls),
                        policy=PolicyEngine(agent_tools={"researcher": []}))
    agent.run("question")

    assert calls == []  # handler never invoked
    result_block = agent.client.messages.calls[1]["messages"][-1]["content"][0]
    assert result_block["is_error"] is True
    assert "Denied by policy" in result_block["content"]


def test_rejected_approval_stops_tool_from_running(tracer):
    calls = []
    responses = [
        FakeResponse([FakeToolUseBlock("echo", {"value": "hi"})], stop_reason="tool_use"),
        FakeResponse([FakeTextBlock("ok")]),
    ]
    agent = build_agent(
        responses, tracer, tool=echo_tool(calls),
        policy=PolicyEngine(agent_tools={"researcher": ["echo"]}, approvals={"echo": "always"}),
        approver=DenyingApprover(),
    )
    agent.run("question")

    assert calls == []
    result_block = agent.client.messages.calls[1]["messages"][-1]["content"][0]
    assert "rejected this action" in result_block["content"]


def test_approved_gate_lets_tool_run(tracer):
    calls = []
    responses = [
        FakeResponse([FakeToolUseBlock("echo", {"value": "hi"})], stop_reason="tool_use"),
        FakeResponse([FakeTextBlock("ok")]),
    ]
    agent = build_agent(
        responses, tracer, tool=echo_tool(calls),
        policy=PolicyEngine(agent_tools={"researcher": ["echo"]}, approvals={"echo": "always"}),
        approver=AutoApprover(),
    )
    agent.run("question")
    assert calls == [{"value": "hi"}]


def test_tool_budget_exhaustion_is_reported_to_the_model(tracer):
    responses = [
        FakeResponse([FakeToolUseBlock("echo", {"value": "1"})], stop_reason="tool_use"),
        FakeResponse([FakeTextBlock("stopping")]),
    ]
    agent = build_agent(responses, tracer, budget=Budget(max_tool_calls=0))
    agent.run("question")

    result_block = agent.client.messages.calls[1]["messages"][-1]["content"][0]
    assert "Budget exceeded" in result_block["content"]


def test_model_budget_exhaustion_ends_the_run(tracer):
    agent = build_agent([FakeResponse([FakeTextBlock("never reached")])], tracer,
                        budget=Budget(max_model_calls=0))
    result = agent.run("question")
    assert result.stopped_because.startswith("budget_exceeded")


def test_loop_stops_at_max_iterations(tracer):
    always_tool_use = [
        FakeResponse([FakeToolUseBlock("echo", {"value": str(i)})], stop_reason="tool_use")
        for i in range(20)
    ]
    agent = build_agent(always_tool_use, tracer, max_iterations=3)
    result = agent.run("question")
    assert result.stopped_because == "max_iterations"
    assert result.iterations == 3


def test_tool_exception_becomes_an_error_result_not_a_crash(tracer):
    def boom(tool_input, ctx):
        raise ValueError("handler blew up")

    exploding = Tool(name="echo", description="d", handler=boom,
                     input_schema={"type": "object", "properties": {}})
    responses = [
        FakeResponse([FakeToolUseBlock("echo", {})], stop_reason="tool_use"),
        FakeResponse([FakeTextBlock("recovered")]),
    ]
    agent = build_agent(responses, tracer, tool=exploding)
    result = agent.run("question")

    assert result.text == "recovered"
    block = agent.client.messages.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True and "handler blew up" in block["content"]


def test_refusal_stop_reason_is_handled(tracer):
    agent = build_agent([FakeResponse([], stop_reason="refusal")], tracer)
    result = agent.run("question")
    assert result.stopped_because.startswith("refusal")


def test_cost_is_charged_and_traced(tracer):
    budget = Budget()
    agent = build_agent([FakeResponse([FakeTextBlock("hi")])], tracer, budget=budget)
    agent.run("question")

    assert budget.model_calls == 1
    assert budget.cost_usd > 0
    rows = tracer.db.execute(
        "SELECT kind, model, input_tokens, cost_usd FROM trace_steps WHERE run_id='test_run'"
    ).fetchall()
    assert rows and rows[0][0] == "model_call" and rows[0][2] == 100


def test_tools_are_only_offered_if_policy_allows_them(tracer):
    agent = build_agent([FakeResponse([FakeTextBlock("hi")])], tracer,
                        policy=PolicyEngine(agent_tools={"researcher": []}))
    agent.run("question")
    assert "tools" not in agent.client.messages.calls[0]
