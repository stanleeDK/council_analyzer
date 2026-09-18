"""Structured-output calls: thinking config, truncation, and budget honesty.

Truncation reaches us two different ways, and the second one is the reason a
critique crashed twice: the SDK validates inside messages.parse(), so a cut-off
response raises pydantic's ValidationError before any response object exists -
no usage, no stop_reason, and no budget charge.
"""
import pydantic
import pytest

from app.models.schemas import CriticReport
from app.observability.traces import Tracer, get_trace_db
from app.runtime.agent import (ADAPTIVE_THINKING_MODELS, Agent, AgentSpec,
                               StructuredOutputError, thinking_kwargs)
from app.runtime.budget import Budget
from app.runtime.policy import PolicyEngine
from app.tools.base import ToolContext, ToolRegistry
from tests.test_agent_loop import FakeResponse, FakeTextBlock, FakeUsage


# -- thinking configuration ----------------------------------------------

def test_thinking_models_get_adaptive():
    for model in ADAPTIVE_THINKING_MODELS:
        assert thinking_kwargs(model, "adaptive") == {"thinking": {"type": "adaptive"}}


def test_models_without_adaptive_support_get_no_parameter():
    """Haiku 4.5 rejects adaptive; omitting it is what it got before anyway."""
    assert thinking_kwargs("claude-haiku-4-5", "adaptive") == {}
    assert thinking_kwargs("some-future-model", "adaptive") == {}


def test_thinking_can_be_turned_off_on_any_model():
    assert thinking_kwargs("claude-sonnet-5", "off") == {"thinking": {"type": "disabled"}}
    assert thinking_kwargs("claude-haiku-4-5", "off") == {"thinking": {"type": "disabled"}}


def test_agents_think_by_default():
    assert AgentSpec(name="a", model="m", system="s").thinking == "adaptive"


# -- the request actually carries it -------------------------------------

def _agent(client, tmp_path, model="claude-sonnet-5", thinking="adaptive", max_tokens=16000,
           budget=None):
    tracer = Tracer(db=get_trace_db(tmp_path / "t.db"), run_id="r", workflow="w", echo=False)
    tracer.start_run("o")
    return Agent(
        spec=AgentSpec(name="critic", model=model, system="s", max_tokens=max_tokens,
                       thinking=thinking),
        client=client, registry=ToolRegistry([]), policy=PolicyEngine(agent_tools={}),
        budget=budget or Budget(), tracer=tracer, ctx=ToolContext(),
    )


class CapturingClient:
    def __init__(self, outer):
        self.outer = outer
        self.messages = self

    def parse(self, **kwargs):
        self.outer.append(kwargs)
        return _ParsedOK()

    def create(self, **kwargs):
        self.outer.append(kwargs)
        return FakeResponse([FakeTextBlock("done")])


class _ParsedOK:
    parsed_output = CriticReport(verdicts=[], needs_more_research=False, follow_up_queries=[])
    stop_reason = "end_turn"
    usage = FakeUsage()


def test_structured_call_sends_the_thinking_setting(tmp_path):
    calls = []
    _agent(CapturingClient(calls), tmp_path).run_structured("p", CriticReport)
    assert calls[0]["thinking"] == {"type": "adaptive"}
    assert calls[0]["max_tokens"] == 16000


def test_tool_loop_sends_it_too(tmp_path):
    calls = []
    _agent(CapturingClient(calls), tmp_path).run("p")
    assert calls[0]["thinking"] == {"type": "adaptive"}


def test_thinking_off_is_sent_as_disabled(tmp_path):
    calls = []
    _agent(CapturingClient(calls), tmp_path, thinking="off").run_structured("p", CriticReport)
    assert calls[0]["thinking"] == {"type": "disabled"}


def test_no_thinking_key_is_sent_to_a_model_that_rejects_it(tmp_path):
    calls = []
    _agent(CapturingClient(calls), tmp_path, model="claude-haiku-4-5").run_structured(
        "p", CriticReport)
    assert "thinking" not in calls[0]


# -- truncation raised from inside the SDK -------------------------------

class TruncatingClient:
    """Reproduces the real failure: validation blows up inside parse()."""

    def __init__(self):
        self.messages = self

    def parse(self, **kwargs):
        # what pydantic raises on a response cut off mid-JSON
        CriticReport.model_validate_json('{"verdicts":[{"claim":"K","status":"sup')


def test_truncation_inside_parse_becomes_a_named_error(tmp_path):
    with pytest.raises(StructuredOutputError) as exc:
        _agent(TruncatingClient(), tmp_path, max_tokens=16000).run_structured("p", CriticReport)

    message = str(exc.value)
    assert "critic" in message
    assert "CriticReport" in message
    assert "max_tokens=16000" in message
    assert "thinking is spent from that same budget" in message


def test_the_original_pydantic_error_is_kept_as_the_cause(tmp_path):
    with pytest.raises(StructuredOutputError) as exc:
        _agent(TruncatingClient(), tmp_path).run_structured("p", CriticReport)
    assert isinstance(exc.value.__cause__, pydantic.ValidationError)


def test_the_failure_admits_the_cost_is_uncounted(tmp_path):
    """The call really did spend money; the trace must not imply otherwise."""
    budget = Budget()
    with pytest.raises(StructuredOutputError, match="NOT counted"):
        _agent(TruncatingClient(), tmp_path, budget=budget).run_structured("p", CriticReport)
    assert budget.cost_usd == 0.0        # we genuinely cannot know it


def test_truncation_is_recorded_in_the_trace(tmp_path):
    agent = _agent(TruncatingClient(), tmp_path)
    with pytest.raises(StructuredOutputError):
        agent.run_structured("p", CriticReport)

    errors = agent.tracer.db.execute(
        "SELECT agent, model, error FROM trace_steps WHERE kind='error'").fetchall()
    assert errors and errors[0][0] == "critic"
    assert errors[0][1] == "claude-sonnet-5"
    assert "could not be parsed" in errors[0][2]


def test_a_successful_call_still_charges_the_budget(tmp_path):
    budget = Budget()
    _agent(CapturingClient([]), tmp_path, budget=budget).run_structured("p", CriticReport)
    assert budget.cost_usd > 0
    assert budget.model_calls == 1
