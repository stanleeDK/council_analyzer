"""Tests for the agent runtime: policy, budgets, approvals, tools, state."""
import pytest

from app.runtime.budget import Budget, BudgetExceeded
from app.runtime.policy import PolicyEngine
from app.runtime.approval import AutoApprover, DenyingApprover
from app.runtime.state import ClaimCheck, Evidence, TaskState
from app.tools.base import Tool, ToolContext, ToolRegistry
from app.tools.calculator import UnsafeExpression, calculate, evaluate
from app.tools.sql import SQLRejected, validate_sql


# -- policy ---------------------------------------------------------------

def test_policy_allows_listed_tool():
    policy = PolicyEngine(agent_tools={"researcher": ["search_transcripts"]})
    decision = policy.decide("researcher", "search_transcripts")
    assert decision.allowed
    assert not decision.requires_approval


def test_policy_denies_unlisted_tool():
    policy = PolicyEngine(agent_tools={"researcher": ["search_transcripts"]})
    decision = policy.decide("researcher", "run_readonly_sql")
    assert not decision.allowed
    assert "not permitted" in decision.reason


def test_policy_denies_unknown_agent_entirely():
    policy = PolicyEngine(agent_tools={"researcher": ["search_transcripts"]})
    assert not policy.decide("critic", "search_transcripts").allowed


def test_policy_flags_tools_needing_approval():
    policy = PolicyEngine(
        agent_tools={"data_analyst": ["run_readonly_sql"]},
        approvals={"run_readonly_sql": "always"},
    )
    decision = policy.decide("data_analyst", "run_readonly_sql")
    assert decision.allowed and decision.requires_approval


# -- budgets --------------------------------------------------------------

def test_budget_blocks_after_max_tool_calls():
    budget = Budget(max_tool_calls=2)
    budget.check_tool_call()
    budget.check_tool_call()
    with pytest.raises(BudgetExceeded, match="max_tool_calls"):
        budget.check_tool_call()


def test_budget_blocks_after_max_model_calls():
    budget = Budget(max_model_calls=1)
    budget.check_model_call()
    with pytest.raises(BudgetExceeded, match="max_model_calls"):
        budget.check_model_call()


def test_budget_charges_by_model_rate():
    budget = Budget()
    cost = budget.charge("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(2.00)
    assert budget.cost_usd == pytest.approx(2.00)
    assert budget.per_model["claude-sonnet-5"] == pytest.approx(2.00)


def test_budget_blocks_once_cost_cap_is_hit():
    budget = Budget(max_cost_usd=0.10)
    budget.charge("claude-opus-5", input_tokens=1_000_000, output_tokens=0)  # $5
    with pytest.raises(BudgetExceeded, match="max_cost_usd"):
        budget.check_model_call()


def test_unknown_model_charges_zero_rather_than_crashing():
    budget = Budget()
    assert budget.charge("some-future-model", 1000, 1000) == 0.0


# -- approvals ------------------------------------------------------------

def test_approvers_have_opposite_verdicts():
    assert AutoApprover().approve("a", "t", {}) is True
    assert DenyingApprover().approve("a", "t", {}) is False


# -- SQL guardrails -------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "DELETE FROM meetings",
    "DROP TABLE meetings",
    "UPDATE meetings SET city = 'x'",
    "INSERT INTO meetings VALUES (1)",
    "ALTER TABLE meetings ADD COLUMN x TEXT",
    "PRAGMA table_info(meetings)",
    "ATTACH DATABASE '/etc/passwd' AS leak",
])
def test_sql_rejects_writes_and_dangerous_statements(sql):
    with pytest.raises(SQLRejected):
        validate_sql(sql)


def test_sql_rejects_stacked_statements():
    with pytest.raises(SQLRejected, match="multiple statements"):
        validate_sql("SELECT 1; DROP TABLE meetings")


def test_sql_rejects_empty():
    with pytest.raises(SQLRejected):
        validate_sql("   ")


def test_sql_allows_select_and_injects_limit():
    out = validate_sql("SELECT city FROM meetings")
    assert out.startswith("SELECT city FROM meetings")
    assert "LIMIT 200" in out


def test_sql_allows_cte():
    out = validate_sql("WITH x AS (SELECT 1 AS n) SELECT n FROM x")
    assert out.lower().startswith("with")


def test_sql_respects_existing_limit():
    out = validate_sql("SELECT city FROM meetings LIMIT 5")
    assert out.count("LIMIT") == 1


# -- calculator -----------------------------------------------------------

def test_calculator_evaluates_arithmetic():
    assert evaluate("(1240 - 980) / 980 * 100") == pytest.approx(26.53, abs=0.01)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "1 if True else 2",
    "[1,2,3]",
    "1/0",
    "2 ** 10000",
])
def test_calculator_rejects_non_arithmetic(expr):
    with pytest.raises(UnsafeExpression):
        evaluate(expr)


def test_calculator_tool_returns_error_string_not_exception():
    out = calculate.run({"expression": "__import__('os')"}, ToolContext())
    assert out.startswith("Rejected:")


# -- tool registry --------------------------------------------------------

def test_registry_builds_api_schemas_for_named_tools_only():
    registry = ToolRegistry([calculate])
    schemas = registry.api_schemas(["calculate", "nonexistent"])
    assert len(schemas) == 1
    assert schemas[0]["name"] == "calculate"
    assert set(schemas[0]) == {"name", "description", "input_schema"}


def test_registry_get_returns_none_for_unknown():
    assert ToolRegistry([calculate]).get("nope") is None


# -- task state -----------------------------------------------------------

def _evidence(cid: str) -> Evidence:
    return Evidence(citation_id=cid, city="Springfield", title="Council", upload_date="2026-01-01",
                    video_id="abc", start_ts=0.0, end_ts=10.0, text=f"text {cid}", score=0.5)


def test_state_dedupes_evidence_by_citation_id():
    state = TaskState(objective="q", workflow="w")
    state.add_evidence([_evidence("C1"), _evidence("C2")])
    added = state.add_evidence([_evidence("C2"), _evidence("C3")])
    assert [e.citation_id for e in state.evidence] == ["C1", "C2", "C3"]
    assert [e.citation_id for e in added] == ["C3"]


def test_state_reports_unsupported_claims():
    state = TaskState(objective="q", workflow="w")
    state.claim_checks = [
        ClaimCheck(claim="a", status="supported", reason=""),
        ClaimCheck(claim="b", status="unsupported", reason="no citation"),
    ]
    assert [c.claim for c in state.unsupported_claims()] == ["b"]


def test_state_serialises_to_json():
    state = TaskState(objective="q", workflow="w")
    state.add_evidence([_evidence("C1")])
    assert '"citation_id": "C1"' in state.to_json()
