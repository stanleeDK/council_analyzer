"""Tests for workflow config, the analytics DB, and the eval harness."""
import json
import sqlite3
from dataclasses import dataclass

import pytest

from app.db.analytics import build_analytics_db, open_readonly
from app.evaluation.retrieval import EvalCase, EvalReport, CaseResult, load_cases, score_case
from app.runtime.workflow import WORKFLOW_DIR, WorkflowConfig


# -- workflow config ------------------------------------------------------

def test_shipped_workflows_all_load():
    names = [p.stem for p in WORKFLOW_DIR.glob("*.yaml")]
    assert {"deep_research", "text2sql", "research_with_approval"} <= set(names)
    for name in names:
        config = WorkflowConfig.load(name)
        assert config.name and config.steps


def test_deep_research_grants_least_privilege():
    config = WorkflowConfig.load("deep_research")
    policy = config.policy()
    # researcher may search but not run SQL
    assert policy.decide("researcher", "search_transcripts").allowed
    assert not policy.decide("researcher", "run_readonly_sql").allowed
    # data analyst is the mirror image
    assert policy.decide("data_analyst", "run_readonly_sql").allowed
    assert not policy.decide("data_analyst", "search_transcripts").allowed
    # planner and critic get no tools at all
    assert policy.tools_for("planner") == []
    assert policy.tools_for("critic") == []


def test_approval_workflow_gates_sql_but_not_search():
    policy = WorkflowConfig.load("research_with_approval").policy()
    assert policy.decide("data_analyst", "run_readonly_sql").requires_approval
    assert not policy.decide("researcher", "search_transcripts").requires_approval


def test_config_drives_budget_and_model_routing():
    config = WorkflowConfig.load("deep_research")
    budget = config.budget()
    assert budget.max_cost_usd == 1.00 and budget.max_tool_calls == 25
    assert config.model_for("planner") == "claude-haiku-4-5"
    assert config.model_for("unlisted_agent") == "claude-sonnet-5"  # falls back to default


def test_missing_workflow_lists_available_ones():
    with pytest.raises(FileNotFoundError, match="deep_research"):
        WorkflowConfig.load("does_not_exist")


def test_malformed_workflow_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: incomplete\n")  # no steps
    with pytest.raises(ValueError, match="steps"):
        WorkflowConfig.load(str(bad))


# -- analytics DB ---------------------------------------------------------

def _corpus_with_chunks(path) -> sqlite3.Connection:
    from app.db.connection import get_db
    db = get_db(path)
    rows = [
        ("Springfield", "2026-01-05", "Council Jan", "vid1", "/raw/Springfield/a.lrc", 0.0, 600.0),
        ("Springfield", "2026-01-05", "Council Jan", "vid1", "/raw/Springfield/a.lrc", 600.0, 1200.0),
        ("Shelbyville", "2025-06-01", "Council Jun", "vid2", "/raw/Shelbyville/b.lrc", 0.0, 300.0),
    ]
    for city, date, title, vid, src, start, end in rows:
        db.execute(
            "INSERT INTO chunks (city, upload_date, title, video_id, source_file, start_ts, end_ts, "
            "text, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
            (city, date, title, vid, src, start, end, "text", "[0.1]"),
        )
    db.commit()
    return db


def test_analytics_db_aggregates_meetings_and_cities(tmp_path):
    corpus = _corpus_with_chunks(tmp_path / "corpus.db")
    out = tmp_path / "analytics.db"
    counts = build_analytics_db(corpus, out)

    assert counts == {"meetings": 2, "cities": 2}  # 2 source files, 2 cities

    db = sqlite3.connect(out)
    springfield = db.execute(
        "SELECT meeting_count, total_chunks, total_hours FROM city_coverage WHERE city='Springfield'"
    ).fetchone()
    assert springfield[0] == 1 and springfield[1] == 2       # 1 meeting, 2 chunks
    assert springfield[2] == pytest.approx(1200 / 3600, abs=0.01)

    year = db.execute("SELECT upload_year FROM meetings WHERE city='Springfield'").fetchone()[0]
    assert year == 2026


def test_analytics_db_is_read_only_when_opened_for_agents(tmp_path):
    corpus = _corpus_with_chunks(tmp_path / "corpus.db")
    out = tmp_path / "analytics.db"
    build_analytics_db(corpus, out)

    db = open_readonly(out)
    assert db.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 2
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM meetings")


def test_analytics_db_rebuild_is_idempotent(tmp_path):
    corpus = _corpus_with_chunks(tmp_path / "corpus.db")
    out = tmp_path / "analytics.db"
    first = build_analytics_db(corpus, out)
    second = build_analytics_db(corpus, out)
    assert first == second


# -- eval harness ---------------------------------------------------------

@dataclass
class FakeChunk:
    text: str
    score: float = 0.5


def test_scores_a_hit_with_its_rank():
    case = EvalCase(id="e1", question="q", expect_any_of=["landusi"])
    result = score_case(case, [FakeChunk("nothing"), FakeChunk("the fifth annual LANDUSI fest")])
    assert result.hit and result.rank == 2


def test_scores_a_miss():
    case = EvalCase(id="e1", question="q", expect_any_of=["landusi"])
    result = score_case(case, [FakeChunk("unrelated"), FakeChunk("also unrelated")])
    assert not result.hit and result.rank is None


def test_any_of_matches_alternate_spellings():
    case = EvalCase(id="e1", question="q", expect_any_of=["landusi", "anduisi"])
    result = score_case(case, [FakeChunk("anduisi sports club east")])
    assert result.hit and result.rank == 1


def test_negative_case_passes_when_nothing_matches():
    case = EvalCase(id="neg", question="q", expect_any_of=["mars colony"], expect_no_match=True)
    assert score_case(case, [FakeChunk("zoning variance")]).hit


def test_negative_case_fails_when_something_matches():
    case = EvalCase(id="neg", question="q", expect_any_of=["mars colony"], expect_no_match=True)
    assert not score_case(case, [FakeChunk("the mars colony lease")]).hit


def test_report_metrics():
    report = EvalReport(results=[
        CaseResult("a", hit=True, rank=1, top_score=0.9),
        CaseResult("b", hit=True, rank=4, top_score=0.7),
        CaseResult("c", hit=False, rank=None, top_score=0.4),
    ])
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1.0 + 0.25 + 0.0) / 3)
    assert [r.case_id for r in report.misses()] == ["c"]


def test_shipped_eval_file_parses():
    cases = load_cases()
    assert len(cases) >= 5
    ids = {c.id for c in cases}
    assert "eval_001_landusi_named_event" in ids
    assert any(c.expect_no_match for c in cases)  # a negative control exists
    for case in cases:
        assert case.question and case.expect_any_of
        assert all(needle == needle.lower() for needle in case.expect_any_of)
