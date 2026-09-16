"""End-to-end workflow orchestration against a fake client.

Proves the pieces actually compose: the planner's subquestions reach the
researcher, the researcher's retrieved evidence reaches the critic, the
critic's verdicts reach the reviser, and every step lands in the trace.
"""
import sqlite3

import pytest

from app.db.connection import get_db
from app.models.schemas import CriticReport, ClaimVerdict, ResearchPlan
from app.observability.traces import get_trace_db
from app.runtime.workflow import WorkflowConfig, WorkflowRunner
from tests.test_agent_loop import FakeResponse, FakeTextBlock, FakeToolUseBlock, FakeUsage


class FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output
        self.usage = FakeUsage()
        self.stop_reason = "end_turn"


class ScriptedMessages:
    """Returns queued responses for create(); queued objects for parse()."""

    def __init__(self, create_responses, parse_outputs):
        self._create = list(create_responses)
        self._parse = list(parse_outputs)
        self.create_calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._create.pop(0) if self._create else FakeResponse([FakeTextBlock("ok")])

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return FakeParsedResponse(self._parse.pop(0))


class ScriptedClient:
    def __init__(self, create_responses, parse_outputs):
        self.messages = ScriptedMessages(create_responses, parse_outputs)


@pytest.fixture
def corpus(tmp_path):
    """A corpus with one retrievable chunk, and a stubbed embedder so the
    test never loads a real model."""
    db = get_db(tmp_path / "corpus.db")
    db.execute(
        "INSERT INTO chunks (city, upload_date, title, video_id, source_file, start_ts, end_ts, "
        "text, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
        ("City of Green Bay", "2021-03-23", "Protection & Policy", "vid1", "/raw/gb/a.lrc",
         1400.0, 1460.0, "request to hold the fifth annual landusi fest", "[1.0, 0.0]"),
    )
    db.commit()
    return db


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    import numpy as np
    monkeypatch.setattr("app.rag.retrieve.embed_text", lambda text: np.array([1.0, 0.0]))


def test_deep_research_threads_state_across_all_agents(corpus, tmp_path):
    plan = ResearchPlan(
        objective="Find festival permits",
        subquestions=["What festival permits were considered?"],
        relevant_cities=["City of Green Bay"],
        needs_quantitative_data=False,   # so data_analysis is skipped
    )
    critique = CriticReport(
        verdicts=[ClaimVerdict(claim="A fest permit was requested", status="supported",
                               reason="stated in [C1]", action="")],
        needs_more_research=False,
        follow_up_queries=[],
    )
    client = ScriptedClient(
        create_responses=[
            # researcher: one search, then its findings
            FakeResponse([FakeToolUseBlock("search_transcripts",
                                           {"query": "festival permit"})], stop_reason="tool_use"),
            FakeResponse([FakeTextBlock("SUBQUESTION: What festival permits were considered?\n"
                                        "FINDING: A fifth annual fest permit was requested [C1]")]),
            FakeResponse([FakeTextBlock("## Summary\nA fest permit was requested [C1]")]),  # draft
            FakeResponse([FakeTextBlock("## Summary\nA fest permit was requested [C1]")]),  # revise
        ],
        parse_outputs=[plan, critique],
    )

    runner = WorkflowRunner(
        config=WorkflowConfig.load("deep_research"),
        corpus_db=corpus,
        analytics_db=None,
        client=client,
        trace_db=get_trace_db(tmp_path / "traces.db"),
        echo=False,
    )
    state = runner.run("What festival permits did Green Bay consider?")

    assert state.status == "complete"
    # planner output reached state
    assert state.subquestions == ["What festival permits were considered?"]
    # the search tool actually ran and evidence landed in shared state
    assert len(state.evidence) == 1
    assert "landusi" in state.evidence[0].text
    assert state.evidence[0].citation_id.startswith("C")
    # researcher's structured finding was parsed out
    assert state.findings and state.findings[0].citation_ids == ["C1"]
    # critic ran over the draft and produced verdicts
    assert state.claim_checks and state.claim_checks[0].status == "supported"
    # final report exists
    assert "fest permit" in state.final_report
    # cost accrued and was bounded
    assert 0 < runner.budget.cost_usd < runner.budget.max_cost_usd


def test_data_analysis_is_skipped_when_planner_says_non_quantitative(corpus, tmp_path):
    plan = ResearchPlan(objective="o", subquestions=["q"], relevant_cities=[],
                        needs_quantitative_data=False)
    critique = CriticReport(verdicts=[], needs_more_research=False, follow_up_queries=[])
    client = ScriptedClient(
        create_responses=[
            FakeResponse([FakeTextBlock("SUBQUESTION: q\nFINDING: nothing found")]),
            FakeResponse([FakeTextBlock("draft")]),
            FakeResponse([FakeTextBlock("final")]),
        ],
        parse_outputs=[plan, critique],
    )
    runner = WorkflowRunner(config=WorkflowConfig.load("deep_research"), corpus_db=corpus,
                            analytics_db=None, client=client,
                            trace_db=get_trace_db(tmp_path / "traces.db"), echo=False)
    state = runner.run("a non-quantitative question")

    assert state.sql_results == []
    assert any("skipped" in n for n in _trace_previews(tmp_path / "traces.db", state.run_id))


def test_critic_can_trigger_one_extra_research_pass(corpus, tmp_path):
    plan = ResearchPlan(objective="o", subquestions=["q"], relevant_cities=[],
                        needs_quantitative_data=False)
    critique = CriticReport(
        verdicts=[ClaimVerdict(claim="unsupported thing", status="unsupported",
                               reason="no citation", action="remove")],
        needs_more_research=True,
        follow_up_queries=["landusi fest"],
    )
    client = ScriptedClient(
        create_responses=[
            FakeResponse([FakeTextBlock("SUBQUESTION: q\nFINDING: thin [C1]")]),   # research
            FakeResponse([FakeTextBlock("draft with unsupported thing")]),         # draft
            FakeResponse([FakeToolUseBlock("search_transcripts",
                                           {"query": "landusi fest"})], stop_reason="tool_use"),
            FakeResponse([FakeTextBlock("SUBQUESTION: q\nFINDING: found it [C1]")]),  # follow-up
            FakeResponse([FakeTextBlock("revised final")]),                        # revise
        ],
        parse_outputs=[plan, critique],
    )
    runner = WorkflowRunner(config=WorkflowConfig.load("deep_research"), corpus_db=corpus,
                            analytics_db=None, client=client,
                            trace_db=get_trace_db(tmp_path / "traces.db"), echo=False)
    state = runner.run("question")

    assert len(state.unsupported_claims()) == 1
    assert state.final_report == "revised final"
    # the follow-up search really ran
    searches = [c for c in client.messages.create_calls if "tools" in c]
    assert len(searches) >= 2


def test_text2sql_workflow_runs_without_retrieval(tmp_path):
    from app.db.analytics import build_analytics_db, open_readonly
    corpus = get_db(tmp_path / "corpus.db")
    corpus.execute(
        "INSERT INTO chunks (city, upload_date, title, video_id, source_file, start_ts, end_ts, "
        "text, embedding) VALUES ('Springfield','2026-01-01','Council','v','/raw/s/a.lrc',0,60,'t','[1.0]')"
    )
    corpus.commit()
    analytics_path = tmp_path / "analytics.db"
    build_analytics_db(corpus, analytics_path)

    client = ScriptedClient(
        create_responses=[
            FakeResponse([FakeToolUseBlock("get_schema", {})], stop_reason="tool_use"),
            FakeResponse([FakeToolUseBlock("run_readonly_sql",
                                           {"sql": "SELECT city, meeting_count FROM city_coverage"})],
                         stop_reason="tool_use"),
            FakeResponse([FakeTextBlock("Springfield held 1 meeting.")]),
            FakeResponse([FakeTextBlock("## Summary\nSpringfield held 1 meeting.")]),
        ],
        parse_outputs=[],
    )
    runner = WorkflowRunner(config=WorkflowConfig.load("text2sql"), corpus_db=None,
                            analytics_db=open_readonly(analytics_path), client=client,
                            trace_db=get_trace_db(tmp_path / "traces.db"), echo=False)
    state = runner.run("How many meetings did each city hold?")

    assert state.status == "complete"
    assert state.sql_results and "Springfield" in state.sql_results[0]["answer"]
    assert "Springfield" in state.final_report


def test_failed_step_marks_run_failed_and_traces_the_error(corpus, tmp_path):
    class ExplodingClient:
        class messages:
            @staticmethod
            def parse(**kwargs):
                raise RuntimeError("API is down")

            @staticmethod
            def create(**kwargs):
                raise RuntimeError("API is down")

    runner = WorkflowRunner(config=WorkflowConfig.load("deep_research"), corpus_db=corpus,
                            analytics_db=None, client=ExplodingClient(),
                            trace_db=get_trace_db(tmp_path / "traces.db"), echo=False)
    state = runner.run("question")

    assert state.status == "failed"
    db = get_trace_db(tmp_path / "traces.db")
    errors = db.execute("SELECT error FROM trace_steps WHERE kind='error' AND run_id=?",
                        (state.run_id,)).fetchall()
    assert errors and "API is down" in errors[0][0]


def _trace_previews(trace_path, run_id) -> list[str]:
    db = get_trace_db(trace_path)
    return [row[0] for row in db.execute(
        "SELECT output_preview FROM trace_steps WHERE run_id = ?", (run_id,)).fetchall()]
