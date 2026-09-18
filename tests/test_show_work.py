"""Every agent narrates what it did into the trace as a `note`.

The model and tool calls already echo, but they only show that something
happened, not what it amounted to. These cover the summaries themselves and
the multi-line echo that renders them.
"""
from app.agents import planner, reporter, researcher
from app.models.schemas import ResearchPlan
from app.observability.traces import (NOTE_ECHO_LIMIT, Tracer, clip, get_trace_db,
                                      plural)
from app.runtime.state import Finding


# -- clip -----------------------------------------------------------------

def test_clip_collapses_whitespace_to_one_line():
    assert clip("a\n  b\tc  ", 100) == "a b c"


def test_clip_cuts_on_a_word_boundary():
    out = clip("alpha beta gamma delta", 12)
    assert out == "alpha beta…"
    assert "gam" not in out


def test_clip_leaves_short_text_alone():
    assert clip("short", 100) == "short"


# -- plural ---------------------------------------------------------------

def test_plural_uses_the_singular_for_one():
    assert plural(1, "search", "searches") == "1 search"
    assert plural(1, "claim") == "1 claim"


def test_plural_defaults_to_adding_s():
    assert plural(3, "claim") == "3 claims"
    assert plural(0, "claim") == "0 claims"


def test_plural_takes_an_irregular_form_rather_than_a_suffix():
    assert plural(2, "query", "queries") == "2 queries"
    assert plural(2, "search", "searches") == "2 searches"


# -- note echo ------------------------------------------------------------

def _tracer(tmp_path, echo=True) -> Tracer:
    t = Tracer(db=get_trace_db(tmp_path / "t.db"), run_id="r", workflow="w", echo=echo)
    t.start_run("o")
    return t


def test_multiline_note_aligns_continuation_lines(tmp_path, capsys):
    _tracer(tmp_path).record("note", agent="planner", output_preview="first\nsecond\nthird")

    lines = capsys.readouterr().out.rstrip("\n").splitlines()
    assert lines[0] == "  [planner] first"
    # continuation lines start under 'first', not under the prefix
    indent = lines[0].index("first")
    assert lines[1] == " " * indent + "second"
    assert lines[2] == " " * indent + "third"


def test_note_echo_is_bounded(tmp_path, capsys):
    _tracer(tmp_path).record("note", agent="a", output_preview="x" * (NOTE_ECHO_LIMIT * 3))
    assert len(capsys.readouterr().out) < NOTE_ECHO_LIMIT * 2


def test_quiet_mode_prints_nothing(tmp_path, capsys):
    _tracer(tmp_path, echo=False).record("note", agent="a", output_preview="hidden")
    assert capsys.readouterr().out == ""


def test_note_is_persisted_even_when_not_echoed(tmp_path):
    tracer = _tracer(tmp_path, echo=False)
    tracer.record("note", agent="planner", output_preview="the plan")
    rows = tracer.db.execute(
        "SELECT output_preview FROM trace_steps WHERE kind='note'").fetchall()
    assert rows == [("the plan",)]


# -- planner --------------------------------------------------------------

def _plan(**kwargs) -> ResearchPlan:
    return ResearchPlan(**{
        "objective": "Track short-term rental policy",
        "subquestions": ["What did the council decide?", "What complaints were raised?"],
        "relevant_cities": ["City of Green Bay"],
        "needs_quantitative_data": False,
        **kwargs,
    })


def test_planner_note_shows_objective_cities_and_numbered_subquestions():
    note = planner._summary(_plan())
    assert "objective: Track short-term rental policy" in note
    assert "cities: City of Green Bay" in note
    assert "quantitative: no" in note
    assert "1. What did the council decide?" in note
    assert "2. What complaints were raised?" in note


def test_planner_note_names_the_unscoped_case():
    assert "all jurisdictions" in planner._summary(_plan(relevant_cities=[]))


def test_planner_note_surfaces_the_quantitative_flag():
    assert "quantitative: yes" in planner._summary(_plan(needs_quantitative_data=True))


# -- researcher -----------------------------------------------------------

class FakeResult:
    def __init__(self, tool_calls=3, stopped_because="end_turn"):
        self.tool_calls = tool_calls
        self.stopped_because = stopped_because
        self.text = ""


def test_researcher_note_counts_this_pass_not_the_whole_run():
    from app.runtime.state import TaskState
    state = TaskState(objective="o", workflow="w")
    state.evidence = [None] * 40          # 40 total after this pass

    note = researcher._summary(state, FakeResult(tool_calls=3), [], new_passages=12)
    assert "3 searches" in note
    assert "12 new passages" in note
    assert "40 total" in note


def test_researcher_note_shows_findings_without_duplicating_their_citations():
    """The finding text already carries [C##] inline - appending them repeats."""
    from app.runtime.state import TaskState
    findings = [Finding(question="q1", finding="A permit was requested [C1] [C2]",
                        citation_ids=["C1", "C2"])]
    note = researcher._summary(TaskState(objective="o", workflow="w"), FakeResult(), findings, 8)
    assert "A permit was requested [C1] [C2]" in note
    assert "[C1] [C2] [C1]" not in note
    assert note.count("[C1]") == 1


def test_researcher_note_flags_an_uncited_finding():
    from app.runtime.state import TaskState
    findings = [Finding(question="q", finding="Something vague", citation_ids=[])]
    note = researcher._summary(TaskState(objective="o", workflow="w"), FakeResult(), findings, 0)
    assert "(uncited)" in note


def test_researcher_note_says_when_nothing_parsed():
    from app.runtime.state import TaskState
    note = researcher._summary(TaskState(objective="o", workflow="w"), FakeResult(), [], 0)
    assert "no SUBQUESTION/FINDING pairs parsed" in note


def test_researcher_note_reports_an_abnormal_stop():
    from app.runtime.state import TaskState
    note = researcher._summary(TaskState(objective="o", workflow="w"),
                               FakeResult(stopped_because="max_iterations"), [], 0)
    assert "stopped: max_iterations" in note


# -- reporter -------------------------------------------------------------

def test_reporter_note_shows_citation_coverage():
    text = "## Summary\nA [C1] and B [C2] and A again [C1]."
    note = reporter._summary("draft", text, evidence_count=60)
    assert "cites 2/60 passages" in note        # distinct IDs, not occurrences


def test_reporter_note_lists_section_headings():
    text = "## Summary\nx\n\n## Key Findings\ny\n\n## Sources\nz"
    note = reporter._summary("draft", text, evidence_count=3)
    assert "sections: Summary | Key Findings | Sources" in note


def test_reporter_note_carries_the_revision_detail():
    note = reporter._summary("final", "## Summary\nx [C1]", 4, extra="applied 7 verdicts")
    assert "final:" in note and "applied 7 verdicts" in note


def test_reporter_note_handles_a_report_with_no_headings():
    note = reporter._summary("draft", "just prose, no headings", evidence_count=1)
    assert "sections:" not in note
    assert "cites 0/1 passages" in note


# -- end to end through the real tracer ----------------------------------

def test_every_agent_note_lands_in_the_trace_db(tmp_path):
    """A workflow run should leave a readable narration behind, not just costs."""
    from tests.test_workflow_e2e import ScriptedClient
    from tests.test_agent_loop import FakeResponse, FakeTextBlock, FakeToolUseBlock
    from app.db.connection import get_db
    from app.models.schemas import ClaimVerdict, CriticReport
    from app.runtime.workflow import WorkflowConfig, WorkflowRunner

    corpus = get_db(tmp_path / "corpus.db")
    corpus.execute(
        "INSERT INTO chunks (city, upload_date, title, video_id, source_file, start_ts, end_ts, "
        "text, embedding) VALUES ('City of Green Bay','2021-03-23','P&P','v','/a.lrc',"
        "1400,1460,'the fifth annual landusi fest','[1.0, 0.0]')"
    )
    corpus.commit()

    import numpy as np
    import app.rag.retrieve as retrieve_mod
    original = retrieve_mod.embed_text
    retrieve_mod.embed_text = lambda text: np.array([1.0, 0.0])
    try:
        client = ScriptedClient(
            create_responses=[
                FakeResponse([FakeToolUseBlock("search_transcripts", {"query": "fest"})],
                             stop_reason="tool_use"),
                FakeResponse([FakeTextBlock("SUBQUESTION: q\nFINDING: a fest permit [C1]")]),
                FakeResponse([FakeTextBlock("## Summary\nA fest permit was requested [C1]")]),
                FakeResponse([FakeTextBlock("## Summary\nA fest permit was requested [C1]")]),
            ],
            parse_outputs=[
                ResearchPlan(objective="Find permits", subquestions=["What permits?"],
                             relevant_cities=["City of Green Bay"], needs_quantitative_data=False),
                CriticReport(verdicts=[ClaimVerdict(claim="A fest permit was requested",
                                                    status="supported", reason="[C1]", action="")],
                             needs_more_research=False, follow_up_queries=[]),
            ],
        )
        trace_db = get_trace_db(tmp_path / "traces.db")
        runner = WorkflowRunner(config=WorkflowConfig.load("deep_research"), corpus_db=corpus,
                                analytics_db=None, client=client, trace_db=trace_db, echo=False)
        state = runner.run("What permits did Green Bay consider?")
    finally:
        retrieve_mod.embed_text = original

    assert state.status == "complete"
    notes = dict(trace_db.execute(
        "SELECT agent, output_preview FROM trace_steps WHERE kind='note' AND run_id=? "
        "GROUP BY agent", (state.run_id,)).fetchall())

    assert "What permits?" in notes["planner"]
    assert "1 search " in notes["researcher"] and "[C1]" in notes["researcher"]
    assert "1 claim checked" in notes["critic"]
    assert "cites 1/1 passages" in notes["reporter"]


def test_data_analyst_note_carries_its_conclusion(tmp_path):
    """The SQL echoes as tool calls; what it concluded does not."""
    from tests.test_workflow_e2e import ScriptedClient
    from tests.test_agent_loop import FakeResponse, FakeTextBlock, FakeToolUseBlock
    from app.db.analytics import build_analytics_db, open_readonly
    from app.db.connection import get_db
    from app.runtime.workflow import WorkflowConfig, WorkflowRunner

    corpus = get_db(tmp_path / "corpus.db")
    corpus.execute(
        "INSERT INTO chunks (city, upload_date, title, video_id, source_file, start_ts, end_ts, "
        "text, embedding) VALUES ('Springfield','2026-01-01','Council','v','/a.lrc',0,60,'t','[1.0]')"
    )
    corpus.commit()
    build_analytics_db(corpus, tmp_path / "analytics.db")

    client = ScriptedClient(
        create_responses=[
            FakeResponse([FakeToolUseBlock("get_schema", {})], stop_reason="tool_use"),
            FakeResponse([FakeToolUseBlock("run_readonly_sql",
                                           {"sql": "SELECT city FROM city_coverage"})],
                         stop_reason="tool_use"),
            FakeResponse([FakeTextBlock("Springfield held 1 meeting in the indexed period.")]),
            FakeResponse([FakeTextBlock("## Summary\nSpringfield held 1 meeting.")]),
        ],
        parse_outputs=[],
    )
    trace_db = get_trace_db(tmp_path / "traces.db")
    runner = WorkflowRunner(config=WorkflowConfig.load("text2sql"), corpus_db=None,
                            analytics_db=open_readonly(tmp_path / "analytics.db"),
                            client=client, trace_db=trace_db, echo=False)
    state = runner.run("How many meetings?")

    note = trace_db.execute(
        "SELECT output_preview FROM trace_steps WHERE kind='note' AND agent='data_analyst' "
        "AND run_id=?", (state.run_id,)).fetchone()[0]
    assert "2 queries" in note
    assert "Springfield held 1 meeting in the indexed period." in note
