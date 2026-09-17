"""Critic batching, prompt shaping, and structured-output failure handling.

The critic's input is deliberately asymmetric: the *complete* citation-ID
manifest (so a fabricated ID is detectable) but full passage text only for
what the draft cites. Its output is batched, because the number of verdicts
scales with the draft and a long draft used to truncate the response
mid-JSON.
"""
import pytest

from app.agents import critic
from app.models.schemas import ClaimVerdict, CriticReport
from app.runtime.state import Evidence, TaskState


# -- helpers --------------------------------------------------------------

class RecordingAgent:
    """Stands in for Agent: records prompts, returns queued reports."""

    def __init__(self, reports):
        self._reports = list(reports)
        self.prompts = []

    def run_structured(self, prompt, output_format):
        self.prompts.append(prompt)
        return self._reports.pop(0)


def _evidence(cid: str, text: str) -> Evidence:
    return Evidence(citation_id=cid, city="Springfield", title="Council",
                    upload_date="2026-01-01", video_id="v", start_ts=0.0, end_ts=60.0,
                    text=text, score=0.5)


def _report(*verdicts, needs_more=False, queries=()):
    return CriticReport(verdicts=list(verdicts), needs_more_research=needs_more,
                        follow_up_queries=list(queries))


def _verdict(claim, status="supported"):
    return ClaimVerdict(claim=claim, status=status, reason="r", action="")


# -- splitting ------------------------------------------------------------

def test_empty_draft_produces_no_segments():
    assert critic.split_draft("") == []
    assert critic.split_draft("   \n  ") == []


def test_short_draft_stays_one_segment():
    assert critic.split_draft("## Summary\nA short draft.") == ["## Summary\nA short draft."]


def test_long_draft_splits_on_headings():
    draft = "\n\n".join(f"## Section {i}\n{'word ' * 400}" for i in range(4))
    segments = critic.split_draft(draft, max_chars=1000)

    assert len(segments) > 1
    # every heading survives somewhere, and none is orphaned from its body
    for i in range(4):
        owner = [s for s in segments if f"## Section {i}" in s]
        assert len(owner) == 1
        assert "word" in owner[0]


def test_split_never_cuts_a_paragraph_in_half():
    paragraph = "This is one indivisible claim. " * 100   # ~3000 chars, no blank lines
    segments = critic.split_draft(f"## A\n{paragraph}", max_chars=500)

    # oversized rather than severed
    assert len(segments) == 1
    assert paragraph.strip() in segments[0]


def test_segments_cover_the_whole_draft():
    draft = "\n\n".join(f"## S{i}\nparagraph {i} " + "x " * 200 for i in range(5))
    segments = critic.split_draft(draft, max_chars=800)
    for i in range(5):
        assert f"paragraph {i}" in "\n\n".join(segments)


# -- citation extraction --------------------------------------------------

def test_cited_ids_dedupe_and_keep_first_appearance_order():
    assert critic._cited_ids("b [C9] a [C2] c [C9] d [C1]") == ["C9", "C2", "C1"]


def test_cited_ids_ignores_non_citation_brackets():
    assert critic._cited_ids("[note] [C12] [TODO] [x1]") == ["C12"]


# -- prompt shape ---------------------------------------------------------

def test_prompt_lists_every_id_but_only_cites_text_for_referenced_passages():
    state = TaskState(objective="o", workflow="w")
    state.add_evidence([
        _evidence("C1", "TEXT OF ONE"),
        _evidence("C2", "TEXT OF TWO"),
        _evidence("C3", "TEXT OF THREE"),
    ])
    state.draft = "## Summary\nThe council acted [C2]."

    agent = RecordingAgent([_report(_verdict("The council acted"))])
    critic.run(agent, state)

    prompt = agent.prompts[0]
    # the manifest is complete - that is what makes a bogus ID detectable
    assert "C1 C2 C3" in prompt
    # but only the cited passage's text is paid for
    assert "TEXT OF TWO" in prompt
    assert "TEXT OF ONE" not in prompt
    assert "TEXT OF THREE" not in prompt


def test_prompt_says_so_when_the_draft_cites_nothing():
    state = TaskState(objective="o", workflow="w")
    state.add_evidence([_evidence("C1", "TEXT OF ONE")])
    state.draft = "## Summary\nThe corpus does not cover this."

    agent = RecordingAgent([_report()])
    critic.run(agent, state)

    assert "cites no passage" in agent.prompts[0]
    assert "C1" in agent.prompts[0]          # manifest still complete


def test_prompt_handles_an_empty_corpus():
    state = TaskState(objective="o", workflow="w")
    state.draft = "## Summary\nNothing was found."

    agent = RecordingAgent([_report()])
    critic.run(agent, state)

    assert "no evidence was retrieved" in agent.prompts[0]


def test_fabricated_citation_gets_no_passage_but_stays_in_the_draft_text():
    state = TaskState(objective="o", workflow="w")
    state.add_evidence([_evidence("C1", "REAL TEXT")])
    state.draft = "## Summary\nInvented fact [C999]."

    agent = RecordingAgent([_report(_verdict("Invented fact", status="unsupported"))])
    critic.run(agent, state)

    prompt = agent.prompts[0]
    assert "[C999]" in prompt                 # the critic can see the bogus ID
    assert "cites no passage" in prompt       # and that nothing backs it
    assert state.claim_checks[0].status == "unsupported"


# -- batching -------------------------------------------------------------

def test_long_draft_is_critiqued_in_several_calls_and_verdicts_merge():
    state = TaskState(objective="o", workflow="w")
    state.add_evidence([_evidence("C1", "T1"), _evidence("C2", "T2")])
    state.draft = "\n\n".join(f"## Section {i}\nClaim {i} [C1] " + "filler " * 400
                              for i in range(4))

    reports = [_report(_verdict(f"claim {i}")) for i in range(4)]
    agent = RecordingAgent(reports)
    report = critic.run(agent, state)

    assert len(agent.prompts) > 1                       # actually batched
    assert len(report.verdicts) == len(agent.prompts)   # all verdicts kept
    assert len(state.claim_checks) == len(report.verdicts)


def test_each_batch_is_labelled_with_its_position():
    state = TaskState(objective="o", workflow="w")
    state.draft = "\n\n".join(f"## S{i}\n" + "word " * 500 for i in range(3))
    agent = RecordingAgent([_report() for _ in range(5)])

    critic.run(agent, state)

    total = len(agent.prompts)
    assert total > 1
    assert f"PART 1 OF {total}" in agent.prompts[0]
    assert f"PART {total} OF {total}" in agent.prompts[-1]


def test_single_batch_prompt_is_not_labelled_as_a_part():
    state = TaskState(objective="o", workflow="w")
    state.draft = "## Summary\nShort."
    agent = RecordingAgent([_report()])

    critic.run(agent, state)
    assert "PART 1 OF" not in agent.prompts[0]


def test_empty_draft_skips_the_model_entirely():
    state = TaskState(objective="o", workflow="w")
    state.draft = ""
    agent = RecordingAgent([])          # would IndexError if called

    report = critic.run(agent, state)

    assert agent.prompts == []
    assert report.verdicts == [] and not report.needs_more_research
    assert any("nothing to verify" in n for n in state.notes)


# -- merging --------------------------------------------------------------

def test_merge_ors_the_research_flag():
    merged = critic._merge([_report(needs_more=False), _report(needs_more=True)])
    assert merged.needs_more_research


def test_merge_dedupes_follow_up_queries_case_insensitively():
    merged = critic._merge([
        _report(queries=["landusi fest", "zoning variance"]),
        _report(queries=["Landusi Fest", "budget hearing"]),
    ])
    assert merged.follow_up_queries == ["landusi fest", "zoning variance", "budget hearing"]


def test_merge_caps_follow_up_queries():
    many = [_report(queries=[f"query {i}"]) for i in range(20)]
    merged = critic._merge(many)
    assert len(merged.follow_up_queries) == critic.MAX_FOLLOW_UP_QUERIES


def test_merge_preserves_verdict_order_across_batches():
    merged = critic._merge([
        _report(_verdict("first"), _verdict("second")),
        _report(_verdict("third")),
    ])
    assert [v.claim for v in merged.verdicts] == ["first", "second", "third"]


# -- structured-output failure -------------------------------------------

def test_truncated_structured_response_raises_a_named_error(tmp_path):
    from app.observability.traces import Tracer, get_trace_db
    from app.runtime.agent import Agent, AgentSpec, StructuredOutputError
    from app.runtime.budget import Budget
    from app.runtime.policy import PolicyEngine
    from app.tools.base import ToolContext, ToolRegistry
    from tests.test_agent_loop import FakeUsage

    class TruncatedResponse:
        parsed_output = None
        stop_reason = "max_tokens"
        usage = FakeUsage()

    class TruncatingClient:
        class messages:
            @staticmethod
            def parse(**kwargs):
                return TruncatedResponse()

    db = get_trace_db(tmp_path / "traces.db")
    tracer = Tracer(db=db, run_id="r1", workflow="w", echo=False)
    tracer.start_run("o")
    agent = Agent(
        spec=AgentSpec(name="critic", model="claude-sonnet-5", system="s", max_tokens=4096),
        client=TruncatingClient(), registry=ToolRegistry([]),
        policy=PolicyEngine(agent_tools={}), budget=Budget(), tracer=tracer, ctx=ToolContext(),
    )

    with pytest.raises(StructuredOutputError) as exc:
        agent.run_structured("prompt", CriticReport)

    message = str(exc.value)
    assert "critic" in message
    assert "CriticReport" in message
    assert "stop_reason=max_tokens" in message
    assert "max_tokens=4096" in message      # names the cap that needs raising

    errors = db.execute("SELECT error FROM trace_steps WHERE kind='error' AND run_id='r1'").fetchall()
    assert errors and "stop_reason=max_tokens" in errors[0][0]


def test_non_truncation_failure_reports_its_own_stop_reason(tmp_path):
    from app.observability.traces import Tracer, get_trace_db
    from app.runtime.agent import Agent, AgentSpec, StructuredOutputError
    from app.runtime.budget import Budget
    from app.runtime.policy import PolicyEngine
    from app.tools.base import ToolContext, ToolRegistry
    from tests.test_agent_loop import FakeUsage

    class RefusedResponse:
        parsed_output = None
        stop_reason = "refusal"
        usage = FakeUsage()

    class RefusingClient:
        class messages:
            @staticmethod
            def parse(**kwargs):
                return RefusedResponse()

    tracer = Tracer(db=get_trace_db(tmp_path / "t.db"), run_id="r2", workflow="w", echo=False)
    tracer.start_run("o")
    agent = Agent(
        spec=AgentSpec(name="planner", model="claude-sonnet-5", system="s"),
        client=RefusingClient(), registry=ToolRegistry([]),
        policy=PolicyEngine(agent_tools={}), budget=Budget(), tracer=tracer, ctx=ToolContext(),
    )

    with pytest.raises(StructuredOutputError, match="stop_reason=refusal"):
        agent.run_structured("prompt", CriticReport)
