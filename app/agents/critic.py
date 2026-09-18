"""Critic agent.

Checks the draft's claims against the evidence actually in state - not
against the world. Returns structured verdicts so the workflow can branch
on them (trigger more research, or strike unsupported claims) instead of
parsing prose.

Two things shape the prompt this builds:

- Detecting a *fabricated* citation needs the complete list of valid IDs,
  even for passages the draft never cites. So the ID manifest is always
  whole.
- Verifying what a passage *says* only needs the passages the draft
  actually cites. So full text is sent for those alone. On a typical run
  that is a handful out of dozens retrieved.

The critic is another probabilistic model, not an oracle. Its verdicts
are recorded in the trace so a human can disagree with them.
"""
from __future__ import annotations

import re

from app.models.schemas import CriticReport
from app.observability.traces import clip, plural
from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import ClaimCheck, Evidence, TaskState

# One verdict costs ~140 output tokens (claim + reason + action + JSON
# scaffolding). A draft section this size yields well under max_tokens'
# worth of them, which is what keeps responses from truncating mid-JSON.
MAX_DRAFT_CHARS_PER_BATCH = 4000

# Merged across batches, so bound it: each query buys a search in the
# follow-up research pass and spends the run's tool budget.
MAX_FOLLOW_UP_QUERIES = 6

_CITATION_RE = re.compile(r"\[(C\d+)\]")

SYSTEM = """You verify a draft answer against the evidence passages that were actually \
retrieved. You are checking grounding, not writing style.

For each substantive factual claim in the draft:
- supported: the cited passages clearly state it.
- partially_supported: the passages point that way but the draft overstates \
certainty, scope, or numbers.
- unsupported: no cited passage backs it, or the citation ID does not exist in the \
evidence list.

Rules:
- A claim citing an ID that is not in the VALID CITATION IDS list is automatically \
unsupported - say so explicitly and name the bogus ID.
- Numbers, dates and outcomes ("passed unanimously", "fifth annual") need a passage \
that actually contains them.
- Transcripts are auto-generated captions with misheard proper nouns; a plausible \
near-spelling in a passage counts as support if the context matches, but note it.
- Saying "the corpus does not cover this" is a supported claim if the searches really \
came back empty - do not flag honest gaps as unsupported claims.
- You are shown the full text only of passages this part of the draft cites. An ID in \
the valid list with no passage shown is a real passage that simply was not cited here; \
do not treat its absence as evidence of anything.

Set needs_more_research only when a targeted follow-up search could plausibly fix an \
unsupported claim."""


def spec(model: str) -> AgentSpec:
    # Headroom, not a target: batching keeps each response far below this,
    # and unused output tokens cost nothing.
    return AgentSpec(name="critic", model=model, system=SYSTEM, max_tokens=8000)


def run(agent: Agent, state: TaskState) -> CriticReport:
    """Critique the draft, one bounded batch at a time, and merge the verdicts."""
    segments = split_draft(state.draft)
    if not segments:
        state.claim_checks = []
        agent.tracer.record("note", agent="critic",
                            output_preview="draft was empty \u2014 nothing to verify")
        state.notes.append("critic: draft was empty, nothing to verify")
        return CriticReport(verdicts=[], needs_more_research=False, follow_up_queries=[])

    by_id = state.evidence_by_id()
    manifest = " ".join(e.citation_id for e in state.evidence) or "(none - no evidence was retrieved)"

    reports = [
        agent.run_structured(
            _prompt(state, segment, manifest, by_id, index, len(segments)), CriticReport
        )
        for index, segment in enumerate(segments, 1)
    ]
    report = _merge(reports)

    state.claim_checks = [
        ClaimCheck(claim=v.claim, status=v.status, reason=v.reason, action=v.action)
        for v in report.verdicts
    ]
    agent.tracer.record("note", agent="critic",
                        output_preview=_summary(report, len(segments)))
    unsupported = len(state.unsupported_claims())
    state.notes.append(f"critic: {len(report.verdicts)} claims checked across {len(segments)} "
                       f"batch(es), {unsupported} unsupported, "
                       f"needs_more_research={report.needs_more_research}")
    return report


def _summary(report: CriticReport, batches: int) -> str:
    """Tallies, then every claim that is not clean.

    Supported claims are the boring majority; listing them would bury the
    handful that are about to change the final report.
    """
    counts = {"supported": 0, "partially_supported": 0, "unsupported": 0}
    for verdict in report.verdicts:
        counts[verdict.status] = counts.get(verdict.status, 0) + 1
    batch_note = f" across {batches} batches" if batches > 1 else ""
    lines = [f"{plural(len(report.verdicts), 'claim')} checked{batch_note} \u00b7 "
             f"{counts['supported']} supported, "
             f"{counts['partially_supported']} partial, "
             f"{counts['unsupported']} unsupported"]

    flagged = [v for v in report.verdicts if v.status != "supported"]
    lines += [f"\u00b7 [{v.status}] {clip(v.claim, 110)}" for v in flagged]
    if report.follow_up_queries:
        lines.append("follow-ups: " + ", ".join(report.follow_up_queries))
    return "\n".join(lines)


# -- prompt construction --------------------------------------------------

def _prompt(state: TaskState, segment: str, manifest: str, by_id: dict[str, Evidence],
            index: int, total: int) -> str:
    cited = _cited_ids(segment)
    passages = "\n\n".join(_format_passage(by_id[cid]) for cid in cited if cid in by_id)
    if not passages:
        passages = "(this part of the draft cites no passage from the evidence list)"

    scope = f"PART {index} OF {total} OF THE DRAFT" if total > 1 else "DRAFT"
    header = f"=== {scope} (verify only the claims below) ==="
    return (
        f"Objective: {state.objective}\n\n"
        f"{header}\n{segment}\n\n"
        f"=== VALID CITATION IDS (any ID not in this list is fabricated) ===\n{manifest}\n\n"
        f"=== PASSAGES CITED ABOVE (full text) ===\n{passages}\n\n"
        "Return a verdict for every substantive factual claim in this part of the draft."
    )


def _format_passage(e: Evidence) -> str:
    return (f"[{e.citation_id}] {e.city} — {e.title} ({e.start_ts:.0f}s-{e.end_ts:.0f}s)\n"
            f"{e.text}")


def _cited_ids(text: str) -> list[str]:
    """Citation IDs in order of first appearance, deduped."""
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in _CITATION_RE.findall(text):
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


# -- batching -------------------------------------------------------------

def split_draft(draft: str, max_chars: int = MAX_DRAFT_CHARS_PER_BATCH) -> list[str]:
    """Split a draft into segments small enough to critique in one call.

    The number of verdicts a critique emits scales with the draft, not the
    evidence, so a long draft is what truncates a response. Splits on
    markdown headings first (a natural claim boundary), then on blank lines
    when a single section is still too long. Never splits mid-paragraph - a
    claim cut in half cannot be verified.
    """
    if not draft.strip():
        return []
    if len(draft) <= max_chars:
        return [draft]

    segments: list[str] = []
    current = ""
    for block in _heading_blocks(draft):
        for piece in _split_oversized(block, max_chars):
            if current and len(current) + len(piece) + 2 > max_chars:
                segments.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        segments.append(current)
    return segments


def _heading_blocks(draft: str) -> list[str]:
    """Split on markdown headings, keeping each heading with its body."""
    blocks: list[str] = []
    current: list[str] = []
    for line in draft.splitlines():
        if line.startswith("#") and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_oversized(block: str, max_chars: int) -> list[str]:
    """Break one over-long block on paragraph boundaries.

    A single paragraph longer than max_chars comes back whole: an oversized
    batch is better than a claim severed in the middle.
    """
    if len(block) <= max_chars:
        return [block]
    pieces: list[str] = []
    current = ""
    for paragraph in block.split("\n\n"):
        if current and len(current) + len(paragraph) + 2 > max_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def _merge(reports: list[CriticReport]) -> CriticReport:
    """Concatenate verdicts; union follow-up queries; OR the research flag."""
    verdicts = [v for report in reports for v in report.verdicts]
    queries: list[str] = []
    seen: set[str] = set()
    for report in reports:
        for query in report.follow_up_queries:
            key = query.strip().lower()
            if key and key not in seen:
                seen.add(key)
                queries.append(query.strip())
    return CriticReport(
        verdicts=verdicts,
        needs_more_research=any(r.needs_more_research for r in reports),
        follow_up_queries=queries[:MAX_FOLLOW_UP_QUERIES],
    )
