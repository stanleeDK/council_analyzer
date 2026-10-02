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
are recorded in the run's state so a human can disagree with them.
"""
from __future__ import annotations

import re

from app.models.schemas import CriticReport
from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import ClaimCheck, Evidence, TaskState
from app.text import clip, plural

# Merged across batches, so bound it: each query buys a search in the
# follow-up research pass and spends the run's tool budget.
MAX_FOLLOW_UP_QUERIES = 6

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
    # Headroom, not a target - unused output tokens cost nothing. It needs to
    # cover thinking *and* the verdicts; a long draft can still eat the whole
    # allowance, so a genuinely long draft may need to raise this.
    return AgentSpec(name="critic", model=model, system=SYSTEM, max_tokens=16000)


def run(agent: Agent, state: TaskState) -> CriticReport:
    """Critique the whole draft in one call and record the verdicts."""
    if not state.draft.strip():
        state.claim_checks = []
        agent.log("  [critic] draft was empty - nothing to verify")
        return CriticReport(verdicts=[], needs_more_research=False, follow_up_queries=[])

    by_id = state.evidence_by_id()
    manifest = " ".join(e.citation_id for e in state.evidence) or "(none - no evidence was retrieved)"
    prompt = _prompt(state, manifest, by_id)

    report = agent.run_structured(prompt, CriticReport)
    report.follow_up_queries = report.follow_up_queries[:MAX_FOLLOW_UP_QUERIES]

    state.claim_checks = []
    for verdict in report.verdicts:
        state.claim_checks.append(ClaimCheck(
            claim=verdict.claim, status=verdict.status,
            reason=verdict.reason, action=verdict.action,
        ))

    agent.log(_summary(report))
    return report


def _summary(report: CriticReport) -> str:
    """Tallies, then every claim that is not clean.

    Supported claims are the boring majority; listing them would bury the
    handful that are about to change the final report.
    """
    counts = {"supported": 0, "partially_supported": 0, "unsupported": 0}
    for verdict in report.verdicts:
        counts[verdict.status] = counts.get(verdict.status, 0) + 1

    lines = [f"  [critic] {plural(len(report.verdicts), 'claim')} checked - "
             f"{counts['supported']} supported, "
             f"{counts['partially_supported']} partial, "
             f"{counts['unsupported']} unsupported"]

    for verdict in report.verdicts:
        if verdict.status != "supported":
            lines.append(f"           [{verdict.status}] {clip(verdict.claim, 110)}")

    if report.follow_up_queries:
        lines.append("           follow-ups: " + ", ".join(report.follow_up_queries))
    return "\n".join(lines)


# -- prompt construction --------------------------------------------------

def _prompt(state: TaskState, manifest: str, by_id: dict[str, Evidence]) -> str:
    cited_ids = _cited_ids(state.draft)
    passages = []
    for citation_id in cited_ids:
        if citation_id in by_id:
            passages.append(_format_passage(by_id[citation_id]))
    passage_text = "\n\n".join(passages) if passages else "(the draft cites no passage from the evidence list)"

    return (
        f"Objective: {state.objective}\n\n"
        f"=== DRAFT (verify every substantive factual claim below) ===\n{state.draft}\n\n"
        f"=== VALID CITATION IDS (any ID not in this list is fabricated) ===\n{manifest}\n\n"
        f"=== PASSAGES CITED ABOVE (full text) ===\n{passage_text}\n\n"
        "Return a verdict for every substantive factual claim in the draft."
    )


def _format_passage(e: Evidence) -> str:
    return (f"[{e.citation_id}] {e.city} — {e.title} ({e.start_ts:.0f}s-{e.end_ts:.0f}s)\n"
            f"{e.text}")


def _cited_ids(text: str) -> list[str]:
    """Citation IDs in order of first appearance, deduped."""
    seen: set[str] = set()
    ordered: list[str] = []
    for citation_id in re.findall(r"\[(C\d+)\]", text):
        if citation_id not in seen:
            seen.add(citation_id)
            ordered.append(citation_id)
    return ordered
