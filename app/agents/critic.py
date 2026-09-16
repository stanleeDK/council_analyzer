"""Critic agent.

Checks the draft's claims against the evidence actually in state - not
against the world. Returns structured verdicts so the workflow can branch
on them (trigger more research, or strike unsupported claims) instead of
parsing prose.

The critic is another probabilistic model, not an oracle. Its verdicts
are recorded in the trace so a human can disagree with them.
"""
from __future__ import annotations

from app.models.schemas import CriticReport
from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import ClaimCheck, TaskState

SYSTEM = """You verify a draft answer against the evidence passages that were actually \
retrieved. You are checking grounding, not writing style.

For each substantive factual claim in the draft:
- supported: the cited passages clearly state it.
- partially_supported: the passages point that way but the draft overstates \
certainty, scope, or numbers.
- unsupported: no cited passage backs it, or the citation ID does not exist in the \
evidence list.

Rules:
- A claim citing a passage that does not appear in the evidence list is automatically \
unsupported - say so explicitly.
- Numbers, dates and outcomes ("passed unanimously", "fifth annual") need a passage \
that actually contains them.
- Transcripts are auto-generated captions with misheard proper nouns; a plausible \
near-spelling in a passage counts as support if the context matches, but note it.
- Saying "the corpus does not cover this" is a supported claim if the searches really \
came back empty - do not flag honest gaps as unsupported claims.

Set needs_more_research only when a targeted follow-up search could plausibly fix an \
unsupported claim."""


def spec(model: str) -> AgentSpec:
    return AgentSpec(name="critic", model=model, system=SYSTEM, max_tokens=4096)


def run(agent: Agent, state: TaskState) -> CriticReport:
    evidence_block = "\n\n".join(
        f"[{e.citation_id}] {e.city} — {e.title} ({e.start_ts:.0f}s-{e.end_ts:.0f}s)\n{e.text}"
        for e in state.evidence
    ) or "(no evidence was retrieved)"

    prompt = (
        f"Objective: {state.objective}\n\n"
        f"=== DRAFT UNDER REVIEW ===\n{state.draft}\n\n"
        f"=== EVIDENCE AVAILABLE (the only valid citation IDs) ===\n{evidence_block}\n\n"
        "Return a verdict for every substantive factual claim in the draft."
    )
    report = agent.run_structured(prompt, CriticReport)

    state.claim_checks = [
        ClaimCheck(claim=v.claim, status=v.status, reason=v.reason, action=v.action)
        for v in report.verdicts
    ]
    unsupported = len(state.unsupported_claims())
    state.notes.append(f"critic: {len(report.verdicts)} claims checked, {unsupported} unsupported, "
                       f"needs_more_research={report.needs_more_research}")
    return report
