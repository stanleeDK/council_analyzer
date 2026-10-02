"""Reporter agent.

Two jobs, same agent: write the first draft from the researcher's
findings, and write the final report after the critic has ruled on it.
Separated from research so the model doing synthesis can't quietly
"remember" facts it never retrieved.
"""
from __future__ import annotations

import re

from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import TaskState
from app.text import clip

SYSTEM = """You write evidence-grounded answers about city/county council meetings.

Hard rules:
- Every factual claim carries a citation ID like [C123], taken from the evidence list \
you are given. Never invent an ID.
- Never add facts that are not in the evidence, even if you believe them.
- Keep fact, inference and uncertainty visibly separate.
- If the evidence does not answer part of the question, say so in its own line rather \
than padding around it.

Structure the answer as:

## Summary
## Key Findings
## Evidence Gaps and Uncertainties
## Sources

Be concise. Council transcripts are verbose; your answer should not be."""


def spec(model: str) -> AgentSpec:
    return AgentSpec(name="reporter", model=model, system=SYSTEM, max_tokens=8000)


def _evidence_block(state: TaskState) -> str:
    if not state.evidence:
        return "(no evidence was retrieved)"
    return "\n\n".join(
        f"[{e.citation_id}] {e.city} — {e.title} "
        f"(uploaded {e.upload_date or 'unknown'}, {e.start_ts:.0f}s-{e.end_ts:.0f}s)\n{e.text}"
        for e in state.evidence
    )


def draft(agent: Agent, state: TaskState, research_notes: str) -> str:
    sections = [
        f"Objective: {state.objective}",
        "",
        "=== RESEARCH NOTES ===",
        research_notes or "(none)",
        "",
        "=== EVIDENCE (the only valid citation IDs) ===",
        _evidence_block(state),
        "",
        "Write the draft answer.",
    ]

    result = agent.run("\n".join(sections))
    state.draft = result.text

    agent.log(_summary("draft", result.text, len(state.evidence)))
    return result.text


def revise(agent: Agent, state: TaskState) -> str:
    """Apply the critic's verdicts. If the critic found nothing, the draft stands."""
    if not state.claim_checks:
        agent.log("  [reporter] no verdicts to apply - the draft stands as final")
        state.final_report = state.draft
        return state.final_report

    verdicts = "\n".join(
        f"- [{c.status}] {c.claim}\n    reason: {c.reason}"
        + (f"\n    action: {c.action}" if c.action else "")
        for c in state.claim_checks
    )
    prompt = (
        f"Objective: {state.objective}\n\n"
        f"=== YOUR DRAFT ===\n{state.draft}\n\n"
        f"=== VERIFICATION RESULTS ===\n{verdicts}\n\n"
        f"=== EVIDENCE (the only valid citation IDs) ===\n{_evidence_block(state)}\n\n"
        "Rewrite the answer applying every verdict: remove or qualify unsupported claims, "
        "soften overstated ones, keep supported ones. Do not add new claims. If removing a "
        "claim leaves a gap, name the gap under Evidence Gaps and Uncertainties."
    )
    result = agent.run(prompt)
    state.final_report = result.text

    extra = (f"applied {len(state.claim_checks)} verdicts "
             f"({len(state.unsupported_claims())} unsupported)")
    agent.log(_summary("final", result.text, len(state.evidence), extra=extra))
    return result.text


def _summary(label: str, text: str, evidence_count: int, extra: str = "") -> str:
    """Shape of what was written: sections, length, and how many distinct
    passages it actually cites. A report citing 2 of 60 retrieved passages is
    worth noticing."""
    headings = []
    for line in text.splitlines():
        if line.startswith("#"):
            headings.append(line.lstrip("# ").strip())

    citation_ids = set(re.findall(r"\[(C\d+)\]", text))

    first_line = f"  [reporter] {label}: {len(text)} chars - cites {len(citation_ids)}/{evidence_count} passages"
    if extra:
        first_line += f" - {extra}"

    lines = [first_line]
    if headings:
        lines.append("             sections: " + clip(" | ".join(headings), 160))
    return "\n".join(lines)
