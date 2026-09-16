"""Reporter agent.

Two jobs, same agent: write the first draft from the researcher's
findings, and write the final report after the critic has ruled on it.
Separated from research so the model doing synthesis can't quietly
"remember" facts it never retrieved.
"""
from __future__ import annotations

from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import TaskState

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
    ]
    if state.sql_results:
        sections += ["", "=== QUANTITATIVE ANALYSIS ==="]
        sections += [r["answer"] for r in state.sql_results]
    sections += ["", "=== EVIDENCE (the only valid citation IDs) ===", _evidence_block(state),
                 "", "Write the draft answer."]

    result = agent.run("\n".join(sections))
    state.draft = result.text
    state.notes.append(f"reporter: draft written ({len(result.text)} chars)")
    return result.text


def revise(agent: Agent, state: TaskState) -> str:
    """Apply the critic's verdicts. If the critic found nothing, the draft stands."""
    if not state.claim_checks:
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
    state.notes.append(f"reporter: revised after {len(state.unsupported_claims())} unsupported claims")
    return result.text
