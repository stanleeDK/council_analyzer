"""Research agent.

The one genuinely agentic step: it decides what to search for, reads what
comes back, and decides whether to search again. Evidence accumulates in
shared state as a side effect of the search tool, so later agents can see
every passage this one retrieved.
"""
from __future__ import annotations

import re

from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import Finding, TaskState
from app.text import clip, plural

SYSTEM = """You research questions using ONLY the search_transcripts tool over a corpus \
of city/county council meeting transcripts.

Method:
- Search for each subquestion. If results are thin or off-topic, search again with \
different phrasing before giving up. Rare proper nouns (event names, company names) \
work better as short literal queries than as long questions.
- Transcripts are auto-generated captions: they contain filler words, misheard proper \
nouns, and no speaker labels. Treat near-miss spellings as possible matches and say so.
- Cite passages by their bracketed IDs, e.g. [C123]. Never invent a citation ID.
- Distinguish what was actually said from what you infer.
- If the corpus genuinely does not cover something, say so plainly. "Not found in the \
corpus" is a correct and useful answer - do not fill the gap with general knowledge.

When you have searched enough, write one short finding per subquestion, in this format:

SUBQUESTION: <the subquestion>
FINDING: <what the evidence shows, with [C##] citations>

Do not write a report or executive summary - that is a later step's job."""


def spec(model: str, tools: list[str]) -> AgentSpec:
    return AgentSpec(name="researcher", model=model, system=SYSTEM, max_tokens=8000, tools=tools)


def run(agent: Agent, state: TaskState, extra_instructions: str = "") -> str:
    lines = [f"Overall objective: {state.objective}", "", "Subquestions to research:"]
    lines += [f"{i}. {q}" for i, q in enumerate(state.subquestions or [state.objective], 1)]
    if state.relevant_cities:
        lines += ["", f"Jurisdictions of interest: {', '.join(state.relevant_cities)}",
                  "(Use the city filter only when it clearly helps; city names must match exactly.)"]
    if extra_instructions:
        lines += ["", extra_instructions]

    # Snapshot first: a follow-up pass appends to both lists, and the note
    # should describe this pass, not the whole run.
    findings_before = len(state.findings)
    evidence_before = len(state.evidence)

    result = agent.run("\n".join(lines))
    state.findings.extend(_parse_findings(result.text))

    agent.log(_summary(state, result, state.findings[findings_before:],
                       len(state.evidence) - evidence_before))
    return result.text


def _summary(state: TaskState, result, new_findings: list[Finding], new_passages: int) -> str:
    """What this pass actually found, one line per subquestion.

    The searches themselves already print as tool calls; what they add up to
    does not, and that is the part worth reading.
    """
    header = (f"  [researcher] {plural(result.tool_calls, 'search', 'searches')} -> "
              f"{plural(new_passages, 'new passage')} "
              f"({len(state.evidence)} total) - stopped: {result.stopped_because}")
    lines = [header]
    if not new_findings:
        lines.append("               (no SUBQUESTION/FINDING pairs parsed from the response)")
        return "\n".join(lines)
    for finding in new_findings:
        # The finding text already carries its citations inline; only say
        # something when it carries none.
        body = clip(finding.finding, 130)
        if finding.citation_ids:
            lines.append(f"               - {body}")
        else:
            lines.append(f"               - {body} (uncited)")
    return "\n".join(lines)


def _parse_findings(text: str) -> list[Finding]:
    """Pull SUBQUESTION/FINDING pairs out of the agent's free text.

    Best-effort: the prose is still kept verbatim in the draft prompt, so a
    parse miss degrades structure, not content.
    """
    findings: list[Finding] = []
    question = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("SUBQUESTION:"):
            question = line.split(":", 1)[1].strip()
        elif line.upper().startswith("FINDING:"):
            body = line.split(":", 1)[1].strip()
            findings.append(Finding(
                question=question,
                finding=body,
                citation_ids=_citation_ids(body),
            ))
            question = ""
    return findings


def _citation_ids(text: str) -> list[str]:
    return re.findall(r"\[(C\d+)\]", text)
