"""Data / Text2SQL agent.

Answers questions that are about counts and coverage rather than what was
said - "how many meetings did each city hold in 2026" is a SQL question,
not a retrieval question. Inspects the schema first, writes SELECTs, and
sanity-checks what comes back.
"""
from __future__ import annotations

from app.observability.traces import clip, plural
from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import TaskState

SYSTEM = """You answer quantitative questions about a council-meeting corpus by writing \
read-only SQL.

Method:
- Always call get_schema first. Never guess column names.
- Write one SELECT at a time. The database is read-only; writes are rejected.
- After results come back, sanity-check them: do the magnitudes make sense? Does the \
row count match what you expected? If a query returns 0 rows, consider whether your \
filter was too narrow (e.g. exact city-name matching) and try again.
- Report the finding, the SQL you ran, and any caveat about what the data does or does \
not cover.

Important caveat to carry into your answer: this database describes *meeting coverage* \
(which meetings were recorded, when, how long), derived from ingested transcripts. It \
does not contain votes, motions, or agenda outcomes. If a question needs those, say so \
rather than approximating."""


def spec(model: str, tools: list[str]) -> AgentSpec:
    return AgentSpec(name="data_analyst", model=model, system=SYSTEM, max_tokens=8000, tools=tools)


def run(agent: Agent, state: TaskState) -> str:
    prompt = (
        f"Question: {state.objective}\n\n"
        "Inspect the schema, then answer with SQL. If the question cannot be answered "
        "from meeting-coverage data alone, say exactly what is missing."
    )
    result = agent.run(prompt)
    state.sql_results.append({"question": state.objective, "answer": result.text})

    # The SQL itself already echoes as tool calls; the conclusion drawn from it
    # does not.
    agent.tracer.record("note", agent="data_analyst", output_preview=(
        f"{plural(result.tool_calls, 'query', 'queries')} \u00b7 "
        f"stopped: {result.stopped_because}\n"
        f"\u00b7 {clip(result.text, 260) or '(no answer text)'}"))
    state.notes.append(f"data_analyst: {result.tool_calls} queries, stopped={result.stopped_because}")
    return result.text
