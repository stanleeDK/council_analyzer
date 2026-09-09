"""Stage 1: structured planner.

Turns a free-text question into a validated ResearchPlan. Deliberately has
no dependency on the corpus/embeddings - it's a standalone building block
(see README.md Stage 1) that later stages call before any retrieval happens.
"""
from __future__ import annotations

from anthropic import Anthropic

from app.models.schemas import ResearchPlan

PLANNER_MODEL = "claude-haiku-4-5"

_SYSTEM = """You are a research planner for a system that analyzes city and \
county council meeting transcripts. Given a user's question, decompose it \
into an objective and a small number of concrete subquestions that could \
each be answered by searching meeting transcripts. If specific cities or \
counties are named or clearly implied, list them in relevant_cities; \
otherwise leave that list empty."""


def make_plan(question: str, client: Anthropic | None = None) -> ResearchPlan:
    client = client or Anthropic()
    response = client.messages.parse(
        model=PLANNER_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": question}],
        output_format=ResearchPlan,
    )
    return response.parsed_output
