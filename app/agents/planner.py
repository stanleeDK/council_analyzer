"""Planner agent.

Turns an ambiguous request into a structured plan. No tools, no corpus
access - it decomposes the question and nothing else. Deliberately the
cheapest model in the workflow: this is a structured-extraction task,
not a reasoning-heavy one.
"""
from __future__ import annotations

from app.models.schemas import ResearchPlan
from app.runtime.agent import Agent, AgentSpec
from app.runtime.state import TaskState

SYSTEM = """You are the planning step of a system that answers questions about \
city and county council meeting transcripts.

Decompose the user's question into an objective plus 3-6 concrete subquestions, \
each of which could plausibly be answered by searching meeting transcripts. \
Subquestions should be specific enough to make good search queries - prefer \
"what did the council decide about the short-term rental ordinance" over \
"what happened with housing".

If specific cities or counties are named or clearly implied, list them exactly as \
named. Set needs_quantitative_data only when the question is fundamentally about \
counts, totals or trends across many meetings rather than what was said in them."""


def spec(model: str) -> AgentSpec:
    return AgentSpec(name="planner", model=model, system=SYSTEM, max_tokens=1024)


def run(agent: Agent, state: TaskState) -> ResearchPlan:
    plan = agent.run_structured(
        f"User question: {state.objective}\n\nProduce the research plan.",
        ResearchPlan,
    )
    state.subquestions = plan.subquestions
    state.relevant_cities = plan.relevant_cities
    state.needs_quantitative_data = plan.needs_quantitative_data
    state.notes.append(f"planner: {len(plan.subquestions)} subquestions, "
                       f"quantitative={plan.needs_quantitative_data}")
    return plan
