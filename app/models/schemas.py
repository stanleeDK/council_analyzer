"""Pydantic schemas for structured model outputs.

These get converted to JSON Schema by the Anthropic SDK and sent as an
output-format constraint, so responses come back as validated objects
rather than prose that has to be parsed.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    objective: str = Field(description="One sentence restating what the user actually wants to know.")
    subquestions: list[str] = Field(description="3-6 concrete questions answerable from meeting transcripts.")
    relevant_cities: list[str] = Field(description="Cities/counties named or clearly implied; empty if none.")
    needs_quantitative_data: bool = Field(
        description="True if answering requires counts/aggregates across meetings rather than quoted discussion."
    )


class ClaimVerdict(BaseModel):
    claim: str = Field(description="The specific claim from the draft, quoted or closely paraphrased.")
    status: str = Field(description="One of: supported, partially_supported, unsupported.")
    reason: str = Field(description="Why - reference the citation IDs that do or don't back it up.")
    action: str = Field(description="What to change in the draft. Empty if nothing needs to change.")


class CriticReport(BaseModel):
    verdicts: list[ClaimVerdict] = Field(description="One entry per substantive factual claim in the draft.")
    needs_more_research: bool = Field(description="True if unsupported claims could be fixed by searching again.")
    follow_up_queries: list[str] = Field(description="Specific searches that might close the gaps; empty if none.")
