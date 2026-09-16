"""Shared task state.

Every agent in a workflow reads from and writes to one of these. Keeping
state explicit (rather than passing chat history around) is what lets the
critic inspect what the researcher actually found, and lets the reporter
cite evidence it never retrieved itself.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict


@dataclass
class Evidence:
    """One retrieved passage, with enough provenance to cite it."""
    citation_id: str
    city: str
    title: str
    upload_date: str | None
    video_id: str | None
    start_ts: float
    end_ts: float
    text: str
    score: float
    retrieved_for: str = ""  # which subquestion / query pulled this in


@dataclass
class Finding:
    """A researcher's conclusion about one subquestion, tied to evidence."""
    question: str
    finding: str
    citation_ids: list[str] = field(default_factory=list)


@dataclass
class ClaimCheck:
    """A critic's verdict on one claim in the draft."""
    claim: str
    status: str  # supported | partially_supported | unsupported
    reason: str
    action: str = ""


@dataclass
class TaskState:
    objective: str
    workflow: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    subquestions: list[str] = field(default_factory=list)
    relevant_cities: list[str] = field(default_factory=list)
    needs_quantitative_data: bool = False
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    sql_results: list[dict] = field(default_factory=list)
    draft: str = ""
    claim_checks: list[ClaimCheck] = field(default_factory=list)
    final_report: str = ""
    status: str = "running"  # running | complete | failed | halted
    notes: list[str] = field(default_factory=list)

    def add_evidence(self, items: list[Evidence]) -> list[Evidence]:
        """Append evidence, skipping citation_ids already collected."""
        seen = {e.citation_id for e in self.evidence}
        added = [e for e in items if e.citation_id not in seen]
        self.evidence.extend(added)
        return added

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.citation_id: e for e in self.evidence}

    def unsupported_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claim_checks if c.status == "unsupported"]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
