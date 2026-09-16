"""Retrieval metrics.

Ground truth is specified as *content* ("a returned passage must contain
this substring"), not as a chunk ID - chunk IDs are reassigned every time
the corpus is re-ingested with different chunking parameters, so an
ID-based eval couldn't compare two configurations. Content-based ground
truth survives re-chunking, which is exactly what a parameter sweep needs.

No LLM calls here: these metrics score the retriever alone, so a sweep
costs only local embedding time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "evals" / "retrieval_evals.json"


@dataclass
class EvalCase:
    id: str
    question: str
    expect_any_of: list[str]          # case-insensitive substrings; any match counts as a hit
    city: str | None = None
    note: str = ""
    expect_no_match: bool = False     # negative case: the corpus should NOT cover this

    @classmethod
    def from_dict(cls, raw: dict) -> "EvalCase":
        expect = raw.get("expect_any_of") or ([raw["expect_text_contains"]]
                                              if raw.get("expect_text_contains") else [])
        return cls(
            id=raw["id"],
            question=raw["question"],
            expect_any_of=[s.lower() for s in expect],
            city=raw.get("city"),
            note=raw.get("note", ""),
            expect_no_match=bool(raw.get("expect_no_match", False)),
        )


@dataclass
class CaseResult:
    case_id: str
    hit: bool
    rank: int | None       # 1-indexed position of the first matching passage
    top_score: float
    matched_text: str = ""


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    label: str = ""

    @property
    def hit_rate(self) -> float:
        return sum(r.hit for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: 1.0 if always first, 0 if never found."""
        if not self.results:
            return 0.0
        return sum((1.0 / r.rank) if (r.hit and r.rank) else 0.0 for r in self.results) / len(self.results)

    def misses(self) -> list[CaseResult]:
        return [r for r in self.results if not r.hit]

    def summary_line(self) -> str:
        return (f"{self.label or 'eval'}: hit_rate={self.hit_rate:.2f} "
                f"mrr={self.mrr:.3f} ({sum(r.hit for r in self.results)}/{len(self.results)} cases)")


def load_cases(path: Path = DEFAULT_EVAL_PATH) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"No eval file at {path}")
    raw = json.loads(path.read_text())
    cases = raw["cases"] if isinstance(raw, dict) else raw
    return [EvalCase.from_dict(c) for c in cases]


def score_case(case: EvalCase, retrieved: list) -> CaseResult:
    """Score one case against a ranked list of RetrievedChunk-like objects."""
    top_score = retrieved[0].score if retrieved else 0.0

    match_rank, matched_text = None, ""
    for rank, chunk in enumerate(retrieved, start=1):
        text = chunk.text.lower()
        if any(needle in text for needle in case.expect_any_of):
            match_rank, matched_text = rank, chunk.text[:160]
            break

    found = match_rank is not None
    # For a negative case, "hit" means correctly finding nothing.
    hit = (not found) if case.expect_no_match else found
    return CaseResult(case_id=case.id, hit=hit, rank=match_rank,
                      top_score=top_score, matched_text=matched_text)
