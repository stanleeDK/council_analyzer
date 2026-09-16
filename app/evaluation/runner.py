"""Eval runner and parameter sweep.

`evaluate_db` scores one corpus database against the eval set.
`sweep` builds a fresh corpus for each (chunk_words, overlap) combination
and scores each one, so chunking parameters get chosen from measurements
instead of intuition.

The sweep is expensive: each grid point re-embeds the entire corpus.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.db.connection import get_db
from app.evaluation.retrieval import EvalCase, EvalReport, score_case
from app.rag.ingest import ingest_directory
from app.rag.retrieve import search_documents


def evaluate_db(db: sqlite3.Connection, cases: list[EvalCase], top_k: int = 8,
                label: str = "") -> EvalReport:
    report = EvalReport(label=label)
    for case in cases:
        retrieved = search_documents(db, case.question, top_k=top_k, city=case.city)
        report.results.append(score_case(case, retrieved))
    return report


@dataclass
class SweepPoint:
    chunk_words: int
    overlap: float
    hit_rate: float
    mrr: float
    db_path: Path


def sweep(
    raw_dir: Path,
    cases: list[EvalCase],
    chunk_sizes: list[int],
    overlaps: list[float],
    out_dir: Path,
    top_k: int = 8,
    echo: bool = True,
) -> list[SweepPoint]:
    """Build one corpus per parameter combination and score each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    points: list[SweepPoint] = []

    for chunk_words in chunk_sizes:
        for overlap in overlaps:
            db_path = out_dir / f"sweep_c{chunk_words}_o{int(overlap * 100)}.db"
            if echo:
                print(f"\n=== chunk_words={chunk_words} overlap={overlap} -> {db_path.name} ===")

            if not db_path.exists():
                db = get_db(db_path)
                ingest_directory(db, raw_dir, chunk_words=chunk_words, overlap_pct=overlap)
            else:
                if echo:
                    print("   (reusing existing database)")
                db = get_db(db_path)

            report = evaluate_db(db, cases, top_k=top_k,
                                 label=f"c{chunk_words}/o{overlap}")
            db.close()
            if echo:
                print("   " + report.summary_line())
            points.append(SweepPoint(chunk_words, overlap, report.hit_rate, report.mrr, db_path))

    return sorted(points, key=lambda p: (-p.mrr, -p.hit_rate))


def format_sweep_table(points: list[SweepPoint]) -> str:
    header = f"{'chunk_words':>12} {'overlap':>8} {'hit_rate':>9} {'mrr':>7}"
    rows = [f"{p.chunk_words:>12} {p.overlap:>8.2f} {p.hit_rate:>9.2f} {p.mrr:>7.3f}" for p in points]
    return "\n".join([header, "-" * len(header), *rows])
