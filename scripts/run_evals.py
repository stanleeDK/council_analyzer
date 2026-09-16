#!/usr/bin/env python3
"""Score a corpus database against the retrieval eval set.

Usage:
    python3 scripts/run_evals.py
    python3 scripts/run_evals.py --db data/processed/experiment_80.db --top-k 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.connection import DB_PATH, get_db
from app.evaluation.retrieval import DEFAULT_EVAL_PATH, load_cases
from app.evaluation.runner import evaluate_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--evals", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No corpus database at {args.db}.")
        raise SystemExit(1)

    cases = load_cases(args.evals)
    report = evaluate_db(get_db(args.db), cases, top_k=args.top_k, label=args.db.name)

    print(f"\n{len(cases)} cases against {args.db} (top_k={args.top_k})\n")
    by_id = {c.id: c for c in cases}
    for result in report.results:
        case = by_id[result.case_id]
        mark = "PASS" if result.hit else "FAIL"
        rank = f"rank {result.rank}" if result.rank else "not in top-k"
        kind = " (negative case)" if case.expect_no_match else ""
        print(f"  [{mark}] {result.case_id}{kind}")
        print(f"         {rank}, top score {result.top_score:.3f}")
        if result.matched_text:
            print(f"         matched: {result.matched_text[:100]}...")

    print(f"\n{report.summary_line()}")
    if report.misses():
        print(f"Misses: {', '.join(r.case_id for r in report.misses())}")


if __name__ == "__main__":
    main()
