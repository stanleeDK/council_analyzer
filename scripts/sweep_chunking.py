#!/usr/bin/env python3
"""Grid-search chunking parameters against the eval set.

Builds one corpus database per (chunk_words, overlap) combination and
scores each. Expensive: every grid point re-embeds the whole corpus.
Existing sweep databases are reused, so an interrupted sweep resumes.

Usage:
    python3 scripts/sweep_chunking.py
    python3 scripts/sweep_chunking.py --chunk-sizes 80 120 180 --overlaps 0 0.15 0.3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.retrieval import DEFAULT_EVAL_PATH, load_cases
from app.evaluation.runner import format_sweep_table, sweep

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SWEEP_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "sweeps"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[80, 180, 300])
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--evals", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=SWEEP_DIR)
    args = parser.parse_args()

    cases = load_cases(args.evals)
    total = len(args.chunk_sizes) * len(args.overlaps)
    print(f"Sweeping {total} combination(s) over {len(cases)} eval cases.")
    print("Each new combination re-embeds the entire corpus - this is slow.\n")

    points = sweep(RAW_DIR, cases, args.chunk_sizes, args.overlaps,
                   out_dir=args.out_dir, top_k=args.top_k)

    print("\n\nResults (best first):\n")
    print(format_sweep_table(points))
    if points:
        best = points[0]
        print(f"\nBest: chunk_words={best.chunk_words} overlap={best.overlap} "
              f"(mrr={best.mrr:.3f}, hit_rate={best.hit_rate:.2f})")
        print(f"Database: {best.db_path}")


if __name__ == "__main__":
    main()
