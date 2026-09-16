#!/usr/bin/env python3
"""Build the analytics database (Text2SQL subsystem) from the ingested corpus.

Derived data only - every row comes from what's actually in the chunks
table. Safe to re-run: the database is rebuilt from scratch each time.

Usage:
    python3 scripts/seed_analytics_db.py
    python3 scripts/seed_analytics_db.py --db data/processed/experiment_80.db
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.analytics import ANALYTICS_PATH, build_analytics_db
from app.db.connection import DB_PATH, get_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Corpus database to derive from")
    parser.add_argument("--out", type=Path, default=ANALYTICS_PATH, help="Analytics database to write")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No corpus database at {args.db}. Run scripts/ingest_documents.py first.")
        raise SystemExit(1)

    counts = build_analytics_db(get_db(args.db), args.out)
    print(f"Built {args.out}")
    print(f"  meetings: {counts['meetings']}")
    print(f"  cities:   {counts['cities']}")


if __name__ == "__main__":
    main()
