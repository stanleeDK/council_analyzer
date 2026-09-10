#!/usr/bin/env python3
"""Ingest all council-meeting LRC files under data/raw/<city>/*.lrc.

Usage:
    python3 scripts/ingest_documents.py
    python3 scripts/ingest_documents.py --chunk-words 80 --db data/processed/experiment_80.db
    python3 scripts/ingest_documents.py --overlap 0.2 --db data/processed/experiment_overlap.db
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.connection import DB_PATH, get_db
from app.rag.chunk import DEFAULT_CHUNK_WORDS
from app.rag.ingest import ingest_directory

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS,
                         help="Words per chunk (default: %(default)s)")
    parser.add_argument("--overlap", type=float, default=0.0,
                         help="Fraction of chunk_words repeated at the start of the next "
                              "chunk, e.g. 0.2 for 20%% overlap (default: %(default)s)")
    parser.add_argument("--db", type=Path, default=DB_PATH,
                         help="Database file to write to (default: %(default)s). "
                              "Point at a new path to run a side-by-side experiment "
                              "without touching your existing database.")
    args = parser.parse_args()

    db = get_db(args.db)
    results = ingest_directory(db, RAW_DIR, chunk_words=args.chunk_words, overlap_pct=args.overlap)

    if not results:
        print("No new files ingested (either none found, or all already ingested).")
        return

    total_chunks = 0
    for source_file, num_chunks in results.items():
        print(f"  {source_file}: {num_chunks} chunks")
        total_chunks += num_chunks

    print(f"\nIngested {len(results)} file(s), {total_chunks} chunks total "
          f"(chunk_words={args.chunk_words}, overlap={args.overlap}) into {args.db}")


if __name__ == "__main__":
    main()
