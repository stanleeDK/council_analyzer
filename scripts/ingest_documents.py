#!/usr/bin/env python3
"""Ingest all council-meeting LRC files under data/raw/<city>/*.lrc.

Usage:
    python scripts/ingest_documents.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.connection import get_db
from app.rag.ingest import ingest_directory

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    db = get_db()
    results = ingest_directory(db, RAW_DIR)

    if not results:
        print("No new files ingested (either none found, or all already ingested).")
        return

    total_chunks = 0
    for source_file, num_chunks in results.items():
        print(f"  {source_file}: {num_chunks} chunks")
        total_chunks += num_chunks

    print(f"\nIngested {len(results)} file(s), {total_chunks} chunks total.")


if __name__ == "__main__":
    main()
