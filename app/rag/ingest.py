"""End-to-end ingestion: LRC file -> dedupe -> chunk -> embed -> store.

Expects files organized as data/raw/<city>/<meeting>.lrc so city can be
read from the directory name (it's not reliably present in the filename
itself).
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.rag.chunk import chunk_transcript
from app.rag.embed import embed_texts
from app.rag.filenames import parse_filename
from app.rag.parse_lrc import parse_lrc_file


def ingest_file(db: sqlite3.Connection, path: Path, city: str) -> int:
    """Ingest one LRC file. Returns number of chunks stored. Skips files
    already ingested (tracked by source_file)."""
    source_file = str(path)

    existing = db.execute(
        "SELECT 1 FROM ingested_files WHERE source_file = ?", (source_file,)
    ).fetchone()
    if existing:
        return 0

    lines = parse_lrc_file(source_file)
    if not lines:
        return 0

    chunks = chunk_transcript(lines)
    meta = parse_filename(path.stem)

    vectors = embed_texts([c.text for c in chunks])

    for chunk, vector in zip(chunks, vectors):
        db.execute(
            """INSERT INTO chunks
               (city, meeting_date, meeting_title, source_file, start_ts, end_ts, text, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                city,
                meta.meeting_date,
                meta.meeting_title,
                source_file,
                chunk.start_ts,
                chunk.end_ts,
                chunk.text,
                json.dumps(vector.tolist()),
            ),
        )

    db.execute(
        "INSERT INTO ingested_files (source_file, city, num_chunks, ingested_at) VALUES (?, ?, ?, ?)",
        (source_file, city, len(chunks), datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    return len(chunks)


def ingest_directory(db: sqlite3.Connection, raw_dir: Path) -> dict[str, int]:
    """Walk raw_dir/<city>/*.lrc and ingest everything found.
    Returns {source_file: num_chunks_added} for files actually processed."""
    results: dict[str, int] = {}
    for city_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        city = city_dir.name
        for lrc_path in sorted(city_dir.glob("*.lrc")):
            added = ingest_file(db, lrc_path, city)
            if added:
                results[str(lrc_path)] = added
    return results
