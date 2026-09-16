"""Analytics database for the Text2SQL subsystem.

Derived entirely from what was actually ingested - no invented data. Each
row in `meetings` is one ingested transcript file; `city_coverage` is a
rollup. This describes *meeting coverage* (what was recorded, when, how
long), not votes or outcomes, and the data agent's system prompt says so
explicitly so it doesn't over-claim.

Opened read-only for agent use (mode=ro), so the SQL guardrails in
app/tools/sql.py are a second line of defence rather than the only one.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

ANALYTICS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "analytics.db"

ANALYTICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    meeting_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    city           TEXT NOT NULL,      -- jurisdiction (from the source folder name)
    title          TEXT,               -- meeting/committee title from the filename
    upload_date    TEXT,               -- YYYY-MM-DD the video was uploaded (NOT the meeting date)
    upload_year    INTEGER,
    video_id       TEXT,               -- YouTube video id
    source_file    TEXT NOT NULL UNIQUE,
    chunk_count    INTEGER NOT NULL,   -- retrievable passages produced
    duration_secs  REAL                -- last timestamp seen in the transcript
);
CREATE INDEX IF NOT EXISTS idx_meetings_city ON meetings(city);
CREATE INDEX IF NOT EXISTS idx_meetings_year ON meetings(upload_year);

CREATE TABLE IF NOT EXISTS city_coverage (
    city            TEXT PRIMARY KEY,
    meeting_count   INTEGER NOT NULL,
    first_upload    TEXT,
    last_upload     TEXT,
    total_chunks    INTEGER NOT NULL,
    total_hours     REAL
);
"""


def build_analytics_db(corpus_db: sqlite3.Connection, path: Path = ANALYTICS_PATH) -> dict[str, int]:
    """(Re)build the analytics DB from an ingested corpus. Returns row counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # derived data - always rebuilt from scratch, never migrated

    out = sqlite3.connect(path)
    out.executescript(ANALYTICS_SCHEMA)

    rows = corpus_db.execute(
        """SELECT source_file,
                  MIN(city)        AS city,
                  MIN(title)       AS title,
                  MIN(upload_date) AS upload_date,
                  MIN(video_id)    AS video_id,
                  COUNT(*)         AS chunk_count,
                  MAX(end_ts)      AS duration_secs
           FROM chunks
           GROUP BY source_file"""
    ).fetchall()

    for source_file, city, title, upload_date, video_id, chunk_count, duration in rows:
        year = int(upload_date[:4]) if upload_date and len(upload_date) >= 4 and upload_date[:4].isdigit() else None
        out.execute(
            """INSERT INTO meetings
               (city, title, upload_date, upload_year, video_id, source_file, chunk_count, duration_secs)
               VALUES (?,?,?,?,?,?,?,?)""",
            (city, title, upload_date, year, video_id, source_file, chunk_count, duration),
        )

    out.execute(
        """INSERT INTO city_coverage (city, meeting_count, first_upload, last_upload, total_chunks, total_hours)
           SELECT city, COUNT(*), MIN(upload_date), MAX(upload_date),
                  SUM(chunk_count), ROUND(SUM(COALESCE(duration_secs, 0)) / 3600.0, 2)
           FROM meetings GROUP BY city"""
    )
    out.commit()

    counts = {
        "meetings": out.execute("SELECT COUNT(*) FROM meetings").fetchone()[0],
        "cities": out.execute("SELECT COUNT(*) FROM city_coverage").fetchone()[0],
    }
    out.close()
    return counts


def open_readonly(path: Path = ANALYTICS_PATH) -> sqlite3.Connection:
    """Open the analytics DB read-only - writes fail at the SQLite level."""
    if not path.exists():
        raise FileNotFoundError(
            f"No analytics database at {path}. Build it with: python3 scripts/seed_analytics_db.py"
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
