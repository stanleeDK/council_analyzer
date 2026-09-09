CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    upload_date TEXT,
    title TEXT,
    video_id TEXT,
    source_file TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_city ON chunks(city);

CREATE TABLE IF NOT EXISTS ingested_files (
    source_file TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    num_chunks INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
