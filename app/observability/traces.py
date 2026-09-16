"""Run tracing.

Every model call and tool call gets a row. This is what makes a run
explainable after the fact: which agent did what, with which model, how
long it took, what it cost, and what came back.

Traces live in their own SQLite file so they survive re-ingesting the
corpus, and so a trace DB can be inspected without touching the chunks.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TRACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    workflow TEXT,
    step_index INTEGER,
    agent TEXT,
    kind TEXT NOT NULL,          -- model_call | tool_call | note | error
    model TEXT,
    tool TEXT,
    tool_input TEXT,
    output_preview TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_steps(run_id);

CREATE TABLE IF NOT EXISTS trace_runs (
    run_id TEXT PRIMARY KEY,
    workflow TEXT,
    objective TEXT,
    status TEXT,
    total_cost_usd REAL,
    total_latency_ms REAL,
    started_at TEXT,
    finished_at TEXT
);
"""

DEFAULT_TRACE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "traces.db"


def get_trace_db(path: Path = DEFAULT_TRACE_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(TRACE_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Tracer:
    db: sqlite3.Connection
    run_id: str
    workflow: str
    step_index: int = 0
    echo: bool = True

    def start_run(self, objective: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO trace_runs (run_id, workflow, objective, status, "
            "total_cost_usd, total_latency_ms, started_at) VALUES (?, ?, ?, ?, 0, 0, ?)",
            (self.run_id, self.workflow, objective, "running", _now()),
        )
        self.db.commit()

    def finish_run(self, status: str, cost_usd: float, latency_ms: float) -> None:
        self.db.execute(
            "UPDATE trace_runs SET status = ?, total_cost_usd = ?, total_latency_ms = ?, "
            "finished_at = ? WHERE run_id = ?",
            (status, cost_usd, latency_ms, _now(), self.run_id),
        )
        self.db.commit()

    def record(
        self,
        kind: str,
        agent: str = "",
        model: str = "",
        tool: str = "",
        tool_input: dict | None = None,
        output_preview: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        error: str = "",
    ) -> None:
        self.step_index += 1
        self.db.execute(
            """INSERT INTO trace_steps
               (run_id, workflow, step_index, agent, kind, model, tool, tool_input,
                output_preview, input_tokens, output_tokens, cost_usd, latency_ms,
                error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.run_id, self.workflow, self.step_index, agent, kind, model, tool,
                json.dumps(tool_input or {})[:4000], output_preview[:2000],
                input_tokens, output_tokens, cost_usd, latency_ms, error, _now(),
            ),
        )
        self.db.commit()
        if self.echo:
            self._echo(kind, agent, model, tool, tool_input, cost_usd, latency_ms, error, output_preview)

    def _echo(self, kind, agent, model, tool, tool_input, cost_usd, latency_ms, error, output_preview) -> None:
        prefix = f"  [{agent}]" if agent else "  "
        if kind == "model_call":
            print(f"{prefix} model {model} — {latency_ms/1000:.1f}s / ${cost_usd:.4f}")
        elif kind == "tool_call":
            arg = json.dumps(tool_input or {})[:100]
            print(f"{prefix} tool  {tool}({arg}) — {latency_ms/1000:.1f}s")
        elif kind == "error":
            print(f"{prefix} ERROR {error}")
        elif kind == "note" and output_preview:
            print(f"{prefix} {output_preview[:200]}")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000
        return False
