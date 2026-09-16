"""Text2SQL tools with guardrails.

The model writes SQL; this module decides whether that SQL is allowed to
touch the database. Guardrails, in order of enforcement:

  1. connection is opened read-only at the SQLite level (mode=ro)
  2. only a single statement is accepted (no stacked `;` statements)
  3. statement must begin with SELECT or WITH
  4. forbidden keywords rejected anywhere in the statement
  5. a LIMIT is injected if the model didn't supply one
  6. a wall-clock timeout interrupts long-running queries

Layer 1 alone would stop writes, but the rest give clear, early errors
the model can actually recover from, instead of an opaque failure.
"""
from __future__ import annotations

import re
import time

from app.tools.base import Tool, ToolContext

FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "truncate", "grant",
)
MAX_ROWS = 200
TIMEOUT_SECONDS = 5.0


class SQLRejected(ValueError):
    pass


def validate_sql(sql: str) -> str:
    """Return a safe, LIMIT-bounded statement, or raise SQLRejected."""
    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise SQLRejected("empty statement")

    if ";" in statement:
        raise SQLRejected("multiple statements are not allowed; submit one SELECT at a time")

    lowered = statement.lower()
    first_word = re.split(r"\s+", lowered, maxsplit=1)[0]
    if first_word not in ("select", "with"):
        raise SQLRejected(f"only SELECT/WITH queries are allowed, got '{first_word.upper()}'")

    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            raise SQLRejected(f"forbidden keyword '{word.upper()}' - this database is read-only")

    if not re.search(r"\blimit\b", lowered):
        statement = f"{statement}\nLIMIT {MAX_ROWS}"
    return statement


def _with_timeout(conn, seconds: float):
    """Interrupt a query that runs longer than `seconds` (SQLite progress hook)."""
    deadline = time.monotonic() + seconds

    def handler():
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(handler, 10_000)


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "Query executed successfully but returned 0 rows."
    header = " | ".join(columns)
    divider = "-" * len(header)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
    return f"{header}\n{divider}\n{body}\n\n({len(rows)} row(s))"


def _run_sql(tool_input: dict, ctx: ToolContext) -> str:
    if ctx.analytics_db is None:
        return "Error: no analytics database is attached to this run."
    raw = tool_input.get("sql") or ""
    try:
        statement = validate_sql(raw)
    except SQLRejected as exc:
        return f"SQL rejected: {exc}"

    try:
        _with_timeout(ctx.analytics_db, TIMEOUT_SECONDS)
        cursor = ctx.analytics_db.execute(statement)
        rows = cursor.fetchmany(MAX_ROWS)
        columns = [d[0] for d in cursor.description] if cursor.description else []
    except Exception as exc:  # surfaced back to the model so it can fix its SQL
        return f"SQL error: {type(exc).__name__}: {exc}"
    finally:
        ctx.analytics_db.set_progress_handler(None, 0)

    return f"SQL executed:\n{statement}\n\n{_format_rows(columns, rows)}"


def _get_schema(tool_input: dict, ctx: ToolContext) -> str:
    if ctx.analytics_db is None:
        return "Error: no analytics database is attached to this run."
    rows = ctx.analytics_db.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    if not rows:
        return "The analytics database has no tables. Run scripts/seed_analytics_db.py first."
    return "\n\n".join(f"-- table: {name}\n{ddl}" for name, ddl in rows)


run_readonly_sql = Tool(
    name="run_readonly_sql",
    description=(
        "Execute a single read-only SQL SELECT against the council analytics database "
        "(meeting counts, dates, durations per city). Only SELECT/WITH are permitted; "
        f"results are capped at {MAX_ROWS} rows. Call get_schema first if you do not "
        "know the table structure."
    ),
    input_schema={
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
        "required": ["sql"],
    },
    handler=_run_sql,
    read_only=True,
)

get_schema = Tool(
    name="get_schema",
    description="Return the CREATE TABLE definitions for every table in the analytics database.",
    input_schema={"type": "object", "properties": {}},
    handler=_get_schema,
    read_only=True,
)
