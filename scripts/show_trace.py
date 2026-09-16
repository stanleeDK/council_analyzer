#!/usr/bin/env python3
"""Show what happened during a run.

Usage:
    python3 scripts/show_trace.py              # list recent runs
    python3 scripts/show_trace.py <run_id>     # step-by-step detail
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.observability.traces import DEFAULT_TRACE_PATH, get_trace_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--trace-db", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    if not args.trace_db.exists():
        print(f"No trace database at {args.trace_db} - run a workflow first.")
        raise SystemExit(1)
    db = get_trace_db(args.trace_db)

    if not args.run_id:
        rows = db.execute(
            "SELECT run_id, workflow, status, total_cost_usd, total_latency_ms, started_at, objective "
            "FROM trace_runs ORDER BY started_at DESC LIMIT ?", (args.limit,)
        ).fetchall()
        if not rows:
            print("No runs recorded yet.")
            return
        print(f"\n{'run_id':<14} {'workflow':<22} {'status':<10} {'cost':>8} {'time':>8}  objective")
        print("-" * 110)
        for run_id, wf, status, cost, ms, _started, objective in rows:
            print(f"{run_id:<14} {wf:<22} {status:<10} ${cost or 0:>7.4f} {(ms or 0)/1000:>7.1f}s  "
                  f"{(objective or '')[:45]}")
        return

    run = db.execute(
        "SELECT workflow, objective, status, total_cost_usd, total_latency_ms FROM trace_runs WHERE run_id = ?",
        (args.run_id,)
    ).fetchone()
    if not run:
        print(f"No run '{args.run_id}'.")
        raise SystemExit(1)

    workflow, objective, status, cost, ms = run
    print(f"\nRun {args.run_id} — {workflow} — {status}")
    print(f"Objective: {objective}")
    print(f"Total: ${cost or 0:.4f} / {(ms or 0)/1000:.1f}s\n")

    steps = db.execute(
        "SELECT step_index, agent, kind, model, tool, tool_input, output_preview, "
        "input_tokens, output_tokens, cost_usd, latency_ms, error "
        "FROM trace_steps WHERE run_id = ? ORDER BY step_index", (args.run_id,)
    ).fetchall()

    for (idx, agent, kind, model, tool, tool_input, preview,
         tok_in, tok_out, step_cost, latency, error) in steps:
        head = f"{idx:>3}. [{agent or '-':<13}] {kind}"
        if kind == "model_call":
            head += f" {model}  {tok_in}->{tok_out} tok  ${step_cost:.4f}  {latency/1000:.1f}s"
        elif kind == "tool_call":
            head += f" {tool}  {latency/1000:.1f}s"
        print(head)
        if tool_input and tool_input != "{}":
            print(f"      in:  {tool_input[:160]}")
        if preview:
            print(f"      out: {preview[:160].replace(chr(10), ' ')}")
        if error:
            print(f"      ERROR: {error[:200]}")


if __name__ == "__main__":
    main()
