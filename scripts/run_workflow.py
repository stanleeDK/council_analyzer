#!/usr/bin/env python3
"""Run an agentic workflow against the corpus.

Usage:
    python3 scripts/run_workflow.py "How has the council's stance on short-term rentals changed?"
    python3 scripts/run_workflow.py "How many meetings did each city hold in 2026?" --workflow text2sql
    python3 scripts/run_workflow.py "..." --workflow research_with_approval   # prompts before SQL
    python3 scripts/run_workflow.py "..." --save-state runs/state.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from app.db.analytics import ANALYTICS_PATH, open_readonly
from app.db.connection import DB_PATH, get_db
from app.runtime.approval import AutoApprover, CLIApprover
from app.runtime.workflow import WorkflowConfig, WorkflowRunner


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--workflow", default="deep_research",
                        help="Workflow name in workflows/ or a path to a YAML file (default: %(default)s)")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Corpus database (default: %(default)s)")
    parser.add_argument("--analytics-db", type=Path, default=ANALYTICS_PATH,
                        help="Analytics database for SQL tools (default: %(default)s)")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Approve gated tool calls without prompting (for unattended runs)")
    parser.add_argument("--save-state", type=Path, help="Write the full TaskState JSON here")
    parser.add_argument("--quiet", action="store_true", help="Suppress the live step-by-step trace")
    args = parser.parse_args()

    config = WorkflowConfig.load(args.workflow)

    corpus_db = get_db(args.db) if args.db.exists() else None
    if corpus_db is None:
        print(f"Warning: no corpus database at {args.db} - retrieval tools will fail. "
              f"Run scripts/ingest_documents.py first.")

    analytics_db = None
    if args.analytics_db.exists():
        analytics_db = open_readonly(args.analytics_db)
    elif "data_analysis" in config.steps:
        print(f"Warning: no analytics database at {args.analytics_db} - SQL tools will fail. "
              f"Run scripts/seed_analytics_db.py first.")

    runner = WorkflowRunner(
        config=config,
        corpus_db=corpus_db,
        analytics_db=analytics_db,
        approver=AutoApprover() if args.auto_approve else CLIApprover(),
        echo=not args.quiet,
    )
    state = runner.run(args.question)

    print("\n" + "=" * 70)
    print(state.final_report or "(no report produced)")
    print("=" * 70)

    if state.evidence:
        print(f"\n{len(state.evidence)} evidence passages cited across "
              f"{len({e.city for e in state.evidence})} jurisdiction(s).")
    if state.claim_checks:
        unsupported = len(state.unsupported_claims())
        print(f"Verification: {len(state.claim_checks)} claims checked, {unsupported} unsupported.")
    print(f"Trace: run_id={state.run_id}  (python3 scripts/show_trace.py {state.run_id})")

    if args.save_state:
        args.save_state.parent.mkdir(parents=True, exist_ok=True)
        args.save_state.write_text(state.to_json())
        print(f"State written to {args.save_state}")


if __name__ == "__main__":
    main()
