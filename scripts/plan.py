#!/usr/bin/env python3
"""Stage 1 CLI: question -> validated ResearchPlan.

Usage:
    python scripts/plan.py "How has the city's stance on short-term rentals changed?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from app.models.llm import make_plan


def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python scripts/plan.py \"<question>\"")
        sys.exit(1)

    question = sys.argv[1]
    plan = make_plan(question)
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
