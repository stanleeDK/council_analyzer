#!/usr/bin/env python3
"""Stage 2 CLI: question -> retrieve top-k chunks -> cited answer.

This is the single-pass RAG baseline (no agent loop, no planner) - the
Milestone 2 definition of done: ask a real question against the corpus and
get an answer whose claims point to actual retrieved passages.

Usage:
    python scripts/ask.py "What did the council decide about short-term rentals?"
    python scripts/ask.py "..." --city springfield
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from anthropic import Anthropic

from app.db.connection import get_db
from app.rag.retrieve import search_documents

ANSWER_MODEL = "claude-sonnet-5"

_SYSTEM = """You answer questions about city/county council meetings using \
ONLY the numbered evidence passages provided below. Every factual claim \
must be followed by its citation tag, e.g. [C14]. If the evidence does not \
support an answer, say so explicitly rather than guessing - do not use \
outside knowledge."""


def format_evidence(chunks) -> str:
    return "\n\n".join(
        f"[{c.citation_id}] {c.city} - {c.meeting_title} ({c.meeting_date or 'date unknown'}), "
        f"{c.start_ts:.0f}s-{c.end_ts:.0f}s:\n{c.text}"
        for c in chunks
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--city", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    db = get_db()
    chunks = search_documents(db, args.question, top_k=args.top_k, city=args.city)

    if not chunks:
        print("No indexed chunks found. Run scripts/ingest_documents.py first.")
        return

    evidence = format_evidence(chunks)
    client = Anthropic()
    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=2048,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Evidence:\n\n{evidence}\n\nQuestion: {args.question}",
        }],
    )

    answer = next(b.text for b in response.content if b.type == "text")
    print(answer)

    print("\n--- Sources ---")
    for c in chunks:
        print(f"[{c.citation_id}] {c.city} - {c.meeting_title} ({c.meeting_date or 'unknown'}), "
              f"{c.start_ts:.0f}s-{c.end_ts:.0f}s  (similarity {c.score:.3f})")


if __name__ == "__main__":
    main()
