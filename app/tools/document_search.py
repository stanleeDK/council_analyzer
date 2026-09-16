"""search_transcripts tool - agentic RAG over the council corpus.

Wraps app/rag/retrieve.search_documents. Results are appended to the
shared TaskState as Evidence with stable citation IDs, so a later agent
(critic, reporter) can reason about passages this agent retrieved.
"""
from __future__ import annotations

from app.rag.retrieve import search_documents
from app.runtime.state import Evidence
from app.tools.base import Tool, ToolContext

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "What to search for, in natural language.",
        },
        "city": {
            "type": "string",
            "description": "Optional exact city/county name to restrict the search to, "
                           "e.g. 'City of Green Bay'. Omit to search all jurisdictions.",
        },
        "top_k": {
            "type": "integer",
            "description": "How many passages to return (1-15). Default 8.",
        },
    },
    "required": ["query"],
}


def _handler(tool_input: dict, ctx: ToolContext) -> str:
    if ctx.corpus_db is None:
        return "Error: no corpus database is attached to this run."

    query = (tool_input.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required and must be non-empty."

    city = tool_input.get("city") or None
    top_k = max(1, min(int(tool_input.get("top_k") or 8), 15))

    chunks = search_documents(ctx.corpus_db, query, top_k=top_k, city=city)
    if not chunks:
        return (f"No passages found for query {query!r}"
                + (f" in city {city!r}." if city else ".")
                + " The corpus may not cover this, or the city name may not match exactly.")

    if ctx.state is not None:
        ctx.state.add_evidence([
            Evidence(
                citation_id=c.citation_id, city=c.city, title=c.title,
                upload_date=c.upload_date, video_id=c.video_id,
                start_ts=c.start_ts, end_ts=c.end_ts, text=c.text,
                score=c.score, retrieved_for=query,
            )
            for c in chunks
        ])

    lines = []
    for c in chunks:
        lines.append(
            f"[{c.citation_id}] {c.city} — {c.title} "
            f"(uploaded {c.upload_date or 'unknown'}, {c.start_ts:.0f}s-{c.end_ts:.0f}s, "
            f"similarity {c.score:.3f})\n{c.text}"
        )
    return "\n\n".join(lines)


search_transcripts = Tool(
    name="search_transcripts",
    description=(
        "Search indexed city/county council meeting transcripts for passages relevant "
        "to a query. Returns passages with citation IDs like [C123]. Call this multiple "
        "times with different phrasings if the first results are thin - rare proper nouns "
        "often need a literal, narrow query."
    ),
    input_schema=SCHEMA,
    handler=_handler,
    read_only=True,
)
