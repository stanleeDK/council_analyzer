"""Small text helpers for printing what an agent did.

Kept separate from any one agent module because the researcher, the critic,
and the workflow runner all print small summaries like "3 searches" or a
trimmed one-line preview of a long string.
"""
from __future__ import annotations


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """Pick the right word for a count: plural(1, "search") -> "1 search"."""
    if count == 1:
        word = singular
    elif plural_form is not None:
        word = plural_form
    else:
        word = singular + "s"
    return f"{count} {word}"


def clip(text: str, limit: int) -> str:
    """Collapse whitespace to one line and cut it to `limit` characters.

    Cuts on the last whole word so the result never ends mid-word.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut + "..."
