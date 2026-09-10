"""Group deduped transcript lines into retrieval-sized chunks.

Raw caption lines are too small and fragment-y to embed individually.
We merge consecutive lines up to a target word count, and keep the
start/end timestamp the chunk spans so it can be cited (and, later,
deep-linked into the source video).

Optional overlap (as a fraction of chunk_words) repeats the tail of one
chunk at the start of the next, so a phrase that would otherwise be
split across a hard chunk boundary has a chance of appearing intact in
at least one of the two chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.rag.parse_lrc import TranscriptLine

DEFAULT_CHUNK_WORDS = 180


@dataclass
class Chunk:
    start_ts: float
    end_ts: float
    text: str


def chunk_transcript(
    lines: list[TranscriptLine],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_pct: float = 0.0,
) -> list[Chunk]:
    overlap_words = min(int(chunk_words * overlap_pct), chunk_words - 1)

    chunks: list[Chunk] = []
    buffer: list[tuple[str, float]] = []  # (word, timestamp of the line it came from)

    for line in lines:
        buffer.extend((word, line.ts) for word in line.text.split())

        while len(buffer) >= chunk_words:
            entries = buffer[:chunk_words]
            chunks.append(Chunk(
                start_ts=entries[0][1],
                end_ts=entries[-1][1],
                text=" ".join(word for word, _ in entries),
            ))
            buffer = buffer[chunk_words - overlap_words:]

    if buffer:
        chunks.append(Chunk(
            start_ts=buffer[0][1],
            end_ts=buffer[-1][1],
            text=" ".join(word for word, _ in buffer),
        ))

    return chunks
