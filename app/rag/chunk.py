"""Group deduped transcript lines into retrieval-sized chunks.

Raw caption lines are too small and fragment-y to embed individually.
We merge consecutive lines up to a target word count, and keep the
start/end timestamp the chunk spans so it can be cited (and, later,
deep-linked into the source video).
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


def chunk_transcript(lines: list[TranscriptLine], chunk_words: int = DEFAULT_CHUNK_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    words: list[str] = []
    start_ts: float | None = None
    end_ts: float = 0.0

    for line in lines:
        if start_ts is None:
            start_ts = line.ts
        words.extend(line.text.split())
        end_ts = line.ts
        if len(words) >= chunk_words:
            chunks.append(Chunk(start_ts=start_ts, end_ts=end_ts, text=" ".join(words)))
            words = []
            start_ts = None

    if words:
        chunks.append(Chunk(start_ts=start_ts if start_ts is not None else end_ts, end_ts=end_ts, text=" ".join(words)))

    return chunks
