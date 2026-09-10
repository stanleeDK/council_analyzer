from app.rag.parse_lrc import TranscriptLine
from app.rag.chunk import chunk_transcript


def _word_lines(words):
    return [TranscriptLine(ts=float(i), text=w) for i, w in enumerate(words)]


def test_no_overlap_matches_original_behavior():
    lines = _word_lines(["one", "two", "three", "four", "five", "six"])
    chunks = chunk_transcript(lines, chunk_words=3, overlap_pct=0.0)
    assert [c.text for c in chunks] == ["one two three", "four five six"]


def test_overlap_repeats_tail_words_in_next_chunk():
    lines = _word_lines([str(i) for i in range(1, 11)])  # "1".."10"
    chunks = chunk_transcript(lines, chunk_words=4, overlap_pct=0.25)
    texts = [c.text for c in chunks]
    # stride is chunk_words - overlap_words = 4 - 1 = 3, so cuts land at
    # 1-4, 4-7, 7-10, with "10" left over as its own final short chunk.
    assert texts == ["1 2 3 4", "4 5 6 7", "7 8 9 10", "10"]


def test_overlap_word_carries_correct_timestamp():
    lines = _word_lines([str(i) for i in range(1, 11)])
    chunks = chunk_transcript(lines, chunk_words=4, overlap_pct=0.25)
    # the repeated word "4" at the start of chunk 2 should carry ts=3 (0-indexed),
    # matching where "4" actually appeared in the source lines - not ts=0.
    assert chunks[1].start_ts == 3.0
