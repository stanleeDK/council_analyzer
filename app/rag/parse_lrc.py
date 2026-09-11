"""Parse council-meeting subtitle files in LRC format.

YouTube auto-caption exports converted to LRC repeat each phrase 2-3 times
in a row (once per timestamp "edge" of the original caption's display
window). This module collapses those consecutive duplicates back into a
clean, continuous transcript, then strips verbal filler words ("uh", "um",
...) so they don't eat into a chunk's fixed word budget with zero semantic
content. Filler-stripping runs after dedup (so duplicate detection still
sees the original repeated phrases) and before the transcript ever reaches
chunking.
"""
import re
from dataclasses import dataclass

_LINE_RE = re.compile(r"\[(\d{2}):(\d{2}\.\d{2})\](.*)")

# Conservative, unambiguous filler tokens only. Deliberately excludes words
# like "like"/"so"/"right" that are also fillers in speech but are real,
# legitimately-used words in plenty of sentences (e.g. "I would like to
# make a motion").
FILLER_WORDS = ["uh", "uhh", "uhm", "um", "umm", "erm"]

# Only matches a filler token with whitespace (or string start/end) on both
# sides, so it never touches a real word that merely contains the letters
# (e.g. "duh") or a token glued to punctuation (e.g. "uh,").
_FILLER_RE = re.compile(
    r"(?<!\S)(" + "|".join(FILLER_WORDS) + r")(?!\S)",
    re.IGNORECASE,
)


@dataclass
class TranscriptLine:
    ts: float  # seconds from meeting start
    text: str


def parse_lrc_file(path: str) -> list[TranscriptLine]:
    with open(path, encoding="utf-8") as f:
        return parse_lrc_text(f.read())


def parse_lrc_text(raw: str) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    for row in raw.splitlines():
        match = _LINE_RE.match(row)
        if not match:
            continue
        minutes, seconds, text = match.groups()
        text = text.strip()
        if not text:
            continue
        ts = int(minutes) * 60 + float(seconds)
        lines.append(TranscriptLine(ts=ts, text=text))
    return _strip_filler_words(_dedupe_consecutive(lines))


def _dedupe_consecutive(lines: list[TranscriptLine]) -> list[TranscriptLine]:
    deduped: list[TranscriptLine] = []
    for line in lines:
        if deduped and deduped[-1].text == line.text:
            continue
        deduped.append(line)
    return deduped


def _strip_filler_words(lines: list[TranscriptLine]) -> list[TranscriptLine]:
    cleaned: list[TranscriptLine] = []
    for line in lines:
        text = re.sub(r"\s+", " ", _FILLER_RE.sub("", line.text)).strip()
        if not text:
            continue
        cleaned.append(TranscriptLine(ts=line.ts, text=text))
    return cleaned
