"""Parse council-meeting subtitle files in LRC format.

YouTube auto-caption exports converted to LRC repeat each phrase 2-3 times
in a row (once per timestamp "edge" of the original caption's display
window). This module collapses those consecutive duplicates back into a
clean, continuous transcript.
"""
import re
from dataclasses import dataclass

_LINE_RE = re.compile(r"\[(\d{2}):(\d{2}\.\d{2})\](.*)")


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
    return _dedupe_consecutive(lines)


def _dedupe_consecutive(lines: list[TranscriptLine]) -> list[TranscriptLine]:
    deduped: list[TranscriptLine] = []
    for line in lines:
        if deduped and deduped[-1].text == line.text:
            continue
        deduped.append(line)
    return deduped
