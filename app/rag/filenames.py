"""Best-effort metadata extraction from council-meeting filenames.

Filenames look like:
    20201120_Protection__Policy_Committee_111920_3_aBAj4WQ1c.enorig.lrc

There's no reliable standard here, so this is heuristic: pull a leading
YYYYMMDD date if present, strip the trailing video-id/language suffix, and
turn the rest into a human-readable title. City is NOT derivable from the
filename in the sample data — it must come from the containing directory
(see ingest.py, which expects data/raw/<city>/*.lrc).
"""
import re
from dataclasses import dataclass

_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_")
# Matches only a single trailing underscore-free token (typically a YouTube
# video ID, e.g. "_aBAj4WQ1c") - deliberately does not eat earlier segments.
_TRAILING_ID_RE = re.compile(r"_[A-Za-z0-9]{8,12}$")


@dataclass
class MeetingMeta:
    meeting_date: str | None
    meeting_title: str


def parse_filename(stem: str) -> MeetingMeta:
    """stem: filename without directory or extension(s), e.g.
    '20201120_Protection__Policy_Committee_111920_3_aBAj4WQ1c.enorig'
    """
    stem = re.sub(r"\.en(orig)?$", "", stem)

    date_match = _DATE_RE.match(stem)
    meeting_date = None
    rest = stem
    if date_match:
        year, month, day = date_match.groups()
        meeting_date = f"{year}-{month}-{day}"
        rest = stem[date_match.end():]

    rest = _TRAILING_ID_RE.sub("", rest)
    title = re.sub(r"[_\s]+", " ", rest).strip()

    return MeetingMeta(meeting_date=meeting_date, meeting_title=title or stem)
