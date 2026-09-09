"""Metadata extraction from council-meeting filenames.

Standard format (confirmed across the real corpus):

    [UPLOAD_DATE] Free-form name, may embed a meeting date in any format [VIDEO_ID].en(-orig).lrc

Examples actually seen:
    [20251118] Finance Committee： Meeting of November 17, 2025 [g_vfAOPT3Y4].en-orig.lrc
    [20250403] Committee of the Whole 4⧸2⧸2025 [KB9ilVA54tA].en-orig.lrc
    [20260507] 2024-11-07 - Annual Budget Adoption [2N3v6eh1IQM].en-orig.lrc
    [20190627] Steering & Rules 6⧸27⧸19 Item 5 [4gYsZ0f_mJg].en-orig.lrc
    [20141217] Waukesha County Monthly Update - August 2014 [tmZywbnaaRQ].en-orig.lrc

The bracketed leading date is the UPLOAD date, not necessarily the meeting
date - they can differ by months (see the 2026/2024 example above). The
free-form name may contain a meeting date in whatever format the uploader
used; we deliberately don't try to parse that out - it's kept verbatim as
the title.

Some older files may not match this format at all (no brackets); those
fall back to upload_date=None, video_id=None, title=whole filename stem,
rather than raising.
"""
import re
from dataclasses import dataclass

_BRACKET_RE = re.compile(
    r"^\[(\d{8})\]\s*(.+?)\s*\[([A-Za-z0-9_-]+)\]\.en(?:-?orig)?\.lrc$"
)


@dataclass
class MeetingMeta:
    upload_date: str | None  # YYYY-MM-DD, from the leading [YYYYMMDD] - NOT the meeting date
    title: str
    video_id: str | None


def parse_filename(filename: str) -> MeetingMeta:
    match = _BRACKET_RE.match(filename)
    if not match:
        # Doesn't match the standard format - don't guess, just fall back.
        title = re.sub(r"\.lrc$", "", filename)
        return MeetingMeta(upload_date=None, title=title, video_id=None)

    raw_date, title, video_id = match.groups()
    upload_date = f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    return MeetingMeta(upload_date=upload_date, title=title, video_id=video_id)


def youtube_url(video_id: str, start_ts: float | None = None) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    if start_ts is not None:
        url += f"&t={int(start_ts)}s"
    return url
