#!/usr/bin/env python3
"""Strip standalone verbal filler words (uh, um, uhh, umm, erm, ...) from raw
LRC transcript files.

Only removes a token when it appears as its own word with whitespace (or
line start/end) on both sides -- never as part of another word (e.g. "duh"
or "yummy" are left untouched), and never when glued to punctuation (e.g.
"uh," is left alone, since that's not "spaces on either side").

By default this is a dry run: it reports what *would* change without
touching any files. Pass --apply to actually rewrite files (a .bak backup
of each modified file is written alongside it, unless --no-backup is set).

Usage:
    python3 scripts/remove_filler_words.py                  # dry run, all of data/raw
    python3 scripts/remove_filler_words.py --apply           # actually rewrite files
    python3 scripts/remove_filler_words.py --apply --no-backup
    python3 scripts/remove_filler_words.py --root data/raw/example_city
    python3 scripts/remove_filler_words.py --stats-json data/processed/filler_stats.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

_LINE_RE = re.compile(r"^(\[\d{2}:\d{2}\.\d{2}\])(.*)$")

# Conservative, unambiguous filler tokens only. Deliberately excludes words
# like "like"/"so"/"right" that are also fillers in speech but are real,
# legitimately-used words in plenty of sentences.
FILLER_WORDS = ["uh", "uhh", "uhm", "um", "umm", "erm"]

_FILLER_RE = re.compile(
    r"(?<!\S)(" + "|".join(FILLER_WORDS) + r")(?!\S)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\S+")


class FileStats:
    def __init__(self, path: Path):
        self.path = path
        self.total_words = 0
        self.removed_words = 0
        self.removed_by_word: Counter[str] = Counter()

    @property
    def pct_removed(self) -> float:
        return (self.removed_words / self.total_words * 100) if self.total_words else 0.0

    def as_dict(self) -> dict:
        return {
            "file": str(self.path),
            "total_words": self.total_words,
            "removed_words": self.removed_words,
            "pct_removed": round(self.pct_removed, 2),
            "removed_by_word": dict(self.removed_by_word),
        }


def clean_text(text: str, stats: FileStats) -> str:
    stats.total_words += len(_WORD_RE.findall(text))

    def _replace(match: re.Match) -> str:
        stats.removed_words += 1
        stats.removed_by_word[match.group(1).lower()] += 1
        return ""

    cleaned = _FILLER_RE.sub(_replace, text)
    # Collapse whitespace left behind by removed tokens (but preserve
    # leading/trailing spacing the LRC format itself gave the line).
    leading = " " if cleaned[:1].isspace() else ""
    trailing = " " if cleaned[-1:].isspace() and len(cleaned) > 1 else ""
    collapsed = re.sub(r"[ \t]+", " ", cleaned.strip())
    return f"{leading}{collapsed}{trailing}" if collapsed else cleaned.strip()


def clean_lrc_text(raw: str, stats: FileStats) -> str:
    out_lines = []
    for row in raw.splitlines():
        match = _LINE_RE.match(row)
        if not match:
            out_lines.append(row)
            continue
        prefix, text = match.groups()
        out_lines.append(prefix + clean_text(text, stats))
    trailing_newline = "\n" if raw.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_newline


def process_file(path: Path, apply: bool, backup: bool) -> FileStats:
    raw = path.read_text(encoding="utf-8")
    stats = FileStats(path)
    cleaned = clean_lrc_text(raw, stats)

    if apply and stats.removed_words:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if not backup_path.exists():
                backup_path.write_text(raw, encoding="utf-8")
        path.write_text(cleaned, encoding="utf-8")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=RAW_DIR,
                         help="Directory to search for .lrc files, recursively (default: %(default)s)")
    parser.add_argument("--apply", action="store_true",
                         help="Actually rewrite files. Without this flag, it's a dry run.")
    parser.add_argument("--no-backup", action="store_true",
                         help="Skip writing a .bak backup of each modified file (only relevant with --apply)")
    parser.add_argument("--stats-json", type=Path, default=None,
                         help="Optional path to write full per-file stats as JSON")
    args = parser.parse_args()

    files = sorted(args.root.rglob("*.lrc"))
    if not files:
        print(f"No .lrc files found under {args.root}")
        return

    all_stats: list[FileStats] = []
    for path in files:
        all_stats.append(process_file(path, apply=args.apply, backup=not args.no_backup))

    total_words = sum(s.total_words for s in all_stats)
    total_removed = sum(s.removed_words for s in all_stats)
    overall_by_word: Counter[str] = Counter()
    for s in all_stats:
        overall_by_word.update(s.removed_by_word)

    mode = "APPLIED" if args.apply else "DRY RUN (pass --apply to write changes)"
    print(f"{mode} -- scanned {len(files)} file(s) under {args.root}\n")

    for s in all_stats:
        if s.removed_words:
            rel = s.path.relative_to(args.root) if args.root in s.path.parents else s.path
            print(f"  {rel}: removed {s.removed_words}/{s.total_words} words ({s.pct_removed:.2f}%)")

    print(f"\nTotal words scanned:  {total_words}")
    print(f"Filler words removed: {total_removed}")
    print(f"Overall percent of content that was filler: "
          f"{(total_removed / total_words * 100) if total_words else 0:.2f}%")
    if overall_by_word:
        breakdown = ", ".join(f"{word}={count}" for word, count in overall_by_word.most_common())
        print(f"Breakdown by word: {breakdown}")

    if args.stats_json:
        report = {
            "root": str(args.root),
            "applied": args.apply,
            "total_words": total_words,
            "total_removed": total_removed,
            "overall_pct_removed": round((total_removed / total_words * 100) if total_words else 0, 2),
            "removed_by_word": dict(overall_by_word),
            "files": [s.as_dict() for s in all_stats],
        }
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote full stats to {args.stats_json}")


if __name__ == "__main__":
    sys.exit(main() or 0)
