# Council Analyzer

An agentic RAG system for analyzing city and county council meeting transcripts —
built as a hands-on learning project. The full staged roadmap this repo follows
lives in [`docs/PLATFORM_PLAN.md`](docs/PLATFORM_PLAN.md); this README covers
what's actually implemented and how to run it.

## Data

Source data is YouTube auto-caption exports (`.lrc` format) of council/committee
meetings across several cities and counties. Auto-caption exports repeat each
caption line 2-3 times (an artifact of how the sliding caption window maps onto
LRC's single-timestamp format) and carry **no speaker labels** — both handled
(or noted as a known limitation) in the ingestion pipeline below.

Organize raw files as:

```
data/raw/<city_slug>/<meeting>.lrc
```

City comes from the directory name since it isn't reliably present in the
filenames themselves.

## Current status

- **Stage 1 — Structured planner**: `app/models/llm.py` (`make_plan`) — question → validated `ResearchPlan`.
- **Stage 2 — RAG**: `app/rag/` — LRC parsing/dedup → chunking → local embeddings → SQLite storage → cosine retrieval → cited answers via `scripts/ask.py`.

Nothing beyond this is wired up yet (no tool-calling agent loop, no critic, no Text2SQL) — see `docs/PLATFORM_PLAN.md` for what comes next.

## Setup

```bash
pip install -e .
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

Embeddings run locally via `sentence-transformers` (`all-MiniLM-L6-v2`) — no
Anthropic embeddings API exists, and this avoids per-chunk API cost during
ingestion. The model downloads from Hugging Face on first use, so you'll need
normal internet access the first time you run ingestion.

## Usage

```bash
# 1. Ingest all .lrc files under data/raw/<city>/
python3 scripts/ingest_documents.py

# 2. Turn a question into a structured research plan (Stage 1, standalone)
python3 scripts/plan.py "How has the city's or council's stance on immigration  changed?"

# 3. Ask a question against the indexed transcripts and get a cited answer (Stage 2)
python scripts/ask.py "What did the council decide about short-term rentals?"
python scripts/ask.py "..." --city springfield --top-k 10
```

## Project layout

```
app/
  models/       # Stage 1: Pydantic schemas + structured-output planner call
  rag/          # Stage 2: parse_lrc, chunk, embed, ingest, retrieve
  db/           # SQLite schema + connection
scripts/        # CLI entry points
data/
  raw/          # <city>/<meeting>.lrc source files (gitignored contents)
  processed/    # council.db (gitignored)
  evals/        # hand-written eval question sets (Stage 7, not yet built)
tests/
docs/
  PLATFORM_PLAN.md   # full staged roadmap (Stage 1 through platform-ification)
```

## Known limitations

- **No speaker attribution.** Auto-captions don't identify who's speaking, so
  "what did council member X say" currently can't be answered — only "what was
  said." Revisit if per-speaker diarization becomes available.
- **Filename metadata is best-effort.** `app/rag/filenames.py` heuristically
  extracts a date and title from filenames; there's no reliable standard across
  sources, so verify metadata after ingesting a new city's files.
- **Retrieval is brute-force cosine similarity** in Python/numpy — fine for a
  corpus of a few thousand chunks (5-8 cities' worth of meetings), but would
  need a real vector index if the corpus grows much larger.
