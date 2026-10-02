# Council Analyzer

**A personal project built to learn how multi-agent orchestration and shared
agent state actually work.** The domain — city and county council meeting
transcripts — was chosen because it's messy, real, and publicly available:
auto-generated captions with no speaker labels, misheard proper nouns, and hours of
procedural filler around the parts that matter.

A question goes in; a plan, a set of searches, a draft, a verification pass, and a
cited report come out.

**Built with:** Python · Anthropic API (Claude) · Pydantic · sentence-transformers · SQLite

This README covers what is actually implemented and how to run it.

## What it does

```
question
   ↓
Planner ────────── decomposes into subquestions          (structured output)
   ↓
Researcher ─────── searches transcripts, decides when      (agent loop + tools)
   ↓                to search again
Reporter ───────── drafts a cited answer
   ↓
Critic ─────────── checks every claim against the          (structured output)
   ↓                evidence actually retrieved; may
   ↓                trigger one more research pass
Reporter ───────── revises per the critic's verdicts
   ↓
cited report
```

Each agent reads and writes one shared `TaskState`, so the critic can inspect
what the researcher actually found and the reporter can only cite evidence that
was really retrieved. `TaskState.notes` is also the run's log: every agent appends
one line describing what it did, in the order it happened, so after a run you can
read `state.notes` and see exactly how the answer was assembled. There is no
separate trace database — the state object *is* the record.

The critic's input is deliberately asymmetric: it gets the **complete** list of
valid citation IDs (detecting a *fabricated* `[C###]` is impossible without
knowing the whole valid set) but full passage text only for what the draft
actually cites — typically a handful out of dozens retrieved.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env    # add your ANTHROPIC_API_KEY
```

Embeddings run locally via `sentence-transformers` (`all-MiniLM-L6-v2`) — no
Anthropic embeddings API exists, and this keeps ingestion free. The model
downloads from Hugging Face on first use.

## Use cases

### 1. Deep research (the main one)

Plan → research → draft → verify → revise, with citations back to video timestamps.

```bash
python3 scripts/run_workflow.py "How has the council's stance on short-term rentals changed?"
python3 scripts/run_workflow.py "..." --save-state runs/state.json --quiet
```

`--save-state` writes the full `TaskState` as JSON, including `notes` — the
clearest way to see the whole run's state after the fact.

### 2. Single-pass RAG baseline

Retrieve and answer in one shot — no agents. Useful as the control in experiments,
and as the simplest possible example of citing retrieved evidence before looking
at the agent loop.

```bash
python3 scripts/ask.py "What did the council decide about short-term rentals?" --city "City of Green Bay"
```

## Getting from raw files to a running system

```bash
# 1. Ingest transcripts (data/raw/<city>/*.lrc)
python3 scripts/ingest_documents.py
python3 scripts/ingest_documents.py --chunk-words 180 --overlap 0.2 --db data/processed/exp.db

# 2. Ask something
python3 scripts/run_workflow.py "your question"
```

## Configuration

Workflows are data, not code. `workflows/deep_research.yaml` sets the step
sequence, per-agent model routing, per-agent tool permissions, and budgets.

Note that `max_tokens` is a budget for everything a model writes, thinking
included — on models that think by default, reasoning and the answer come out of
the same allowance. Each agent sets its thinking mode explicitly (`adaptive` or
`off`) rather than inheriting a per-model default.

```yaml
steps: [plan, research, draft, critique, revise]

models:
  planner: claude-haiku-4-5      # cheap model for structured decomposition
  default: claude-sonnet-5

tools:                            # least privilege: only the researcher
  researcher: [search_transcripts]  # can touch the corpus

limits:
  max_tool_calls: 25
  max_cost_usd: 1.00
```

Adding a workflow means adding a YAML file with a different step sequence,
model routing, or tool permissions. No runtime changes.

## Project layout

```
app/
  runtime/        agent loop, workflow engine, policy, budgets, state
  agents/         planner, researcher, critic, reporter
  tools/          search_transcripts
  rag/            parse_lrc, chunk, embed, ingest, retrieve
  db/             corpus schema/connection
  text.py         small formatting helpers (plural, clip)
workflows/        deep_research.yaml
scripts/          CLI entry points
data/
  raw/            <city>/<meeting>.lrc  (gitignored)
  processed/      corpus database (gitignored)
```

## Safety and controls

- **Least privilege** — an agent can only call tools its workflow grants it.
  A denied call comes back as a readable error the model can react to, not a crash.
- **Budgets** — per-run caps on tool calls, model calls and dollar cost, enforced
  before each call. Exceeding one ends the run cleanly.
- **Untrusted corpus** — transcripts are treated as data. Agents are instructed to
  cite only retrieved passages and never to act on instructions found inside them.

## Known limitations

- **No speaker attribution.** Auto-captions don't identify who's speaking, so
  "what did council member X say" can't be answered — only "what was said."
- **Retrieval misses rare proper nouns.** Dense embeddings dilute a short distinctive
  phrase inside a long procedural chunk. Hybrid keyword + vector search is the likely
  fix and is not built yet.
- **Chunking is word-count based**, with no awareness of agenda-item boundaries, so a
  chunk can straddle two unrelated topics.
- **Retrieval is brute-force cosine similarity** in numpy — fine for thousands of
  chunks, would need a real vector index beyond that.
- **The critic is another probabilistic model**, not an oracle. Its verdicts are
  recorded in `state.notes` so a human can disagree.
- **A structured call that gets truncated costs money the budget never sees.** The
  SDK validates inside `messages.parse()`, so a response cut off at `max_tokens`
  raises before any usage is returned. The run fails with a clear error naming the
  cap, but that call's cost cannot be recovered or charged.
- **A single, very long draft could still truncate the critic's response**, since
  the critic now checks the whole draft in one call rather than splitting it into
  batches. Raise the critic's `max_tokens` in the workflow YAML if that happens.
- **`sentence-transformers` is capped below 3.0** because newer releases require
  `torch>=2.5`, which has no Intel-Mac wheel. Drop the cap on Apple Silicon or Linux.
