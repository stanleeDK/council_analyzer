# Council Analyzer

**A personal project built to learn and experiment with multi-agent orchestration,
RAG, tool use, evaluation and agent controls.** The domain — city and county council
meeting transcripts — was chosen because it's messy, real, and publicly available:
auto-generated captions with no speaker labels, misheard proper nouns, and hours of
procedural filler around the parts that matter.

A question goes in; a plan, a set of searches, a draft, a verification pass, and a
cited report come out — with a full trace of every model and tool call.

**Built with:** Python · Anthropic API (Claude) · Pydantic · sentence-transformers · SQLite

The staged roadmap this repo follows is [`docs/PLATFORM_PLAN.md`](docs/PLATFORM_PLAN.md).
This README covers what is actually implemented and how to run it.

## What it does

```
question
   ↓
Planner ────────── decomposes into subquestions          (structured output)
   ↓
Researcher ─────── searches transcripts, decides when      (agent loop + tools)
   ↓                to search again
Data Analyst ───── writes read-only SQL, only if the       (agent loop + tools)
   ↓                planner flagged the question as
   ↓                quantitative
Reporter ───────── drafts a cited answer
   ↓
Critic ─────────── checks every claim against the          (structured output)
   ↓                evidence actually retrieved; may
   ↓                trigger one more research pass
Reporter ───────── revises per the critic's verdicts
   ↓
cited report + trace + cost
```

Each agent reads and writes one shared `TaskState`, so the critic can inspect
what the researcher actually found and the reporter can only cite evidence that
was really retrieved.

## Example run

<!-- TODO(stanley): replace the two blocks below with real captured output.
     Capture with:
       python3 scripts/run_workflow.py "How has the council's stance on short-term rentals changed?" | tee /tmp/run.txt
       python3 scripts/show_trace.py <run_id> | tee /tmp/trace.txt
     Keep it short — trim the report to a Summary + 2-3 findings, and the trace
     to the step list. Do not paste anything not actually produced by a run. -->

> **Note:** this section is a placeholder pending a captured run. Everything below
> is illustrative structure, not real output.

**Question:** *"How has the council's stance on short-term rentals changed?"*

```
── plan ──
  [planner] model claude-haiku-4-5 — 1.2s / $0.0004
── research ──
  [researcher] tool  search_transcripts({"query": "short-term rental ordinance"})
  [researcher] tool  search_transcripts({"query": "Airbnb licensing complaints"})
── draft ──
── critique ──
   critic requested 1 follow-up search
── revise ──
── done ── 8 model calls, 5 tool calls, $0.08 / 31s (status: complete)
```

```markdown
## Summary
The council moved from exploratory discussion toward a licensing framework [C412],
though no ordinance had passed as of the most recent indexed meeting [C588].

## Evidence Gaps and Uncertainties
No transcript covers the committee session referenced in [C588], so the outcome
of that referral is unknown.

## Sources
[C412] City of Green Bay — Protection & Policy, 2025-04-15, 1240s–1310s
[C588] City of Green Bay — Common Council, 2026-02-03, 880s–944s
```

The SQL guardrails need no API key, so these are verbatim from `app/tools/sql.py`:

```
>>> run_readonly_sql({"sql": "DELETE FROM meetings"})
SQL rejected: only SELECT/WITH queries are allowed, got 'DELETE'

>>> run_readonly_sql({"sql": "SELECT 1; DROP TABLE meetings"})
SQL rejected: multiple statements are not allowed; submit one SELECT at a time

>>> run_readonly_sql({"sql": "SELECT nonexistent FROM meetings"})
SQL error: OperationalError: no such column: nonexistent
```

The last one matters as much as the rejections: a failed query comes back as text
the model can read and correct, rather than crashing the run.

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

### 2. Quantitative analysis (Text2SQL)

No retrieval — the agent inspects a schema, writes read-only SQL, sanity-checks it.

```bash
python3 scripts/run_workflow.py "How many meetings did each city hold in 2026?" --workflow text2sql
```

### 3. Research with a human approval gate

Same as deep research, but every SQL query stops and asks you first.

```bash
python3 scripts/run_workflow.py "..." --workflow research_with_approval
```

### 4. Single-pass RAG baseline

Retrieve and answer in one shot — no agents. Useful as the control in experiments.

```bash
python3 scripts/ask.py "What did the council decide about short-term rentals?" --city "City of Green Bay"
```

## Getting from raw files to a running system

```bash
# 1. Ingest transcripts (data/raw/<city>/*.lrc)
python3 scripts/ingest_documents.py
python3 scripts/ingest_documents.py --chunk-words 180 --overlap 0.2 --db data/processed/exp.db

# 2. Build the analytics DB the SQL agent queries (derived from the corpus)
python3 scripts/seed_analytics_db.py

# 3. Ask something
python3 scripts/run_workflow.py "your question"

# 4. See exactly what happened
python3 scripts/show_trace.py            # list recent runs
python3 scripts/show_trace.py <run_id>   # every model + tool call, with cost
```

## Retrieval evaluation

This measures the **retriever**, not the final generated report. Answer quality,
groundedness and citation correctness are not yet scored — see Known limitations.

Retrieval is scored against hand-verified cases in `data/evals/retrieval_evals.json`.
Ground truth is a *substring a correct passage must contain*, not a chunk ID —
chunk IDs change whenever the corpus is re-chunked, so content-based truth is
what makes a parameter sweep possible.

```bash
python3 scripts/run_evals.py                          # hit_rate + MRR for one corpus
python3 scripts/run_evals.py --db data/processed/exp.db --top-k 10

# Grid-search chunking against the evals (slow: re-embeds the corpus per point)
python3 scripts/sweep_chunking.py --chunk-sizes 80 180 300 --overlaps 0 0.2
```

**Add more eval cases.** Five cases prove almost nothing; 15-30 across different
cities and question styles is the minimum for tuning against.

## Configuration — what makes this a platform

Workflows are data, not code. `workflows/*.yaml` sets the step sequence, per-agent
model routing, per-agent tool permissions, approval gates and budgets. The same
runtime serves all three shipped workflows:

```yaml
steps: [plan, research, data_analysis, draft, critique, revise]

models:
  planner: claude-haiku-4-5      # cheap model for structured decomposition
  default: claude-sonnet-5

tools:                            # least privilege: researcher can't touch SQL
  researcher: [search_transcripts]
  data_analyst: [get_schema, run_readonly_sql, calculate]

approvals:
  run_readonly_sql: always        # <- flip to gate this tool behind a human

limits:
  max_tool_calls: 25
  max_cost_usd: 1.00
```

Adding a workflow means adding a YAML file. No runtime changes.

## Project layout

```
app/
  runtime/        agent loop, workflow engine, policy, budgets, approvals, state
  agents/         planner, researcher, data_analyst, critic, reporter
  tools/          search_transcripts, run_readonly_sql, get_schema, calculate
  rag/            parse_lrc, chunk, embed, ingest, retrieve
  db/             corpus schema/connection + derived analytics DB
  evaluation/     retrieval metrics, eval runner, parameter sweep
  observability/  trace recording
workflows/        deep_research, text2sql, research_with_approval
scripts/          CLI entry points
data/
  raw/            <city>/<meeting>.lrc  (gitignored)
  processed/      corpus, analytics and trace databases (gitignored)
  evals/          hand-verified eval cases
```

## Safety and controls

- **Least privilege** — an agent can only call tools its workflow grants it.
  A denied call comes back as a readable error the model can react to, not a crash.
- **Read-only SQL** — the analytics DB is opened `mode=ro`, and SQL is additionally
  validated (single statement, SELECT/WITH only, forbidden keywords, injected LIMIT,
  query timeout) so the model gets clear errors instead of opaque failures.
- **No code execution** — `calculate` walks the Python AST and permits only
  arithmetic; `eval()` is never used on model output.
- **Budgets** — per-run caps on tool calls, model calls and dollar cost, enforced
  before each call. Exceeding one ends the run cleanly.
- **Human approval** — any tool can be gated behind a terminal prompt via config.
- **Untrusted corpus** — transcripts are treated as data. Agents are instructed to
  cite only retrieved passages and never to act on instructions found inside them.

## Known limitations

- **No speaker attribution.** Auto-captions don't identify who's speaking, so
  "what did council member X say" can't be answered — only "what was said."
- **Retrieval misses rare proper nouns.** Dense embeddings dilute a short distinctive
  phrase inside a long procedural chunk. `eval_001` documents a real, reproducible
  miss. Hybrid keyword + vector search is the likely fix and is not built yet.
- **Chunking is word-count based**, with no awareness of agenda-item boundaries, so a
  chunk can straddle two unrelated topics.
- **The analytics DB covers meeting coverage, not outcomes** — which meetings were
  recorded, when, how long. It has no votes or motions; the data agent is told to say
  so rather than approximate.
- **Retrieval is brute-force cosine similarity** in numpy — fine for thousands of
  chunks, would need a real vector index beyond that.
- **The critic is another probabilistic model**, not an oracle. Its verdicts are
  recorded in the trace so a human can disagree.
- **`sentence-transformers` is capped below 3.0** because newer releases require
  `torch>=2.5`, which has no Intel-Mac wheel. Drop the cap on Apple Silicon or Linux.
