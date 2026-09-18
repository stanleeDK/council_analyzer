# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

An agentic RAG platform over city/county council meeting transcripts — a personal
project for learning multi-agent orchestration, tool use, evals and agent controls.
`README.md` covers what's built; `docs/PLATFORM_PLAN.md` is the roadmap it follows.

## Environment

- **Activate the venv first:** `source .venv/bin/activate`. Inside it, `python` is
  the project's Python 3.12 and is the normal thing to type.
- Outside the venv, `python` is Python 2.7 on the dev machine and dies on modern
  syntax. The commands below say `python3` because that at least gets a Python 3 if
  someone forgets to activate — but activation is the real requirement, since the
  system Python 3 has none of the dependencies.
- `pip install -e .` fails here (PEP 660). Install dependencies directly instead;
  the scripts already `sys.path.insert`, so an editable install isn't needed.
- `sentence-transformers` is pinned `<3.0` and `numpy` `<2`. **Do not bump either.**
  Newer releases need `torch>=2.5`, which has no Intel-Mac wheel; the machine is
  capped at torch 2.2.2. The cap is a hardware constraint, not staleness.

## Commands

```bash
python3 -m pytest tests/ -q                  # the whole suite
python3 scripts/ingest_documents.py          # data/raw/*.lrc -> corpus DB
python3 scripts/seed_analytics_db.py         # corpus -> analytics DB
python3 scripts/run_workflow.py "question"   # the agent workflow
python3 scripts/show_trace.py <run_id>       # every model + tool call, with cost
python3 scripts/run_evals.py                 # retrieval hit_rate + MRR
```

## Data

**`data/raw/` is gitignored and exists only on the local disk.** It is not in the
repo and is not recoverable from git. Never delete it, and never assume a fresh
clone has transcripts. `data/processed/*.db` is likewise gitignored and rebuildable.

## Architecture notes

- Workflows are data: `workflows/*.yaml` sets steps, per-agent model routing, tool
  allowlists, approval gates and budgets. Adding a workflow means adding a YAML file,
  not changing the runtime.
- The agent loop in `app/runtime/agent.py` is hand-written rather than the SDK's
  `tool_runner`, deliberately — policy, budget, approval and tracing are enforced at
  each hop, and that's the point of the exercise.
- `max_tokens` is a budget for everything a model writes, **thinking included**. On
  models that think by default, reasoning and the answer share one allowance; that's
  what truncated the critic's JSON twice. Each agent sets `thinking` explicitly.
- Agents narrate what they did via `tracer.record("note", ...)`. `TaskState.notes` is
  *not* echoed — a note only reaches the terminal through the tracer.

## Working style

### Making code changes

- **Ask before changing code.** Describe the change, then wait for a go-ahead.
- **Use the Edit/Write tools, not shell scripts that do string replacement.** I want
  to see the actual before/after, not a command that describes one. `sed -i` and
  Python `str.replace()` also fail silently on a bad match.
- If a bulk mechanical edit genuinely needs a script, say so first and show
  `git diff` before committing.

### Verifying

- **Passing tests are not evidence the feature works.** Where the thing can be run,
  run it and look at the real output. Two formatting bugs in this repo got past a
  green suite and were caught only by rendering a run.
- **Never present invented output as captured output.** If a real run hasn't
  happened, label the example illustrative or leave a TODO.
- Say plainly what was verified and what is inferred.
- Report failures with the actual error text, not a summary of it.

### Explaining

- This is a learning project. Explain why a design is the way it is, including
  trade-offs I didn't ask about.
- Lead with the plain-English version; add the jargon after, if it's needed.
- When I question a design, answer the design question before fixing the bug.
