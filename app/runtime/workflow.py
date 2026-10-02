"""Workflow engine.

Loads a workflow definition (YAML) and runs it. A workflow is a mostly
deterministic sequence of steps, with a small number of bounded agentic
decisions inside it:

  - the researcher decides how many searches to run (agentic)
  - the workflow decides whether to re-research (critic's verdicts)

That mix is the point: the sequence is known, the judgement calls inside
each step are the model's.

Making this config-driven means a new workflow is a YAML file, not a code
change: a different step sequence, different per-agent models, different
tool assignments and budgets, all served by the same runtime.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from anthropic import Anthropic

from app.agents import critic as critic_agent
from app.agents import planner as planner_agent
from app.agents import reporter as reporter_agent
from app.agents import researcher as researcher_agent
from app.runtime.agent import Agent
from app.runtime.budget import Budget
from app.runtime.state import TaskState
from app.text import plural
from app.tools.base import ToolContext, ToolRegistry
from app.tools.document_search import search_transcripts

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflows"

DEFAULT_TOOLS = [search_transcripts]


@dataclass
class WorkflowConfig:
    name: str
    steps: list[str]
    models: dict[str, str] = field(default_factory=dict)
    agent_tools: dict[str, list[str]] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str) -> "WorkflowConfig":
        path = Path(name_or_path)
        if not path.exists():
            path = WORKFLOW_DIR / f"{name_or_path}.yaml"
        if not path.exists():
            available = ", ".join(sorted(p.stem for p in WORKFLOW_DIR.glob("*.yaml")))
            raise FileNotFoundError(f"No workflow '{name_or_path}'. Available: {available}")

        raw = yaml.safe_load(path.read_text()) or {}
        missing = [k for k in ("name", "steps") if k not in raw]
        if missing:
            raise ValueError(f"{path} is missing required key(s): {', '.join(missing)}")
        return cls(
            name=raw["name"],
            steps=list(raw["steps"]),
            models=raw.get("models", {}),
            agent_tools=raw.get("tools", {}),
            limits=raw.get("limits", {}),
        )

    def tools_for(self, agent: str) -> list[str]:
        return list(self.agent_tools.get(agent, []))

    def budget(self) -> Budget:
        return Budget(
            max_tool_calls=int(self.limits.get("max_tool_calls", 20)),
            max_model_calls=int(self.limits.get("max_model_calls", 30)),
            max_cost_usd=float(self.limits.get("max_cost_usd", 1.00)),
        )

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.models.get("default", "claude-sonnet-5"))


class WorkflowRunner:
    def __init__(
        self,
        config: WorkflowConfig,
        corpus_db=None,
        client: Anthropic | None = None,
        echo: bool = True,
    ):
        self.config = config
        self.client = client or Anthropic()
        self.registry = ToolRegistry(DEFAULT_TOOLS)
        self.budget = config.budget()
        self.ctx = ToolContext(corpus_db=corpus_db)
        self.echo = echo

    def run(self, objective: str) -> TaskState:
        state = TaskState(objective=objective, workflow=self.config.name)
        self.ctx.state = state

        if self.echo:
            print(f"\nRun {state.run_id} — workflow '{self.config.name}'")
            print(f"Objective: {objective}\n")

        started = time.perf_counter()
        research_notes = ""
        try:
            for step in self.config.steps:
                if self.echo:
                    print(f"\n── {step} ──")
                research_notes = self._run_step(step, state, research_notes)
            state.status = "complete"
        except Exception as exc:
            state.status = "failed"
            state.log(f"error: {type(exc).__name__}: {exc}")
            if self.echo:
                print(f"\nRun failed: {type(exc).__name__}: {exc}")
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if self.echo:
                print(f"\n── done ── {self.budget.summary()} / {elapsed_ms / 1000:.1f}s "
                      f"(status: {state.status})")

        if not state.final_report:
            state.final_report = state.draft
        return state

    # -- steps -----------------------------------------------------------

    def _agent(self, spec) -> Agent:
        return Agent(spec=spec, client=self.client, registry=self.registry,
                     budget=self.budget, ctx=self.ctx, echo=self.echo)

    def _run_step(self, step: str, state: TaskState, research_notes: str) -> str:
        if step == "plan":
            return self._run_plan(state, research_notes)
        if step == "research":
            return self._run_research(state, research_notes)
        if step == "draft":
            return self._run_draft(state, research_notes)
        if step == "critique":
            return self._run_critique(state, research_notes)
        if step == "revise":
            return self._run_revise(state, research_notes)
        raise ValueError(f"Unknown workflow step '{step}'. "
                         f"Known: plan, research, draft, critique, revise")

    def _run_plan(self, state: TaskState, research_notes: str) -> str:
        planner_spec = planner_agent.spec(self.config.model_for("planner"))
        planner = self._agent(planner_spec)
        plan = planner_agent.run(planner, state)
        state.log(f"plan: {plan.objective}")
        return research_notes

    def _run_research(self, state: TaskState, research_notes: str) -> str:
        researcher_tools = self.config.tools_for("researcher")
        researcher_spec = researcher_agent.spec(self.config.model_for("researcher"), researcher_tools)
        researcher = self._agent(researcher_spec)
        return researcher_agent.run(researcher, state)

    def _run_draft(self, state: TaskState, research_notes: str) -> str:
        reporter_spec = reporter_agent.spec(self.config.model_for("reporter"))
        reporter = self._agent(reporter_spec)
        reporter_agent.draft(reporter, state, research_notes)
        return research_notes

    def _run_critique(self, state: TaskState, research_notes: str) -> str:
        critic_spec = critic_agent.spec(self.config.model_for("critic"))
        critic = self._agent(critic_spec)
        report = critic_agent.run(critic, state)

        # bounded re-research: exactly one extra pass, only if the critic asked
        if not (report.needs_more_research and report.follow_up_queries):
            return research_notes

        if self.echo:
            print("   critic requested "
                  f"{plural(len(report.follow_up_queries), 'follow-up search', 'follow-up searches')}")

        researcher_tools = self.config.tools_for("researcher")
        researcher_spec = researcher_agent.spec(self.config.model_for("researcher"), researcher_tools)
        researcher = self._agent(researcher_spec)

        gaps = "\n".join(f"- {q}" for q in report.follow_up_queries)
        follow_up = researcher_agent.run(
            researcher,
            state,
            extra_instructions=(
                "These specific gaps were flagged during verification. Search for "
                f"each one:\n{gaps}"
            ),
        )
        return research_notes + "\n\n=== FOLLOW-UP RESEARCH ===\n" + follow_up

    def _run_revise(self, state: TaskState, research_notes: str) -> str:
        reporter_spec = reporter_agent.spec(self.config.model_for("reporter"))
        reporter = self._agent(reporter_spec)
        reporter_agent.revise(reporter, state)
        return research_notes
