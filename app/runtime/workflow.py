"""Workflow engine.

Loads a workflow definition (YAML) and runs it. A workflow is a mostly
deterministic sequence of steps, with a small number of bounded agentic
decisions inside it:

  - the researcher decides how many searches to run (agentic)
  - the workflow decides whether to run the data agent (planner's flag)
  - the workflow decides whether to re-research (critic's verdicts)

That mix is the point: the sequence is known, the judgement calls inside
each step are the model's.

Making this config-driven is what turns the app into a platform - the
same runtime serves deep_research.yaml and text2sql.yaml with different
agents, tools, permissions and budgets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from anthropic import Anthropic

from app.agents import critic as critic_agent
from app.agents import data_analyst as data_agent
from app.agents import planner as planner_agent
from app.agents import reporter as reporter_agent
from app.agents import researcher as researcher_agent
from app.observability.traces import Tracer, get_trace_db, plural
from app.runtime.agent import Agent
from app.runtime.approval import Approver, AutoApprover
from app.runtime.budget import Budget
from app.runtime.policy import PolicyEngine
from app.runtime.state import TaskState
from app.tools.base import ToolContext, ToolRegistry
from app.tools.calculator import calculate
from app.tools.document_search import search_transcripts
from app.tools.sql import get_schema, run_readonly_sql

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflows"

DEFAULT_TOOLS = [search_transcripts, run_readonly_sql, get_schema, calculate]


@dataclass
class WorkflowConfig:
    name: str
    steps: list[str]
    models: dict[str, str] = field(default_factory=dict)
    agent_tools: dict[str, list[str]] = field(default_factory=dict)
    approvals: dict[str, str] = field(default_factory=dict)
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
            approvals=raw.get("approvals", {}),
            limits=raw.get("limits", {}),
        )

    def policy(self) -> PolicyEngine:
        return PolicyEngine(agent_tools=self.agent_tools, approvals=self.approvals)

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
        analytics_db=None,
        client: Anthropic | None = None,
        approver: Approver | None = None,
        trace_db=None,
        echo: bool = True,
    ):
        self.config = config # yaml file steps changed into an internal class
        self.client = client or Anthropic()
        self.registry = ToolRegistry(DEFAULT_TOOLS)
        self.policy = config.policy()
        self.budget = config.budget()
        self.approver = approver or AutoApprover()
        self.ctx = ToolContext(corpus_db=corpus_db, analytics_db=analytics_db)
        self.trace_db = trace_db or get_trace_db()
        self.echo = echo #should the app print to console or not

    # this function loads up the runner class's attributes which is the orchestrator 
    # start the workflow with _run
    def run(self, objective: str) -> TaskState:
        state = TaskState(objective=objective, workflow=self.config.name)
        self.ctx.state = state
        tracer = Tracer(db=self.trace_db, run_id=state.run_id,
                        workflow=self.config.name, echo=self.echo)
        tracer.start_run(objective)

        if self.echo:
            print(f"\nRun {state.run_id} — workflow '{self.config.name}'")
            print(f"Objective: {objective}\n")

        started = time.perf_counter()
        research_notes = ""
        try:
            for step in self.config.steps:
                if self.echo:
                    print(f"\n── {step} ──")
                research_notes = self._run_step(step, state, tracer, research_notes)
            state.status = "complete"
        except Exception as exc:
            state.status = "failed"
            tracer.record("error", error=f"{type(exc).__name__}: {exc}")
            if self.echo:
                print(f"\nRun failed: {type(exc).__name__}: {exc}")
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            tracer.finish_run(state.status, self.budget.cost_usd, elapsed_ms)
            if self.echo:
                print(f"\n── done ── {self.budget.summary()} / {elapsed_ms/1000:.1f}s "
                      f"(status: {state.status})")

        if not state.final_report:
            state.final_report = state.draft
        return state

    # -- steps -----------------------------------------------------------

    def _agent(self, spec, tracer: Tracer) -> Agent:
        return Agent(spec=spec, client=self.client, registry=self.registry,
                     policy=self.policy, budget=self.budget, tracer=tracer,
                     ctx=self.ctx, approver=self.approver)

    def _run_step(self, step: str, state: TaskState, tracer: Tracer, research_notes: str) -> str:
        cfg = self.config

        if step == "plan":
            plan = planner_agent.run(
                self._agent(
                    planner_agent.spec(cfg.model_for("planner")), tracer),
                    state
                )
            
            state.notes.append(f"plan: {plan.objective}")
            return research_notes

        if step == "research":
            return researcher_agent.run(
                self._agent(researcher_agent.spec(cfg.model_for("researcher"), self.policy.tools_for("researcher")), tracer),
                state)

        if step == "data_analysis":
            # If a planner ran and judged this question non-quantitative, skip the
            # step rather than paying for SQL nobody asked for.
            if "plan" in cfg.steps and not state.needs_quantitative_data:
                tracer.record("note", agent="data_analyst",
                              output_preview="skipped: planner judged question non-quantitative")
                return research_notes
            data_agent.run(
                self._agent(data_agent.spec(
                    cfg.model_for("data_analyst"), self.policy.tools_for("data_analyst")), tracer),
                state)
            return research_notes

        if step == "draft":
            reporter_agent.draft(
                self._agent(reporter_agent.spec(cfg.model_for("reporter")), tracer),
                state, research_notes)
            return research_notes

        if step == "critique":
            report = critic_agent.run(
                self._agent(critic_agent.spec(cfg.model_for("critic")), tracer), state)
            # bounded re-research: exactly one extra pass, only if the critic asked
            if report.needs_more_research and report.follow_up_queries:
                if self.echo:
                    print("   critic requested "
                          f"{plural(len(report.follow_up_queries), 'follow-up search', 'follow-up searches')}")
                follow_up = researcher_agent.run(
                    self._agent(researcher_agent.spec(
                        cfg.model_for("researcher"), self.policy.tools_for("researcher")), tracer),
                    state,
                    extra_instructions=(
                        "These specific gaps were flagged during verification. Search for "
                        "each one:\n" + "\n".join(f"- {q}" for q in report.follow_up_queries)
                    ),
                )
                return research_notes + "\n\n=== FOLLOW-UP RESEARCH ===\n" + follow_up
            return research_notes

        if step == "revise":
            reporter_agent.revise(
                self._agent(reporter_agent.spec(cfg.model_for("reporter")), tracer), state)
            return research_notes

        raise ValueError(f"Unknown workflow step '{step}'. "
                         f"Known: plan, research, data_analysis, draft, critique, revise")
