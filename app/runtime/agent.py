"""The agent loop.

A hand-written tool-use loop against the Anthropic Messages API:

    model call -> stop_reason?
        end_turn  -> done, return text
        tool_use  -> for each requested tool:
                       policy check -> approval gate -> budget check -> execute
                     feed all results back as one user message, loop
        refusal / max_tokens / pause_turn -> handled explicitly

Deliberately not using the SDK's tool_runner helper: every hop through
this loop is where permissions, budgets, human approval and tracing get
enforced, and those are the point of the exercise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from anthropic import Anthropic

from app.runtime.approval import Approver, AutoApprover
from app.runtime.budget import Budget, BudgetExceeded
from app.runtime.policy import PolicyEngine
from app.observability.traces import Tracer, Timer
from app.tools.base import ToolContext, ToolRegistry

MAX_ITERATIONS = 12


class StructuredOutputError(RuntimeError):
    """A structured call came back without a valid object.

    Almost always truncation: the model hit max_tokens mid-JSON, so schema
    validation produced nothing. Raised instead of returning None so the
    failure names its own cause rather than surfacing as an AttributeError
    three frames away.
    """


@dataclass
class AgentResult:
    text: str
    stopped_because: str
    iterations: int
    tool_calls: int = 0


@dataclass
class AgentSpec:
    """Static configuration of an agent - what it is, not what it's doing."""
    name: str
    model: str
    system: str
    max_tokens: int = 4096
    tools: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        client: Anthropic,
        registry: ToolRegistry,
        policy: PolicyEngine,
        budget: Budget,
        tracer: Tracer,
        ctx: ToolContext,
        approver: Approver | None = None,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.spec = spec
        self.client = client
        self.registry = registry
        self.policy = policy
        self.budget = budget
        self.tracer = tracer
        self.ctx = ctx
        self.approver = approver or AutoApprover()
        self.max_iterations = max_iterations

    # -- public API ------------------------------------------------------

    def run(self, prompt: str) -> AgentResult:
        """Run the loop until the model stops asking for tools."""
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_schemas = self.registry.api_schemas(self.policy.tools_for(self.spec.name))
        tool_calls = 0

        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self._call_model(messages, tool_schemas)
            except BudgetExceeded as exc:
                self.tracer.record("error", agent=self.spec.name, error=str(exc))
                return AgentResult(self._last_text(messages), f"budget_exceeded: {exc}", iteration, tool_calls)

            if response.stop_reason == "refusal":
                detail = getattr(response, "stop_details", None)
                reason = f"refusal: {getattr(detail, 'category', 'unknown')}"
                self.tracer.record("error", agent=self.spec.name, error=reason)
                return AgentResult("", reason, iteration, tool_calls)

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            if response.stop_reason != "tool_use":
                return AgentResult(_text_of(response), response.stop_reason or "end_turn", iteration, tool_calls)

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                results.append(self._execute_tool_block(block))
            messages.append({"role": "user", "content": results})

        self.tracer.record("note", agent=self.spec.name,
                           output_preview=f"hit max_iterations={self.max_iterations}")
        return AgentResult(self._last_text(messages), "max_iterations", self.max_iterations, tool_calls)

    def run_structured(self, prompt: str, output_format):
        """One-shot call returning a validated Pydantic object (no tools)."""
        self.budget.check_model_call()
        with Timer() as timer:
            response = self.client.messages.parse(
                model=self.spec.model,
                max_tokens=self.spec.max_tokens,
                system=self.spec.system,
                messages=[{"role": "user", "content": prompt}],
                output_format=output_format,
            )
        cost = self.budget.charge(self.spec.model, response.usage.input_tokens, response.usage.output_tokens)
        self.tracer.record(
            "model_call", agent=self.spec.name, model=self.spec.model,
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            cost_usd=cost, latency_ms=timer.ms,
            output_preview=str(response.parsed_output)[:500],
        )

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise StructuredOutputError(self._structured_failure(response, output_format))
        return parsed

    def _structured_failure(self, response, output_format) -> str:
        stop = getattr(response, "stop_reason", None) or "unknown"
        message = (
            f"{self.spec.name}: the model returned no valid {output_format.__name__} "
            f"(stop_reason={stop})."
        )
        if stop == "max_tokens":
            message += (
                f" The response was cut off mid-JSON at max_tokens={self.spec.max_tokens}. "
                "Either raise this agent's max_tokens or ask it for fewer items per call."
            )
        self.tracer.record("error", agent=self.spec.name, error=message)
        return message

    # -- internals -------------------------------------------------------

    def _call_model(self, messages: list[dict], tool_schemas: list[dict]):
        self.budget.check_model_call()
        kwargs = dict(
            model=self.spec.model,
            max_tokens=self.spec.max_tokens,
            system=self.spec.system,
            messages=messages,
        )
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        with Timer() as timer:
            response = self.client.messages.create(**kwargs)

        cost = self.budget.charge(self.spec.model, response.usage.input_tokens, response.usage.output_tokens)
        self.tracer.record(
            "model_call", agent=self.spec.name, model=self.spec.model,
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            cost_usd=cost, latency_ms=timer.ms, output_preview=_text_of(response)[:500],
        )
        return response

    def _execute_tool_block(self, block) -> dict:
        """Policy -> approval -> budget -> execute. Every rejection is a
        tool_result the model can read and react to, not a crash."""
        decision = self.policy.decide(self.spec.name, block.name)
        if not decision.allowed:
            self.tracer.record("tool_call", agent=self.spec.name, tool=block.name,
                               tool_input=dict(block.input), error=decision.reason)
            return _tool_error(block.id, f"Denied by policy: {decision.reason}")

        if decision.requires_approval:
            approved = self.approver.approve(self.spec.name, block.name, dict(block.input))
            if not approved:
                self.tracer.record("tool_call", agent=self.spec.name, tool=block.name,
                                   tool_input=dict(block.input), error="rejected by human")
                return _tool_error(block.id, "A human reviewer rejected this action. Do not retry it.")

        try:
            self.budget.check_tool_call()
        except BudgetExceeded as exc:
            self.tracer.record("tool_call", agent=self.spec.name, tool=block.name,
                               tool_input=dict(block.input), error=str(exc))
            return _tool_error(block.id, f"Budget exceeded: {exc}. Summarise what you have and stop.")

        tool = self.registry.get(block.name)
        if tool is None:
            return _tool_error(block.id, f"Unknown tool '{block.name}'.")

        with Timer() as timer:
            try:
                output = tool.run(dict(block.input), self.ctx)
                error = ""
            except Exception as exc:  # a broken tool shouldn't kill the run
                output, error = f"Tool raised {type(exc).__name__}: {exc}", str(exc)

        self.tracer.record("tool_call", agent=self.spec.name, tool=block.name,
                           tool_input=dict(block.input), output_preview=output[:500],
                           latency_ms=timer.ms, error=error)
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
            **({"is_error": True} if error else {}),
        }

    @staticmethod
    def _last_text(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message["role"] != "assistant":
                continue
            content = message["content"]
            if isinstance(content, str):
                return content
            texts = [b.text for b in content if getattr(b, "type", "") == "text"]
            if texts:
                return "\n".join(texts)
        return ""


def _text_of(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text")


def _tool_error(tool_use_id: str, message: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": message, "is_error": True}
