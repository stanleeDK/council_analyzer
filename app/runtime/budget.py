"""Cost / iteration budgets.

An agent loop without limits can spin forever or run up a bill. Every
loop checks a Budget before each model call and each tool call, and
raises BudgetExceeded rather than silently continuing.

Prices are a cached snapshot (USD per million tokens) - update if
Anthropic's pricing changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRICES_USD_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_tool_calls: int = 20
    max_model_calls: int = 30
    max_cost_usd: float = 1.00

    tool_calls: int = 0
    model_calls: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    per_model: dict[str, float] = field(default_factory=dict)

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rate_in, rate_out = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
        cost = (input_tokens / 1_000_000) * rate_in + (output_tokens / 1_000_000) * rate_out
        self.cost_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.per_model[model] = self.per_model.get(model, 0.0) + cost
        return cost

    def check_model_call(self) -> None:
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded(f"max_model_calls={self.max_model_calls} reached")
        if self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"max_cost_usd=${self.max_cost_usd:.2f} reached (spent ${self.cost_usd:.4f})")
        self.model_calls += 1

    def check_tool_call(self) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded(f"max_tool_calls={self.max_tool_calls} reached")
        self.tool_calls += 1

    def summary(self) -> str:
        return (f"{self.model_calls} model calls, {self.tool_calls} tool calls, "
                f"{self.input_tokens} in / {self.output_tokens} out tokens, "
                f"${self.cost_usd:.4f}")
