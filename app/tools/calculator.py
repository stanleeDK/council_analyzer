"""calculate tool - arithmetic without letting the model run code.

Uses Python's AST to evaluate only a whitelist of numeric operations.
eval() on model-supplied text would be arbitrary code execution; this
walks the parse tree and rejects anything that isn't arithmetic.
"""
from __future__ import annotations

import ast
import operator

from app.tools.base import Tool, ToolContext

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class UnsafeExpression(ValueError):
    pass


def evaluate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"could not parse: {exc}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpression("only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise UnsafeExpression("division by zero")
        if type(node.op) is ast.Pow and abs(right) > 100:
            raise UnsafeExpression("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise UnsafeExpression(f"unsupported expression element: {type(node).__name__}")


def _handler(tool_input: dict, ctx: ToolContext) -> str:
    expression = (tool_input.get("expression") or "").strip()
    if not expression:
        return "Error: 'expression' is required."
    try:
        return f"{expression} = {evaluate(expression)}"
    except UnsafeExpression as exc:
        return f"Rejected: {exc}"


calculate = Tool(
    name="calculate",
    description=(
        "Evaluate a numeric arithmetic expression (+, -, *, /, //, %, **). "
        "Use this instead of doing arithmetic mentally when precision matters."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '(1240 - 980) / 980 * 100'"}
        },
        "required": ["expression"],
    },
    handler=_handler,
    read_only=True,
)
