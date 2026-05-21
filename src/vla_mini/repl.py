"""Sandboxed execution of user-edited Python against the live env session."""

from __future__ import annotations

import ast
import builtins
import io
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from vla_mini.env.base import ToyEnv


class UnsafeCodeError(ValueError):
    pass


_BLOCKED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "license",
        "credits",
        "input",
        "memoryview",
    }
)


class _SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        raise UnsafeCodeError("不允许 import（请使用已注入的 env, np, Image, predict）")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise UnsafeCodeError("不允许 import（请使用已注入的 env, np, Image, predict）")

    def visit_Global(self, node: ast.Global) -> None:
        raise UnsafeCodeError("不支持 global")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise UnsafeCodeError("不支持 nonlocal")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr not in {"__name__"}:
            raise UnsafeCodeError(f"禁止访问魔术属性: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES:
            raise UnsafeCodeError(f"禁止使用内置函数: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_NAMES:
            raise UnsafeCodeError(f"禁止调用: {node.func.id}")
        self.generic_visit(node)


def validate_code(source: str) -> None:
    if len(source) > 12_000:
        raise UnsafeCodeError("代码过长（上限 12000 字符）")
    tree = ast.parse(source, mode="exec")
    _SafetyVisitor().visit(tree)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    error: str | None
    namespace: dict[str, Any] = field(default_factory=dict)
    obs: np.ndarray | None = None
    instruction: str | None = None
    action: np.ndarray | None = None
    overlay_base: Image.Image | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def console_text(self) -> str:
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append(self.stderr.rstrip())
        if self.error:
            parts.append(f"ERROR:\n{self.error}")
        return "\n".join(parts) if parts else "(no output)"


def _safe_builtins() -> dict[str, Any]:
    allowed = (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "round",
        "str",
        "sum",
        "tuple",
        "zip",
        "True",
        "False",
        "None",
    )
    out: dict[str, Any] = {"__builtins__": {}}
    inner = out["__builtins__"]
    for name in allowed:
        inner[name] = getattr(builtins, name)
    return out


@dataclass
class CodeSession:
    env: ToyEnv
    predict_fn: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def namespace(self) -> dict[str, Any]:
        ns: dict[str, Any] = {
            "env": self.env,
            "np": np,
            "numpy": np,
            "Image": Image,
            "obs": None,
            "instruction": None,
            "action": None,
            "result": None,
        }
        if self.predict_fn is not None:
            ns["predict"] = self.predict_fn
        ns.update(self.extra)
        return ns

    def execute(self, source: str) -> ExecResult:
        validate_code(source)
        ns = self.namespace()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error: str | None = None

        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf
            merged = _safe_builtins()
            merged.update(ns)
            exec(compile(source, "<vla-mini>", "exec"), merged, merged)  # noqa: S102
            ns = merged
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout, sys.stderr = old_out, old_err

        obs = ns.get("obs")
        if obs is None:
            result = ns.get("result")
            if result is not None and hasattr(result, "observation"):
                obs = result.observation
        if isinstance(obs, np.ndarray) and obs.dtype != np.uint8:
            obs = obs.astype(np.uint8)

        instruction = ns.get("instruction")
        if instruction is None:
            result = ns.get("result")
            if result is not None and hasattr(result, "instruction"):
                instruction = result.instruction
            elif hasattr(self.env, "instruction"):
                instruction = self.env.instruction

        action = ns.get("action")
        if action is not None:
            action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]

        overlay_base = None
        if isinstance(obs, np.ndarray):
            overlay_base = Image.fromarray(obs.astype(np.uint8))

        return ExecResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=error,
            namespace=ns,
            obs=obs if isinstance(obs, np.ndarray) else None,
            instruction=str(instruction) if instruction is not None else None,
            action=action,
            overlay_base=overlay_base,
        )


DEFAULT_SNIPPET = """# 可用: env, np, Image, predict(可选), obs, instruction, action, result
obs, instruction = env.reset()
action = env.expert_action()
result = env.step(action)
print("instruction:", instruction)
print("info:", result.info)
"""

PUSH_SNIPPET = """# L1 push_block — 推彩色方块进绿色 zone
obs, instruction = env.reset()
action = env.expert_action()
result = env.step(action)
print("instruction:", instruction)
print("info:", result.info)
"""

DRY_RUN_SNIPPET = DEFAULT_SNIPPET
