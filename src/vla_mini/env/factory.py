"""Create teaching environments by task name."""

from __future__ import annotations

from typing import Literal

from vla_mini.env.base import ToyEnv
from vla_mini.env.toy_push import ToyPushEnv
from vla_mini.env.toy_reach import ToyReachEnv

TaskName = Literal["reach", "push"]
TASK_NAMES: tuple[str, ...] = ("reach", "push")


def make_env(task: str = "reach", seed: int | None = None, **kwargs) -> ToyEnv:
    """Build a 2D toy env. ``task``: ``reach`` (L0) or ``push`` (L1 push_block)."""
    key = task.strip().lower()
    if key == "reach":
        return ToyReachEnv(seed=seed, **kwargs)
    if key == "push":
        return ToyPushEnv(seed=seed, **kwargs)
    raise ValueError(f"unknown task {task!r}; choose from {TASK_NAMES}")
