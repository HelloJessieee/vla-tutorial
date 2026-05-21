"""Create teaching environments by task name."""

from __future__ import annotations

from vla_mini.env.base import ToyEnv
from vla_mini.env.tasks import TASK_NAMES, TaskName, TaskSpec, get_task_spec
from vla_mini.env.toy_grasp import ToyGraspEnv
from vla_mini.env.toy_push import ToyPushEnv
from vla_mini.env.toy_reach import ToyReachEnv


def make_env(task: str = "reach", seed: int | None = None, **kwargs) -> ToyEnv:
    """Build a 2D toy env: ``reach`` (L0), ``push`` (L1 push_t), ``grasp`` (L2)."""
    key = task.strip().lower()
    if key == "reach":
        return ToyReachEnv(seed=seed, **kwargs)
    if key == "push":
        return ToyPushEnv(seed=seed, **kwargs)
    if key == "grasp":
        return ToyGraspEnv(seed=seed, **kwargs)
    raise ValueError(f"unknown task {task!r}; choose from {TASK_NAMES}")


__all__ = ["make_env", "get_task_spec", "TaskSpec", "TASK_NAMES", "TaskName"]
