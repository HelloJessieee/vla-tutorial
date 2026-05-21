"""Per-task action layout for L0/L1/L2 teaching stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TaskSpec:
    """VLA output layout: flat vector length = action_dim * action_chunk."""

    name: str
    level: str
    action_dim: int
    action_chunk: int
    description: str

    @property
    def output_dim(self) -> int:
        return self.action_dim * self.action_chunk


TASKS: dict[str, TaskSpec] = {
    "reach": TaskSpec(
        name="reach",
        level="L0",
        action_dim=2,
        action_chunk=1,
        description="image + language → single approach step (dx, dy)",
    ),
    "push": TaskSpec(
        name="push",
        level="L1",
        action_dim=2,
        action_chunk=4,
        description="image + language → push_t chunk (K× dx,dy) while contacting block",
    ),
    "grasp": TaskSpec(
        name="grasp",
        level="L2",
        action_dim=3,
        action_chunk=1,
        description="image + language → (dx, dy, gripper) with gripper ∈ [-1, 1]",
    ),
}

TASK_NAMES: tuple[str, ...] = tuple(TASKS.keys())
TaskName = Literal["reach", "push", "grasp"]


def get_task_spec(task: str) -> TaskSpec:
    key = task.strip().lower()
    if key not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {TASK_NAMES}")
    return TASKS[key]
