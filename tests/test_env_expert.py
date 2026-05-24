"""Expert rollouts for L0/L1/L2 — no torch, no network."""

from __future__ import annotations

import pytest

from vla_mini.dry_run import expert_rollout
from vla_mini.env.tasks import get_task_spec


@pytest.mark.parametrize("task", ["reach", "push", "grasp"])
def test_expert_rollout_success_rate(task: str) -> None:
    stats = expert_rollout(episodes=25, seed=0, task=task)
    assert stats["task"] == task
    assert stats["success_rate"] >= 0.8, (
        f"{task} expert success {stats['success_rate']:.1%} below 80%"
    )


def test_task_specs_output_dim() -> None:
    assert get_task_spec("reach").output_dim == 2
    assert get_task_spec("push").output_dim == 8
    assert get_task_spec("grasp").output_dim == 3
