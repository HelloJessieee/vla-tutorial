"""Synthetic collect produces correct action vector sizes."""

from __future__ import annotations

from pathlib import Path

import pytest

from vla_mini.data.synthetic import collect_episodes, load_manifest
from vla_mini.env.tasks import get_task_spec
from vla_mini.model.checkpoint import validate_manifest_dim
from vla_mini.model.vla import VLAConfig


@pytest.mark.parametrize(
    ("task", "output_dim"),
    [("reach", 2), ("push", 8), ("grasp", 3)],
)
def test_collect_manifest_action_dim(tmp_path: Path, task: str, output_dim: int) -> None:
    out = tmp_path / task
    manifest = collect_episodes(num_episodes=2, seed=0, out_dir=out, task=task)
    rows = load_manifest(manifest)
    assert len(rows) > 0
    for row in rows[:10]:
        assert len(row["action"]) == output_dim
        assert row.get("task") == task
    spec = get_task_spec(task)
    cfg = VLAConfig(
        task=task,
        action_dim=spec.action_dim,
        action_chunk=spec.action_chunk,
    )
    validate_manifest_dim(manifest, cfg)
