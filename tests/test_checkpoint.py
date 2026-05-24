"""Checkpoint vs config validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vla_mini.model.checkpoint import CheckpointMismatchError, validate_checkpoint
from vla_mini.model.vla import VLAConfig


def test_validate_checkpoint_rejects_wrong_task(tmp_path: Path) -> None:
    """No network: only checks saved config metadata, not CLIP weights."""
    reach_cfg = VLAConfig(task="reach", action_dim=2, action_chunk=1)
    ckpt = tmp_path / "action_head.pt"
    torch.save(
        {"action_head": {}, "config": reach_cfg.__dict__},
        ckpt,
    )
    push_cfg = VLAConfig(task="push", action_dim=2, action_chunk=4)
    with pytest.raises(CheckpointMismatchError, match="mismatch"):
        validate_checkpoint(ckpt, push_cfg)
