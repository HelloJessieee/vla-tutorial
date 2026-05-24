"""Checkpoint ↔ config validation and startup banners."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from vla_mini.data.synthetic import load_manifest
from vla_mini.env.tasks import TaskSpec, get_task_spec
from vla_mini.model.vla import VLAConfig


class CheckpointMismatchError(ValueError):
    """Raised when action_head.pt does not match the active task config."""


def print_task_banner(
    model_cfg: VLAConfig,
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    config_path: Path | str | None = None,
    ckpt_path: Path | str | None = None,
) -> None:
    spec = get_task_spec(model_cfg.task)
    lines = [
        "",
        "[vla-mini] ── task ─────────────────────────────",
        f"  config file : {config_path or '(inline)'}",
        f"  task        : {model_cfg.task} ({spec.level})",
        f"  output_dim  : {model_cfg.output_dim}  "
        f"(action_dim={model_cfg.action_dim} × chunk={model_cfg.action_chunk})",
        f"  data_dir    : {data_dir}",
        f"  output_dir  : {output_dir}",
    ]
    if ckpt_path is not None:
        lines.append(f"  checkpoint  : {ckpt_path}")
    lines.append("────────────────────────────────────────────")
    print("\n".join(lines), file=sys.stderr)


def load_checkpoint_config(ckpt_path: Path) -> VLAConfig:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = payload.get("config")
    if raw is None:
        raise CheckpointMismatchError(f"checkpoint missing 'config': {ckpt_path}")
    return VLAConfig(**raw)


def validate_checkpoint(ckpt_path: Path, expected: VLAConfig) -> VLAConfig:
    """Ensure checkpoint matches expected task layout; return saved config."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}\n"
            f"  Train first: python -m vla_mini.train --config ... --collect",
        )
    saved = load_checkpoint_config(ckpt_path)
    if saved.output_dim != expected.output_dim or saved.task != expected.task:
        raise CheckpointMismatchError(
            f"checkpoint task/output_dim mismatch:\n"
            f"  checkpoint: task={saved.task!r}, output_dim={saved.output_dim} "
            f"(action_dim={saved.action_dim}, chunk={saved.action_chunk})\n"
            f"  config:     task={expected.task!r}, output_dim={expected.output_dim} "
            f"(action_dim={expected.action_dim}, chunk={expected.action_chunk})\n"
            f"  Fix: use the matching configs/*.yaml and runs/ folder, or re-train:\n"
            f"    python -m vla_mini.train --config configs/{expected.task}.yaml --collect",
        )
    return saved


def validate_manifest_dim(manifest: Path, expected: VLAConfig | TaskSpec) -> None:
    """Fail fast if collected data does not match model output size."""
    spec = expected if isinstance(expected, TaskSpec) else get_task_spec(expected.task)
    rows = load_manifest(manifest)
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")
    bad: list[str] = []
    for i, row in enumerate(rows[:20]):
        n = len(row.get("action", []))
        if n != spec.output_dim:
            bad.append(f"  row {i}: action len {n}")
    if bad:
        raise ValueError(
            f"manifest action size does not match task={spec.name!r} "
            f"(expected {spec.output_dim}):\n" + "\n".join(bad[:5])
            + ("\n  ..." if len(bad) > 5 else "")
            + f"\n  Re-collect: python -m vla_mini.train --config configs/{spec.name}.yaml --collect",
        )
