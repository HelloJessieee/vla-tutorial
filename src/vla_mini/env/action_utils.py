"""Roll out short expert horizons for push_t without mutating the live env state."""

from __future__ import annotations

import numpy as np

from vla_mini.env.base import ToyEnv


def expert_action_chunk(env: ToyEnv, chunk_size: int) -> np.ndarray:
    """Return concatenated [a0, a1, ...] each of shape (env.action_dim,)."""
    if chunk_size <= 1:
        return env.expert_action().astype(np.float32)

    if not hasattr(env, "snapshot_state") or not hasattr(env, "restore_state"):
        base = env.expert_action().astype(np.float32)
        return np.tile(base, chunk_size)

    snap = env.snapshot_state()
    parts: list[np.ndarray] = []
    try:
        for _ in range(chunk_size):
            a = env.expert_action().astype(np.float32)
            parts.append(a)
            env.apply_action(a)
    finally:
        env.restore_state(snap)
    return np.concatenate(parts).astype(np.float32)
