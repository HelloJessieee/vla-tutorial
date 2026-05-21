"""Map flat policy vectors to env steps (supports push_t chunks)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vla_mini.env.base import ToyEnv
from vla_mini.env.tasks import TaskSpec
from vla_mini.model.vla import MiniVLA


def policy_vector_to_steps(vector: np.ndarray, spec: TaskSpec) -> list[np.ndarray]:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    expected = spec.output_dim
    if flat.size < expected:
        pad = np.zeros(expected - flat.size, dtype=np.float32)
        flat = np.concatenate([flat, pad])
    flat = flat[:expected]
    return [a for a in flat.reshape(spec.action_chunk, spec.action_dim)]


def rollout_predicted_actions(
    env: ToyEnv,
    model: MiniVLA,
    obs: np.ndarray,
    instruction: str,
    spec: TaskSpec,
) -> tuple[np.ndarray, bool, dict]:
    """Execute one model call (possibly multi-step chunk); return last obs, done, last info."""
    vector = model.predict(Image.fromarray(obs), instruction).numpy()
    info: dict = {}
    done = False
    for action in policy_vector_to_steps(vector, spec):
        result = env.step(action)
        obs = result.observation
        info = result.info
        done = result.done
        if done:
            break
    return obs, done, info
