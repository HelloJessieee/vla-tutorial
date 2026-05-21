"""Generate (image, language, action) tuples with a scripted expert — no external datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from vla_mini.env.base import ToyEnv
from vla_mini.env.factory import make_env


def collect_episodes(
    num_episodes: int = 200,
    seed: int = 0,
    out_dir: Path | str = "data/synthetic",
    task: str = "reach",
    env: ToyEnv | None = None,
    env_factory: Callable[[int], ToyEnv] | None = None,
) -> Path:
    """Roll out expert demos. Pass ``env`` for a single instance, or ``task`` / ``env_factory`` per episode."""
    out = Path(out_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def _new_env(ep: int) -> ToyEnv:
        if env_factory is not None:
            return env_factory(seed + ep)
        if env is not None:
            return env
        return make_env(task, seed=seed + ep)

    for ep in tqdm(range(num_episodes), desc=f"collect-{task}"):
        episode_env = _new_env(ep)
        obs, instruction = episode_env.reset()
        step_idx = 0
        while True:
            action = episode_env.expert_action()
            img_path = images_dir / f"ep{ep:04d}_s{step_idx:03d}.png"
            Image.fromarray(obs).save(img_path)
            records.append(
                {
                    "image": str(img_path.relative_to(out)).replace("\\", "/"),
                    "instruction": instruction,
                    "action": action.tolist(),
                    "episode": ep,
                    "step": step_idx,
                    "task": getattr(episode_env, "task_name", task),
                }
            )
            result = episode_env.step(action)
            obs = result.observation
            step_idx += 1
            if result.done:
                break

    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def load_manifest(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
