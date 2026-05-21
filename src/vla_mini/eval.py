"""Roll out the policy in the toy env and report success rate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image

from vla_mini.env.toy_reach import ToyReachEnv
from vla_mini.model.vla import MiniVLA, VLAConfig


def load_model(ckpt_path: Path, device: str) -> MiniVLA:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = VLAConfig(**payload["config"])
    model = MiniVLA(cfg).to(device)
    model.action_head.load_state_dict(payload["action_head"])
    model.eval()
    return model


def evaluate(
    ckpt_path: Path,
    episodes: int = 50,
    seed: int = 100,
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    env = ToyReachEnv()
    successes = 0
    for ep in range(episodes):
        obs, instruction = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = model.predict(Image.fromarray(obs), instruction).numpy()
            result = env.step(action)
            obs = result.observation
            done = result.done
            if result.info.get("success"):
                successes += 1
                break
    rate = successes / episodes
    print(f"success_rate={rate:.1%} ({successes}/{episodes})")
    return {"success_rate": rate, "successes": successes, "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate expert policy only (no checkpoint / VLM)",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        from vla_mini.dry_run import expert_rollout

        stats = expert_rollout(
            episodes=cfg.get("eval_episodes", 50),
            seed=cfg.get("eval_seed", 100),
        )
        print(
            f"[dry-run expert] success_rate={stats['success_rate']:.1%} "
            f"({stats['successes']}/{stats['episodes']})"
        )
        return

    ckpt = args.ckpt or Path(cfg["output_dir"]) / "action_head.pt"
    evaluate(
        ckpt,
        episodes=cfg.get("eval_episodes", 50),
        seed=cfg.get("eval_seed", 100),
        device=cfg.get("device"),
    )


if __name__ == "__main__":
    main()
