"""Roll out the policy in the toy env and report success rate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image

from vla_mini.env import get_task_spec, make_env
from vla_mini.model.action_exec import policy_vector_to_steps, rollout_predicted_actions
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
    task: str = "reach",
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    spec = get_task_spec(task)
    successes = 0
    for ep in range(episodes):
        env = make_env(task, seed=seed + ep)
        obs, instruction = env.reset()
        done = False
        while not done:
            obs, done, info = rollout_predicted_actions(
                env, model, obs, instruction, spec
            )
            if info.get("success"):
                successes += 1
                break
    rate = successes / episodes
    print(f"success_rate={rate:.1%} ({successes}/{episodes})  task={task}")
    return {"success_rate": rate, "successes": successes, "episodes": episodes}


def expert_step_once(env, spec) -> bool:
    from vla_mini.env.action_utils import expert_action_chunk

    if spec.action_chunk > 1:
        flat = expert_action_chunk(env, spec.action_chunk)
        steps = policy_vector_to_steps(flat, spec)
    else:
        steps = [env.expert_action()]
    done = False
    success = False
    for action in steps:
        result = env.step(action)
        done = result.done
        if result.info.get("success"):
            success = True
            break
        if done:
            break
    return success


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

    task = cfg.get("task", "reach")

    if args.dry_run:
        from vla_mini.dry_run import expert_rollout

        stats = expert_rollout(
            episodes=cfg.get("eval_episodes", 50),
            seed=cfg.get("eval_seed", 100),
            task=task,
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
        task=task,
    )


if __name__ == "__main__":
    main()
