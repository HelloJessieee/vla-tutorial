"""L0 smoke test: env + expert only — no torch, no VLM download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vla_mini.data.synthetic import collect_episodes, load_manifest
from vla_mini.env.toy_reach import ToyReachEnv


def expert_rollout(episodes: int = 30, seed: int = 0, max_steps: int | None = None) -> dict:
    env = ToyReachEnv()
    if max_steps is not None:
        env.max_steps = max_steps
    successes = 0
    total_steps = 0
    for ep in range(episodes):
        obs, instruction = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done:
            action = env.expert_action()
            result = env.step(action)
            obs = result.observation
            done = result.done
            steps += 1
            if result.info.get("success"):
                successes += 1
                break
        total_steps += steps
    rate = successes / max(episodes, 1)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": rate,
        "avg_steps": total_steps / max(episodes, 1),
        "sample_instruction": instruction,
        "observation_shape": tuple(obs.shape),
    }


def verify_manifest(manifest: Path, data_root: Path, sample_rows: int = 5) -> dict:
    rows = load_manifest(manifest)
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")
    checked = 0
    for row in rows[:sample_rows]:
        img_path = data_root / row["image"]
        if not img_path.is_file():
            raise FileNotFoundError(f"missing image: {img_path}")
        if "action" not in row or len(row["action"]) != 2:
            raise ValueError(f"bad action in row: {row}")
        if not row.get("instruction"):
            raise ValueError(f"missing instruction in row: {row}")
        checked += 1
    return {"manifest_rows": len(rows), "samples_verified": checked}


def run(
    episodes: int = 30,
    seed: int = 0,
    collect: bool = False,
    data_dir: Path | str = "data/synthetic",
    collect_episodes_count: int = 10,
) -> int:
    print("== vla-mini dry-run (no VLM, no GPU) ==\n")

    print("[1/3] ToyReachEnv reset + render ...")
    env = ToyReachEnv(seed=seed)
    obs, instruction = env.reset()
    assert obs.shape == (env.size, env.size, 3), f"unexpected shape: {obs.shape}"
    print(f"  OK  observation {obs.shape}, dtype={obs.dtype}")
    print(f"  OK  instruction: {instruction!r}")

    print(f"\n[2/3] Expert policy rollout ({episodes} episodes) ...")
    stats = expert_rollout(episodes=episodes, seed=seed)
    print(
        f"  OK  success_rate={stats['success_rate']:.1%} "
        f"({stats['successes']}/{stats['episodes']}), "
        f"avg_steps={stats['avg_steps']:.1f}"
    )
    if stats["success_rate"] < 0.5:
        print("  WARN  expert success < 50% — check env logic", file=sys.stderr)

    if collect:
        print(f"\n[3/3] Collect {collect_episodes_count} synthetic episodes ...")
        out = Path(data_dir)
        manifest = collect_episodes(
            num_episodes=collect_episodes_count,
            seed=seed,
            out_dir=out,
        )
        meta = verify_manifest(manifest, out)
        print(f"  OK  {meta['manifest_rows']} rows, verified {meta['samples_verified']} samples")
        print(f"  OK  manifest -> {manifest}")
    else:
        print("\n[3/3] Data collect skipped (pass --collect to test manifest)")

    print("\n== dry-run passed ==\n")
    print("Next: pip install -e .  &&  python -m vla_mini.train --collect")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test env + expert without downloading any VLM weights",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="also write a small synthetic dataset and validate manifest",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--collect-episodes", type=int, default=10)
    args = parser.parse_args()
    try:
        code = run(
            episodes=args.episodes,
            seed=args.seed,
            collect=args.collect,
            data_dir=args.data_dir,
            collect_episodes_count=args.collect_episodes,
        )
    except Exception as exc:
        print(f"\n== dry-run FAILED ==\n{exc}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
