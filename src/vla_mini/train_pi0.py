"""Train EduPI0Policy (LeRobot-style batch) on synthetic toy data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vla_mini.data.synthetic import collect_episodes, load_manifest
from vla_mini.policy.batch import obs_to_batch
from vla_mini.policy.configuration import EduPI0Config
from vla_mini.policy.modeling_edu_pi0 import EduPI0Policy


class Pi0ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path, config: EduPI0Config) -> None:
        self.rows = load_manifest(manifest_path)
        self.root = data_root
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = Image.open(self.root / row["image"]).convert("RGB")
        obs = __import__("numpy").array(img)
        action = row["action"]
        state = [0.0, 0.0, float(action[0]), float(action[1])]
        return {
            "obs": obs,
            "instruction": row["instruction"],
            "state": state,
            "action": action,
        }


def collate_pi0(batch: list[dict], config: EduPI0Config, device: str) -> dict[str, torch.Tensor]:
    return obs_to_batch(
        images=[b["obs"] for b in batch],
        instructions=[b["instruction"] for b in batch],
        states=[b["state"] for b in batch],
        actions=[b["action"] for b in batch],
        config=config,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train educational π₀-style policy")
    parser.add_argument("--config", type=Path, default=Path("configs/edu_pi0.yaml"))
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pi0_cfg = EduPI0Config(
        vlm_backbone=cfg.get("vlm_backbone", "smolvlm"),
        vlm_pretrained=cfg.get("vlm_pretrained"),
        chunk_size=cfg.get("chunk_size", 1),
        action_dim=cfg.get("action_dim", 2),
        max_action_dim=cfg.get("max_action_dim", 8),
        max_state_dim=cfg.get("max_state_dim", 8),
        freeze_vlm=cfg.get("freeze_vlm", True),
        image_resolution=tuple(cfg.get("image_resolution", [128, 128])),
    )

    data_dir = Path(cfg["data_dir"])
    if args.collect or not (data_dir / "manifest.jsonl").exists():
        collect_episodes(num_episodes=cfg.get("num_episodes", 120), out_dir=data_dir)

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = EduPI0Policy(pi0_cfg).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.get("lr", 1e-3))

    ds = Pi0ManifestDataset(data_dir / "manifest.jsonl", data_dir, pi0_cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        collate_fn=lambda b: collate_pi0(b, pi0_cfg, device),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.get("epochs", 3)):
        policy.train()
        total = 0.0
        for batch in tqdm(loader, desc=f"edu_pi0 epoch {epoch + 1}"):
            loss, _ = policy(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1} loss={total / max(len(loader), 1):.4f}")

    ckpt = out_dir / "edu_pi0.pt"
    torch.save(
        {
            "action_expert": policy.action_expert.state_dict(),
            "config": pi0_cfg.__dict__,
        },
        ckpt,
    )
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
