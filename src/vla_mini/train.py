"""Train only the action head on synthetic demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vla_mini.data.synthetic import collect_episodes, load_manifest
from vla_mini.model.vla import MiniVLA, VLAConfig


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path) -> None:
        self.rows = load_manifest(manifest_path)
        self.root = data_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = Image.open(self.root / row["image"]).convert("RGB")
        action = torch.tensor(row["action"], dtype=torch.float32)
        return {"image": img, "instruction": row["instruction"], "action": action}


def collate(batch: list[dict]) -> dict:
    return {
        "images": [b["image"] for b in batch],
        "instructions": [b["instruction"] for b in batch],
        "actions": torch.stack([b["action"] for b in batch]),
    }


def train_loop(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str | None = None,
    vlm_name: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = VLAConfig(vlm_name=vlm_name) if vlm_name else VLAConfig()
    model = MiniVLA(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    loss_fn = torch.nn.MSELoss()

    dataset = ManifestDataset(manifest, data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    history: list[dict] = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            actions = batch["actions"].to(device)
            pred = model(batch["images"], batch["instructions"])
            loss = loss_fn(pred, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            history.append({"step": global_step, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / max(len(loader), 1)
        print(f"epoch {epoch + 1} avg_loss={avg:.4f}")

    ckpt = output_dir / "action_head.pt"
    torch.save(
        {
            "action_head": model.action_head.state_dict(),
            "config": cfg.__dict__,
        },
        ckpt,
    )
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f)
    print(f"saved checkpoint -> {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vla-mini action head")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--collect", action="store_true", help="regenerate synthetic data")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only test env + expert (+ optional --collect); no VLM download",
    )
    args = parser.parse_args()

    if args.dry_run:
        from vla_mini.dry_run import run

        with args.config.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        import sys

        sys.exit(
            run(
                episodes=cfg.get("dry_run_episodes", 30),
                seed=cfg.get("seed", 0),
                collect=args.collect,
                data_dir=Path(cfg["data_dir"]),
                collect_episodes_count=min(cfg.get("num_episodes", 10), 10),
                task=cfg.get("task", "reach"),
            )
        )

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    if args.collect or not (data_dir / "manifest.jsonl").exists():
        print("collecting synthetic expert demos...")
        collect_episodes(
            num_episodes=cfg.get("num_episodes", 200),
            seed=cfg.get("seed", 0),
            out_dir=data_dir,
            task=cfg.get("task", "reach"),
        )

    train_loop(
        manifest=data_dir / "manifest.jsonl",
        data_root=data_dir,
        output_dir=Path(cfg["output_dir"]),
        epochs=cfg.get("epochs", 3),
        batch_size=cfg.get("batch_size", 8),
        lr=cfg.get("lr", 1e-3),
        device=cfg.get("device"),
        vlm_name=cfg.get("vlm_name"),
    )


if __name__ == "__main__":
    main()
