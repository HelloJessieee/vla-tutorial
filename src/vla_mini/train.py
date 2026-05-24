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
from vla_mini.env.tasks import get_task_spec
from vla_mini.model.checkpoint import print_task_banner, validate_manifest_dim
from vla_mini.model.vla import MiniVLA, VLAConfig


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path, output_dim: int) -> None:
        self.rows = load_manifest(manifest_path)
        self.root = data_root
        self.output_dim = output_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = Image.open(self.root / row["image"]).convert("RGB")
        action = torch.tensor(row["action"], dtype=torch.float32)
        if action.numel() != self.output_dim:
            raise ValueError(
                f"row {idx} action len {action.numel()} != expected {self.output_dim}",
            )
        return {"image": img, "instruction": row["instruction"], "action": action}


def collate(batch: list[dict]) -> dict:
    return {
        "images": [b["image"] for b in batch],
        "instructions": [b["instruction"] for b in batch],
        "actions": torch.stack([b["action"] for b in batch]),
    }


def vla_config_from_yaml(cfg: dict) -> VLAConfig:
    task = cfg.get("task", "reach")
    spec = get_task_spec(task)
    return VLAConfig(
        vlm_backbone=cfg.get("vlm_backbone", "clip"),
        vlm_name=cfg.get("vlm_name"),
        action_dim=cfg.get("action_dim", spec.action_dim),
        action_chunk=cfg.get("action_chunk", spec.action_chunk),
        hidden_size=cfg.get("hidden_size", 256),
        freeze_vlm=cfg.get("freeze_vlm", True),
        task=task,
    )


def train_loop(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    model_cfg: VLAConfig,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MiniVLA(model_cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    loss_fn = torch.nn.MSELoss()

    dataset = ManifestDataset(manifest, data_root, model_cfg.output_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    history: list[dict] = []
    global_step = 0
    print(
        f"task={model_cfg.task}  output_dim={model_cfg.output_dim} "
        f"(action_dim={model_cfg.action_dim} x chunk={model_cfg.action_chunk})",
    )
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
            "config": model_cfg.__dict__,
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

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task = cfg.get("task", "reach")

    if args.dry_run:
        from vla_mini.dry_run import run

        import sys

        sys.exit(
            run(
                episodes=cfg.get("dry_run_episodes", 30),
                seed=cfg.get("seed", 0),
                collect=args.collect,
                data_dir=Path(cfg["data_dir"]),
                collect_episodes_count=min(cfg.get("num_episodes", 10), 10),
                task=task,
            )
        )

    model_cfg = vla_config_from_yaml(cfg)
    data_dir = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    print_task_banner(
        model_cfg,
        data_dir=data_dir,
        output_dir=output_dir,
        config_path=args.config,
        ckpt_path=output_dir / "action_head.pt",
    )

    manifest = data_dir / "manifest.jsonl"
    if args.collect or not manifest.exists():
        print(f"collecting synthetic expert demos (task={task})...")
        collect_episodes(
            num_episodes=cfg.get("num_episodes", 200),
            seed=cfg.get("seed", 0),
            out_dir=data_dir,
            task=task,
            action_chunk=cfg.get("action_chunk"),
        )

    validate_manifest_dim(manifest, model_cfg)

    train_loop(
        manifest=manifest,
        data_root=data_dir,
        output_dir=output_dir,
        model_cfg=model_cfg,
        epochs=cfg.get("epochs", 3),
        batch_size=cfg.get("batch_size", 8),
        lr=cfg.get("lr", 1e-3),
        device=cfg.get("device"),
    )


if __name__ == "__main__":
    main()
