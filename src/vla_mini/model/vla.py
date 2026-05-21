"""Frozen small VLM + trainable action head (2D continuous actions)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image

from vla_mini.policy.backends import CLIPEncoder, build_encoder

DEFAULT_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


@dataclass
class VLAConfig:
    vlm_backbone: str = "clip"  # clip | smolvlm
    vlm_name: str | None = None
    action_dim: int = 2
    hidden_size: int = 256
    freeze_vlm: bool = True


class MiniVLA(nn.Module):
    """Maps (RGB image, text instruction) -> continuous action in [-1, 1]."""

    def __init__(self, config: VLAConfig | None = None) -> None:
        super().__init__()
        self.config = config or VLAConfig()
        model_id = self.config.vlm_name
        if model_id is None:
            model_id = None if self.config.vlm_backbone == "clip" else DEFAULT_VLM
        self.encoder = build_encoder(
            self.config.vlm_backbone,
            model_id or "",
            freeze=self.config.freeze_vlm,
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        images: list[Image.Image],
        instructions: list[str],
    ) -> torch.Tensor:
        pooled = self.encoder.encode(images, instructions)
        return self.action_head(pooled)

    @torch.inference_mode()
    def predict(self, image: Image.Image, instruction: str) -> torch.Tensor:
        self.eval()
        return self.forward([image], [instruction])[0].cpu()
