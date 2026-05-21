"""Educational π₀-style policy: LeRobot batch layout + small VLM + action expert."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
from PIL import Image

from vla_mini.policy.backends import VLMEncoder, build_encoder
from vla_mini.policy.batch import pad_vector
from vla_mini.policy.configuration import EduPI0Config
from vla_mini.policy.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class ActionExpert(nn.Module):
    """Lightweight 'action expert' (MSE head), not full Gemma flow-matching."""

    def __init__(self, config: EduPI0Config, vlm_hidden: int) -> None:
        super().__init__()
        self.config = config
        in_dim = vlm_hidden + config.max_state_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.expert_hidden),
            nn.GELU(),
            nn.Linear(config.expert_hidden, config.expert_hidden),
            nn.GELU(),
            nn.Linear(config.expert_hidden, config.chunk_size * config.max_action_dim),
        )

    def forward(self, vlm_feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([vlm_feat, state], dim=-1)
        out = self.mlp(x)
        return out.view(-1, self.config.chunk_size, self.config.max_action_dim).tanh()


class EduPI0Policy(nn.Module):
    """
    Teaching policy with the same *call pattern* as ``lerobot.policies.pi0.PI0Policy``:

    - ``forward(batch)`` → loss for training
    - ``select_action(batch)`` → one action step (uses action queue)
    - ``predict_action_chunk(batch)`` → (B, chunk_size, action_dim)
    """

    config_class = EduPI0Config
    name = "edu_pi0"

    def __init__(self, config: EduPI0Config | None = None) -> None:
        super().__init__()
        self.config = config or EduPI0Config()
        self.encoder: VLMEncoder = build_encoder(
            self.config.vlm_backbone,
            self.config.vlm_model_id,
            freeze=self.config.freeze_vlm,
        )
        self.action_expert = ActionExpert(self.config, self.encoder.hidden_size)
        self._action_queue: deque[torch.Tensor] = deque()
        self._tokenizer = getattr(self.encoder, "tokenizer", None) or getattr(
            self.encoder, "processor", None
        )
        if hasattr(self._tokenizer, "tokenizer"):
            self._tokenizer = self._tokenizer.tokenizer

    @property
    def image_features(self) -> list[str]:
        return [self.config.image_feature]

    def _encode_vlm(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        texts = batch.get("_instruction_text")
        if texts is None:
            raise KeyError("batch must include _instruction_text for VLM encoding")
        key = self.config.image_feature
        imgs_chw = batch[key]
        pil_list = []
        for i in range(imgs_chw.shape[0]):
            arr = (imgs_chw[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil_list.append(Image.fromarray(arr))
        return self.encoder.encode(pil_list, list(texts))

    def prepare_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = batch[ACTION]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
        return pad_vector(actions, self.config.max_action_dim)

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        vlm_feat = self._encode_vlm(batch)
        state = self.prepare_state(batch)
        chunk = self.action_expert(vlm_feat, state)
        return chunk[:, :, : self.config.action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)
            steps = chunk[:, : self.config.n_action_steps]
            for b in range(steps.shape[0]):
                for t in range(steps.shape[1]):
                    self._action_queue.append(steps[b, t])
        return self._action_queue.popleft()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict]:
        vlm_feat = self._encode_vlm(batch)
        state = self.prepare_state(batch)
        pred = self.action_expert(vlm_feat, state)
        target = self.prepare_action(batch)
        pred = pred[:, :, : self.config.action_dim]
        target = target[:, :, : self.config.action_dim]
        losses = (pred - target).pow(2).mean(dim=-1)  # (B, chunk)
        loss_dict = {"loss": losses.mean().item()}
        if reduction == "none":
            return losses.mean(dim=1), loss_dict
        return losses.mean(), loss_dict

    def reset_action_queue(self) -> None:
        self._action_queue.clear()
