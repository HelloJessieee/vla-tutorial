"""Educational PI0-style config (inspired by lerobot PI0Config, simplified)."""

from __future__ import annotations

from dataclasses import dataclass, field

from vla_mini.policy.constants import ACTION, CAMERA_MAIN, OBS_STATE


@dataclass
class EduPI0Config:
    """Teaching policy config — same *batch layout* as LeRobot π₀, lighter training."""

    # VLM backbone: clip (default, stable) | smolvlm | minimind2-small-v
    vlm_backbone: str = "clip"
    vlm_pretrained: str | None = None  # override HF id

    dtype: str = "float32"
    image_resolution: tuple[int, int] = (128, 128)
    image_feature: str = CAMERA_MAIN

    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    max_state_dim: int = 8
    max_action_dim: int = 8
    action_dim: int = 2  # unpadded dim for toy env

    freeze_vlm: bool = True
    expert_hidden: int = 256

    # Educational: MSE on action chunk (not flow-matching like full π₀)
    use_flow_matching: bool = False

    device: str | None = None

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.action_dim > self.max_action_dim:
            raise ValueError("action_dim cannot exceed max_action_dim")

    @property
    def vlm_model_id(self) -> str:
        if self.vlm_pretrained:
            return self.vlm_pretrained
        if self.vlm_backbone == "minimind2-small-v":
            return "jingyaogong/MiniMind2-Small-V"
        if self.vlm_backbone == "clip":
            return "openai/clip-vit-base-patch32"
        return "HuggingFaceTB/SmolVLM-256M-Instruct"
