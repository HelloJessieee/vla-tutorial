"""Build LeRobot-style batch dicts from teaching env observations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vla_mini.policy.configuration import EduPI0Config
from vla_mini.policy.constants import (
    ACTION,
    CAMERA_MAIN,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def pad_vector(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = dim - x.shape[-1]
    return F.pad(x, (0, pad))


def numpy_rgb_to_chw_float(img: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    pil = Image.fromarray(img.astype(np.uint8)).resize(size)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tokenize_instruction(
    texts: list[str],
    tokenizer,
    max_length: int = 48,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def obs_to_batch(
    images: list[np.ndarray],
    instructions: list[str],
    states: list[np.ndarray] | None,
    actions: list[np.ndarray] | None,
    config: EduPI0Config,
    tokenizer=None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a π₀-compatible batch for EduPI0Policy."""
    b = len(images)
    dev = torch.device(device)
    key = config.image_feature or CAMERA_MAIN

    imgs = torch.stack(
        [numpy_rgb_to_chw_float(im, config.image_resolution) for im in images],
        dim=0,
    ).to(dev)

    batch: dict[str, torch.Tensor] = {key: imgs}

    if states is not None:
        st = torch.tensor(np.stack(states), dtype=torch.float32, device=dev)
        batch[OBS_STATE] = pad_vector(st, config.max_state_dim)
    else:
        batch[OBS_STATE] = torch.zeros(b, config.max_state_dim, device=dev)

    if actions is not None:
        act = torch.tensor(np.stack(actions), dtype=torch.float32, device=dev)
        batch[ACTION] = pad_vector(act, config.max_action_dim)
    else:
        batch[ACTION] = torch.zeros(b, config.max_action_dim, device=dev)

    if tokenizer is not None:
        tokens, mask = tokenize_instruction(instructions, tokenizer, device=dev)
        batch[OBS_LANGUAGE_TOKENS] = tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = mask

    batch["_instruction_text"] = instructions  # extension for smolvlm path
    return batch
