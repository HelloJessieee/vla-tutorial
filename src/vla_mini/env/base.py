"""Shared types and protocol for 2D teaching environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StepResult:
    observation: np.ndarray
    instruction: str
    action: np.ndarray
    reward: float
    done: bool
    info: dict


class ToyEnv(Protocol):
    """Minimal interface used by collect / eval / demo / repl."""

    size: int
    max_steps: int

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, str]: ...

    def step(self, action: np.ndarray) -> StepResult: ...

    def expert_action(self) -> np.ndarray: ...

    @property
    def instruction(self) -> str: ...
