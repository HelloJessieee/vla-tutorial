"""2D reach-to-target environment — no physics engine, pure NumPy + Pillow render."""

from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw

COLOR_NAMES = ("red", "green", "blue")
COLOR_RGB = {
    "red": (220, 60, 60),
    "green": (60, 200, 90),
    "blue": (70, 120, 240),
}


@dataclass
class StepResult:
    observation: np.ndarray
    instruction: str
    action: np.ndarray
    reward: float
    done: bool
    info: dict


class ToyReachEnv:
    """Agent moves a dot toward a colored target; language names the target color."""

    def __init__(
        self,
        size: int = 128,
        max_steps: int = 40,
        action_scale: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.rng = random.Random(seed)
        self.agent = np.array([0.5, 0.5], dtype=np.float32)
        self.target_color: str = "red"
        self.targets: dict[str, np.ndarray] = {}
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, str]:
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.agent = np.array(
            [self.rng.uniform(0.15, 0.35), self.rng.uniform(0.15, 0.85)],
            dtype=np.float32,
        )
        positions = [
            np.array([self.rng.uniform(0.55, 0.9), self.rng.uniform(0.15, 0.45)], dtype=np.float32),
            np.array([self.rng.uniform(0.55, 0.9), self.rng.uniform(0.55, 0.85)], dtype=np.float32),
            np.array([self.rng.uniform(0.35, 0.75), self.rng.uniform(0.35, 0.65)], dtype=np.float32),
        ]
        self.rng.shuffle(positions)
        self.targets = {c: positions[i] for i, c in enumerate(COLOR_NAMES)}
        self.target_color = self.rng.choice(COLOR_NAMES)
        return self.render(), self.instruction

    @property
    def instruction(self) -> str:
        return f"Move the white dot to the {self.target_color} circle."

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        self.agent = np.clip(self.agent + action * self.action_scale, 0.02, 0.98)
        self.step_count += 1
        target = self.targets[self.target_color]
        dist = float(np.linalg.norm(self.agent - target))
        reward = -dist
        done = dist < 0.04 or self.step_count >= self.max_steps
        success = dist < 0.04
        obs = self.render()
        return StepResult(
            observation=obs,
            instruction=self.instruction,
            action=action,
            reward=reward,
            done=done,
            info={"distance": dist, "success": success, "target_color": self.target_color},
        )

    def expert_action(self) -> np.ndarray:
        """Normalized direction toward the active target (for synthetic demos)."""
        delta = self.targets[self.target_color] - self.agent
        norm = np.linalg.norm(delta) + 1e-6
        return np.clip(delta / norm, -1.0, 1.0).astype(np.float32)

    def render(self) -> np.ndarray:
        img = Image.new("RGB", (self.size, self.size), (24, 28, 36))
        draw = ImageDraw.Draw(img)
        for name, pos in self.targets.items():
            x, y = self._to_px(pos)
            r = 10
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLOR_RGB[name], outline=(255, 255, 255))
        ax, ay = self._to_px(self.agent)
        draw.ellipse((ax - 7, ay - 7, ax + 7, ay + 7), fill=(250, 250, 250))
        return np.array(img, dtype=np.uint8)

    def _to_px(self, pos: np.ndarray) -> tuple[int, int]:
        return int(pos[0] * self.size), int(pos[1] * self.size)
