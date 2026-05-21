"""L1 push-block: move a pusher dot to contact a colored block and push it into a green zone."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from vla_mini.env.base import StepResult
from vla_mini.env.toy_reach import COLOR_NAMES, COLOR_RGB

CONTACT_DIST = 0.10
ZONE_HALF_W = 0.14
ZONE_HALF_H = 0.12
BLOCK_HALF = 0.045
PUSHER_OFFSET = 0.11


class ToyPushEnv:
    """Pusher (white dot) couples to a colored block; goal is block center inside green zone."""

    task_name = "push"

    def __init__(
        self,
        size: int = 128,
        max_steps: int = 70,
        action_scale: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.rng = random.Random(seed)
        self.pusher = np.array([0.2, 0.5], dtype=np.float32)
        self.block_color: str = "red"
        self.block = np.array([0.45, 0.5], dtype=np.float32)
        self.zone_center = np.array([0.78, 0.5], dtype=np.float32)
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, str]:
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.block_color = self.rng.choice(COLOR_NAMES)
        self.pusher = np.array(
            [self.rng.uniform(0.08, 0.22), self.rng.uniform(0.2, 0.8)],
            dtype=np.float32,
        )
        self.block = np.array(
            [self.rng.uniform(0.32, 0.52), self.rng.uniform(0.22, 0.78)],
            dtype=np.float32,
        )
        self.zone_center = np.array(
            [self.rng.uniform(0.68, 0.88), self.rng.uniform(0.25, 0.75)],
            dtype=np.float32,
        )
        return self.render(), self.instruction

    @property
    def instruction(self) -> str:
        return f"Push the {self.block_color} block into the green zone."

    def _block_in_zone(self) -> bool:
        dx = abs(float(self.block[0] - self.zone_center[0]))
        dy = abs(float(self.block[1] - self.zone_center[1]))
        return dx < ZONE_HALF_W and dy < ZONE_HALF_H

    def _distance_to_zone(self) -> float:
        delta = self.zone_center - self.block
        return float(np.linalg.norm(delta))

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        delta = action * self.action_scale
        self.pusher = np.clip(self.pusher + delta, 0.02, 0.98)
        if float(np.linalg.norm(self.pusher - self.block)) < CONTACT_DIST:
            self.block = np.clip(self.block + delta * 1.05, 0.05, 0.95)
        self.step_count += 1
        dist = self._distance_to_zone()
        success = self._block_in_zone()
        done = success or self.step_count >= self.max_steps
        return StepResult(
            observation=self.render(),
            instruction=self.instruction,
            action=action,
            reward=-dist,
            done=done,
            info={
                "distance": dist,
                "success": success,
                "block_color": self.block_color,
                "task": self.task_name,
            },
        )

    def expert_action(self) -> np.ndarray:
        """Approach block from behind (w.r.t. zone), then push block into zone."""
        to_zone = self.zone_center - self.block
        zone_norm = float(np.linalg.norm(to_zone)) + 1e-6
        zone_unit = to_zone / zone_norm
        behind = self.block - zone_unit * PUSHER_OFFSET
        to_behind = behind - self.pusher
        dist_behind = float(np.linalg.norm(to_behind))
        dist_pb = float(np.linalg.norm(self.pusher - self.block))

        if dist_pb > CONTACT_DIST and dist_behind > 0.04:
            direction = to_behind / (dist_behind + 1e-6)
        elif dist_pb > CONTACT_DIST:
            direction = (self.block - self.pusher) / (dist_pb + 1e-6)
        else:
            direction = zone_unit
        return np.clip(direction, -1.0, 1.0).astype(np.float32)

    def render(self) -> np.ndarray:
        img = Image.new("RGB", (self.size, self.size), (24, 28, 36))
        draw = ImageDraw.Draw(img)
        zx, zy = self._to_px(self.zone_center)
        zw, zh = int(ZONE_HALF_W * self.size), int(ZONE_HALF_H * self.size)
        draw.rectangle(
            (zx - zw, zy - zh, zx + zw, zy + zh),
            outline=(60, 200, 90),
            fill=(40, 90, 55),
        )
        draw.text((zx - zw + 4, zy - zh + 2), "zone", fill=(180, 255, 200))
        bx, by = self._to_px(self.block)
        bw, bh = int(BLOCK_HALF * self.size), int(BLOCK_HALF * self.size * 0.75)
        draw.rectangle(
            (bx - bw, by - bh, bx + bw, by + bh),
            fill=COLOR_RGB[self.block_color],
            outline=(255, 255, 255),
        )
        px, py = self._to_px(self.pusher)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(250, 250, 250))
        return np.array(img, dtype=np.uint8)

    def _to_px(self, pos: np.ndarray) -> tuple[int, int]:
        return int(pos[0] * self.size), int(pos[1] * self.size)
