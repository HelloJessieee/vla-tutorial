"""L2 grasp: (dx, dy, gripper) — approach, close, transport, release."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from vla_mini.env.base import StepResult
from vla_mini.env.toy_reach import COLOR_NAMES, COLOR_RGB

GRASP_DIST = 0.09
ZONE_HALF_W = 0.14
ZONE_HALF_H = 0.12
OBJECT_HALF = 0.04
GRIP_CLOSE = 0.4
GRIP_OPEN = -0.4


class ToyGraspEnv:
    """Gripper dot picks a colored object and releases it in the green zone."""

    task_name = "grasp"
    action_dim = 3

    def __init__(
        self,
        size: int = 128,
        max_steps: int = 90,
        action_scale: float = 0.085,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.rng = random.Random(seed)
        self.gripper = np.array([0.12, 0.5], dtype=np.float32)
        self.object_color: str = "red"
        self.object_pos = np.array([0.48, 0.4], dtype=np.float32)
        self.zone_center = np.array([0.78, 0.55], dtype=np.float32)
        self.holding = False
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, str]:
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.holding = False
        self.object_color = self.rng.choice(COLOR_NAMES)
        self.gripper = np.array(
            [self.rng.uniform(0.06, 0.18), self.rng.uniform(0.25, 0.75)],
            dtype=np.float32,
        )
        self.object_pos = np.array(
            [self.rng.uniform(0.32, 0.52), self.rng.uniform(0.22, 0.78)],
            dtype=np.float32,
        )
        self.zone_center = np.array(
            [self.rng.uniform(0.7, 0.88), self.rng.uniform(0.28, 0.72)],
            dtype=np.float32,
        )
        return self.render(), self.instruction

    @property
    def instruction(self) -> str:
        return f"Grasp the {self.object_color} object and place it in the green zone."

    def _in_zone(self, pos: np.ndarray) -> bool:
        dx = abs(float(pos[0] - self.zone_center[0]))
        dy = abs(float(pos[1] - self.zone_center[1]))
        return dx < ZONE_HALF_W and dy < ZONE_HALF_H

    def snapshot_state(self) -> tuple:
        return (
            self.gripper.copy(),
            self.object_pos.copy(),
            self.zone_center.copy(),
            self.object_color,
            self.holding,
            self.step_count,
        )

    def restore_state(self, snap: tuple) -> None:
        (
            self.gripper,
            self.object_pos,
            self.zone_center,
            self.object_color,
            self.holding,
            self.step_count,
        ) = snap

    def apply_action(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:3], -1.0, 1.0)
        delta = action[:2] * self.action_scale
        grip = float(action[2])
        self.gripper = np.clip(self.gripper + delta, 0.02, 0.98)

        dist_go = float(np.linalg.norm(self.gripper - self.object_pos))
        if not self.holding and dist_go < GRASP_DIST and grip > 0.0:
            self.holding = True

        if self.holding:
            self.object_pos = self.gripper.copy()
            if grip < GRIP_OPEN and self._in_zone(self.object_pos):
                self.holding = False

        self.step_count += 1

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:3], -1.0, 1.0)
        self.apply_action(action)
        dist = float(np.linalg.norm(self.zone_center - self.object_pos))
        success = (not self.holding) and self._in_zone(self.object_pos)
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
                "holding": self.holding,
                "object_color": self.object_color,
                "task": self.task_name,
            },
        )

    def expert_action(self) -> np.ndarray:
        if not self.holding:
            to_obj = self.object_pos - self.gripper
            dist = float(np.linalg.norm(to_obj)) + 1e-6
            move = to_obj / dist
            grip = GRIP_CLOSE if dist < GRASP_DIST * 1.8 else GRIP_OPEN
            return np.clip(np.array([move[0], move[1], grip], dtype=np.float32), -1.0, 1.0)

        to_zone = self.zone_center - self.gripper
        dist_z = float(np.linalg.norm(to_zone)) + 1e-6
        move = to_zone / dist_z
        if self._in_zone(self.gripper):
            return np.array([0.0, 0.0, GRIP_OPEN], dtype=np.float32)
        return np.array([move[0], move[1], GRIP_CLOSE], dtype=np.float32)

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
        ox, oy = self._to_px(self.object_pos)
        ow = int(OBJECT_HALF * self.size)
        draw.rectangle(
            (ox - ow, oy - ow, ox + ow, oy + ow),
            fill=COLOR_RGB[self.object_color],
            outline=(255, 255, 255),
        )
        gx, gy = self._to_px(self.gripper)
        col = (255, 200, 80) if self.holding else (250, 250, 250)
        draw.ellipse((gx - 8, gy - 8, gx + 8, gy + 8), fill=col, outline=(40, 40, 40))
        return np.array(img, dtype=np.uint8)

    def _to_px(self, pos: np.ndarray) -> tuple[int, int]:
        return int(pos[0] * self.size), int(pos[1] * self.size)
