"""Gradio UI — left: editable code execution, right: vision + language + training."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw

from vla_mini.env import make_env
from vla_mini.repl import (
    CodeSession,
    DEFAULT_SNIPPET,
    DRY_RUN_SNIPPET,
    PUSH_SNIPPET,
    ExecResult,
)

_torch = None
_load_model = None
_train_loop = None
_collect_episodes = None


def _ensure_train_deps():
    global _torch, _load_model, _train_loop, _collect_episodes
    if _torch is None:
        import torch

        from vla_mini.data.synthetic import collect_episodes
        from vla_mini.eval import load_model
        from vla_mini.train import train_loop

        _torch = torch
        _load_model = load_model
        _train_loop = train_loop
        _collect_episodes = collect_episodes


DEMO_CSS = """
.gradio-container { max-width: 100% !important; padding: 0 12px 12px; }
#vla-header {
  background: linear-gradient(90deg, #0d2137 0%, #1a4a7a 100%);
  color: #e8f4ff;
  padding: 14px 20px;
  border-radius: 8px;
  margin-bottom: 10px;
}
#left-panel {
  background: #1e1e1e;
  border-radius: 8px;
  padding: 8px;
  border: 1px solid #2d3a4a;
}
#right-panel {
  background: #0f1419;
  border-radius: 8px;
  padding: 10px;
  border: 1px solid #2d4a6a;
}
#bottom-bar {
  background: #152535;
  border-radius: 8px;
  padding: 10px 14px;
  margin-top: 8px;
  border: 1px solid #2a5080;
}
.footer-status textarea { font-family: ui-monospace, monospace; font-size: 0.85rem; }
"""


def _draw_action_arrow(img: Image.Image, action: np.ndarray) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cx, cy = w // 2, h // 2
    dx, dy = float(action[0]) * 40, float(action[1]) * 40
    draw.line((cx, cy, cx + dx, cy + dy), fill=(255, 220, 80), width=3)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 220, 80))
    return out


def _sync_ui_from_exec(
    exec_result: ExecResult,
    fallback_obs: np.ndarray | None,
    fallback_instr: str | None,
    fallback_action: np.ndarray | None,
) -> tuple:
    obs = exec_result.obs if exec_result.obs is not None else fallback_obs
    instr = exec_result.instruction if exec_result.instruction is not None else (fallback_instr or "")
    action = exec_result.action if exec_result.action is not None else fallback_action

    overlay = None
    if exec_result.overlay_base is not None and action is not None:
        overlay = _draw_action_arrow(exec_result.overlay_base, action)
    elif obs is not None and action is not None:
        overlay = _draw_action_arrow(Image.fromarray(np.asarray(obs, dtype=np.uint8)), action)

    frame_out = obs if obs is not None else fallback_obs
    return frame_out, instr, overlay, exec_result.console_text()


def build_demo(
    config_path: Path = Path("configs/default.yaml"),
    dry_run: bool = False,
) -> gr.Blocks:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task = cfg.get("task", "reach")
    ckpt_path = Path(cfg["output_dir"]) / "action_head.pt"
    env = make_env(task)
    model = None
    device = None
    mode_label = f"DRY-RUN · {task} · expert only" if dry_run else f"VLA · {task}"
    if task == "push":
        default_code = PUSH_SNIPPET
    else:
        default_code = DRY_RUN_SNIPPET if dry_run else DEFAULT_SNIPPET

    if not dry_run:
        _ensure_train_deps()
        device = cfg.get("device") or (
            "cuda" if _torch.cuda.is_available() else "cpu"
        )

    def ensure_model():
        nonlocal model
        if dry_run or model is not None:
            return
        _ensure_train_deps()
        if ckpt_path.exists():
            model = _load_model(ckpt_path, device)

    def make_predict_fn():
        def predict(image, instruction: str):
            ensure_model()
            if model is None:
                raise RuntimeError("无 checkpoint，请用 env.expert_action() 或先训练")
            pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
            return model.predict(pil, instruction).numpy()

        return predict if not dry_run and ckpt_path.exists() else None

    session = CodeSession(env=env, predict_fn=make_predict_fn())

    def refresh_predict_fn():
        session.predict_fn = make_predict_fn()

    def reset_env(code: str):
        obs, instr = env.reset()
        session.predict_fn = make_predict_fn()
        status = (
            f"[{mode_label}] 环境已重置\n"
            f"Instruction: {instr}\n"
            f"编辑左侧代码后点击「运行代码 ▶」"
        )
        return np.array(obs), instr, status, code or default_code, None

    def run_user_code(code: str, cur_obs, cur_instr):
        refresh_predict_fn()
        try:
            result = session.execute(code)
        except Exception as exc:
            return (
                cur_obs,
                cur_instr,
                None,
                f"安全检查失败: {exc}",
            )
        fb_obs = cur_obs if isinstance(cur_obs, np.ndarray) else None
        fb_instr = cur_instr
        fb_action = session.namespace().get("action")
        if isinstance(fb_action, np.ndarray):
            pass
        elif result.action is not None:
            fb_action = result.action
        frame_out, instr, overlay, log = _sync_ui_from_exec(
            result, fb_obs, fb_instr, fb_action
        )
        if result.ok:
            log = f"[运行成功]\n{log}"
        return frame_out, instr, overlay, log

    def predict_step(image: np.ndarray, instruction: str, code: str):
        pil = Image.fromarray(image.astype(np.uint8))
        if dry_run or not ckpt_path.exists():
            action = env.expert_action()
            label = "expert"
        else:
            ensure_model()
            if model is None:
                action = env.expert_action()
                label = "expert (no checkpoint)"
            else:
                action = model.predict(pil, instruction).numpy()
                label = "policy"
        overlay = _draw_action_arrow(pil, action)
        result = env.step(action)
        log = (
            f"[{label} 单步] action=[{action[0]:.3f}, {action[1]:.3f}]  "
            f"dist={result.info['distance']:.4f}  success={result.info['success']}"
        )
        return (
            np.array(result.observation),
            instruction,
            overlay,
            log,
            code,
        )

    def run_training():
        if dry_run:
            return None, "Dry-run 模式不训练。请: npx vla-mini train --collect"
        _ensure_train_deps()
        data_dir = Path(cfg["data_dir"])
        _collect_episodes(
            num_episodes=cfg.get("num_episodes", 80),
            out_dir=data_dir,
            task=task,
        )
        _train_loop(
            manifest=data_dir / "manifest.jsonl",
            data_root=data_dir,
            output_dir=Path(cfg["output_dir"]),
            epochs=cfg.get("epochs", 2),
            batch_size=cfg.get("batch_size", 4),
            lr=cfg.get("lr", 1e-3),
            device=device,
            vlm_name=cfg.get("vlm_name"),
        )
        nonlocal model
        model = _load_model(ckpt_path, device)
        refresh_predict_fn()
        hist_path = Path(cfg["output_dir"]) / "history.json"
        fig = None
        if hist_path.exists():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="#0f1419")
            ax.set_facecolor("#0f1419")
            ax.plot(
                [h["step"] for h in hist],
                [h["loss"] for h in hist],
                color="#4da3ff",
                linewidth=2,
            )
            ax.set_xlabel("step", color="#aaa")
            ax.set_ylabel("MSE", color="#aaa")
            ax.tick_params(colors="#888")
            ax.set_title("Training loss", color="#cce4ff")
            for spine in ax.spines.values():
                spine.set_color("#334")
            fig.tight_layout()
        return fig, "训练完成，可使用 predict(image, instruction) 或重新运行代码。"

    with gr.Blocks(title="vla-mini") as demo:
        gr.Markdown(
            f"# vla-mini 开发台 · **{mode_label}**\n"
            "左侧 **可编辑 Python** 并真实执行（沙箱，共享 `env`）；右侧同步画面与动作箭头。"
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_id="left-panel"):
                gr.Markdown("#### 代码编辑器")
                code_box = gr.Code(
                    value=default_code,
                    language="python",
                    label="编辑后点击「运行代码」",
                    lines=16,
                    interactive=True,
                )
                with gr.Row():
                    run_code_btn = gr.Button("运行代码 ▶", variant="primary")
                    reset_btn = gr.Button("重置环境", variant="secondary")
                    step_btn = gr.Button("VLA 单步", variant="secondary")
                train_btn = gr.Button(
                    "采集 + 训练动作头",
                    variant="secondary",
                    interactive=not dry_run,
                )
                console = gr.Textbox(
                    label="运行日志",
                    lines=6,
                    elem_classes=["footer-status"],
                    interactive=False,
                )
                gr.Markdown(
                    "可用变量: `env`, `np`, `Image`, `predict`（有权重时）, "
                    "`obs`, `instruction`, `action`, `result` · 禁止 `import`"
                )

            with gr.Column(scale=6, elem_id="right-panel"):
                gr.Markdown("#### 视觉 · 语言 · 训练")
                with gr.Row():
                    frame = gr.Image(label="RGB 观测", type="numpy", height=280)
                    overlay = gr.Image(label="动作箭头", type="pil", height=280)
                instr_box = gr.Textbox(label="语言指令", interactive=False)
                loss_plot = gr.Plot(label="Loss")

        gr.Markdown(
            elem_id="bottom-bar",
            value="**npm:** `npx vla-mini demo --dry-run` · "
            "**Python:** `python -m vla_mini.demo --dry-run`",
        )

        reset_btn.click(
            reset_env,
            inputs=[code_box],
            outputs=[frame, instr_box, console, code_box, overlay],
        )
        run_code_btn.click(
            run_user_code,
            inputs=[code_box, frame, instr_box],
            outputs=[frame, instr_box, overlay, console],
        )
        step_btn.click(
            predict_step,
            inputs=[frame, instr_box, code_box],
            outputs=[frame, instr_box, overlay, console, code_box],
        )
        train_btn.click(run_training, outputs=[loss_plot, console])

        demo.load(
            lambda: reset_env(default_code),
            outputs=[frame, instr_box, console, code_box, overlay],
        )

    return demo


def main() -> None:
    import argparse
    import os

    # Avoid system HTTP proxy breaking localhost Gradio on Windows.
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if os.environ.get(key):
            os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0a0e14",
        block_background_fill="#121820",
        body_text_color="#e0e8f0",
        block_title_text_color="#9ec8ff",
    )
    build_demo(args.config, dry_run=args.dry_run).launch(
        share=args.share,
        server_name="127.0.0.1",
        server_port=args.port,
        theme=theme,
        css=DEMO_CSS,
        show_error=True,
    )


if __name__ == "__main__":
    main()
