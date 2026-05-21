# vla-mini

仓库：<https://github.com/HelloJessieee/vla-tutorial>

面向教学的**最小 VLA 工作台**：2D 仿真 + 自动生成数据 + 网页 Demo + 可选训练（CLIP 小模型 + 动作头）。  
**不依赖 LeRobot / MuJoCo 安装**；可选 **Edu-π₀** 对齐 [LeRobot π₀ policy 格式](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/pi0)。

npm 包：https://www.npmjs.com/package/@jxhs/vla-mini

---

## 三种用法（一个项目，三个档位）

| 档位 | 干什么 | 要 GPU？ | 要联网？ |
|------|--------|----------|----------|
| **① 演示** `demo --dry-run` | 网页 + 改代码 + 2D 画面；**规则专家**动，不训练神经网络 | 否 | 否 |
| **② 真 VLA** `train` → `demo` | 自动生成数据 → 训练 → 模型推理 | 建议有 | 训练时要（下 CLIP） |
| **③ Edu-π₀** `train_pi0` | 与 ② 同类，代码格式像 LeRobot π₀ | 建议有 | 训练时要 |

**不是三个软件**，是同一仓库里的三条命令路径。

---

## 教学任务阶梯（仿真）

同一套 **train / eval / demo / collect**；只改 `task` 与输出维度。

| 级别 | `task` | 看图+指令 → 输出 | `output_dim` | 配置 |
|------|--------|------------------|--------------|------|
| **L0** | `reach` | 单步靠近 `(dx, dy)` | 2 | `configs/default.yaml` |
| **L1** | `push` | **push_t** 连续推动 `K×(dx,dy)`，默认 K=4 | 8 | `configs/push.yaml` |
| **L2** | `grasp` | `(dx, dy, gripper)`，夹爪 ∈ [-1,1] | 3 | `configs/grasp.yaml` |
| **课外** | LIBERO 等 | 3D / 真机 | — | 见升级路径 |

仍 **无 Bullet**、**自动 collect + expert**。切换示例：

```cmd
.\.venv\Scripts\python.exe -m vla_mini.dry_run --task reach
.\.venv\Scripts\python.exe -m vla_mini.dry_run --task push
.\.venv\Scripts\python.exe -m vla_mini.dry_run --task grasp

.\.venv\Scripts\python.exe -m vla_mini.train --config configs\push.yaml --collect
.\.venv\Scripts\python.exe -m vla_mini.train --config configs\grasp.yaml --collect
.\.venv\Scripts\python.exe -m vla_mini.demo --config configs\grasp.yaml --dry-run
```

---

## 环境要求

- **Python 3.12 或 3.13**
- **Windows / macOS / Linux**（演示模式 Windows 可跑）
- **GPU**：仅 ① 不需要；②③ 建议 NVIDIA（CPU 也能跑，慢）
- **Node 18+**：仅在使用 `npx @jxhs/vla-mini` 时需要

---

## 一、本地安装（首次必做）

在 **CMD** 或 **PowerShell** 中，**必须在 `d:\vla` 根目录**操作（不要在 `npm\vla-mini` 里用系统 `python`）。

### CMD（推荐你当前环境）

```cmd
cd /d d:\vla
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
```

### PowerShell

```powershell
cd d:\vla
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

### 检查安装是否成功

```cmd
cd /d d:\vla
.\.venv\Scripts\python.exe -c "import vla_mini; print('OK')"
```

---

## 二、① 演示模式（啥都不懂先跑这个）

无需 GPU、无需下载 VLM 权重。

```cmd
cd /d d:\vla
.\.venv\Scripts\python.exe -m vla_mini.dry_run
```

可选：顺带生成/校验合成数据：

```cmd
.\.venv\Scripts\python.exe -m vla_mini.dry_run --collect
```

打开网页 Demo（浏览器访问 http://127.0.0.1:7860 ）：

```cmd
scripts\run-demo.cmd --dry-run
```

等价命令：

```cmd
set NO_PROXY=127.0.0.1,localhost
.\.venv\Scripts\python.exe -m vla_mini.demo --dry-run
```

换端口（7860 被占用时）：

```cmd
.\.venv\Scripts\python.exe -m vla_mini.demo --dry-run --port 7861
```

**网页操作：** 重置环境 → 运行代码 / 单步执行。  
**说明：** 这是**演示**，不是训练好的 VLA 大模型。

---

## 三、② 真 VLA：训练 + 推理

### 1. 生成训练数据（约 120 局，已有数据可跳过）

```cmd
cd /d d:\vla
.\.venv\Scripts\python.exe -m vla_mini.train --collect
```

数据位置：`data\synthetic\manifest.jsonl` 与 `data\synthetic\images\`。

只用已有数据、不再生成：

```cmd
.\.venv\Scripts\python.exe -m vla_mini.train
```

### 2. 训练（首次会从 Hugging Face 下载 CLIP，需联网）

```cmd
.\.venv\Scripts\python.exe -m vla_mini.train --collect
```

权重输出：`runs\default\action_head.pt`

### 3. 评测

```cmd
.\.venv\Scripts\python.exe -m vla_mini.eval
```

### 4. 用训练好的模型打开 Demo

```cmd
scripts\run-demo.cmd
```

或：

```cmd
.\.venv\Scripts\python.exe -m vla_mini.demo
```

---

## 四、③ Edu-π₀（LeRobot 风格 policy）

```cmd
cd /d d:\vla
.\.venv\Scripts\python.exe -m vla_mini.train_pi0 --collect
```

配置：`configs\edu_pi0.yaml`（可改 `vlm_backbone: clip` / `minimind2-small-v`）  
权重输出：`runs\edu_pi0\edu_pi0.pt`  
说明文档：[docs/EDU_PI0.md](docs/EDU_PI0.md)

---

## 五、用 npm / npx（给别人一键装）

发布包：`@jxhs/vla-mini`（npm ≥0.1.1 含 `train-pi0` 子命令）。

```bash
# 全局安装（推荐，不用每次 npx）
npm install -g @jxhs/vla-mini
vla-mini install
vla-mini demo --dry-run

# 或一次性 npx（不写入全局）
npx @jxhs/vla-mini install
npx @jxhs/vla-mini dry-run
npx @jxhs/vla-mini demo --dry-run
npx @jxhs/vla-mini train --collect
npx @jxhs/vla-mini train-pi0 --collect
npx @jxhs/vla-mini eval
npx @jxhs/vla-mini demo
```

开发者在仓库内调试 npm CLI：

```cmd
cd /d d:\vla\npm\vla-mini
node bin\vla-mini.js dry-run
node bin\vla-mini.js demo --dry-run
node bin\vla-mini.js train --collect
```

---

## 六、推荐完整跑通顺序（复制执行）

```cmd
cd /d d:\vla
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m vla_mini.dry_run --collect
scripts\run-demo.cmd --dry-run
.\.venv\Scripts\python.exe -m vla_mini.train --collect
.\.venv\Scripts\python.exe -m vla_mini.eval
scripts\run-demo.cmd
```

有 GPU 时训练更快；无 GPU 时前两步（dry-run + demo --dry-run）即可完成课堂演示。

---

## 七、常见问题

| 现象 | 处理 |
|------|------|
| `No module named 'vla_mini'` | 未在 `d:\vla` 安装，或用了 `C:\Python313\python` 而非 `.\.venv\Scripts\python.exe` |
| `$env:...` 报错 | 你在 **CMD** 里；改用 `set NO_PROXY=...` 或 `scripts\run-demo.cmd` |
| Gradio 502 | 代理问题；用 `run-demo.cmd` 或关系统 HTTP 代理 |
| `train` 很慢 / 超时 | 无 GPU 或网络下 HF 模型；可先只跑 ① |
| 注释行 `# ...` 报错 | CMD 不要粘贴以 `#` 开头的行 |

---

## 项目结构

```
d:\vla\
  src/vla_mini/
    env/toy_reach.py      # L0 reach
    env/toy_push.py       # L1 push_t
    env/toy_grasp.py      # L2 grasp (+ gripper dim)
    env/factory.py        # make_env(task=reach|push|grasp)
    env/tasks.py          # action_dim × action_chunk 规格
  configs/push.yaml
  configs/grasp.yaml
    data/synthetic.py     # 合成数据
    model/vla.py          # 基础 VLA（CLIP + 动作头）
    policy/               # EduPI0Policy（π₀ 格式）
    train.py / train_pi0.py / eval.py / demo.py / dry_run.py
  configs/default.yaml
  configs/edu_pi0.yaml
  data/synthetic/         # 运行 collect 后生成
  runs/                   # 训练后生成
  scripts/run-demo.cmd    # Windows 启动 Demo
  npm/vla-mini/           # npm 封装
```

---

## 数据说明

- **自带数据**：运行 `--collect` 后在本地 **自动生成**（不是 LIBERO / `lerobot/*`）。
- **不会**在安装时自动下载 Hugging Face 机器人大数据集。
- 训练时下载的是 **模型权重**（如 CLIP），不是示范数据集。

---

## 升级路径

1. **本仓库内**：`pick`、多任务混合（`make_env` / `collect_episodes` 已支持扩展）
2. 换仿真：PyBullet / LIBERO（单独环境，课堂外）
3. 换数据：接入 `lerobot/*` 或 HDF5
4. 换大 VLM / LoRA
5. 完整 LeRobot：`lerobot[libero]` 等

---

## 许可

Apache-2.0
