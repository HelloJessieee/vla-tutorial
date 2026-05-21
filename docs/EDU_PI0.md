# 教学用 Edu-π₀（对齐 LeRobot policy 格式）

参考 [LeRobot PI0Policy](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/pi0)，在 **不安装 lerobot** 的前提下，实现同一套 **batch 键名 + 调用方式**，便于以后无缝迁移。

## 与真 π₀ 的差异（诚实说明）

| 项目 | LeRobot π₀ | vla-mini `EduPI0Policy` |
|------|------------|-------------------------|
| VLM | PaliGemma + Gemma action expert | SmolVLM / MiniMind2-Small-V |
| 动作预测 | Flow matching + 扩散去噪 | MSE + 小型 MLP（action expert） |
| 算力 | 高 | 低（冻结 VLM，只训 expert） |
| 环境 | 真机 / LIBERO | 2D `ToyReachEnv` |

## VLM 选型

| 后端 | HF 模型 | 参数量 | 说明 |
|------|---------|--------|------|
| `clip`（默认） | `openai/clip-vit-base-patch32` | ~150M | 最稳，教学 batch 跑通首选 |
| `smolvlm` | `HuggingFaceTB/SmolVLM-256M-Instruct` | ~256M | 部分 transformers 版本需额外配置 |
| `minimind2-small-v` | `jingyaogong/MiniMind2-Small-V` | ~26M | 对齐 MiniMind 生态，需 `trust_remote_code` |

更小的 [MiniMind-V 3](https://github.com/jingyaogong/minimind-v)（65M）教学也很好，但权重流程偏 ModelScope；后续可加 `minimind-3v` 后端。

## Batch 格式（与 LeRobot 一致）

```python
batch = {
    "observation.images.main": tensor,  # (B, 3, H, W) float [0,1]
    "observation.state": tensor,        # (B, max_state_dim) padded
    "action": tensor,                   # (B, max_action_dim) padded
    "_instruction_text": list[str],     # vla-mini 扩展，供 VLM 编码
}
```

## API（与 PI0Policy 同名方法）

```python
from vla_mini.policy import EduPI0Config, EduPI0Policy
from vla_mini.policy.batch import obs_to_batch

policy = EduPI0Policy(EduPI0Config(vlm_backbone="clip"))
batch = obs_to_batch([obs], [instruction], states=[[0,0,0,0]], actions=None, config=policy.config)
chunk = policy.predict_action_chunk(batch)   # (1, chunk_size, action_dim)
action = policy.select_action(batch)         # 单步，带 action queue
loss, info = policy.forward(batch)           # 训练
```

## 训练

```bash
python -m vla_mini.train_pi0 --collect
# MiniMind 后端：
# 编辑 configs/edu_pi0.yaml → vlm_backbone: minimind2-small-v
python -m vla_mini.train_pi0 --collect
```

## 升级到真 LeRobot

1. 把 `observation.images.main` 换成数据集里真实相机键名  
2. 用 `OBS_LANGUAGE_TOKENS` 走 tokenizer（我们已预留常量）  
3. 替换 `EduPI0Policy` → `lerobot.policies.pi0.PI0Policy`  
4. 单独环境安装 `lerobot[pi]`
