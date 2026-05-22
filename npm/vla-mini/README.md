# vla-mini (npm)

Node 封装层：自动检测 Python 3.12/3.13、创建 `~/.vla-mini/venv`、安装 [vla-tutorial](https://github.com/HelloJessieee/vla-tutorial) Python 包并启动 Demo。

## 安装

```bash
# 本地开发（在仓库根目录）
cd npm/vla-mini
npm install
npm link

# 或从路径安装
npm install -g ../../npm/vla-mini
```

## 使用（已发布）

```bash
npm install -g @jxhs/vla-mini
vla-mini install
vla-mini demo --dry-run
vla-mini dry-run --task reach
vla-mini dry-run --task push
vla-mini train --config configs/push.yaml --collect
vla-mini train --config configs/grasp.yaml --collect
vla-mini train-pi0 --collect
vla-mini demo --config configs/push.yaml --dry-run
```

npm 0.1.2+: L0 reach / L1 push_t (8-dim chunk) / L2 grasp (3-dim gripper).

https://www.npmjs.com/package/@jxhs/vla-mini

## 发布到 npm

```bash
npm run bundle   # 同步 vendor/
npm publish
```

## 环境变量

- `VLA_MINI_HOME` — 覆盖默认 `~/.vla-mini` 缓存目录
