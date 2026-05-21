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
vla-mini dry-run --collect
vla-mini train --collect
vla-mini train-pi0 --collect
vla-mini demo
```

https://www.npmjs.com/package/@jxhs/vla-mini

## 发布到 npm

```bash
npm run bundle   # 同步 vendor/
npm publish
```

## 环境变量

- `VLA_MINI_HOME` — 覆盖默认 `~/.vla-mini` 缓存目录
