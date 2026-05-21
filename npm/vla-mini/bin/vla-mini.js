#!/usr/bin/env node
"use strict";

const { runModule } = require("../lib/python");

const argv = process.argv.slice(2);
const cmd = argv[0] || "help";

const map = {
  demo: ["vla_mini.demo", ...argv.slice(1)],
  "dry-run": ["vla_mini.dry_run", ...argv.slice(1)],
  train: ["vla_mini.train", ...argv.slice(1)],
  "train-pi0": ["vla_mini.train_pi0", ...argv.slice(1)],
  eval: ["vla_mini.eval", ...argv.slice(1)],
  install: [],
  help: [],
};

function printHelp() {
  console.log(`
vla-mini — 教学用最小 VLA 工作台（npm 封装 Python 后端）

用法:
  npx vla-mini demo [--dry-run] [--share]   启动 Gradio（左侧可编辑并执行代码）
  npx vla-mini dry-run [--collect]          无 VLM 下载的秒级验收
  npx vla-mini train [--collect] [--dry-run]
  npx vla-mini train-pi0 [--collect]        Edu-π₀（LeRobot 风格）
  npx vla-mini eval [--dry-run]
  npx vla-mini install                      仅安装/更新 Python 虚拟环境

环境变量:
  VLA_MINI_HOME   虚拟环境与缓存目录（默认 ~/.vla-mini）

要求: Node >= 18, Python 3.12 或 3.13
`);
}

if (cmd === "help" || cmd === "-h" || cmd === "--help") {
  printHelp();
  process.exit(0);
}

if (cmd === "install") {
  const { ensureVenv } = require("../lib/python");
  ensureVenv();
  console.log("[vla-mini] Python 环境就绪。");
  process.exit(0);
}

const moduleArgs = map[cmd];
if (!moduleArgs) {
  console.error(`未知命令: ${cmd}\n`);
  printHelp();
  process.exit(1);
}

runModule(moduleArgs);
