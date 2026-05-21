"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const { getProjectRoot, getVenvDir, getVenvPython } = require("./paths");

function runCapture(cmd, args, opts = {}) {
  const shell = opts.shell ?? process.platform === "win32";
  return spawnSync(cmd, args, {
    encoding: "utf8",
    shell,
    windowsHide: true,
    ...opts,
  });
}

function versionOk(out) {
  const m = String(out).match(/3\.(12|13)/);
  return Boolean(m);
}

function findSystemPython() {
  const tries =
    process.platform === "win32"
      ? [
          ["py", ["-3.12", "-c", "import sys; print(sys.version)"]],
          ["py", ["-3.13", "-c", "import sys; print(sys.version)"]],
          ["python", ["-c", "import sys; print(sys.version)"]],
          ["python3", ["-c", "import sys; print(sys.version)"]],
        ]
      : [
          ["python3.12", ["-c", "import sys; print(sys.version)"]],
          ["python3.13", ["-c", "import sys; print(sys.version)"]],
          ["python3", ["-c", "import sys; print(sys.version)"]],
        ];

  for (const [cmd, args] of tries) {
    const r = runCapture(cmd, args);
    if (r.status === 0 && versionOk(r.stdout || r.stderr)) {
      return { launcher: cmd, venvArgs: cmd === "py" ? [args[0]] : [] };
    }
  }
  return null;
}

function findRepoVenv(root) {
  const repoVenv = path.join(root, ".venv");
  const py = getVenvPython(repoVenv);
  if (fs.existsSync(py)) return { root, venvDir: repoVenv, venvPy: py };
  return null;
}

function run(cmd, args, opts = {}) {
  const shell = process.platform === "win32";
  const r = spawnSync(cmd, args, { stdio: "inherit", shell, windowsHide: true, ...opts });
  if (r.error) throw r.error;
  if (r.status !== 0) process.exit(r.status ?? 1);
}

function ensureVenv() {
  const root = getProjectRoot();
  const globalVenv = getVenvDir();
  const globalPy = getVenvPython(globalVenv);

  if (fs.existsSync(globalPy)) {
    return { root, venvDir: globalVenv, venvPy: globalPy };
  }

  const repo = findRepoVenv(root);
  if (repo) {
    console.log(`[vla-mini] 使用仓库内虚拟环境 -> ${repo.venvDir}`);
    return repo;
  }

  const py = findSystemPython();
  if (!py) {
    console.error(
      "需要 Python 3.12 或 3.13。\n" +
        "  Windows: 安装后执行  py -3.12 -m pip install -e <vla-mini路径>\n" +
        "  或先在仓库运行: python -m venv .venv && pip install -e .",
    );
    process.exit(1);
  }

  fs.mkdirSync(path.dirname(globalVenv), { recursive: true });
  console.log(`[vla-mini] 创建虚拟环境 -> ${globalVenv}`);
  if (py.launcher === "py") {
    run("py", [...py.venvArgs, "-m", "venv", globalVenv]);
  } else {
    run(py.launcher, ["-m", "venv", globalVenv]);
  }

  console.log("[vla-mini] 安装 Python 依赖（首次较慢，含 PyTorch）...");
  run(globalPy, ["-m", "pip", "install", "-U", "pip"]);
  run(globalPy, ["-m", "pip", "install", "-e", root]);

  return { root, venvDir: globalVenv, venvPy: globalPy };
}

function runModule(moduleArgs, extraEnv = {}) {
  const { root, venvPy } = ensureVenv();
  const env = { ...process.env, VLA_MINI_ROOT: root, ...extraEnv };
  run(venvPy, ["-m", ...moduleArgs], { cwd: root, env });
}

module.exports = { ensureVenv, runModule, findSystemPython, getProjectRoot };
