"use strict";

const fs = require("fs");
const path = require("path");

function hasPythonProject(dir) {
  return (
    fs.existsSync(path.join(dir, "pyproject.toml")) &&
    fs.existsSync(path.join(dir, "src", "vla_mini"))
  );
}

/** Monorepo root when developing; vendor/ when installed from npm pack. */
function getProjectRoot() {
  const pkgRoot = path.join(__dirname, "..");
  const candidates = [
    path.join(pkgRoot, "..", ".."), // d:\vla when CLI lives in npm/vla-mini
    path.join(pkgRoot, "vendor"),
    process.env.VLA_MINI_ROOT,
    process.cwd(),
  ].filter(Boolean);

  for (const dir of candidates) {
    const resolved = path.resolve(dir);
    if (hasPythonProject(resolved)) return resolved;
  }

  throw new Error(
    "找不到 vla-mini Python 项目。请: npm install ./npm/vla-mini 或设置 VLA_MINI_ROOT",
  );
}

function getVenvDir() {
  const home =
    process.env.VLA_MINI_HOME ||
    path.join(process.env.USERPROFILE || process.env.HOME || ".", ".vla-mini");
  return path.join(home, "venv");
}

function getVenvPython(venvDir) {
  const win = process.platform === "win32";
  return path.join(
    venvDir,
    win ? "Scripts" : "bin",
    win ? "python.exe" : "python",
  );
}

module.exports = { getProjectRoot, getVenvDir, getVenvPython, hasPythonProject };
