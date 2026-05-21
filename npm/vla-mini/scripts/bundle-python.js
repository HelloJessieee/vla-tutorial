#!/usr/bin/env node
"use strict";
/** Copy Python sources into vendor/ before npm pack publish. */

const fs = require("fs");
const path = require("path");

const pkgRoot = path.join(__dirname, "..");
const repoRoot = path.join(pkgRoot, "..", "..");
const vendor = path.join(pkgRoot, "vendor");

const SKIP_DIRS = new Set(["__pycache__", ".pytest_cache", ".ruff_cache", ".venv"]);

function rimraf(dir) {
  if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true, force: true });
}

function copyRecursive(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const name of fs.readdirSync(src)) {
    if (SKIP_DIRS.has(name) || name.endsWith(".pyc")) continue;
    const s = path.join(src, name);
    const d = path.join(dest, name);
    if (fs.statSync(s).isDirectory()) copyRecursive(s, d);
    else fs.copyFileSync(s, d);
  }
}

if (!fs.existsSync(path.join(repoRoot, "pyproject.toml"))) {
  console.error("bundle-python: repo root not found at", repoRoot);
  process.exit(1);
}

rimraf(vendor);
fs.mkdirSync(vendor, { recursive: true });
fs.copyFileSync(
  path.join(repoRoot, "pyproject.toml"),
  path.join(vendor, "pyproject.toml"),
);
copyRecursive(path.join(repoRoot, "src"), path.join(vendor, "src"));
copyRecursive(path.join(repoRoot, "configs"), path.join(vendor, "configs"));
if (fs.existsSync(path.join(repoRoot, "README.md"))) {
  fs.copyFileSync(path.join(repoRoot, "README.md"), path.join(vendor, "README.md"));
}
console.log("[bundle-python] vendor/ updated at", vendor);
