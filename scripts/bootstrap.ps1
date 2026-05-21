# One-shot setup for Windows (PowerShell)
Set-Location $PSScriptRoot\..
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\pip install -e .
Write-Host "`nL0 (no VLM download):"
Write-Host "  .\.venv\Scripts\python -m vla_mini.dry_run"
Write-Host "`nL1+:"
Write-Host "  .\.venv\Scripts\activate"
Write-Host "  python -m vla_mini.train --collect"
Write-Host "  python -m vla_mini.demo"
Write-Host "  python -m vla_mini.demo --dry-run"
