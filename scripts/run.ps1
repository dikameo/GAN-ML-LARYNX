# Menjalankan perintah di dalam kontainer.
#   .\scripts\run.ps1 python -m src.env_check --config configs/base.yaml
#   .\scripts\run.ps1 -Jupyter            # JupyterLab di http://localhost:8888
# Dataset dipasang BACA-SAJA; outputs dapat ditulis; kode di-bind-mount untuk pengembangan.
param(
  [switch]$Jupyter,
  [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Cmd
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force "$root\outputs" | Out-Null

$commit = (git -C $root rev-parse --short HEAD 2>$null)
if (-not $commit) { $commit = "tanpa-commit" }
if (git -C $root status --porcelain) { $commit = "$commit-dirty" }
$imgId = docker image inspect larynx-pipeline:latest --format '{{.Id}}'

$args_ = @("run", "--rm",
  "-v", "$root\data\Larynx_CE-NBI_Dataset:/data/Larynx_CE-NBI_Dataset:ro",
  "-v", "$root\outputs:/outputs",
  "-v", "$root\src:/app/src", "-v", "$root\configs:/app/configs",
  "-v", "$root\tests:/app/tests", "-v", "$root\notebooks:/app/notebooks",
  "-e", "GIT_COMMIT=$commit", "-e", "IMAGE_DIGEST=$imgId")

if ($Jupyter) {
  $args_ += @("-p", "8888:8888", "larynx-pipeline:latest",
    "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks")
} else {
  $args_ += @("larynx-pipeline:latest") + $Cmd
}
& docker @args_
exit $LASTEXITCODE
