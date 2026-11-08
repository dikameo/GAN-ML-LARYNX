# Menjalankan perintah di dalam kontainer.
#   .\scripts\run.ps1 python -m src.env_check '--config' configs/base.yaml     (kutip argumen berawalan --)
#   .\scripts\run.ps1 -Gpu python -m src.run_all '--stage' all
#   .\scripts\run.ps1 -Jupyter            # JupyterLab di http://localhost:8888
# Dataset dipasang BACA-SAJA; outputs dapat ditulis; kode di-bind-mount untuk pengembangan.
param(
  [switch]$Jupyter,
  [switch]$Gpu,
  [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Cmd
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force "$root\outputs" | Out-Null

$tag = if ($Gpu) { "larynx-pipeline-gpu" } else { "larynx-pipeline" }
$commit = (git -C $root rev-parse --short HEAD 2>$null)
if (-not $commit) { $commit = "tanpa-commit" }
if (git -C $root status --porcelain) { $commit = "$commit-dirty" }
$imgId = docker image inspect "${tag}:latest" --format '{{.Id}}'

$args_ = @("run", "--rm", "--shm-size=2g")
if ($Gpu) { $args_ += @("--gpus", "all") }
$args_ += @(
  "-v", "$root\data\Larynx_CE-NBI_Dataset:/data/Larynx_CE-NBI_Dataset:ro",
  "-v", "$root\outputs:/outputs",
  "-v", "$root\src:/app/src", "-v", "$root\configs:/app/configs",
  "-v", "$root\tests:/app/tests", "-v", "$root\notebooks:/app/notebooks", "-v", "$root\scripts:/app/scripts",
  "-e", "GIT_COMMIT=$commit", "-e", "IMAGE_DIGEST=$imgId")

if ($Jupyter) {
  $args_ += @("-p", "8888:8888", "${tag}:latest",
    "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks")
} else {
  $args_ += @("${tag}:latest") + $Cmd
}
$ErrorActionPreference = "Continue"
& docker @args_
exit $LASTEXITCODE
