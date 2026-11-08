# Membangun citra pipeline. Digest citra dasar dibaca dari hasil pull dan dikunci di tag.
#   .\scripts\build.ps1          # CPU
#   .\scripts\build.ps1 -Gpu     # GPU NVIDIA (requirements.gpu.lock)
param([switch]$Gpu)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$req = if ($Gpu) { "requirements.gpu.lock" } else { "requirements.lock" }
$tag = if ($Gpu) { "larynx-pipeline-gpu" } else { "larynx-pipeline" }
$base = "python:3.11-slim"
docker pull $base | Out-Null
$digest = docker image inspect $base --format '{{index .RepoDigests 0}}'
if (-not $digest) { throw "digest citra dasar tidak terbaca" }

$commit = (git rev-parse --short HEAD 2>$null)
if (-not $commit) { $commit = "tanpa-commit" }
if (git status --porcelain) { $commit = "$commit-dirty" }

docker build --build-arg "BASE_IMAGE=$digest" --build-arg "REQUIREMENTS=$req" --build-arg "GIT_COMMIT=$commit" -t "${tag}:$commit" -t "${tag}:latest" .
if ($LASTEXITCODE -ne 0) { throw "docker build gagal" }

Write-Host "Citra: ${tag}:$commit"
Write-Host "Citra dasar: $digest"
Write-Host ("ID citra: " + (docker image inspect "${tag}:latest" --format '{{.Id}}'))
