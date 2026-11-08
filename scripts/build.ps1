# Membangun citra pipeline. Digest citra dasar dibaca dari hasil pull dan dikunci di tag.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$base = "python:3.11-slim"
docker pull $base | Out-Null
$digest = docker image inspect $base --format '{{index .RepoDigests 0}}'
if (-not $digest) { throw "digest citra dasar tidak terbaca" }

$commit = (git rev-parse --short HEAD 2>$null)
if (-not $commit) { $commit = "tanpa-commit" }
$dirty = (git status --porcelain)
if ($dirty) { $commit = "$commit-dirty" }

docker build --build-arg "BASE_IMAGE=$digest" --build-arg "GIT_COMMIT=$commit" -t "larynx-pipeline:$commit" -t "larynx-pipeline:latest" .
if ($LASTEXITCODE -ne 0) { throw "docker build gagal" }

$imgId = docker image inspect larynx-pipeline:latest --format '{{.Id}}'
Write-Host "Citra: larynx-pipeline:$commit"
Write-Host "Citra dasar: $digest"
Write-Host "ID citra: $imgId"
