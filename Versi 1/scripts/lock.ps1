# Menghasilkan requirements.lock dari requirements.in DI DALAM kontainer Python bersih.
# Ini satu-satunya tempat pip dipakai di luar Dockerfile, dan hanya untuk menyelesaikan versi.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
docker run --rm -v "${root}:/work" -w /work python:3.11-slim sh -c `
  "pip install --quiet pip-tools && pip-compile --quiet --strip-extras --resolver=backtracking --output-file=requirements.lock requirements.in"
if ($LASTEXITCODE -ne 0) { throw "pip-compile gagal" }
Write-Host "requirements.lock dibuat. Periksa isinya sebelum commit."
