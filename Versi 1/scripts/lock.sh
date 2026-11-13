#!/usr/bin/env bash
# Hasilkan berkas lock dari .in DI DALAM kontainer Python bersih.  ./scripts/lock.sh [--gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
IN=requirements.in; OUT=requirements.lock
if [[ "${1:-}" == "--gpu" ]]; then IN=requirements.gpu.in; OUT=requirements.gpu.lock; fi
docker run --rm -v "$PWD:/work" -w /work python:3.11-slim sh -c \
  "pip install --quiet pip-tools && pip-compile --quiet --strip-extras --resolver=backtracking --output-file=$OUT $IN"
echo "$OUT dibuat. Periksa isinya sebelum commit."
