#!/usr/bin/env bash
# Bangun citra pipeline di Linux/WSL.  ./scripts/build.sh [--gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
REQ=requirements.lock; TAG=larynx-pipeline
if [[ "${1:-}" == "--gpu" ]]; then REQ=requirements.gpu.lock; TAG=larynx-pipeline-gpu; fi
BASE=python:3.11-slim
docker pull "$BASE" >/dev/null
DIGEST=$(docker image inspect "$BASE" --format '{{index .RepoDigests 0}}')
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo tanpa-commit)
[[ -n "$(git status --porcelain 2>/dev/null)" ]] && COMMIT="$COMMIT-dirty"
docker build --build-arg "BASE_IMAGE=$DIGEST" --build-arg "REQUIREMENTS=$REQ" --build-arg "GIT_COMMIT=$COMMIT" \
  -t "$TAG:$COMMIT" -t "$TAG:latest" .
echo "Citra: $TAG:$COMMIT"; echo "Citra dasar: $DIGEST"
echo "ID citra: $(docker image inspect "$TAG:latest" --format '{{.Id}}')"
