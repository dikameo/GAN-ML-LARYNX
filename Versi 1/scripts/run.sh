#!/usr/bin/env bash
# Jalankan perintah di kontainer (Linux/WSL).
#   ./scripts/run.sh [--gpu] python -m src.run_all --stage all
#   ./scripts/run.sh [--gpu] --jupyter           # JupyterLab di http://localhost:8888
# Dataset dipasang BACA-SAJA. Atur DATASET_DIR (default ../data/Larynx_CE-NBI_Dataset) dan OUTPUTS_DIR (default ./outputs).
# Untuk beberapa GPU: GPUS='"device=0"' ./scripts/run.sh --gpu ...
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=larynx-pipeline; GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then TAG=larynx-pipeline-gpu; GPU_ARGS=(--gpus "${GPUS:-all}"); shift; fi
JUP=0; if [[ "${1:-}" == "--jupyter" ]]; then JUP=1; shift; fi
DATASET_DIR=${DATASET_DIR:-$(dirname "$PWD")/data/Larynx_CE-NBI_Dataset}      # dataset di Training/data, di luar Versi 1
OUTPUTS_DIR=${OUTPUTS_DIR:-$PWD/outputs}
mkdir -p "$OUTPUTS_DIR"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo tanpa-commit)
[[ -n "$(git status --porcelain 2>/dev/null)" ]] && COMMIT="$COMMIT-dirty"
IMG_ID=$(docker image inspect "$TAG:latest" --format '{{.Id}}')
COMMON=(--rm "${GPU_ARGS[@]}" --shm-size=2g
  -v "$DATASET_DIR:/data/Larynx_CE-NBI_Dataset:ro" -v "$OUTPUTS_DIR:/outputs"
  -v "$PWD/src:/app/src" -v "$PWD/configs:/app/configs" -v "$PWD/tests:/app/tests"
  -v "$PWD/notebooks:/app/notebooks" -v "$PWD/scripts:/app/scripts"
  -e "GIT_COMMIT=$COMMIT" -e "IMAGE_DIGEST=$IMG_ID")
if [[ $JUP == 1 ]]; then
  exec docker run "${COMMON[@]}" -p 8888:8888 "$TAG:latest" jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app/notebooks
else
  exec docker run "${COMMON[@]}" "$TAG:latest" "$@"
fi
