# Citra pipeline. PRD §13: versi dasar dikunci dengan digest, dependensi dari berkas lock.
#   CPU : docker build --build-arg REQUIREMENTS=requirements.lock     (default)
#   GPU : docker build --build-arg REQUIREMENTS=requirements.gpu.lock  (roda dengan --gpus all; butuh driver NVIDIA di host)
# Roda CUDA ikut di dalam wheel torch, jadi citra dasar tetap python:3.11-slim.
# BASE_IMAGE diisi tag@sha256:... oleh scripts/build.* setelah digest dibaca dari hasil pull.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG REQUIREMENTS=requirements.lock
ARG GIT_COMMIT=tidak-diketahui
LABEL org.opencontainers.image.revision=${GIT_COMMIT}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/outputs/hf_cache \
    GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app

# Dependensi dikunci. Tidak ada pip install manual di dalam kontainer (RULES §2).
COPY ${REQUIREMENTS} /app/requirements.lock
RUN pip install -r /app/requirements.lock

# Kode disalin agar citra mandiri; skrip run menimpanya dengan bind mount saat pengembangan.
COPY src /app/src
COPY configs /app/configs
COPY tests /app/tests
COPY notebooks /app/notebooks
COPY scripts /app/scripts

EXPOSE 8888
CMD ["python", "-m", "src.env_check", "--config", "configs/base.yaml", "--skip-weights", "--skip-dataset"]
