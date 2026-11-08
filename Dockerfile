# Citra pipeline (CPU). PRD §13: versi dasar dikunci dengan digest, dependensi dari requirements.lock.
# BASE_IMAGE diisi dengan tag@sha256:... oleh scripts/build.ps1 setelah digest dibaca dari hasil pull.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

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
COPY requirements.lock /app/requirements.lock
RUN pip install -r /app/requirements.lock

# Kode disalin agar citra mandiri; run.ps1 menimpanya dengan bind mount saat pengembangan.
COPY src /app/src
COPY configs /app/configs
COPY tests /app/tests
COPY notebooks /app/notebooks

EXPOSE 8888
CMD ["python", "-m", "src.env_check", "--config", "configs/base.yaml", "--skip-weights", "--skip-dataset"]
