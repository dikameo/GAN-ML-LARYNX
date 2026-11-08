"""Utilitas umum: konfigurasi, seed, penulisan atomik, penanda DONE, manifest (FR-12)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash stabil dari isi konfigurasi (urutan kunci tidak berpengaruh)."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def job_seed(base_seed: int, *parts: Any) -> int:
    """Seed deterministik per job dari (seed dasar, lipatan, skenario, ...). Dicatat di tiap job."""
    key = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Tulis ke berkas sementara di direktori yang sama, lalu ganti nama (atomik)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def atomic_torch_save(obj: Any, path: str | os.PathLike) -> None:
    import io

    import torch

    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_write_bytes(path, buf.getvalue())


def is_done(job_dir: str | os.PathLike, cfg_hash: str) -> bool:
    """Job dilewati hanya bila DONE ada DAN hash konfigurasinya sama (RULES §3)."""
    done = Path(job_dir) / "DONE"
    return done.exists() and done.read_text(encoding="utf-8").strip() == cfg_hash


def mark_done(job_dir: str | os.PathLike, cfg_hash: str) -> None:
    atomic_write_bytes(Path(job_dir) / "DONE", cfg_hash.encode("utf-8"))


def write_manifest(out_dir: str | os.PathLike, files: list[str | os.PathLike], extra: dict | None = None) -> Path:
    """manifest.json: daftar keluaran beserta hash (PRD §13). Berkas harus sudah ada."""
    out_dir = Path(out_dir)
    entries = {}
    for f in files:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Keluaran checkpoint tidak ditemukan: {p}")
        entries[str(p.relative_to(out_dir)) if p.is_relative_to(out_dir) else str(p)] = {
            "sha256": file_sha256(p),
            "bytes": p.stat().st_size,
        }
    manifest = {"files": entries, **(extra or {})}
    mp = out_dir / "manifest.json"
    atomic_write_json(mp, manifest)
    return mp
