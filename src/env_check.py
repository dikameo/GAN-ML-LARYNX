"""Uji lingkungan CP0: versi pustaka, perangkat, dataset terpasang baca-saja, bobot Swin-T bisa dimuat.

Jalankan: python -m src.env_check --config configs/base.yaml [--skip-weights]
Keluar dengan kode 1 bila ada pemeriksaan yang gagal. Tidak ada hasil yang dipalsukan:
pemeriksaan yang tidak bisa dijalankan dilaporkan sebagai GAGAL atau DILEWATI, bukan LULUS.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

from src.utils import atomic_write_json, load_config

REQUIRED = ["torch", "torchvision", "transformers", "numpy", "pandas", "scipy", "sklearn",
            "imblearn", "skimage", "PIL", "yaml", "openpyxl", "matplotlib", "psutil", "jupyterlab"]


def check_imports() -> dict:
    out = {}
    for name in REQUIRED:
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            out[name] = f"GAGAL: {e!r}"
    return out


def check_device() -> dict:
    import torch

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
    }


def check_dataset_readonly(dataset_dir: str) -> dict:
    p = Path(dataset_dir)
    if not p.is_dir():
        return {"status": "GAGAL", "detail": f"folder tidak ada: {dataset_dir}"}
    probe = p / ".write_probe"
    try:
        probe.write_text("x")
        probe.unlink()
        return {"status": "GAGAL", "detail": "dataset BISA ditulis; harus dipasang :ro (RULES §2)"}
    except OSError:
        return {"status": "LULUS", "detail": "dataset baca-saja"}


def check_weights(hf_name: str, cache_dir: str) -> dict:
    try:
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(
            hf_name, cache_dir=cache_dir, num_labels=2, ignore_mismatched_sizes=True)
        n = sum(p.numel() for p in model.parameters())
        return {"status": "LULUS", "detail": f"bobot {hf_name} termuat, {n:,} parameter"}
    except Exception as e:  # noqa: BLE001
        return {"status": "GAGAL", "detail": f"bobot tidak dapat dimuat/diunduh: {e!r}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-weights", action="store_true", help="lewati pengunduhan bobot (tanpa jaringan)")
    ap.add_argument("--skip-dataset", action="store_true", help="lewati cek dataset (mis. saat uji tiruan)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    report = {"python": sys.version.split()[0], "platform": platform.platform(),
              "git_commit": os.environ.get("GIT_COMMIT", "tidak-diketahui"),
              "image_digest": os.environ.get("IMAGE_DIGEST", "tidak-diketahui")}
    report["imports"] = check_imports()
    report["device"] = check_device()
    report["dataset"] = ({"status": "DILEWATI"} if args.skip_dataset
                         else check_dataset_readonly(cfg["paths"]["dataset"]))
    report["weights"] = ({"status": "DILEWATI"} if args.skip_weights
                         else check_weights(cfg["model"]["hf_name"], cfg["paths"]["hf_cache"]))

    failed = [k for k, v in report["imports"].items() if str(v).startswith("GAGAL")]
    failed += [k for k in ("dataset", "weights") if report[k]["status"] == "GAGAL"]
    report["lulus"] = not failed
    report["gagal"] = failed

    for k, v in report.items():
        print(f"{k}: {v}")
    if args.out:
        atomic_write_json(args.out, report)
    return 0 if report["lulus"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
