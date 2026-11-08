"""D2 / CP4: pilot lambda_roi pada satu lipatan, DIPILIH dari pasien VALIDASI di dalam lipatan latih.

Aturan pemilihan ditetapkan sebelum eksperimen (pilot.criterion di konfigurasi): lambda dengan FID validasi terendah;
seri -> lambda lebih kecil. Citra uji lipatan itu TIDAK dipakai. Hasil: outputs/cp4/lambda_selection.json.

Jalankan: python -m src.pilot_lambda --config configs/s6.yaml
"""
from __future__ import annotations

import argparse
import copy

from src.config import P, load_merged
from src.jobio import job_dir
from src.train_gan import run_gan_job
from src.utils import atomic_write_json


def run_pilot(cfg: dict) -> dict:
    pl = cfg["pilot"]
    if pl["criterion"] != "val_fid":
        raise ValueError("kriteria pilot yang didukung: val_fid")
    rows = []
    for lam in pl["lambdas"]:
        c = copy.deepcopy(cfg)
        c["gan"]["lambda_roi"] = lam
        d = job_dir(c, "pilot", pl["rep"], pl["fold"]) / f"lam{lam}"
        m = run_gan_job(c, pl["rep"], pl["fold"], float(lam), d, mode="pilot", total_kimg=pl["total_kimg"])
        rows.append({"lambda_roi": lam, "fid_val": m["fid_all"], "fid_val_malignant": m.get("fid_malignant"),
                     "vessel_w1_malignant": m.get("vessel_w1_malignant"),
                     "lpips_diversity_malignant": m.get("lpips_diversity_malignant"), "dir": str(d)})
    best = min(rows, key=lambda r: (r["fid_val"], r["lambda_roi"]))
    sel = {"chosen": best["lambda_roi"], "criterion": pl["criterion"], "table": rows,
           "source": f"pasien validasi dalam lipatan latih (rep{pl['rep']}, lipatan {pl['fold']}); uji tidak dipakai",
           "peringatan": "FID Inception-ImageNet kurang andal pada CE-NBI (PRD §11); periksa juga grid sampel dan LPIPS"}
    atomic_write_json(P(cfg, "cp4", "lambda_selection.json"), sel)
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    sel = run_pilot(load_merged(a.config))
    print("lambda_roi terpilih:", sel["chosen"])
    for r in sel["table"]:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
