"""Menjalankan tahap demi tahap (notebook 00..07) dan BERHENTI di checkpoint pertama yang belum lulus.

  python -m src.run_all --stage all                # dari awal; idempoten, job DONE dilewati
  python -m src.run_all --from cp3 --to cp5
  python -m src.run_all --stage cp2

Checkpoint punya item yang hanya bisa diputuskan manusia (D2-D8, parameter ROI, kualitas visual GAN, tinjauan orang
kedua). Program berhenti dan mencetak perintah `python -m src.approve ...` yang diperlukan; setelah itu jalankan ulang.
Kode keluar: 0 semua lulus, 2 berhenti di checkpoint belum lulus, 1 notebook gagal.

--self-review NAMA menandai REVIEWED untuk tahap yang sedang berjalan SEBELUM hasil dilihat. Itu melewati ketentuan
RULES §3 (orang kedua); pakai hanya bila kamu sadar akan konsekuensinya. Tercatat di berkas REVIEWED.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

from src.config import P, load_merged
from src.run_notebook import main as run_nb

STAGES = [("cp0", "00_setup"), ("cp1", "01_audit"), ("cp2", "02_praproses_split"), ("cp3", "03_baseline"),
          ("cp4", "04_pilot_gan"), ("cp5", "05_s5_s6"), ("cp6", "06_s7"), ("cp7", "07_evaluasi_akhir")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="cp0..cp7 atau all")
    ap.add_argument("--from", dest="start", default="cp0")
    ap.add_argument("--to", dest="end", default="cp7")
    ap.add_argument("--self-review", default=None, metavar="NAMA")
    a = ap.parse_args()
    names = [s for s, _ in STAGES]
    if a.stage and a.stage != "all":
        a.start = a.end = a.stage
    todo = STAGES[names.index(a.start): names.index(a.end) + 1]
    cfg = load_merged("configs/base.yaml")
    for cp, nb in todo:
        print(f"\n{'=' * 70}\n{cp.upper()}  <-  notebooks/{nb}.ipynb\n{'=' * 70}", flush=True)
        d = P(cfg, cp)
        if a.self_review:
            d.mkdir(parents=True, exist_ok=True)
            (d / "REVIEWED").write_text(f"SELF-REVIEW oleh {a.self_review} {dt.datetime.now().isoformat(timespec='seconds')} "
                                        f"(bukan orang kedua; RULES §3)\n", encoding="utf-8")
        if run_nb(f"notebooks/{nb}.ipynb") != 0:
            print(f"\nNotebook {nb} gagal. Lihat keluaran di notebooks/{nb}.ipynb")
            return 1
        rp = d / f"{cp}_check.json"
        if not rp.exists():
            print(f"\n{cp.upper()}: {rp} tidak ditulis; periksa sel terakhir notebook.")
            return 1
        rep = json.loads(rp.read_text(encoding="utf-8"))
        if not rep["lulus"]:
            print(f"\n>>> BERHENTI: {cp.upper()} belum lulus. Penghambat:")
            for x in rep["penghambat"]:
                print("   -", x)
            for i in rep["items"]:
                if i["status"] in ("MENUNGGU", "GAGAL"):
                    print(f"     [{i['status']}] {i['item']}: {i['detail']}")
            print("\nSelesaikan penghambat (bila perlu dengan python -m src.approve ...), lalu jalankan ulang perintah yang sama.")
            return 2
        print(f"\n{cp.upper()} LULUS.")
    print("\nSemua tahap yang diminta lulus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
