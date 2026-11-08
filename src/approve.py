"""Pencatatan keputusan dan persetujuan MANUSIA (bukan otomatis). Setiap perintah menulis berkas/baris yang terlihat
dan menambah satu baris ke LOG.md. Program tidak pernah memanggil ini sendiri.

  python -m src.approve decide D2 D3 D4 --by NAMA          # konfirmasi keputusan di configs/decisions.yaml
  python -m src.approve roi --by NAMA                       # kunci parameter Frangi setelah memeriksa PNG (CP2)
  python -m src.approve gan-visual CP4 --by NAMA            # pemeriksaan visual GAN (CP4) / mode collapse (CP5)
  python -m src.approve review CP1 --by NAMA                # tinjauan orang kedua untuk checkpoint (RULES §3)
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

from src.config import P, load_merged

DEC = Path("configs/decisions.yaml")
BASE = Path("configs/base.yaml")
LOG = Path("LOG.md")


def _log(who: str, what: str) -> None:
    line = f"| {dt.date.today().isoformat()} | {who} | {what} | disetujui | via src.approve |\n"
    if LOG.exists():
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line)


def decide(ids: list[str], by: str) -> None:
    text = DEC.read_text(encoding="utf-8")
    for d in ids:
        pat = re.compile(rf"(^{d}:\n  status: )menunggu_konfirmasi", re.M)
        if not pat.search(text):
            print(f"{d}: tidak berstatus menunggu_konfirmasi (dilewati)")
            continue
        text = pat.sub(rf"\1diputuskan\n  dikonfirmasi_oleh: {by}\n  tanggal_konfirmasi: {dt.date.today().isoformat()}", text)
        _log(by, f"Konfirmasi keputusan {d}")
        print(f"{d}: diputuskan oleh {by}")
    DEC.write_text(text, encoding="utf-8")


def roi(by: str, out_root: str | None) -> None:
    t = BASE.read_text(encoding="utf-8")
    t2 = re.sub(r"(  roi:.*?\n    locked: )false", r"\1true", t, count=1, flags=re.S)
    if t2 == t and "locked: true" not in t:
        raise SystemExit("preprocess.roi.locked tidak ditemukan di base.yaml")
    BASE.write_text(t2, encoding="utf-8")
    cfg = load_merged(BASE)
    f = P(cfg, "cp2") / "ROI_APPROVED"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"{by} {dt.datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
    _log(by, "Parameter Frangi dikunci setelah pemeriksaan visual 50 citra (FR-03)")
    print("preprocess.roi.locked=true; ditulis", f)


def flag(cp: str, name: str, by: str, what: str) -> None:
    cfg = load_merged(BASE)
    f = P(cfg, cp.lower()) / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"{by} {dt.datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
    _log(by, f"{what} ({cp})")
    print("ditulis", f)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("decide"); a.add_argument("ids", nargs="+"); a.add_argument("--by", required=True)
    b = sub.add_parser("roi"); b.add_argument("--by", required=True)
    c = sub.add_parser("gan-visual"); c.add_argument("cp", choices=["CP4", "CP5"]); c.add_argument("--by", required=True)
    d = sub.add_parser("review"); d.add_argument("cp", choices=[f"CP{i}" for i in range(8)]); d.add_argument("--by", required=True)
    args = ap.parse_args()
    if args.cmd == "decide":
        decide(args.ids, args.by)
    elif args.cmd == "roi":
        roi(args.by, None)
    elif args.cmd == "gan-visual":
        flag(args.cp, "GAN_VISUAL_APPROVED" if args.cp == "CP4" else "MODE_COLLAPSE_CHECKED", args.by,
             "Pemeriksaan visual sampel GAN" if args.cp == "CP4" else "Pemeriksaan mode collapse GAN")
    elif args.cmd == "review":
        flag(args.cp, "REVIEWED", args.by, "Tinjauan orang kedua")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
