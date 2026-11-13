"""FR-01: audit data dari gambar lokal -> audit.csv.

Jalankan: python -m src.audit --config configs/base.yaml
Semua angka dihitung dari gambar di disk (ukuran dibaca dari header). Selisih terhadap Tabel 3.2
dilaporkan apa adanya, tidak ada yang disesuaikan.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import P, load_merged
from src.utils import atomic_write_bytes, atomic_write_json

IMG_EXT = {".jpg", ".jpeg", ".png"}
CLASSES = (("Benign", 0), ("Malignant", 1))


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def scan_dataset(root: str | Path) -> pd.DataFrame:
    """Satu baris per citra: Benign|Malignant / <histopatologi> / <PatientXXX> / berkas."""
    root = Path(root)
    files = []
    for class_name, label in CLASSES:
        cdir = root / class_name
        for histo in sorted(p for p in cdir.iterdir() if p.is_dir()):
            for pdir in sorted(p for p in histo.iterdir() if p.is_dir()):
                for f in sorted(pdir.iterdir()):
                    if f.suffix.lower() in IMG_EXT:
                        files.append((f, class_name, label, histo.name, pdir.name))
    with ThreadPoolExecutor(8) as ex:
        sizes = list(ex.map(lambda t: _size(t[0]), files))
    rows = [dict(rel_path=str(f.relative_to(root)).replace("\\", "/"), label=label, class_name=cn,
                 histo=h, patient_id=pid, width=w, height=hh, bytes=f.stat().st_size)
            for (f, cn, label, h, pid), (w, hh) in zip(files, sizes)]
    return pd.DataFrame(rows)


def assign_uid(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """ID unik pasien. Bila satu nama folder pasien muncul di kedua kelas, diberi awalan kelas dan dilaporkan."""
    labels_per_id = df.groupby("patient_id")["label"].nunique()
    collisions = sorted(labels_per_id[labels_per_id > 1].index)
    df = df.copy()
    df["patient_uid"] = [f"{c}_{p}" if p in collisions else p for c, p in zip(df.class_name, df.patient_id)]
    return df, collisions


def patients_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["resolution"] = df.width.astype(str) + "x" + df.height.astype(str)
    g = df.groupby("patient_uid")
    t = pd.DataFrame({
        "label": g["label"].first(),
        "class_name": g["class_name"].first(),
        "histo": g["histo"].first(),
        "n_histo_folders": g["histo"].nunique(),
        "n_images": g.size(),
        "n_resolutions": g["resolution"].nunique(),
        "resolution": g["resolution"].agg(lambda s: s.value_counts().index[0]),
    })
    t["label_consistent"] = g["label"].nunique() == 1
    return t.reset_index()


def reference_comparison(pt: pd.DataFrame, img: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    loc = {
        ("patients", "benign"): int(((pt.label == 0)).sum()), ("patients", "malignant"): int(((pt.label == 1)).sum()),
        ("images", "benign"): int((img.label == 0).sum()), ("images", "malignant"): int((img.label == 1).sum()),
    }
    rows = []
    for (unit, cls), v in loc.items():
        r = {"unit": unit, "kelas": cls, "lokal": v}
        for name in ("reference_tabel_3_2", "reference_kiani"):
            ref = cfg["audit"][name][unit][cls]
            r[name] = ref
            r[f"selisih_{name}"] = v - ref
        rows.append(r)
    rows.append({"unit": "images", "kelas": "total", "lokal": len(img),
                 "reference_tabel_3_2": sum(cfg["audit"]["reference_tabel_3_2"]["images"].values()),
                 "reference_kiani": sum(cfg["audit"]["reference_kiani"]["images"].values())})
    rows[-1]["selisih_reference_tabel_3_2"] = rows[-1]["lokal"] - rows[-1]["reference_tabel_3_2"]
    rows[-1]["selisih_reference_kiani"] = rows[-1]["lokal"] - rows[-1]["reference_kiani"]
    return pd.DataFrame(rows)


def s7_check(pt: pd.DataFrame, min_per_class: int) -> dict:
    """Syarat S7 (PRD §6): latih pada resolusi mayoritas, uji pada gabungan resolusi lain; minimal N pasien per kelas."""
    single = pt[pt.n_resolutions == 1]
    per_res = []
    for res, g in single.groupby("resolution"):
        b, m = int((g.label == 0).sum()), int((g.label == 1).sum())
        per_res.append({"resolution": res, "benign_patients": b, "malignant_patients": m,
                        "images": int(g.n_images.sum()), "malignant_fraction": round(m / max(b + m, 1), 3),
                        "memenuhi_syarat": b >= min_per_class and m >= min_per_class})
    per_res.sort(key=lambda r: -(r["benign_patients"] + r["malignant_patients"]))
    out = {"per_resolusi": per_res, "pasien_multi_resolusi": int((pt.n_resolutions > 1).sum())}
    if per_res:
        major = per_res[0]["resolution"]
        minor = single[single.resolution != major]
        b, m = int((minor.label == 0).sum()), int((minor.label == 1).sum())
        out["mayoritas"] = per_res[0]
        out["minor_gabungan"] = {"benign_patients": b, "malignant_patients": m,
                                 "memenuhi_syarat": b >= min_per_class and m >= min_per_class}
        out["s7_dapat_dijalankan"] = bool(per_res[0]["memenuhi_syarat"] and out["minor_gabungan"]["memenuhi_syarat"]
                                          and out["pasien_multi_resolusi"] == 0)
    return out


def _pid_num(s) -> int | None:
    m = re.findall(r"\d+", str(s))
    return int(m[-1]) if m else None


def cross_check_xlsx(xlsx_path: str | Path, pt: pd.DataFrame) -> dict:
    """Bandingkan daftar pasien/label di xlsx dengan folder. Kolom dideteksi dari nama; yang tidak terdeteksi dilaporkan."""
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None)
    except Exception as e:  # noqa: BLE001
        return {"status": "tidak_dapat_dibaca", "detail": repr(e)}
    out = {"status": "dibaca", "sheet": {k: list(map(str, v.columns)) for k, v in sheets.items()}}
    df = max(sheets.values(), key=len)
    cols = {str(c).lower(): c for c in df.columns}
    pcol = next((c for k, c in cols.items() if "patient" in k or k in ("id", "pid", "no")), None)
    lcol = next((c for k, c in cols.items() if any(w in k for w in ("class", "label", "malig", "benign", "diagn", "type", "categ"))), None)
    out["kolom_pasien"], out["kolom_label"] = (str(pcol) if pcol is not None else None), (str(lcol) if lcol is not None else None)
    if pcol is None:
        out["detail"] = "kolom ID pasien tidak terdeteksi; pemeriksaan silang dilewati"
        return out
    x_ids = {_pid_num(v) for v in df[pcol].dropna()}
    f_ids = {_pid_num(v) for v in pt.patient_uid}
    out.update(pasien_xlsx=len(x_ids), pasien_folder=len(f_ids),
               hanya_di_xlsx=sorted(i for i in x_ids - f_ids if i is not None),
               hanya_di_folder=sorted(i for i in f_ids - x_ids if i is not None))
    if lcol is not None:
        def to_label(v):
            s = str(v).lower()
            return 1 if ("malig" in s or s.strip() in ("1", "ganas")) else (0 if ("benign" in s or "jinak" in s or s.strip() == "0") else None)
        xl = {_pid_num(a): to_label(b) for a, b in zip(df[pcol], df[lcol]) if _pid_num(a) is not None}
        fl = {_pid_num(a): b for a, b in zip(pt.patient_uid, pt.label)}
        diff = sorted(i for i in xl if i in fl and xl[i] is not None and xl[i] != fl[i])
        out["label_beda_xlsx_vs_folder"] = diff
        out["label_xlsx_tidak_terbaca"] = sum(v is None for v in xl.values())
    else:
        out["detail"] = "kolom label tidak terdeteksi; label tidak dibandingkan"

    # Jumlah citra dan resolusi per pasien (kolom 'Number of Images' dan 'Image Size')
    ncol = next((c for k, c in cols.items() if "number" in k and "image" in k), None)
    scol = next((c for k, c in cols.items() if "size" in k), None)
    fp = {_pid_num(a): r for a, r in zip(pt.patient_uid, pt.itertuples())}
    if ncol is not None:
        xn = {_pid_num(a): b for a, b in zip(df[pcol], df[ncol]) if _pid_num(a) is not None and pd.notna(b)}
        out["jumlah_citra_beda"] = sorted(
            [{"pasien": i, "xlsx": int(xn[i]), "folder": int(fp[i].n_images)}
             for i in xn if i in fp and int(xn[i]) != int(fp[i].n_images)], key=lambda d: d["pasien"])
        out["jumlah_citra_total_xlsx"] = int(sum(xn.values()))
    if scol is not None:
        def norm(v):
            m = re.findall(r"\d+", str(v))
            return f"{m[0]}x{m[1]}" if len(m) >= 2 else None
        xs = {_pid_num(a): norm(b) for a, b in zip(df[pcol], df[scol]) if _pid_num(a) is not None and pd.notna(b)}
        out["resolusi_beda"] = sorted(
            [{"pasien": i, "xlsx": xs[i], "folder": fp[i].resolution}
             for i in xs if i in fp and xs[i] != fp[i].resolution], key=lambda d: d["pasien"])
    return out


def run_audit(cfg: dict, out_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir or P(cfg, "cp1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["paths"]["dataset"])
    img = scan_dataset(root)
    img, collisions = assign_uid(img)
    pt = patients_table(img)
    ref = reference_comparison(pt, img, cfg)
    s7 = s7_check(pt, cfg["audit"]["s7_min_patients_per_class"])
    xlsx = root / "Patients_List_Updated_Final.xlsx"
    xc = cross_check_xlsx(xlsx, pt) if xlsx.exists() else {"status": "tidak_ada"}

    atomic_write_bytes(out_dir / "audit.csv", pt.to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(out_dir / "audit_images.csv", img.to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(out_dir / "audit_reference.csv", ref.to_csv(index=False).encode("utf-8"))
    summary = {
        "n_images": len(img), "n_patients": len(pt),
        "id_bertabrakan_antar_kelas": collisions,
        "pasien_label_tidak_konsisten": int((~pt.label_consistent).sum()),
        "pasien_di_lebih_dari_satu_folder_histo": int((pt.n_histo_folders > 1).sum()),
        "perbandingan_referensi": ref.to_dict(orient="records"),
        "s7": s7, "xlsx": xc,
        "resolusi_per_citra": img.assign(r=img.width.astype(str) + "x" + img.height.astype(str)).r.value_counts().to_dict(),
    }
    atomic_write_json(out_dir / "audit_summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    s = run_audit(load_merged(args.config))
    print(f"citra={s['n_images']} pasien={s['n_patients']} tabrakan_id={s['id_bertabrakan_antar_kelas']}")
    print(pd.DataFrame(s["perbandingan_referensi"]).to_string(index=False))
    print("S7 dapat dijalankan:", s["s7"].get("s7_dapat_dijalankan"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
