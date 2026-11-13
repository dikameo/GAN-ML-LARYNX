"""Orkestrasi job (FR-12): meluncurkan `python -m <modul> ...` sebagai proses terpisah.

- Idempoten (modul melewati job DONE) dan dapat dilanjutkan dari last.pt.
- Paralel task-parallel (PRD §8): `devices=[0,1,...]` menjalankan satu rantai per GPU lewat CUDA_VISIBLE_DEVICES.
- Rantai GAN: GAN lipatan k -> klasifikator tiap rasio lipatan k (urutan wajib).
- Job yang gagal dua kali berturut-turut menghentikan antrean dan dicatat (RULES §4); tidak ada seed yang diganti
  atau lipatan yang dibuang.
"""
from __future__ import annotations

import datetime as dt
import os
import queue
import subprocess
import sys
import threading
import time

from src.config import P, load_merged
from src.jobio import job_dir
from src.splits import load_splits

Chain = list[tuple[str, list[str]]]


def job_status(cfg: dict, scenario: str) -> list[dict]:
    splits = load_splits(cfg)
    rows = []
    for r in splits["repeats"]:
        for f in r["folds"]:
            d = job_dir(cfg, scenario, r["rep"], f["fold"])
            rows.append({"rep": r["rep"], "fold": f["fold"], "DONE": (d / "DONE").exists(), "ada_last_pt": (d / "last.pt").exists()})
    return rows


def _argv(module: str, config: str, rep: int, fold: int, ratio: float | None = None) -> list[str]:
    a = [sys.executable, "-m", module, "--config", config, "--rep", str(rep), "--fold", str(fold)]
    return a + (["--ratio", str(ratio)] if ratio is not None else [])


def run_chains(cfg: dict, chains: list[Chain], devices: list[int] | None = None, max_attempts: int = 2) -> list[dict]:
    log = P(cfg, "jobs", "launch.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue()
    for c in chains:
        q.put(c)
    stop, lock = threading.Event(), threading.Lock()
    results: list[dict] = []

    def note(line: str) -> None:
        with lock, open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{line}\n")

    def worker(dev):
        env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": str(dev)} if dev is not None else {})}
        while not stop.is_set():
            try:
                chain = q.get_nowait()
            except queue.Empty:
                return
            for label, argv in chain:
                if stop.is_set():
                    return
                t0, rc, attempts = time.time(), 1, 0
                while rc != 0 and attempts < max_attempts:
                    attempts += 1
                    rc = subprocess.run(argv, env=env).returncode
                    note(f"{label}\tgpu={dev}\tpercobaan={attempts}\tkode={rc}\t{time.time() - t0:.0f}s")
                with lock:
                    results.append({"job": label, "kode": rc, "percobaan": attempts})
                if rc != 0:
                    stop.set()
                    return

    threads = [threading.Thread(target=worker, args=(d,)) for d in (devices or [None])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [r for r in results if r["kode"] != 0]
    if bad:
        raise RuntimeError(f"Job gagal {bad[0]['percobaan']}x berturut-turut: {bad[0]['job']}. Antrean dihentikan. "
                           f"Catat penyebabnya di LOG.md sebelum dilanjutkan (RULES §4). Lihat {log}.")
    return results


def _selected(splits: dict, reps, folds):
    for r in splits["repeats"]:
        if reps is not None and r["rep"] not in reps:
            continue
        for f in r["folds"]:
            if folds is None or f["fold"] in folds:
                yield r["rep"], f["fold"]


def launch_scenario(config: str, module: str, reps: list[int] | None = None, folds: list[int] | None = None,
                    devices: list[int] | None = None) -> list[dict]:
    """Satu job per (rep, lipatan): S1 (src.train), S2 (src.train_s2), S3/S4 (src.train_s3), S7_S1 (src.train)."""
    cfg = load_merged(config)
    sc = cfg["scenario"]
    chains = [[(f"{sc}/rep{r}_fold{k}", _argv(module, config, r, k))] for r, k in _selected(load_splits(cfg), reps, folds)]
    return run_chains(cfg, chains, devices)


def launch_gan_chain(config: str, ratios: list | None = None, reps: list[int] | None = None,
                     folds: list[int] | None = None, devices: list[int] | None = None) -> list[dict]:
    """S5/S6: GAN lipatan k, lalu klasifikator untuk tiap rasio. ratios=[None] untuk S7 (rasio 'best' dari konfigurasi)."""
    cfg = load_merged(config)
    sc = cfg["scenario"]
    ratios = ratios if ratios is not None else cfg["gan"]["ratios"]
    chains = []
    for r, k in _selected(load_splits(cfg), reps, folds):
        ch: Chain = [(f"{sc}_gan/rep{r}_fold{k}", _argv("src.train_gan", config, r, k))]
        for ra in ratios:
            tag = f"_r{int(round(ra * 100))}" if ra is not None else ""
            ch.append((f"{sc}{tag}/rep{r}_fold{k}", _argv("src.train", config, r, k, ra)))
        chains.append(ch)
    return run_chains(cfg, chains, devices)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Luncurkan job semua (rep x lipatan) untuk satu skenario")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", required=True)
        p.add_argument("--reps", type=int, nargs="*", default=None, help="batasi ulangan yang diluncurkan (default semua)")
        p.add_argument("--folds", type=int, nargs="*", default=None, help="batasi lipatan yang diluncurkan (default semua)")
        p.add_argument("--devices", type=int, nargs="*", default=None, help="nomor GPU; satu rantai per GPU")

    s = sub.add_parser("scenario", help="S1/S2/S3/S4/S7_S1: satu job per (rep, lipatan)")
    common(s)
    s.add_argument("--module", required=True, help="src.train | src.train_s2 | src.train_s3")
    g = sub.add_parser("gan-chain", help="S5/S6/S7_S5/S7_S6: GAN lipatan k lalu klasifikator tiap rasio")
    common(g)
    g.add_argument("--ratios", nargs="*", default=None, help="mis. 0.25 0.5 1.0 (default dari konfigurasi); 'none' untuk S7")
    st = sub.add_parser("status", help="tampilkan status DONE per job")
    st.add_argument("--config", required=True)
    st.add_argument("--scenario", default=None, help="mis. S5_r25 atau S6_gan (default: skenario di konfigurasi)")
    a = ap.parse_args()
    if a.cmd == "scenario":
        launch_scenario(a.config, a.module, a.reps, a.folds, a.devices)
    elif a.cmd == "gan-chain":
        ratios = None if a.ratios is None else [None if r.lower() == "none" else float(r) for r in a.ratios]
        launch_gan_chain(a.config, ratios, a.reps, a.folds, a.devices)
    else:
        cfg = load_merged(a.config)
        sc = a.scenario or cfg["scenario"]
        rows = job_status(cfg, sc)
        for r in rows:
            print(f"{sc} rep{r['rep']} fold{r['fold']}: {'DONE' if r['DONE'] else 'belum'}")
        print(f"{sum(r['DONE'] for r in rows)}/{len(rows)} selesai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
