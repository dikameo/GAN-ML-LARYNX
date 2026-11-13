"""Peluncur skrip dari notebook. Pelatihan dan pekerjaan berat TIDAK dijalankan di dalam sel;
sel hanya memanggil fungsi ini yang menjalankan `python -m ...` sebagai proses terpisah, keluarannya dialirkan langsung."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_NOISE = ("Loading weights", "it/s]")


def run_module(module: str, args: list[str], skip_if_exists: str | Path | None = None) -> int:
    """Jalankan `python -m module args`, tampilkan keluarannya baris demi baris. Dilewati bila keluaran sudah ada (idempoten)."""
    if skip_if_exists is not None and Path(skip_if_exists).exists():
        print(f"[dilewati] {skip_if_exists} sudah ada")
        return 0
    cmd = [sys.executable, "-u", "-m", module, *args]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if not any(n in line for n in _NOISE):
            print(line, end="", flush=True)
    return proc.wait()
