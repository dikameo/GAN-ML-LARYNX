"""Menjalankan notebook dari awal sampai akhir dan menyimpan hasilnya di tempat.
Jalankan: python -m src.run_notebook notebooks/00_setup.ipynb
Direktori kerja kernel = akar repositori (configs/base.yaml ditemukan dari direktori kerja ke atas)."""
from __future__ import annotations

import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src.config import repo_root


def main(path: str) -> int:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(repo_root())}})
    code = 0
    try:
        client.execute()
    except CellExecutionError as e:
        print("Notebook gagal pada sel:\n", str(e)[-1500:])
        code = 1
    except Exception as e:  # noqa: BLE001
        print("Notebook gagal:", repr(e)[:500])
        code = 1
    nbformat.write(nb, path)  # simpan keluaran apa adanya, termasuk saat gagal
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
