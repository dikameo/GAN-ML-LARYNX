"""Uji utilitas CP0. Hanya membuktikan kode berjalan dengan benar pada data tiruan kecil."""
import json

import pytest

from src.utils import (atomic_write_bytes, config_hash, is_done, job_seed, load_config, mark_done,
                       write_manifest)


def test_config_hash_stable_and_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_job_seed_deterministic_and_distinct():
    assert job_seed(42, 0, "S1") == job_seed(42, 0, "S1")
    assert job_seed(42, 0, "S1") != job_seed(42, 1, "S1")


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "x" / "f.bin"
    atomic_write_bytes(p, b"abc")
    assert p.read_bytes() == b"abc"
    assert [q.name for q in p.parent.iterdir()] == ["f.bin"]


def test_done_marker_respects_config_hash(tmp_path):
    assert not is_done(tmp_path, "h1")
    mark_done(tmp_path, "h1")
    assert is_done(tmp_path, "h1")
    assert not is_done(tmp_path, "h2")  # konfigurasi berubah -> job dijalankan ulang


def test_manifest_hashes_files_and_fails_on_missing(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    mp = write_manifest(tmp_path, [tmp_path / "a.txt"])
    data = json.loads(mp.read_text())
    assert "a.txt" in data["files"] and len(data["files"]["a.txt"]["sha256"]) == 64
    with pytest.raises(FileNotFoundError):
        write_manifest(tmp_path, [tmp_path / "tidak_ada.txt"])


def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg["train"]["batch_size"] == 32 and cfg["train"]["max_epochs"] == 30
