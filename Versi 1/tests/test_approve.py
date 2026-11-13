"""Uji pencatatan persetujuan manusia (src.approve) pada salinan di direktori sementara."""
import shutil
from pathlib import Path

import pytest

from src import approve
from src.config import load_merged


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    for f in ("decisions.yaml", "base.yaml"):
        shutil.copy(Path(__file__).resolve().parent.parent / "configs" / f, tmp_path / "configs" / f)
    (tmp_path / "LOG.md").write_text("| tanggal | siapa | kejadian | hasil | keputusan |\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LARYNX_OUTPUTS", str(tmp_path / "out"))
    return tmp_path


def test_decide_changes_only_requested_and_logs(sandbox):
    before = (sandbox / "configs" / "decisions.yaml").read_text(encoding="utf-8")
    approve.decide(["D4"], "Tester")
    after = (sandbox / "configs" / "decisions.yaml").read_text(encoding="utf-8")
    assert "dikonfirmasi_oleh: Tester" in after and after.count("menunggu_konfirmasi") == before.count("menunggu_konfirmasi") - 1
    assert "D4" in (sandbox / "LOG.md").read_text(encoding="utf-8")
    approve.decide(["D4"], "Tester")  # sudah diputuskan -> tidak berubah
    assert (sandbox / "configs" / "decisions.yaml").read_text(encoding="utf-8") == after


def test_roi_locks_parameters_and_writes_flag(sandbox):
    assert load_merged("configs/base.yaml")["preprocess"]["roi"]["locked"] is False
    approve.roi("Tester", None)
    assert load_merged("configs/base.yaml")["preprocess"]["roi"]["locked"] is True
    assert (sandbox / "out" / "cp2" / "ROI_APPROVED").exists()


def test_review_and_gan_visual_flags(sandbox):
    approve.flag("CP3", "REVIEWED", "Tester", "Tinjauan")
    approve.flag("CP4", "GAN_VISUAL_APPROVED", "Tester", "Visual")
    assert (sandbox / "out" / "cp3" / "REVIEWED").exists() and (sandbox / "out" / "cp4" / "GAN_VISUAL_APPROVED").exists()
