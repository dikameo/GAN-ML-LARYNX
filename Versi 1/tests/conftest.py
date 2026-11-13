"""Fixture data TIRUAN. Uji ini hanya membuktikan kode berjalan dan aturan kebocoran ditegakkan;
tidak ada klaim tentang kinerja pada data asli."""
import copy

import numpy as np
import pytest
from PIL import Image

from src.config import load_merged

N_BENIGN, N_MALIGNANT, N_IMG = 30, 15, 3
SIZES = [(160, 128), (200, 150)]  # dua "resolusi" tiruan


def _fake_image(rng, w, h, malignant: bool) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(w, h) * 0.45) ** 2
    img = np.zeros((h, w, 3), np.float32)
    base = np.array([150, 90, 90] if malignant else [140, 110, 110], np.float32)
    img[inside] = base + rng.normal(0, 12, (inside.sum(), 3))
    for _ in range(3):  # garis gelap mirip pembuluh
        x0 = rng.integers(w // 4, 3 * w // 4)
        img[:, x0:x0 + 2][inside[:, x0:x0 + 2]] *= 0.55
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    ds, out = root / "ds", root / "out"
    rng = np.random.default_rng(0)
    pid = 0
    for cname, n, histo in (("Benign", N_BENIGN, "Cyst"), ("Malignant", N_MALIGNANT, "SCC")):
        for _ in range(n):
            pid += 1
            w, h = SIZES[pid % 2]
            d = ds / cname / histo / f"Patient{pid:03d}"
            d.mkdir(parents=True)
            for k in range(N_IMG):
                _fake_image(rng, w, h, cname == "Malignant").save(d / f"P{pid:03d} ({k + 1}).jpg", quality=95)

    cfg = load_merged("configs/s1.yaml")
    cfg["paths"] = {"dataset": str(ds), "outputs": str(out), "hf_cache": str(root / "hf")}
    cfg["model"].update(pretrained=False, arch_overrides={"embed_dim": 16, "depths": [1, 1, 1, 1], "num_heads": [1, 2, 2, 4]})
    cfg["preprocess"]["workers"] = 1
    cfg["train"].update(max_epochs=2, batch_size=8, early_stop_patience=2)
    cfg["s2"].update(c_grid=[0.1, 1.0], embed_batch_size=8, smote_k=3)
    cfg["eval"]["bootstrap_reps"] = 100
    cfg["compute"] = {"torch_threads": 2}
    cfg["audit"]["s7_min_patients_per_class"] = 5
    return cfg, root


@pytest.fixture(scope="session")
def cfg_s2(synth):
    cfg = copy.deepcopy(synth[0])
    cfg["scenario"] = "S2"
    return cfg
