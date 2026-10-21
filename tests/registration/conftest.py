"""Ortak fixture'lar ve yardımcılar (sentetik çiftler, doğruluk ölçümü)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config import AppConfig
from core.contracts import ImageInput, PreparedImage
from core.registration import create_preprocessor
from scripts.make_synthetic_pair import PairParams, SyntheticPair, make_pair

_cfg = AppConfig()
HAVE_MODELS = _cfg.paths.superpoint_weights.is_file() and _cfg.paths.lightglue_weights.is_file()
requires_models = pytest.mark.skipif(
    not HAVE_MODELS, reason="SuperPoint/LightGlue ağırlıkları yok (CHANGESCOPE_MODELS_DIR?)")


def sift_config() -> AppConfig:
    cfg = AppConfig()
    cfg.registration.matcher = "sift"
    return cfg


def write_pair(pair: SyntheticPair, out_dir: Path) -> tuple[Path, Path]:
    """Çifti diske yazar: before PNG, after JPEG (zaten sıkıştırılmış, tekrar bozma)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    b, a = out_dir / "before.png", out_dir / "after.jpg"
    cv2.imwrite(str(b), cv2.cvtColor(pair.before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(a), cv2.cvtColor(pair.after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return b, a


def prepare_pair(pair: SyntheticPair, out_dir: Path, cfg: AppConfig | None = None
                 ) -> tuple[PreparedImage, PreparedImage]:
    b, a = write_pair(pair, out_dir)
    pre = create_preprocessor(cfg)
    return pre.prepare(ImageInput(path=b)), pre.prepare(ImageInput(path=a))


def corner_error_px(H_est: np.ndarray, H_true: np.ndarray, size_wh: tuple[int, int]) -> float:
    """After görüntüsünün 4 köşesinin iki homografiyle before'daki konumları arasındaki en büyük fark."""
    w, h = size_wh
    pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    e = cv2.perspectiveTransform(pts, H_est).reshape(-1, 2)
    t = cv2.perspectiveTransform(pts, H_true).reshape(-1, 2)
    return float(np.linalg.norm(e - t, axis=1).max())


@pytest.fixture(scope="session")
def default_pair() -> SyntheticPair:
    return make_pair(PairParams())


@pytest.fixture(scope="session")
def scale_pair() -> SyntheticPair:
    return make_pair(PairParams(scale=0.4, rotate_deg=5.0, shift_px=(10.0, 5.0), seed=3))


@pytest.fixture(scope="session")
def other_scene() -> SyntheticPair:
    return make_pair(PairParams(seed=999, n_new_objects=0))
