"""Çözünürlük farkı telafisi ve alan-ölçekli minimum bölge testleri."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import AppConfig
from core.detection import create_detector
from core.detection.ssim_detector import estimate_scale
from scripts.make_synthetic_pair import PairParams
from tests.detection.helpers import make_registration, pair_registration

pytestmark = pytest.mark.filterwarnings("ignore")


def test_estimate_scale_pure_scale_and_rotation() -> None:
    """after->before homografisi 2x büyütme + dönme ise ölçek 2 çıkar; yoksa 1."""
    th = np.deg2rad(20)
    R = np.array([[np.cos(th), -np.sin(th), 5], [np.sin(th), np.cos(th), 9], [0, 0, 1]])
    S = np.diag([2.0, 2.0, 1.0])
    assert estimate_scale(S @ R, (600, 800)) == pytest.approx(2.0, rel=1e-6)
    assert estimate_scale(R, (600, 800)) == pytest.approx(1.0, rel=1e-6)
    assert estimate_scale(None, (600, 800)) == 1.0
    assert estimate_scale(np.diag([0.5, 0.5, 1.0]), (600, 800)) == pytest.approx(0.5)


@pytest.mark.parametrize("scale", [0.5, 0.4, 0.3])
def test_resolution_gap_without_changes_gives_no_regions(scale: float) -> None:
    """Nesnesiz çiftte after düşük çözünürlüklü olsa da sahte bölge çıkmamalı."""
    _, reg = pair_registration(PairParams(scale=scale, n_new_objects=0))
    res = create_detector().detect(reg)
    assert res.regions == [], [(r.bbox, r.area_px) for r in res.regions][:5]
    assert res.params["registration_scale"] > 1.5
    assert res.params["working_scale"] == res.params["registration_scale"]


@pytest.mark.parametrize("scale", [0.5, 0.3])
def test_resolution_gap_with_objects_detected(scale: float) -> None:
    """Ölçek farkında da eklenen nesneler bulunur."""
    pair, reg = pair_registration(PairParams(scale=scale))
    res = create_detector().detect(reg)
    for x, y, w, h in pair.added_bboxes:
        cx, cy = x + w // 2, y + h // 2
        assert any(r.bbox.x <= cx <= r.bbox.x + r.bbox.width and r.bbox.y <= cy <= r.bbox.y + r.bbox.height
                   for r in res.regions)


def test_native_resolution_not_downscaled() -> None:
    """Ölçek 1 ise çalışma ölçeği 1 (küçültme yok)."""
    _, reg = pair_registration(PairParams(scale=1.0))
    assert create_detector().detect(reg).params["working_scale"] == 1.0


def test_min_area_scales_with_valid_area() -> None:
    """Etkin minimum alan geçerli alanla büyür; config değeri alt sınırdır."""
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(small, 8, 0), 8, 1)   # 2400x2400
    cfg = AppConfig()
    r_small = create_detector(cfg).detect(make_registration(small, small.copy()))
    r_big = create_detector(cfg).detect(make_registration(big, big.copy()))
    floor = cfg.detection.min_region_area_px
    assert r_small.params["min_region_area_effective_px"] == floor
    assert r_big.params["min_region_area_effective_px"] > floor
    assert r_big.params["min_region_area_effective_px"] == round(
        r_big.params["min_region_area_fraction"] * 2400 * 2400)


def test_unreliable_flag_absent_for_clean_pair() -> None:
    """Temiz çiftte 'unreliable' 0."""
    _, reg = pair_registration(PairParams(n_new_objects=0))
    assert create_detector().detect(reg).params["unreliable"] == 0
