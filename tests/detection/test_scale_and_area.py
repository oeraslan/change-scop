"""Çözünürlük farkı telafisi ve alan-ölçekli minimum bölge testleri."""

from __future__ import annotations

import cv2
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


def test_min_area_fraction_only_with_working_downscale() -> None:
    """Çözünürlük farkı yokken taban config değeri; küçültme varken alan oranına göre yükselir."""
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(small, 8, 0), 8, 1)   # 2400x2400
    cfg = AppConfig()
    floor = cfg.detection.min_region_area_px
    r_same = create_detector(cfg).detect(make_registration(big, big.copy()))
    assert r_same.params["min_region_area_effective_px"] == floor
    H = np.diag([2.0, 2.0, 1.0])     # after before'a 2x büyütülmüş -> çalışma küçültmesi
    r_big = create_detector(cfg).detect(make_registration(big, big.copy(), homography=H))
    assert r_big.params["working_scale"] == pytest.approx(2.0)
    assert r_big.params["min_region_area_effective_px"] == round(
        r_big.params["min_region_area_fraction"] * 2400 * 2400)


def test_40px_object_in_large_same_resolution_pair_is_found() -> None:
    """3200x2400 aynı çözünürlükte 40x40 px belirgin nesne bulunur (alan tabanı config değeri)."""
    rng = np.random.default_rng(1)
    base = cv2.resize(rng.random((60, 80, 3)).astype(np.float32), (3200, 2400), interpolation=cv2.INTER_CUBIC)
    before = np.clip(base * 120 + 70, 0, 255).astype(np.uint8)
    after = before.copy()
    cv2.rectangle(after, (1500, 1100), (1539, 1139), (230, 30, 30), -1)
    res = create_detector().detect(make_registration(before, after))
    assert any(r.bbox.x <= 1520 <= r.bbox.x + r.bbox.width and r.bbox.y <= 1120 <= r.bbox.y + r.bbox.height
               for r in res.regions)


def test_unreliable_flag_absent_for_clean_pair() -> None:
    """Temiz çiftte 'unreliable' 0."""
    _, reg = pair_registration(PairParams(n_new_objects=0))
    assert create_detector().detect(reg).params["unreliable"] == 0
