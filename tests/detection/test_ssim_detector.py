"""SSIM değişiklik dedektörü testleri."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from core.config import AppConfig
from core.contracts import AlignmentQuality, DetectionResult
from core.detection import create_detector
from scripts.make_synthetic_pair import PairParams
from tests.detection.helpers import make_registration, pair_registration

pytestmark = pytest.mark.filterwarnings("ignore")


def _full_mask(result: DetectionResult, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, bool)
    for r in result.regions:
        b = r.bbox
        out[b.y:b.y + b.height, b.x:b.x + b.width] |= r.mask
    return out


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def synthetic():
    pair, reg = pair_registration()
    t0 = time.perf_counter()
    result = create_detector().detect(reg)
    elapsed = time.perf_counter() - t0
    return pair, reg, result, elapsed


def test_synthetic_objects_detected(synthetic):
    pair, _, result, _ = synthetic
    truth = pair.truth_mask > 0
    pred = _full_mask(result, truth.shape)
    # Her eklenen nesne (bbox) en az %50 kapsanmalı
    for (x, y, w, h) in pair.added_bboxes:
        obj = truth[y:y + h, x:x + w]
        cover = pred[y:y + h, x:x + w][obj].mean()
        assert cover > 0.5, f"nesne {(x, y, w, h)} kapsama {cover:.2f}"
    tp = (pred & truth).sum()
    precision = tp / max(pred.sum(), 1)
    recall = tp / truth.sum()
    print(f"\npiksel precision={precision:.3f} recall={recall:.3f} bölge={len(result.regions)}")
    assert precision > 0.5
    assert recall > 0.6


def test_no_large_false_regions(synthetic):
    pair, _, result, _ = synthetic
    truth = cv2.dilate((pair.truth_mask > 0).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    for r in result.regions:
        b = r.bbox
        near_truth = (truth[b.y:b.y + b.height, b.x:b.x + b.width] & r.mask).sum() / r.mask.sum()
        if near_truth < 0.5:
            assert r.area_px < 1500, f"yanlış bölge {r.bbox} alan {r.area_px} skor {r.score:.2f}"


def test_speed_1600x1200(synthetic):
    _, reg, result, elapsed = synthetic
    assert reg.before.pixels.shape[:2] == (1200, 1600)
    print(f"\nsüre={elapsed:.2f}s")
    assert elapsed < 6.0


def test_output_contract(synthetic):
    _, reg, result, _ = synthetic
    h, w = reg.before.pixels.shape[:2]
    assert result.change_map.shape == (h, w)
    assert result.change_map.dtype == np.float32
    assert 0.0 <= result.change_map.min() and result.change_map.max() <= 1.0
    assert result.method == "ssim"
    scores = [r.score for r in result.regions]
    assert scores == sorted(scores, reverse=True)
    assert [r.region_id for r in result.regions] == list(range(1, len(scores) + 1))
    for r in result.regions:
        assert r.mask.dtype == bool
        assert r.mask.shape == (r.bbox.height, r.bbox.width)
        assert r.area_px == int(r.mask.sum())
        assert 0.0 <= r.score <= 1.0
    for key in ("threshold", "sensitivity", "ssim_window", "blur_sigma", "light_normalization"):
        assert key in result.params


def test_brightness_contrast_only_no_regions():
    """Yalnızca parlaklık/kontrast + JPEG farkı olan kopya bölge üretmemeli."""
    pair, _ = pair_registration(PairParams(n_new_objects=0))
    before = pair.before
    after = _jpeg(np.clip(before.astype(np.float32) * 1.15 + 25, 0, 255).astype(np.uint8), 70)
    result = create_detector().detect(make_registration(before, after))
    assert result.regions == [] or max(r.score for r in result.regions) < 0.2


def test_jpeg_q60_no_false_alarm():
    pair, _ = pair_registration(PairParams(n_new_objects=0))
    result = create_detector().detect(make_registration(pair.before, _jpeg(pair.before, 60)))
    assert result.regions == []


def test_warped_pair_without_objects_no_false_alarm():
    """Geometri + parlaklık + JPEG70 var ama nesne yok: yanlış alarm olmamalı."""
    _, reg = pair_registration(PairParams(n_new_objects=0))
    result = create_detector().detect(reg)
    assert sum(r.area_px for r in result.regions) < 800


def test_diff_outside_valid_mask_ignored():
    pair, _ = pair_registration(PairParams(n_new_objects=0))
    before = pair.before
    after = before.copy()
    h, w = before.shape[:2]
    valid = np.ones((h, w), bool)
    valid[:, w // 2:] = False
    after[300:500, w // 2 + 100:w // 2 + 300] = 255          # geçersiz alanda büyük fark
    result = create_detector().detect(make_registration(before, after, valid))
    assert result.regions == []
    assert np.all(result.change_map[~valid] == 0)


def test_valid_mask_zero_outside_even_with_change_inside(synthetic):
    _, reg, result, _ = synthetic
    assert np.all(result.change_map[~reg.valid_mask] == 0)
    for r in result.regions:
        b = r.bbox
        assert not (r.mask & ~reg.valid_mask[b.y:b.y + b.height, b.x:b.x + b.width]).any()


def test_rectangle_added_exact_region():
    pair, _ = pair_registration(PairParams(n_new_objects=0))
    before = pair.before
    after = before.copy()
    cv2.rectangle(after, (700, 500), (819, 579), (230, 40, 40), -1)
    result = create_detector().detect(make_registration(before, after))
    assert len(result.regions) == 1
    b = result.regions[0].bbox
    assert abs(b.x - 700) <= 10 and abs(b.y - 500) <= 10
    assert abs(b.width - 120) <= 20 and abs(b.height - 80) <= 20


def test_failed_raises():
    pair, _ = pair_registration(PairParams(n_new_objects=0))
    reg = make_registration(pair.before, pair.before, quality=AlignmentQuality.FAILED)
    with pytest.raises(ValueError):
        create_detector().detect(reg)


def test_sensitivity_scales_threshold():
    pair, reg = pair_registration(PairParams(n_new_objects=0, width=800, height=600))
    low, high = AppConfig(), AppConfig()
    low.detection.sensitivity, high.detection.sensitivity = 0.1, 0.9
    t_low = create_detector(low).detect(reg).params["threshold"]
    t_high = create_detector(high).detect(reg).params["threshold"]
    assert t_high < t_low
