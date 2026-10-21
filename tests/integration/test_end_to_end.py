"""Gerçek `build_default_pipeline()` ile sentetik uçtan uca senaryolar."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.contracts import AlignmentQuality, ImageInput
from scripts.make_synthetic_pair import PairParams, make_pair
from tests.integration.helpers import evaluate, record, run_pair, write_pair

SCENARIOS: dict[str, PairParams] = {
    "varsayılan": PairParams(),
    "seed1": PairParams(seed=1),
    "seed7": PairParams(seed=7),
    "seed123": PairParams(seed=123),
    "döndürme20": PairParams(rotate_deg=20.0),
    "jpeg50": PairParams(jpeg_quality=50),
    "ölçek0.6": PairParams(scale=0.6),
    "ölçek0.5": PairParams(scale=0.5),
    "ölçek0.4": PairParams(scale=0.4),
    "birleşik(s7,0.6,20°,jpeg50)": PairParams(seed=7, scale=0.6, rotate_deg=20.0, jpeg_quality=50),
}

_XFAIL_SPURIOUS = {
    "ölçek0.5": "Çözünürlük farkı (after 2x büyütülüyor) telafi edilmiyor: onlarca sahte bölge",
    "ölçek0.4": "Çözünürlük farkı (after 2.5x büyütülüyor) telafi edilmiyor: yüzlerce sahte bölge",
}

_CACHE: dict = {}


def _get(pipeline, name: str, tmp_path_factory):
    """Her senaryoyu bir kez çalıştırır (iki test aynı sonucu kullanır)."""
    if name not in _CACHE:
        pair, result = run_pair(pipeline, SCENARIOS[name], tmp_path_factory.mktemp("e2e"), "s")
        ev = evaluate(pair, result)
        record(name, ev)
        _CACHE[name] = (pair, result, ev)
    return _CACHE[name]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_added_objects_are_detected(name, real_pipeline, tmp_path_factory) -> None:
    """Eklenen her nesnenin merkezi bir bölgenin içinde kalmalı; hizalama FAILED olmamalı."""
    _, result, ev = _get(real_pipeline, name, tmp_path_factory)
    assert result.registration.quality is not AlignmentQuality.FAILED, result.registration.notes
    assert result.detection is not None
    assert all(ev.hits), f"{name}: isabetler: {ev.hits}"


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=_XFAIL_SPURIOUS[n])) if n in _XFAIL_SPURIOUS else n
    for n in SCENARIOS
])
def test_no_large_spurious_regions(name, real_pipeline, tmp_path_factory) -> None:
    """Eklenen nesneler dışında büyük (>= görüntünün %0.1'i) bölge çıkmamalı."""
    _, _, ev = _get(real_pipeline, name, tmp_path_factory)
    assert ev.n_large_extra == 0, f"{name}: {ev.n_large_extra} büyük sahte bölge ({ev.n_extra} toplam sahte)"


def test_light_only_pair_has_no_regions(real_pipeline, tmp_path_factory) -> None:
    """Yalnızca parlaklık/kontrast farkı (nesne yok) → hiç bölge çıkmamalı."""
    params = PairParams(n_new_objects=0, brightness=30.0, contrast=1.15)
    pair, result = run_pair(real_pipeline, params, tmp_path_factory.mktemp("light"), "l")
    record("yalnız_ışık", evaluate(pair, result))
    assert result.registration.quality is AlignmentQuality.GOOD
    assert result.detection is not None and result.detection.regions == []


@pytest.mark.xfail(strict=True, reason="Güçlü ışık farkı (-35/x0.85) + JPEG 50 + 20° dönmede 3 küçük (400-650 px) sahte bölge çıkıyor")
def test_strong_light_shift_with_rotation_has_no_regions(real_pipeline, tmp_path_factory) -> None:
    """Güçlü ışık farkı + 20° dönme + JPEG 50: yine bölge yok."""
    params = PairParams(n_new_objects=0, brightness=-35.0, contrast=0.85, rotate_deg=20.0, jpeg_quality=50)
    pair, result = run_pair(real_pipeline, params, tmp_path_factory.mktemp("light2"), "l2")
    record("güçlü_ışık+20°+jpeg50", evaluate(pair, result))
    assert result.detection is not None
    assert result.detection.regions == [], [(r.bbox, r.area_px) for r in result.detection.regions][:5]


def test_same_image_twice_has_no_regions(real_pipeline, tmp_path_factory) -> None:
    """Aynı dosya iki kez → GOOD, sıfır bölge, homografi ≈ birim."""
    pair = make_pair(PairParams(n_new_objects=0))
    before, _ = write_pair(pair, tmp_path_factory.mktemp("same"), "x")
    result = real_pipeline.run(ImageInput(before), ImageInput(before))
    assert result.registration.quality is AlignmentQuality.GOOD
    assert result.detection is not None and result.detection.regions == []
    assert np.allclose(result.registration.homography, np.eye(3), atol=1e-2)


def test_result_contract_invariants(real_pipeline, tmp_path_factory) -> None:
    """`core/contracts.py` değişmezleri: boyutlar, dtype, bbox sınırları, maske uyumu."""
    pair, result, _ = _get(real_pipeline, "varsayılan", tmp_path_factory)
    reg, det = result.registration, result.detection
    h, w = reg.before.pixels.shape[:2]
    assert (h, w) == pair.before.shape[:2]
    assert reg.after_warped.shape == (h, w, 3) and reg.after_warped.dtype == np.uint8
    assert reg.valid_mask.shape == (h, w) and reg.valid_mask.dtype == bool
    assert reg.homography is not None and reg.homography.shape == (3, 3)
    assert det is not None and det.change_map.shape == (h, w) and det.change_map.dtype == np.float32
    assert 0.0 <= float(det.change_map.min()) and float(det.change_map.max()) <= 1.0
    assert not det.change_map[~reg.valid_mask].any(), "geçersiz alanda değişiklik skoru var"
    ids = [r.region_id for r in det.regions]
    assert len(ids) == len(set(ids))
    for r in det.regions:
        b = r.bbox
        assert 0 <= b.x and 0 <= b.y and b.x + b.width <= w and b.y + b.height <= h
        assert r.mask.shape == (b.height, b.width) and r.mask.dtype == bool
        assert r.area_px == int(r.mask.sum())
        assert 0.0 <= r.score <= 1.0
        full = np.zeros((h, w), bool)
        full[b.y:b.y + b.height, b.x:b.x + b.width] = r.mask
        assert not (full & ~reg.valid_mask).any(), "bölge örtüşmeyen alana taşıyor"
    assert result.llm_used is False and result.summary is None


def test_homography_matches_truth(real_pipeline, tmp_path_factory) -> None:
    """Tahmin edilen homografi gerçek (after→before) homografiyle iç noktalarda <3 px uyuşmalı."""
    pair, result, _ = _get(real_pipeline, "varsayılan", tmp_path_factory)
    H = result.registration.homography
    assert H is not None
    ah, aw = pair.after.shape[:2]
    pts = np.array([[[aw * 0.2, ah * 0.2]], [[aw * 0.8, ah * 0.2]], [[aw * 0.8, ah * 0.8]], [[aw * 0.2, ah * 0.8]]],
                   np.float64)
    est = cv2.perspectiveTransform(pts, H)
    ref = cv2.perspectiveTransform(pts, pair.homography_after_to_before)
    assert float(np.linalg.norm(est - ref, axis=2).max()) < 3.0


def test_progress_stages_are_monotonic(real_pipeline, tmp_path_factory) -> None:
    """İlerleme geri çağrısı: sıralı aşamalar, oran azalmaz, son aşama 'done'."""
    pair = make_pair(PairParams())
    before, after = write_pair(pair, tmp_path_factory.mktemp("prog"), "g")
    events: list[tuple[str, float]] = []
    real_pipeline.run(ImageInput(before), ImageInput(after), lambda s, f, m="": events.append((s, f)))
    stages = [s for s, _ in events]
    assert stages == ["preprocess", "register", "detect", "release", "done"]
    fractions = [f for _, f in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
