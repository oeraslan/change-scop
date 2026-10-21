"""Küçük görüntüde ölçekli min alan ve CPU'da otomatik SSIM testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import AppConfig
from core.contracts import AlignmentQuality, ImageInput
from core.detection import create_detector
from scripts.make_synthetic_pair import PairParams
from tests.detection.helpers import pair_registration, ssim_config

pytestmark = pytest.mark.filterwarnings("ignore")

PIKNIK = Path(__file__).resolve().parents[1] / "data" / "real" / "piknik"


def _overlaps(bbox, x, y, w, h) -> bool:
    return not (bbox.x + bbox.width < x or x + w < bbox.x or bbox.y + bbox.height < y or y + h < bbox.y)


def test_piknik_small_objects_found() -> None:
    """367x415 çiftte top ve limon bulunur; geçerli alan dışı bölge yok; sayı makul."""
    from core.pipeline import build_default_pipeline

    cfg = AppConfig()
    cfg.detection.method = "ssim"
    res = build_default_pipeline(cfg).run(ImageInput(PIKNIK / "before.jpg"), ImageInput(PIKNIK / "after.jpg"))
    assert res.registration.quality != AlignmentQuality.FAILED
    regions = res.detection.regions
    print([(r.bbox, round(r.score, 3), r.area_px) for r in regions])
    assert res.detection.params["min_region_area_effective_px"] < 100
    assert any(_overlaps(r.bbox, 231, 241, 24, 22) for r in regions), "kırmızı top"
    assert any(_overlaps(r.bbox, 110, 295, 30, 45) for r in regions), "limon"
    assert all(r.bbox.x >= 70 for r in regions), "geçerli alan dışı bölge"
    assert len(regions) <= 8


def test_small_pair_without_changes_stays_empty() -> None:
    """Küçük, değişiklikgiz çiftte ölçekli taban sahte alarm üretmez."""
    _, reg = pair_registration(PairParams(width=400, height=300, n_new_objects=0))
    res = create_detector(ssim_config()).detect(reg)
    assert res.regions == []


def test_sensitivity_scales_min_area() -> None:
    """Hassasiyet yükselince etkin min alan küçülür."""
    _, reg = pair_registration(PairParams())
    vals = {}
    for s in (0.0, 0.5, 1.0):
        cfg = ssim_config()
        cfg.detection.sensitivity = s
        vals[s] = create_detector(cfg).detect(reg).params["min_region_area_effective_px"]
    assert vals[1.0] < vals[0.5] <= vals[0.0]


def test_cpu_default_method_uses_ssim() -> None:
    """CPU + varsayılan yöntem: DINOv2 atlanır, method=ssim."""
    _, reg = pair_registration(PairParams(width=400, height=300))
    cfg = AppConfig()
    cfg.runtime.device = "cpu"
    res = create_detector(cfg).detect(reg)
    assert res.method == "ssim"
    assert res.params["fallback_reason"] == "CPU: DINOv2 hız için devre dışı"
    assert res.params["method_requested"] == AppConfig().detection.method


def test_cpu_explicit_dinov2_runs() -> None:
    """CPU + açıkça dinov2: DINOv2 çalışır (model varsa)."""
    _, reg = pair_registration(PairParams(width=400, height=300))
    cfg = AppConfig()
    cfg.runtime.device = "cpu"
    cfg.detection.method = "dinov2"
    det = create_detector(cfg)
    res = det.detect(reg)
    det.release()
    if "fallback_reason" in res.params:
        pytest.skip("DINOv2 modeli yok: " + str(res.params["fallback_reason"]))
    assert res.method == "dinov2"
