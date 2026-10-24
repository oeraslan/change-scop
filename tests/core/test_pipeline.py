"""core/pipeline.py testleri — sahte (fake) bileşenlerle, model gerektirmez."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from core.config import AppConfig
from core.contracts import (
    AlignmentQuality,
    BoundingBox,
    ChangeRegion,
    DetectionResult,
    ImageInput,
    PreparedImage,
    RegistrationResult,
)
from core.pipeline import (
    STAGE_DONE,
    STAGE_RELEASE,
    ComparisonCancelled,
    Pipeline,
    run_comparison,
    set_default_pipeline,
)


class Recorder:
    """Çağrı sırasını kaydeder (VRAM kuralını doğrulamak için)."""

    def __init__(self) -> None:
        self.events: list[str] = []


class FakePreprocessor:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    def prepare(self, image: ImageInput) -> PreparedImage:
        self.rec.events.append(f"prepare:{image.path.name}")
        return PreparedImage(image, np.zeros((32, 32, 3), np.uint8), 1.0, False)


class FakeRegistrar:
    def __init__(self, rec: Recorder, quality: AlignmentQuality = AlignmentQuality.GOOD) -> None:
        self.rec = rec
        self.quality = quality

    def register(self, before: PreparedImage, after: PreparedImage) -> RegistrationResult:
        self.rec.events.append("register")
        return RegistrationResult(
            before=before, after_warped=after.pixels, valid_mask=np.ones((32, 32), bool),
            homography=None if self.quality is AlignmentQuality.FAILED else np.eye(3),
            num_matches=100, num_inliers=80, reprojection_error_px=0.5, quality=self.quality,
        )

    def release(self) -> None:
        self.rec.events.append("release:registrar")


class FakeDetector:
    def __init__(self, rec: Recorder, n_regions: int = 2) -> None:
        self.rec = rec
        self.n_regions = n_regions

    def detect(self, registration: RegistrationResult) -> DetectionResult:
        self.rec.events.append("detect")
        regions = [
            ChangeRegion(i + 1, BoundingBox(0, 0, 4, 4), np.ones((4, 4), bool), 0.9, 16)
            for i in range(self.n_regions)
        ]
        return DetectionResult(np.zeros((32, 32), np.float32), regions, "ssim")

    def release(self) -> None:
        self.rec.events.append("release:detector")


def _inputs() -> tuple[ImageInput, ImageInput]:
    return ImageInput(Path("before.png")), ImageInput(Path("after.jpg"))


def _pipeline(rec: Recorder, *, quality=AlignmentQuality.GOOD, n_regions=2) -> Pipeline:
    return Pipeline(
        preprocessor=FakePreprocessor(rec),
        registrar=FakeRegistrar(rec, quality),
        detector=FakeDetector(rec, n_regions),
        config=AppConfig(),
    )


def test_cv_only_flow_releases_models_and_reports_progress():
    rec = Recorder()
    progress: list[tuple[str, float]] = []
    result = _pipeline(rec).run(*_inputs(), lambda s, f, m="": progress.append((s, f)))

    assert result.detection is not None and len(result.detection.regions) == 2
    assert rec.events[:4] == ["prepare:before.png", "prepare:after.jpg", "register", "detect"]
    assert {"release:registrar", "release:detector"} <= set(rec.events)
    fractions = [f for _, f in progress]
    assert fractions == sorted(fractions) and progress[-1] == (STAGE_DONE, 1.0)


def test_failed_alignment_skips_detection():
    rec = Recorder()
    result = _pipeline(rec, quality=AlignmentQuality.FAILED).run(*_inputs())
    assert result.detection is None
    assert "detect" not in rec.events
    assert result.registration.match_percent == 80.0


def test_match_percent_is_inlier_ratio():
    rec = Recorder()
    reg = FakeRegistrar(rec).register(*[FakePreprocessor(rec).prepare(i) for i in _inputs()])
    assert reg.match_percent == 80.0                      # 80 inlier / 100 eşleşme
    reg.num_matches, reg.num_inliers = 0, 0
    assert reg.match_percent == 0.0                       # eşleşme yoksa sıfıra bölme yok
    reg.num_matches, reg.num_inliers = 10, 12             # tutarsız sayaçlar %100'ü aşmaz
    assert reg.match_percent == 100.0


def test_pipeline_has_no_llm_api():
    import core.pipeline as cp

    for name in ("interpret_comparison", "stop_llm_server", "llm_status"):
        assert not hasattr(cp, name)
    with pytest.raises(TypeError):
        _pipeline(Recorder()).run(*_inputs(), None, use_llm=True)


def test_cancel_raises_and_still_releases():
    rec = Recorder()
    cancel = threading.Event()

    def progress(stage, fraction, message=""):
        if stage == "register":
            cancel.set()

    pipe = _pipeline(rec)
    # İptal, register aşaması bildirildikten SONRA bir sonraki aşama sınırında gerçekleşir.
    with pytest.raises(ComparisonCancelled):
        pipe.run(*_inputs(), progress, cancel=cancel)
    assert "detect" not in rec.events
    assert "release:detector" in rec.events


def test_errors_propagate_but_release_happens():
    rec = Recorder()
    pipe = _pipeline(rec)

    def boom(_):
        raise ValueError("bozuk görüntü")

    pipe.registrar.register = lambda b, a: boom(b)
    stages: list[str] = []
    with pytest.raises(ValueError):
        pipe.run(*_inputs(), lambda s, f, m="": stages.append(s))
    assert STAGE_RELEASE in stages and "release:detector" in rec.events


def test_run_comparison_uses_default_pipeline():
    rec = Recorder()
    set_default_pipeline(_pipeline(rec))
    try:
        stages: list[str] = []
        result = run_comparison(*_inputs(), lambda s, f, m="": stages.append(s))
        assert result.detection is not None
        assert stages[-1] == STAGE_DONE
    finally:
        set_default_pipeline(None)


def test_configure_and_shutdown_default_pipeline(monkeypatch):
    import core.pipeline as cp

    rec = Recorder()
    built: list[AppConfig] = []

    def fake_build(config=None):
        built.append(config)
        return _pipeline(rec)

    monkeypatch.setattr(cp, "build_default_pipeline", fake_build)
    monkeypatch.setattr(cp, "_default_config", None)
    set_default_pipeline(None)
    try:
        cp.shutdown_default_pipeline()                 # kurulmamışken no-op
        assert built == [] and rec.events == []

        cfg = AppConfig()
        cfg.detection.sensitivity = 0.9
        first = cp.get_default_pipeline()
        cp.configure_default_pipeline(cfg)             # eskisi kapatılır
        assert "release:registrar" in rec.events
        second = cp.get_default_pipeline()
        assert second is not first and built[-1] is cfg

        rec.events.clear()
        cp.shutdown_default_pipeline()
        assert "release:detector" in rec.events
        assert cp._default_pipeline is None
    finally:
        set_default_pipeline(None)


def test_retire_while_running_defers_release_to_end_of_run():
    rec = Recorder()
    pipe = _pipeline(rec)
    pipe.config.runtime.release_cv_models_after_run = False   # sadece retire kaynaklı release'i gör
    started, proceed = threading.Event(), threading.Event()
    original = pipe.registrar.register

    def slow_register(b, a):
        started.set()
        proceed.wait(10)
        return original(b, a)

    pipe.registrar.register = slow_register
    t = threading.Thread(target=lambda: pipe.run(*_inputs(), None))
    t.start()
    assert started.wait(10)
    assert pipe.retire() is False                     # meşgul → beklemeden döner
    assert "release:registrar" not in rec.events
    proceed.set()
    t.join(10)
    assert "release:registrar" in rec.events          # koşu bitince bırakıldı


def test_retire_when_idle_releases_immediately():
    rec = Recorder()
    pipe = _pipeline(rec)
    assert pipe.retire() is True
    assert "release:detector" in rec.events


def test_ignore_mask_removed_from_valid_mask_before_detection():
    rec = Recorder()
    pipe = _pipeline(rec)
    seen: list[np.ndarray] = []
    original = pipe.detector.detect

    def capture(registration):
        seen.append(registration.valid_mask.copy())
        return original(registration)

    pipe.detector.detect = capture
    mask = np.zeros((64, 64), bool)        # farklı çözünürlük: 32x32 valid_mask'a ölçeklenir
    mask[:32, :] = True                    # üst yarıyı yok say
    pipe.run(*_inputs(), None, ignore_mask=mask)
    assert not seen[0][:16].any() and seen[0][16:].all()


def test_ignore_mask_must_be_2d():
    from core.pipeline import apply_ignore_mask

    rec = Recorder()
    reg = FakeRegistrar(rec).register(*[FakePreprocessor(rec).prepare(i) for i in _inputs()])
    with pytest.raises(ValueError):
        apply_ignore_mask(reg, np.zeros((4, 4, 3), bool))
