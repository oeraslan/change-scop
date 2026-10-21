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
    ChangeVerdict,
    DetectionResult,
    ImageInput,
    PreparedImage,
    RegistrationResult,
)
from core.pipeline import (
    STAGE_DONE,
    STAGE_LLM_ANALYZE,
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


class FakeAnalyst:
    def __init__(self, rec: Recorder, available: bool = True, fail: bool = False) -> None:
        self.rec = rec
        self.available = available
        self.fail = fail

    def is_available(self) -> bool:
        return self.available

    def analyze(self, registration: RegistrationResult, detection: DetectionResult) -> DetectionResult:
        self.rec.events.append("llm:analyze")
        if self.fail:
            raise RuntimeError("sunucu çöktü")
        for r in detection.regions:
            r.verdict = ChangeVerdict.REAL_CHANGE
            r.description = "yeni yapı"
        return detection

    def summarize(self, registration: RegistrationResult, detection: DetectionResult) -> str | None:
        self.rec.events.append("llm:summarize")
        return "Merkezde yeni bir yapı var."


def _inputs() -> tuple[ImageInput, ImageInput]:
    return ImageInput(Path("before.png")), ImageInput(Path("after.jpg"))


def _pipeline(rec: Recorder, *, quality=AlignmentQuality.GOOD, analyst=None, n_regions=2,
              factory_calls: list | None = None) -> Pipeline:
    def factory():
        if factory_calls is not None:
            factory_calls.append(1)
        rec.events.append("llm:factory")
        return analyst

    return Pipeline(
        preprocessor=FakePreprocessor(rec),
        registrar=FakeRegistrar(rec, quality),
        detector=FakeDetector(rec, n_regions),
        analyst_factory=factory if analyst is not None else None,
        config=AppConfig(),
    )


def test_cv_only_flow_releases_models_and_reports_progress():
    rec = Recorder()
    progress: list[tuple[str, float]] = []
    result = _pipeline(rec).run(*_inputs(), lambda s, f, m="": progress.append((s, f)), use_llm=False)

    assert result.detection is not None and len(result.detection.regions) == 2
    assert result.llm_used is False and result.summary is None
    assert rec.events[:4] == ["prepare:before.png", "prepare:after.jpg", "register", "detect"]
    assert {"release:registrar", "release:detector"} <= set(rec.events)
    fractions = [f for _, f in progress]
    assert fractions == sorted(fractions) and progress[-1] == (STAGE_DONE, 1.0)


def test_llm_runs_only_after_cv_models_released():
    rec = Recorder()
    result = _pipeline(rec, analyst=FakeAnalyst(rec)).run(*_inputs(), None, use_llm=True)

    assert result.llm_used is True
    assert result.summary == "Merkezde yeni bir yapı var."
    assert all(r.verdict is ChangeVerdict.REAL_CHANGE for r in result.detection.regions)
    last_release = max(i for i, e in enumerate(rec.events) if e.startswith("release:"))
    first_llm = min(i for i, e in enumerate(rec.events) if e.startswith("llm:"))
    assert last_release < first_llm, rec.events


def test_analyst_is_created_lazily_and_cached():
    rec = Recorder()
    calls: list[int] = []
    pipe = _pipeline(rec, analyst=FakeAnalyst(rec), factory_calls=calls)

    pipe.run(*_inputs(), None, use_llm=False)
    assert calls == []                                   # LLM istenmedi → hiç oluşturulmadı
    pipe.run(*_inputs(), None, use_llm=True)
    pipe.run(*_inputs(), None, use_llm=True)
    assert calls == [1]                                  # bir kez oluşturuldu, sonra önbellekten


def test_llm_disabled_in_config_skips_llm():
    rec = Recorder()
    pipe = _pipeline(rec, analyst=FakeAnalyst(rec))
    pipe.config.llm.enabled = False
    result = pipe.run(*_inputs(), None, use_llm=True)
    assert result.llm_used is False
    assert not any(e.startswith("llm:") for e in rec.events)


def test_failed_alignment_skips_detection_and_llm():
    rec = Recorder()
    result = _pipeline(rec, quality=AlignmentQuality.FAILED, analyst=FakeAnalyst(rec)).run(
        *_inputs(), None, use_llm=True)
    assert result.detection is None and result.llm_used is False
    assert "detect" not in rec.events
    assert not any(e.startswith("llm:") for e in rec.events)


def test_no_regions_skips_llm():
    rec = Recorder()
    result = _pipeline(rec, analyst=FakeAnalyst(rec), n_regions=0).run(*_inputs(), None, use_llm=True)
    assert result.llm_used is False
    assert "llm:analyze" not in rec.events


@pytest.mark.parametrize("analyst_kwargs", [{"available": False}, {"fail": True}])
def test_llm_unavailable_or_failing_keeps_cv_result(analyst_kwargs):
    rec = Recorder()
    result = _pipeline(rec, analyst=FakeAnalyst(rec, **analyst_kwargs)).run(*_inputs(), None, use_llm=True)
    assert result.llm_used is False and result.summary is None
    assert result.detection is not None
    assert all(r.verdict is ChangeVerdict.UNREVIEWED for r in result.detection.regions)


def test_factory_exception_is_tolerated():
    rec = Recorder()
    pipe = _pipeline(rec)

    def broken():
        raise FileNotFoundError("llama-server.exe yok")

    pipe.analyst_factory = broken
    result = pipe.run(*_inputs(), None, use_llm=True)
    assert result.llm_used is False and result.detection is not None


def test_cancel_raises_and_still_releases():
    rec = Recorder()
    cancel = threading.Event()

    def progress(stage, fraction, message=""):
        if stage == "register":
            cancel.set()

    pipe = _pipeline(rec)
    # İptal, register aşaması bildirildikten SONRA bir sonraki aşama sınırında gerçekleşir.
    with pytest.raises(ComparisonCancelled):
        pipe.run(*_inputs(), progress, use_llm=False, cancel=cancel)
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
        pipe.run(*_inputs(), lambda s, f, m="": stages.append(s), use_llm=False)
    assert STAGE_RELEASE in stages and "release:detector" in rec.events


def test_run_comparison_uses_default_pipeline():
    rec = Recorder()
    set_default_pipeline(_pipeline(rec, analyst=FakeAnalyst(rec)))
    try:
        stages: list[str] = []
        result = run_comparison(*_inputs(), lambda s, f, m="": stages.append(s), use_llm=True)
        assert result.llm_used is True
        assert STAGE_LLM_ANALYZE in stages
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
    pipe.config.runtime.release_cv_models_before_llm = False   # sadece retire kaynaklı release'i gör
    started, proceed = threading.Event(), threading.Event()
    original = pipe.registrar.register

    def slow_register(b, a):
        started.set()
        proceed.wait(10)
        return original(b, a)

    pipe.registrar.register = slow_register
    t = threading.Thread(target=lambda: pipe.run(*_inputs(), None, use_llm=False))
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
    pipe.run(*_inputs(), None, use_llm=False, ignore_mask=mask)
    assert not seen[0][:16].any() and seen[0][16:].all()


def test_ignore_mask_must_be_2d():
    from core.pipeline import apply_ignore_mask

    rec = Recorder()
    reg = FakeRegistrar(rec).register(*[FakePreprocessor(rec).prepare(i) for i in _inputs()])
    with pytest.raises(ValueError):
        apply_ignore_mask(reg, np.zeros((4, 4, 3), bool))


class ServerAnalyst(FakeAnalyst):
    """Sunucu durumu ve ilerleme kancası olan sahte analist."""

    def __init__(self, rec: Recorder) -> None:
        super().__init__(rec)
        self.is_server_running = False
        self.progress_callback = None
        self.seen_regions: list = []

    def analyze(self, registration, detection):
        self.is_server_running = True
        self.seen_regions = detection.regions
        if self.progress_callback:
            self.progress_callback(0, len(detection.regions), "Yapay zekâ modeli yükleniyor…")
            for i in range(len(detection.regions)):
                self.progress_callback(i + 1, len(detection.regions), "")
        return super().analyze(registration, detection)

    def release(self) -> None:
        self.rec.events.append("release:llm")
        self.is_server_running = False


def test_on_cv_result_called_before_llm_and_not_mutated():
    rec = Recorder()
    analyst = ServerAnalyst(rec)
    partial: list = []

    def on_cv(res):
        rec.events.append("partial")
        partial.append(res)

    result = _pipeline(rec, analyst=analyst).run(*_inputs(), None, use_llm=True, on_cv_result=on_cv)
    assert rec.events.index("partial") < rec.events.index("llm:analyze")
    assert partial[0].llm_used is False
    # Kısmi sonuçtaki bölgeler LLM tarafından değiştirilmedi (kopya üzerinde çalışıldı)
    assert all(r.verdict is ChangeVerdict.UNREVIEWED for r in partial[0].detection.regions)
    assert all(r.verdict is ChangeVerdict.REAL_CHANGE for r in result.detection.regions)


def test_on_cv_result_not_called_without_llm():
    rec = Recorder()
    called: list = []
    _pipeline(rec, analyst=FakeAnalyst(rec)).run(*_inputs(), None, use_llm=False, on_cv_result=called.append)
    assert called == []


def test_llm_server_released_before_next_cv_step():
    rec = Recorder()
    analyst = ServerAnalyst(rec)
    pipe = _pipeline(rec, analyst=analyst)
    pipe.run(*_inputs(), None, use_llm=True)
    assert analyst.is_server_running
    rec.events.clear()
    pipe.run(*_inputs(), None, use_llm=False)
    assert rec.events.index("release:llm") < rec.events.index("register")


def test_region_progress_forwarded():
    rec = Recorder()
    msgs: list[tuple[str, float, str]] = []
    _pipeline(rec, analyst=ServerAnalyst(rec)).run(
        *_inputs(), lambda s, f, m="": msgs.append((s, f, m)), use_llm=True)
    llm = [(f, m) for s, f, m in msgs if s == "llm_analyze"]
    assert any("yükleniyor" in m for _, m in llm)
    fr = [f for f, _ in llm]
    assert fr == sorted(fr) and len(fr) >= 3


def test_interpret_runs_only_llm():
    rec = Recorder()
    analyst = ServerAnalyst(rec)
    pipe = _pipeline(rec, analyst=analyst)
    cv = pipe.run(*_inputs(), None, use_llm=False)
    rec.events.clear()
    out = pipe.interpret(cv)
    assert "register" not in rec.events and "detect" not in rec.events
    assert out.llm_used and out.summary
    assert all(r.verdict is ChangeVerdict.UNREVIEWED for r in cv.detection.regions)   # girdi korunur


def test_interpret_without_llm_returns_input():
    rec = Recorder()
    pipe = _pipeline(rec)
    cv = pipe.run(*_inputs(), None, use_llm=False)
    assert pipe.interpret(cv) is cv


def test_stop_llm_server_releases_running_analyst():
    import core.pipeline as cp

    rec = Recorder()
    analyst = ServerAnalyst(rec)
    pipe = _pipeline(rec, analyst=analyst)
    pipe.run(*_inputs(), None, use_llm=True)
    set_default_pipeline(pipe)
    try:
        cp.stop_llm_server()
        assert "release:llm" in rec.events and not analyst.is_server_running
    finally:
        set_default_pipeline(None)
    cp.stop_llm_server()   # pipeline yokken no-op
