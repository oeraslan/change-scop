"""Faz 3: AI paneli + durum çubuğu gerçek (burada sahte) Gemma akışına bağlı."""

from __future__ import annotations

import threading

import pytest

from core import pipeline as core_pipeline
from core.config import AppConfig
from core.contracts import ChangeVerdict
from core.pipeline import Pipeline, set_default_pipeline
from tests.ui.helpers import (
    SUMMARY,
    InstantAnalyst,
    InstantDetector,
    InstantPreprocessor,
    InstantRegistrar,
    make_pipeline,
)


class GatedAnalyst(InstantAnalyst):
    """`analyze` kapıyı bekler; ilerleme geri çağrısını gerçek analist gibi çağırır."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.progress_callback = None
        self.calls = 0

    def analyze(self, registration, detection):
        self.calls += 1
        if self.progress_callback:
            self.progress_callback(0, 3, "Yapay zekâ modeli yükleniyor…")
        self.entered.set()
        self.gate.wait(10)
        if self.progress_callback:
            self.progress_callback(1, 3, "Bölge 1/3 yorumlandı")
        return super().analyze(registration, detection)


def _enable_llm(window, tmp_path):
    """Sahte model/exe dosyaları oluşturur (llm_ready() True olur)."""
    cfg = window.config
    spec = cfg.llm.model_spec
    for path in (spec.gguf_path(cfg.paths.models_dir), spec.mmproj_path(cfg.paths.models_dir),
                 cfg.paths.llama_server_exe):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    window._on_llm_toggled()


def _gated_pipeline() -> tuple[Pipeline, GatedAnalyst]:
    analyst = GatedAnalyst()
    cfg = AppConfig()
    cfg.runtime.release_cv_models_before_llm = False
    pipe = Pipeline(InstantPreprocessor(), InstantRegistrar(), InstantDetector(),
                    analyst_factory=lambda: analyst, config=cfg)
    return pipe, analyst


def _idle(qtbot, cv) -> None:
    qtbot.waitUntil(lambda: not cv.is_busy() and cv._worker is None, timeout=15000)


def test_partial_result_draws_boxes_then_final_fills_panel(window, populated, tmp_path, qtbot) -> None:
    _, _, recs = populated
    _enable_llm(window, tmp_path)
    pipe, analyst = _gated_pipeline()
    set_default_pipeline(pipe)
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])

    # CV sonucu geldi, LLM kapıda bekliyor: kutular + kartlar hemen, panel yükleniyor.
    qtbot.waitUntil(analyst.entered.is_set, timeout=15000)
    qtbot.waitUntil(lambda: cv.visible_region_count() == (3, 3), timeout=5000)
    panel = cv.ai_panel
    assert panel.state() == "loading" and panel.is_interpreting()
    assert len(panel.cards()) == 3
    assert all(c.region.verdict is ChangeVerdict.UNREVIEWED for c in panel.cards().values())
    qtbot.waitUntil(lambda: "yükleniyor" in panel.stage_label.text(), timeout=3000)
    assert "2 dk" in panel.stage_label.text()
    assert window.status_bar.llm_label.text() == "Gemma: yükleniyor…"
    assert cv.is_llm_active()

    analyst.gate.set()
    _idle(qtbot, cv)
    assert panel.state() == "ready" and panel.summary_label.text() == SUMMARY
    verdicts = {rid: c.region.verdict for rid, c in panel.cards().items()}
    assert verdicts == {1: ChangeVerdict.REAL_CHANGE, 2: ChangeVerdict.REAL_CHANGE, 3: ChangeVerdict.FALSE_ALARM}
    assert "yanlış" in panel.cards()[3].description_label.text().lower()
    assert "Güven %93" in panel.cards()[1].confidence_label.text()
    assert not cv.is_llm_active()
    assert window.status_bar.llm_label.text().startswith("Gemma: hazır")


def test_final_result_keeps_zoom(window, populated, tmp_path, qtbot) -> None:
    _, _, recs = populated
    _enable_llm(window, tmp_path)
    pipe, analyst = _gated_pipeline()
    set_default_pipeline(pipe)
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(analyst.entered.is_set, timeout=15000)
    qtbot.waitUntil(lambda: cv.visible_region_count() == (3, 3), timeout=5000)
    qtbot.wait(150)                      # bekleyen fit_all zamanlayıcıları işlensin
    scale, center = cv.before_view.view_state()
    cv.before_view.apply_state(scale * 2, center)
    before = cv.before_view.transform().m11()
    analyst.gate.set()
    _idle(qtbot, cv)
    assert cv.before_view.transform().m11() == pytest.approx(before)


def test_interpret_button_runs_interpret_and_saves(window, populated, tmp_path, qtbot, monkeypatch) -> None:
    library, _, recs = populated
    _enable_llm(window, tmp_path)
    set_default_pipeline(make_pipeline(with_llm=False))
    cv = window.compare_view
    cv.use_llm = False
    window.open_compare(recs[0], recs[1])
    cv.use_llm = False
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    assert cv.ai_panel.state() == "off" and cv.ai_panel.interpret_enabled()
    assert library.find_comparison(recs[0], recs[1]).summary in (None, "")

    calls: list[object] = []
    release = threading.Event()
    analyst = InstantAnalyst()

    def fake_interpret(result, progress, cancel=None):
        calls.append(result)
        progress("llm_analyze", 0.6, "Bölge 2/3 yorumlandı")
        release.wait(10)
        from dataclasses import replace
        det = analyst.analyze(result.registration, replace(
            result.detection, regions=[replace(r) for r in result.detection.regions]))
        return replace(result, detection=det, llm_used=True, summary=SUMMARY)

    monkeypatch.setattr(core_pipeline, "interpret_comparison", fake_interpret)
    cv.ai_panel.interpretRequested.emit()
    qtbot.waitUntil(lambda: cv.ai_panel.is_interpreting(), timeout=5000)
    qtbot.waitUntil(lambda: "Bölge 2/3" in cv.ai_panel.stage_label.text(), timeout=5000)
    assert len(cv.ai_panel.cards()) == 3                   # kartlar yerinde kalır
    release.set()
    _idle(qtbot, cv)
    assert len(calls) == 1
    assert cv.ai_panel.state() == "ready" and cv.ai_panel.summary_label.text() == SUMMARY
    saved = library.find_comparison(recs[0], recs[1])
    assert saved.summary == SUMMARY
    assert library.load_comparison(saved).detection.regions[2].verdict is ChangeVerdict.FALSE_ALARM


def test_interpret_works_on_cached_result_and_is_cancellable(window, populated, tmp_path, qtbot, monkeypatch) -> None:
    library, _, recs = populated
    _enable_llm(window, tmp_path)
    set_default_pipeline(make_pipeline(with_llm=False))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    # Önbellekten yeniden aç
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)

    started = threading.Event()

    def slow_interpret(result, progress, cancel=None):
        started.set()
        cancel.wait(10)
        raise core_pipeline.ComparisonCancelled()

    monkeypatch.setattr(core_pipeline, "interpret_comparison", slow_interpret)
    cv.interpret()
    qtbot.waitUntil(started.is_set, timeout=5000)
    cv.cancel()
    qtbot.waitUntil(lambda: not cv.has_running_workers(), timeout=10000)
    assert cv.ai_panel.state() == "off" and len(cv.ai_panel.cards()) == 3
    assert not cv.is_llm_active()


def test_hide_false_alarms_with_real_verdicts(window, populated, tmp_path, qtbot) -> None:
    _, _, recs = populated
    _enable_llm(window, tmp_path)
    pipe, analyst = _gated_pipeline()
    analyst.gate.set()
    set_default_pipeline(pipe)
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None and cv._result.llm_used, timeout=15000)
    _idle(qtbot, cv)
    cv.hide_false_check.setChecked(True)
    assert sorted(i.region_id for i in cv.before_view.visible_region_items()) == [1, 2]
    assert cv.ai_panel.visible_card_ids() == [1, 2]
    cv.hide_false_check.setChecked(False)
    assert len(cv.before_view.visible_region_items()) == 3
    # Kutu rengi verdict rengi
    from app.ui.theme import VERDICT_COLORS
    item = cv.before_view._region_items[3]
    assert item.spec.verdict is ChangeVerdict.FALSE_ALARM
    assert VERDICT_COLORS[ChangeVerdict.FALSE_ALARM].lower() == "#94a3b8"


def test_status_bar_texts(window, tmp_path, monkeypatch) -> None:
    sb = window.status_bar
    window._refresh_llm_status()
    assert sb.llm_label.text() == "Gemma: kapalı (model bulunamadı)"
    _enable_llm(window, tmp_path)
    monkeypatch.setattr(core_pipeline, "llm_status",
                        lambda: {"available": True, "running": False, "ngl": None, "model": "Gemma 4 12B"})
    window._refresh_llm_status()
    assert sb.llm_label.text() == "Gemma: hazır (Gemma 4 12B)"
    monkeypatch.setattr(core_pipeline, "llm_status",
                        lambda: {"available": True, "running": True, "ngl": 28, "model": "Gemma 4 12B"})
    window._refresh_llm_status()
    assert sb.llm_label.text() == "Gemma: çalışıyor · GPU katmanı 28/48"
    window._on_llm_activity(True)
    assert sb.llm_label.text() == "Gemma: yükleniyor…"
    window._on_llm_activity(False)
    assert sb.llm_label.text() == "Gemma: çalışıyor · GPU katmanı 28/48"
    window.settings_view.llm_switch.setChecked(False)
    assert sb.llm_label.text() == "Gemma: kapalı"
    assert not window.compare_view.use_llm


def test_settings_llm_controls_reconfigure(window, qtbot, monkeypatch) -> None:
    calls: list[AppConfig] = []
    monkeypatch.setattr(core_pipeline, "configure_default_pipeline", lambda cfg: calls.append(cfg))
    sv = window.settings_view
    sv.llm_model_combo.setCurrentIndex(1)
    qtbot.waitUntil(lambda: len(calls) >= 1, timeout=3000)
    assert calls[-1].llm.model == "gemma-4-e4b"
    assert "gemma-e4b" in sv.llm_model_card.contentLabel.text()      # dosya yok: indirme ipucu
    assert window.status_bar.llm_label.text().startswith("Gemma: kapalı")
    sv.max_regions_slider.setValue(12)
    sv.gpu_mode_combo.setCurrentIndex(1)
    sv.gpu_spin.setValue(20)
    qtbot.waitUntil(lambda: calls[-1].llm.max_regions == 12 and calls[-1].llm.n_gpu_layers == 20, timeout=3000)
    sv.gpu_mode_combo.setCurrentIndex(0)
    qtbot.waitUntil(lambda: calls[-1].llm.n_gpu_layers == "auto", timeout=3000)
    assert not sv.gpu_spin.isEnabled()


def test_close_while_llm_running_cancels_and_shuts_down(window, populated, tmp_path, qtbot, monkeypatch) -> None:
    _, _, recs = populated
    _enable_llm(window, tmp_path)
    pipe, analyst = _gated_pipeline()
    set_default_pipeline(pipe)
    shut: list[int] = []
    monkeypatch.setattr(core_pipeline, "shutdown_default_pipeline", lambda: shut.append(1))
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(analyst.entered.is_set, timeout=15000)
    killed = []
    monkeypatch.setattr(type(window), "_kill_llm_server", staticmethod(lambda: (killed.append(1), analyst.gate.set())))
    window.close()
    assert shut == [1]
    assert killed == [1]
    assert not window.compare_view.has_running_workers()


@pytest.mark.llm
def test_real_gemma_interpret_button(qtbot, tmp_path) -> None:
    """Gerçek Gemma: piknik çiftinde Yorumla -> en az bir kart verdict'li (~3 dk; model yoksa atlanır)."""
    import os
    from datetime import datetime
    from pathlib import Path

    from app.prefs import UiPrefs
    from app.storage import Library
    from app.ui.main_window import MainWindow
    from core.config import load_config
    from core.contracts import ImageSource

    root = Path(__file__).resolve().parents[2]
    data = root / "tests" / "data" / "real" / "piknik"
    cfg = load_config()
    spec = cfg.llm.model_spec
    if not (spec.gguf_path(cfg.paths.models_dir).is_file() and cfg.paths.llama_server_exe.is_file()
            and (data / "before.jpg").is_file()):
        pytest.skip("Gemma modeli / llama-server / piknik verisi yok")
    cfg.llm.enabled = True
    cfg.paths.library_dir = tmp_path / "library"
    lib = Library(tmp_path / "library")
    folder = lib.repo.create_folder("Piknik")
    b = lib.import_image(data / "before.jpg", folder.id, datetime(2026, 1, 1), ImageSource.DRONE)
    a = lib.import_image(data / "after.jpg", folder.id, datetime(2026, 1, 8), ImageSource.DRONE)
    core_pipeline.configure_default_pipeline(cfg)
    win = MainWindow(lib, cfg, UiPrefs(), tmp_path / "c.json", tmp_path / "p.json", probe_device=False,
                     reconfigure_pipeline=False)
    qtbot.addWidget(win)
    win.show()
    try:
        cv = win.compare_view
        cv.use_llm = False
        win.switchTo(cv)
        cv.use_llm = False
        cv.open_pair(b, a)
        qtbot.waitUntil(lambda: cv._result is not None and not cv.is_busy(), timeout=240000)
        assert cv.ai_panel.interpret_enabled()
        cv.interpret()
        qtbot.waitUntil(lambda: not cv.is_busy() and cv._worker is None, timeout=600000)
        cards = cv.ai_panel.cards()
        assert cards and any(c.region.verdict is not ChangeVerdict.UNREVIEWED for c in cards.values())
        assert cv.ai_panel.state() == "ready"
    finally:
        win.close()
        lib.close()
