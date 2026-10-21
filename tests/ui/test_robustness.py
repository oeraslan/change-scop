"""Faz 1 düzeltme turu: UI thread'i bloklanmaz, ayarlar pipeline'a ulaşır, kapanış/iptal temiz, Yorumla durumu."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from app.ui import workers
from app.ui.qt_utils import heatmap_to_qimage
from core import pipeline as core_pipeline
from core.pipeline import Pipeline, set_default_pipeline
from tests.ui.helpers import (
    BlockingRegistrar,
    InstantDetector,
    InstantPreprocessor,
    make_pipeline,
)


def _wait_idle(qtbot, view, timeout: int = 15000) -> None:
    qtbot.waitUntil(lambda: not view.has_running_workers() and view._worker is None, timeout=timeout)


class _Ticker:
    """UI olay döngüsünün ne kadar bloklandığını ölçer (en büyük boşluk)."""

    def __init__(self) -> None:
        self.last = time.monotonic()
        self.max_gap = 0.0
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self) -> None:
        now = time.monotonic()
        self.max_gap = max(self.max_gap, now - self.last)
        self.last = now

    def stop(self) -> None:
        self.timer.stop()


# ------------------------------------------------------------ 1. ağır disk işi worker'da

def test_save_and_cache_load_run_off_the_ui_thread(window, populated, qtbot, monkeypatch) -> None:
    library, _, recs = populated
    main_thread = threading.get_ident()
    seen: dict[str, int] = {}
    save, load = library.save_comparison, library.load_comparison

    def slow_save(*a, **k):
        seen["save"] = threading.get_ident()
        time.sleep(0.6)
        return save(*a, **k)

    def slow_load(*a, **k):
        seen["load"] = threading.get_ident()
        time.sleep(0.6)
        return load(*a, **k)

    monkeypatch.setattr(library, "save_comparison", slow_save)
    monkeypatch.setattr(library, "load_comparison", slow_load)
    set_default_pipeline(make_pipeline())
    cv = window.compare_view
    ticker = _Ticker()
    try:
        window.open_compare(recs[0], recs[1])
        qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
        _wait_idle(qtbot, cv)
        assert library.find_comparison(recs[0], recs[1]) is not None
        window.open_compare(recs[0], recs[1])               # önbellekten
        qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
        _wait_idle(qtbot, cv)
    finally:
        ticker.stop()
    assert seen["save"] != main_thread and seen["load"] != main_thread
    assert ticker.max_gap < 0.45, f"UI olay döngüsü {ticker.max_gap:.2f} s bloklandı"


def test_previews_load_in_worker_and_show_before_result(window, populated, qtbot) -> None:
    _, _, recs = populated
    blocking = BlockingRegistrar()
    set_default_pipeline(Pipeline(InstantPreprocessor(), blocking, InstantDetector(),
                                  config=make_pipeline().config))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: not cv.before_view.image_rect.isEmpty(), timeout=10000)
    assert cv._result is None and cv.is_busy()
    blocking.gate.set()
    _wait_idle(qtbot, cv)


def test_library_repository_is_safe_across_threads(populated) -> None:
    library, folders, _ = populated
    errors: list[Exception] = []

    def hammer() -> None:
        try:
            for _ in range(60):
                library.repo.list_images()
                library.repo.list_folders()
                library.repo.total_image_count()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(20):
        library.repo.create_folder(f"eşzamanlı {i}", folders["a"].id)
    for t in threads:
        t.join()
    assert not errors


# ------------------------------------------------------------ 2. ayarlar pipeline'a ulaşır

def _spy_configure(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(core_pipeline, "configure_default_pipeline", lambda cfg: calls.append(cfg))
    return calls


def test_settings_change_calls_configure_with_new_config(window, qtbot, monkeypatch) -> None:
    calls = _spy_configure(monkeypatch)
    sv = window.settings_view
    sv.sens_slider.setValue(77)
    sv.sens_slider.setValue(78)                       # kaydırma: tek çağrıya birleşir
    qtbot.waitUntil(lambda: len(calls) >= 1, timeout=3000)
    assert len(calls) == 1 and calls[0].detection.sensitivity == 0.78
    sv.tile_combo.setCurrentIndex(0)
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=3000)
    assert calls[1].registration.tile_size_px == 512
    sv.device_combo.setCurrentIndex(2)
    qtbot.waitUntil(lambda: len(calls) == 3, timeout=3000)
    assert calls[2].runtime.device == "cpu"
    sv.llm_switch.setChecked(False)
    qtbot.waitUntil(lambda: len(calls) == 4, timeout=3000)
    assert calls[3].llm.enabled is False
    assert "bir sonraki açılışta" not in " ".join(
        w.text() for w in sv.findChildren(type(sv.sens_label)) if hasattr(w, "text"))


def test_configure_does_not_wait_for_running_comparison(window, populated, qtbot, monkeypatch) -> None:
    _, _, recs = populated
    calls = _spy_configure(monkeypatch)
    blocking = BlockingRegistrar()
    set_default_pipeline(Pipeline(InstantPreprocessor(), blocking, InstantDetector(),
                                  config=make_pipeline().config))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(blocking.entered.is_set, timeout=10000)
    window.settings_view.tile_combo.setCurrentIndex(1)
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=3000)   # pipeline artık koşuyu beklemez
    assert calls[0].registration.tile_size_px == 768
    blocking.gate.set()
    _wait_idle(qtbot, cv)


def test_main_configures_default_pipeline(tmp_path: Path, qtbot, monkeypatch) -> None:
    from app import main as app_main

    calls = _spy_configure(monkeypatch)
    cfg_path = tmp_path / "cfg.json"
    win = app_main.build_window(tmp_path / "lib", cfg_path, tmp_path / "prefs.json", probe_device=False)
    qtbot.addWidget(win)
    assert len(calls) == 1 and calls[0].paths.library_dir == tmp_path / "lib"
    win.close()


def test_demo_mode_keeps_demo_pipeline(tmp_path: Path, qtbot, monkeypatch) -> None:
    from app import main as app_main

    calls = _spy_configure(monkeypatch)
    win = app_main.build_window(tmp_path / "lib", tmp_path / "cfg.json", tmp_path / "prefs.json",
                                probe_device=False, demo=True)
    qtbot.addWidget(win)
    assert calls == []
    win.settings_view.tile_combo.setCurrentIndex(0)
    qtbot.wait(700)
    assert calls == []                                 # demo pipeline'ı ezilmez
    win.close()


# ------------------------------------------------------------ 3. kapanış

def test_close_event_shuts_down_pipeline(window, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(core_pipeline, "shutdown_default_pipeline", lambda: calls.append(1))
    window.close()
    window.close()
    assert calls == [1]


def test_close_cancels_running_worker_without_hanging(window, populated, qtbot, monkeypatch) -> None:
    _, _, recs = populated
    calls: list = []
    monkeypatch.setattr(core_pipeline, "shutdown_default_pipeline", lambda: calls.append(1))
    blocking = BlockingRegistrar()
    set_default_pipeline(Pipeline(InstantPreprocessor(), blocking, InstantDetector(),
                                  config=make_pipeline().config))
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(blocking.entered.is_set, timeout=10000)
    threading.Timer(0.4, blocking.gate.set).start()
    t0 = time.monotonic()
    window.close()
    assert time.monotonic() - t0 < 5
    assert not window.compare_view.has_running_workers()
    assert calls == [1]


# ------------------------------------------------------------ 4. iptal bloklamaz

def test_cancel_does_not_block_and_late_result_is_ignored(window, populated, qtbot) -> None:
    _, _, recs = populated
    blocking = BlockingRegistrar()
    set_default_pipeline(Pipeline(InstantPreprocessor(), blocking, InstantDetector(),
                                  config=make_pipeline().config))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(blocking.entered.is_set, timeout=10000)
    t0 = time.monotonic()
    cv.cancel()
    assert time.monotonic() - t0 < 0.5
    assert ("info", "İptal edildi") in cv.active_infobars()
    assert not cv.progress_host.isVisible() and cv._worker is None
    assert cv.has_running_workers()                    # iş hâlâ arka planda; referans tutuluyor
    blocking.gate.set()
    _wait_idle(qtbot, cv)
    assert cv._result is None                          # geç gelen sonuç yok sayıldı


def test_new_comparison_waits_for_cancelled_one(window, populated, qtbot) -> None:
    library, _, recs = populated
    blocking = BlockingRegistrar()
    set_default_pipeline(Pipeline(InstantPreprocessor(), blocking, InstantDetector(),
                                  config=make_pipeline().config))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(blocking.entered.is_set, timeout=10000)
    cv.cancel()
    window.open_compare(recs[2], recs[3])              # eski iş sürerken yenisi: ertelenir
    assert cv.is_busy() and cv._worker is None
    blocking.gate.set()
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _wait_idle(qtbot, cv)
    assert library.find_comparison(recs[2], recs[3]) is not None
    assert library.find_comparison(recs[0], recs[1]) is None   # iptal edilen kaydedilmedi


# ------------------------------------------------------------ 5. ön ısıtma

def test_device_probe_warms_up_after_resolving(qtbot, monkeypatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(workers, "warm_up_imports", lambda: order.append("warm"))
    probe = workers.DeviceProbe("cpu")
    probe.resolved.connect(lambda text: order.append(text), Qt.ConnectionType.DirectConnection)
    probe.start()
    assert probe.wait(10000)
    QApplication.processEvents()
    assert order == ["CPU", "warm"]


def test_warm_up_imports_targets() -> None:
    import inspect

    src = inspect.getsource(workers.warm_up_imports)
    for name in ("torch", "lightglue", "core.registration", "core.detection"):
        assert name in src


# ------------------------------------------------------------ 6. Yorumla durumu

def test_interpret_disabled_when_model_missing(window) -> None:
    panel = window.compare_view.ai_panel
    assert window.config.llm.enabled
    assert not panel.interpret_enabled()
    assert not panel.off_state.button.isEnabled()
    assert "scripts/download_models.py --only gemma-12b" in panel.off_state.text_label.text()
    assert "Gemma modeli indirilmemiş" in panel.off_state.text_label.text()


def test_interpret_message_when_llm_disabled_in_settings(window) -> None:
    window.settings_view.llm_switch.setChecked(False)
    panel = window.compare_view.ai_panel
    assert not panel.off_state.button.isEnabled()
    assert "ayarlarda kapalı" in panel.off_state.title_label.text()


def test_interpret_enabled_when_model_and_server_exist(window, tmp_path: Path) -> None:
    cfg = window.config
    models = tmp_path / "models"
    gguf = cfg.llm.model_spec.gguf_path(models)
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"x")
    cfg.llm.model_spec.mmproj_path(models).write_bytes(b"x")
    bin_dir = tmp_path / "bin"
    (bin_dir / "llama").mkdir(parents=True)
    (bin_dir / "llama" / "llama-server.exe").write_bytes(b"x")
    cfg.paths.models_dir, cfg.paths.bin_dir = models, bin_dir
    window._refresh_llm_status()
    panel = window.compare_view.ai_panel
    assert panel.interpret_enabled() and panel.off_state.button.isEnabled()
    assert window.status_bar.llm_label.text().startswith("Gemma: hazır")


def test_interpret_noop_when_disabled(window, populated) -> None:
    _, _, recs = populated
    cv = window.compare_view
    cv._before_rec, cv._after_rec = recs[0], recs[1]
    cv.interpret()
    assert not cv.is_busy()


# ------------------------------------------------------------ 9. ısı haritası

def test_heatmap_is_visible_with_low_valued_map() -> None:
    cm = np.zeros((100, 100), dtype=np.float32)
    cm[20:40, 20:40] = 0.3                              # gerçek haritalarda tipik düşük değerler
    cm[60:80, 60:80] = 0.02                             # gürültü
    img = heatmap_to_qimage(cm)
    assert img.pixelColor(30, 30).alpha() > 200
    assert img.pixelColor(70, 70).alpha() < 20
    assert img.pixelColor(5, 5).alpha() == 0
    hot = img.pixelColor(30, 30)
    assert hot.red() != hot.blue()                      # renk haritası uygulanmış


def test_heatmap_layer_opacity_is_about_half(window) -> None:
    from app.ui.sync_graphics_view import HEAT_OPACITY

    assert 0.45 <= HEAT_OPACITY <= 0.55
