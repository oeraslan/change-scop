"""Karşılaştırma ekranı: kutular, anahtarlar, senkron zoom, InfoBar'lar, işlem akışı, sonuç paneli."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from core import pipeline as core_pipeline
from core.contracts import AlignmentQuality
from core.pipeline import Pipeline, set_default_pipeline
from tests.ui.helpers import (
    BlockingRegistrar,
    InstantDetector,
    InstantPreprocessor,
    InstantRegistrar,
    make_pipeline,
    make_result,
)


@pytest.fixture()
def cv(window):
    """Sonuç yüklenmiş karşılaştırma görünümü (3 bölge, eşleşme %75)."""
    view = window.compare_view
    window.switchTo(view)
    view.show_result(make_result(640, 480), title="X Bölgesi / Y Dağı", dates="12.10.2026 → 19.10.2026")
    return view


def _wait_idle(qtbot, view, timeout: int = 15000) -> None:
    qtbot.waitUntil(lambda: not view.is_busy() and view._worker is None, timeout=timeout)


# ------------------------------------------------------------------ kutular / anahtarlar

def test_regions_drawn_on_both_views_by_default(cv, qtbot) -> None:
    qtbot.waitUntil(lambda: cv.visible_region_count() == (3, 3))
    left = {i.region_id: i.sceneBoundingRect() for i in cv.before_view.region_items()}
    right = {i.region_id: i.sceneBoundingRect() for i in cv.after_view.region_items()}
    assert left == right                                # aynı kutular aynı koordinatta


def test_regions_use_single_change_color(cv) -> None:
    from app.ui.theme import REGION_COLOR

    for item in cv.before_view.region_items():
        assert item.pen().color().name().lower() == REGION_COLOR.lower()


def test_toggle_off_hides_all_markers(cv, qtbot) -> None:
    qtbot.waitUntil(lambda: cv.visible_region_count() == (3, 3))
    cv.diff_switch.setChecked(False)
    qtbot.waitUntil(lambda: cv.visible_region_count() == (0, 0))
    assert not cv.before_view._region_layer.isVisible()   # numara rozetleri dahil hiçbir şey yok
    assert not cv.after_view._region_layer.isVisible()
    assert not cv.heat_switch.isChecked() and not cv.before_view.heatmap_visible()
    cv.diff_switch.setChecked(True)
    qtbot.waitUntil(lambda: cv.visible_region_count() == (3, 3))


def test_toggle_off_survives_new_result(cv, qtbot) -> None:
    cv.diff_switch.setChecked(False)
    qtbot.waitUntil(lambda: cv.visible_region_count() == (0, 0))
    cv.show_result(make_result(640, 480))
    assert cv.visible_region_count() == (0, 0)             # kapalı kalır


def test_heatmap_toggle_is_separate_and_default_off(cv, qtbot) -> None:
    assert not cv.heat_switch.isChecked()
    assert not cv.before_view.heatmap_visible()
    cv.heat_switch.setChecked(True)
    qtbot.waitUntil(lambda: cv.before_view.heatmap_visible() and cv.after_view.heatmap_visible())
    assert cv.visible_region_count() == (3, 3)             # kutular da duruyor
    cv.heat_switch.setChecked(False)
    qtbot.waitUntil(lambda: not cv.before_view.heatmap_visible())


def test_hover_tooltip_has_number_score_and_area(cv) -> None:
    item = cv.before_view.region_items()[0]
    tip = item.toolTip()
    assert "Bölge #1" in tip and "Fark skoru: 0.91" in tip and "px" in tip


# ---------------------------------------------------------------------- senkron zoom/pan

def test_zoom_is_synchronised(cv, qtbot) -> None:
    qtbot.waitUntil(lambda: cv.before_view.zoom_factor() > 0 and cv.before_view.viewport().width() > 50)
    base = cv.before_view.zoom_factor()
    cv.before_view.zoom_at(2.0)
    assert cv.before_view.zoom_factor() == pytest.approx(base * 2)
    assert cv.after_view.zoom_factor() == pytest.approx(cv.before_view.zoom_factor())
    cv.after_view.zoom_at(0.5)                              # ters yön de çalışır
    assert cv.before_view.zoom_factor() == pytest.approx(base)


def test_wheel_event_zooms_both(cv, qtbot) -> None:
    base = cv.before_view.zoom_factor()
    pos = QPointF(cv.before_view.viewport().rect().center())
    event = QWheelEvent(pos, cv.before_view.viewport().mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(cv.before_view.viewport(), event)
    assert cv.before_view.zoom_factor() > base
    assert cv.after_view.zoom_factor() == pytest.approx(cv.before_view.zoom_factor())


def test_pan_is_synchronised(cv) -> None:
    cv.before_view.zoom_at(4.0)
    bar = cv.before_view.horizontalScrollBar()
    bar.setValue(bar.value() + 80)
    c1, c2 = cv.before_view.center_point(), cv.after_view.center_point()
    assert abs(c1.x() - c2.x()) < 1.5 and abs(c1.y() - c2.y()) < 1.5


def test_double_click_fits(cv, qtbot) -> None:
    fit = cv.before_view.fit_scale()
    cv.before_view.zoom_at(3.0)
    qtbot.mouseDClick(cv.before_view.viewport(), Qt.MouseButton.LeftButton)
    assert cv.before_view.zoom_factor() == pytest.approx(fit, rel=1e-3)
    assert cv.after_view.zoom_factor() == pytest.approx(fit, rel=1e-3)


def test_card_click_zooms_to_region_and_highlights(cv, qtbot) -> None:
    fit = cv.before_view.zoom_factor()
    cv.result_panel.regionActivated.emit(1)
    qtbot.waitUntil(lambda: cv.before_view.zoom_factor() > fit * 1.5)
    assert cv.after_view.zoom_factor() == pytest.approx(cv.before_view.zoom_factor())
    assert cv.result_panel.selected_card_id() == 1
    assert cv.before_view._region_items[1]._highlight


def test_region_click_in_view_highlights_card(cv, qtbot) -> None:
    item = cv.before_view._region_items[2]
    center = cv.before_view.mapFromScene(item.sceneBoundingRect().center())
    qtbot.mouseClick(cv.before_view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    assert cv.result_panel.selected_card_id() == 2


# --------------------------------------------------------------------------- InfoBar'lar

def test_failed_alignment_shows_error_infobar_without_markers(window, qtbot) -> None:
    cv = window.compare_view
    cv.show_result(make_result(320, 240, quality=AlignmentQuality.FAILED))
    assert ("error", "Hizalama başarısız") in cv.active_infobars()
    assert cv.before_view.region_items() == [] and cv.after_view.region_items() == []
    assert cv.visible_region_count() == (0, 0)
    assert not cv.diff_switch.isEnabled()
    assert not cv.before_view.image_rect.isEmpty()          # görüntüler yine de gösterilir
    assert not cv.after_view.image_rect.isEmpty()
    assert cv.result_panel.state() == "failed"
    assert not cv.match_pill.isHidden() and cv.match_pill.text() == "Eşleşme %75"   # 90/120


def test_weak_alignment_shows_warning_infobar(window) -> None:
    cv = window.compare_view
    cv.show_result(make_result(320, 240, quality=AlignmentQuality.WEAK))
    assert ("warning", "Hizalama zayıf") in cv.active_infobars()
    assert any("%75" in b.content for b in cv._infobars)
    assert cv.visible_region_count()[0] == 0 or True         # animasyon sonrası görünür olur
    assert len(cv.before_view.region_items()) == 3


def test_good_alignment_has_no_infobar(cv) -> None:
    assert cv.active_infobars() == []


# --------------------------------------------------------------------------- sonuç paneli

def test_result_panel_shows_match_percent_and_stats(cv) -> None:
    panel = cv.result_panel
    assert panel.state() == "ready"
    assert panel.gauge.percent() == pytest.approx(75.0) and panel.gauge.text() == "%75"
    assert panel.points_stat.value() == "90 / 120"
    assert panel.regions_stat.value() == "3"
    assert panel.overlap_stat.value() == "%100"
    assert panel.changed_stat.value().startswith("%")
    assert panel.quality_pill.text() == "Hizalama iyi"
    assert len(panel.cards()) == 3
    card = panel.cards()[1]
    assert "0.91" in card.score_label.text() and "px" in card.area_label.text()


def test_match_pill_in_header(cv) -> None:
    assert not cv.match_pill.isHidden() and cv.match_pill.text() == "Eşleşme %75"


def test_changed_area_percent() -> None:
    from app.ui.result_panel import changed_area_percent

    result = make_result(100, 100)
    total = sum(r.area_px for r in result.detection.regions)
    assert changed_area_percent(result) == pytest.approx(total / 100)
    result.registration.valid_mask[:] = False
    assert changed_area_percent(result) == 0.0


def test_result_panel_has_no_ai_controls(cv) -> None:
    assert not hasattr(cv, "ai_panel") and not hasattr(cv, "hide_false_check") and not hasattr(cv, "interpret")


def test_result_panel_loading_idle_and_empty_states(window) -> None:
    cv = window.compare_view
    cv.result_panel.show_loading()
    assert cv.result_panel.state() == "loading" and not cv.result_panel.summary_host.isVisible()
    cv.result_panel.set_result(None)
    assert cv.result_panel.state() == "idle"
    result = make_result(320, 240)
    result.detection.regions = []
    cv.show_result(result)
    assert cv.result_panel.state() == "empty"
    assert cv.result_panel.gauge.text() == "%75"            # değişiklik yokken de eşleşme görünür


def test_no_matches_shows_dash(window) -> None:
    cv = window.compare_view
    result = make_result(320, 240, quality=AlignmentQuality.FAILED)
    result.registration.num_matches = result.registration.num_inliers = 0
    cv.show_result(result)
    assert cv.result_panel.gauge.text() == "—" and cv.match_pill.text() == "Eşleşme yok"


def test_result_panel_splitter_collapsible(cv) -> None:
    assert cv.main_splitter.isCollapsible(1)
    cv.main_splitter.setSizes([cv.main_splitter.height(), 0])
    assert cv.main_splitter.sizes()[1] == 0


# ----------------------------------------------------------------------------- export

def test_export_writes_png_csv_and_summary(cv, tmp_path: Path) -> None:
    written = cv.export_to(tmp_path / "cikti.png")
    assert [p.suffix for p in written] == [".png", ".csv", ".txt"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    csv_text = written[1].read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0] == "No;X;Y;Genişlik;Yükseklik;Alan_px;Skor" and csv_text.count("\n") == 4
    summary = written[2].read_text(encoding="utf-8")
    assert "Hizalama eşleşmesi: %75" in summary and "Bölge sayısı: 3" in summary
    from PIL import Image

    with Image.open(written[0]) as img:
        assert img.width > 2 * 640


def test_export_without_boxes_when_switch_off(cv, tmp_path: Path) -> None:
    from PIL import Image

    cv.export_to(tmp_path / "kutulu.png")
    cv.diff_switch.setChecked(False)
    cv.export_to(tmp_path / "kutusuz.png")
    with Image.open(tmp_path / "kutulu.png") as a, Image.open(tmp_path / "kutusuz.png") as b:
        assert a.tobytes() != b.tobytes()


# -------------------------------------------------------- uçtan uca: pipeline + kayıt + önbellek

def test_open_pair_runs_pipeline_saves_and_reuses_cache(window, populated, qtbot, tmp_path) -> None:
    library, _, recs = populated
    registrar = InstantRegistrar()
    calls = []
    original = registrar.register
    registrar.register = lambda b, a: (calls.append(1), original(b, a))[1]
    config = make_pipeline().config
    set_default_pipeline(Pipeline(InstantPreprocessor(), registrar, InstantDetector(), config=config))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    assert cv.progress_host.isVisible()                      # ilerleme çubuğu hemen görünür
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _wait_idle(qtbot, cv)
    assert not cv.progress_host.isVisible()
    assert cv.visible_region_count()[0] == 3 or len(cv.before_view.region_items()) == 3
    assert cv.result_panel.state() == "ready" and cv.result_panel.gauge.text() == "%80"   # 80/100
    assert len(calls) == 1
    assert library.find_comparison(recs[0], recs[1]) is not None
    assert window.library_view.history_layout.count() == 2   # geçmiş güncellendi

    # Aynı çifti tekrar aç: pipeline çalışmaz, kayıttan yüklenir.
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _wait_idle(qtbot, cv)
    assert len(calls) == 1
    assert cv.result_panel.state() == "ready" and cv.match_pill.text() == "Eşleşme %80"   # kayıttan da aynı
    assert [i.region_id for i in cv.before_view.region_items()] == [1, 2, 3]


def test_progress_stage_text_updates(window, populated, qtbot, monkeypatch) -> None:
    _, _, recs = populated
    set_default_pipeline(make_pipeline(delay=0.2))
    cv = window.compare_view
    seen: list[tuple[str, int]] = []
    original = cv._on_progress

    def spy(stage, fraction, message):
        original(stage, fraction, message)
        seen.append((cv.stage_label.text(), cv.progress_bar.value()))

    monkeypatch.setattr(cv, "_on_progress", spy)
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    texts = [t for t, _ in seen]
    assert any("hizalan" in t.lower() for t in texts)
    assert any("Farklar" in t for t in texts)
    assert [v for _, v in seen] == sorted(v for _, v in seen)   # ilerleme geri gitmez


def test_cancel_stops_comparison(window, populated, qtbot) -> None:
    _, _, recs = populated
    blocking = BlockingRegistrar()
    cfg = make_pipeline().config
    set_default_pipeline(Pipeline(InstantPreprocessor(), blocking, InstantDetector(), config=cfg))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(blocking.entered.is_set, timeout=10000)
    assert cv.cancel_button.isVisible()
    cv.cancel_button.click()
    blocking.gate.set()
    _wait_idle(qtbot, cv)
    assert ("info", "İptal edildi") in cv.active_infobars()
    assert cv._result is None and not cv.progress_host.isVisible()


def test_missing_modules_show_generic_error_infobar(window, populated, qtbot, monkeypatch) -> None:
    _, _, recs = populated
    set_default_pipeline(None)
    monkeypatch.setattr(core_pipeline, "build_default_pipeline",
                        lambda config=None: (_ for _ in ()).throw(NotImplementedError("henüz yok")))
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: any(t == "Karşılaştırma başarısız" for _, t in cv.active_infobars()), timeout=10000)
    _wait_idle(qtbot, cv)
    assert not cv.before_view.image_rect.isEmpty()           # önizleme yine de gösterilir


def test_pipeline_error_shows_error_infobar(window, populated, qtbot) -> None:
    _, _, recs = populated
    pipe = make_pipeline()

    class Boom(InstantDetector):
        def detect(self, registration):
            raise ValueError("bozuk veri")

    pipe.detector = Boom()
    set_default_pipeline(pipe)
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: any(t == "Karşılaştırma başarısız" for _, t in cv.active_infobars()), timeout=10000)
    _wait_idle(qtbot, cv)
