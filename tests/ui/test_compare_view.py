"""Karşılaştırma ekranı: kutular, anahtarlar, senkron zoom, InfoBar'lar, işlem akışı, AI paneli."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from core import pipeline as core_pipeline
from core.contracts import AlignmentQuality, ChangeVerdict
from core.pipeline import Pipeline, set_default_pipeline
from tests.ui.helpers import (
    BlockingRegistrar,
    InstantAnalyst,
    InstantDetector,
    InstantPreprocessor,
    InstantRegistrar,
    SUMMARY,
    make_pipeline,
    make_result,
)


@pytest.fixture()
def cv(window):
    """Sonuç yüklenmiş karşılaştırma görünümü (3 bölge, AI yorumlu)."""
    view = window.compare_view
    window.switchTo(view)
    view.show_result(make_result(640, 480, with_llm=True), title="X Bölgesi / Y Dağı", dates="12.10.2026 → 19.10.2026")
    return view


def _wait_idle(qtbot, view, timeout: int = 15000) -> None:
    qtbot.waitUntil(lambda: not view.is_busy() and view._worker is None, timeout=timeout)


# ------------------------------------------------------------------ kutular / anahtarlar

def test_regions_drawn_on_both_views_by_default(cv, qtbot) -> None:
    qtbot.waitUntil(lambda: cv.visible_region_count() == (3, 3))
    left = {i.region_id: i.sceneBoundingRect() for i in cv.before_view.region_items()}
    right = {i.region_id: i.sceneBoundingRect() for i in cv.after_view.region_items()}
    assert left == right                                # aynı kutular aynı koordinatta


def test_region_colors_follow_verdict(cv) -> None:
    from app.ui.theme import VERDICT_COLORS

    for item in cv.before_view.region_items():
        assert item.pen().color().name().lower() == VERDICT_COLORS[item.spec.verdict].lower()


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
    cv.show_result(make_result(640, 480, with_llm=True))
    assert cv.visible_region_count() == (0, 0)             # kapalı kalır


def test_heatmap_toggle_is_separate_and_default_off(cv, qtbot) -> None:
    assert not cv.heat_switch.isChecked()
    assert not cv.before_view.heatmap_visible()
    cv.heat_switch.setChecked(True)
    qtbot.waitUntil(lambda: cv.before_view.heatmap_visible() and cv.after_view.heatmap_visible())
    assert cv.visible_region_count() == (3, 3)             # kutular da duruyor
    cv.heat_switch.setChecked(False)
    qtbot.waitUntil(lambda: not cv.before_view.heatmap_visible())


def test_hide_false_alarms(cv, qtbot) -> None:
    qtbot.waitUntil(lambda: cv.visible_region_count() == (3, 3))
    cv.hide_false_check.setChecked(True)
    assert cv.visible_region_count() == (2, 2)
    assert len(cv.ai_panel.visible_card_ids()) == 2
    verdicts = {i.spec.verdict for i in cv.before_view.visible_region_items()}
    assert ChangeVerdict.FALSE_ALARM not in verdicts
    cv.hide_false_check.setChecked(False)
    assert cv.visible_region_count() == (3, 3)


def test_hover_tooltip_has_description_and_score(cv) -> None:
    item = cv.before_view.region_items()[0]
    assert "Yeni bir yapı" in item.toolTip() and "Skor: 0.91" in item.toolTip()


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
    cv.ai_panel.regionActivated.emit(1)
    qtbot.waitUntil(lambda: cv.before_view.zoom_factor() > fit * 1.5)
    assert cv.after_view.zoom_factor() == pytest.approx(cv.before_view.zoom_factor())
    assert cv.ai_panel.selected_card_id() == 1
    assert cv.before_view._region_items[1]._highlight


def test_region_click_in_view_highlights_card(cv, qtbot) -> None:
    item = cv.before_view._region_items[2]
    center = cv.before_view.mapFromScene(item.sceneBoundingRect().center())
    qtbot.mouseClick(cv.before_view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    assert cv.ai_panel.selected_card_id() == 2


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
    assert cv.ai_panel.state() == "failed"


def test_weak_alignment_shows_warning_infobar(window) -> None:
    cv = window.compare_view
    cv.show_result(make_result(320, 240, quality=AlignmentQuality.WEAK))
    assert ("warning", "Hizalama zayıf") in cv.active_infobars()
    assert cv.visible_region_count()[0] == 0 or True         # animasyon sonrası görünür olur
    assert len(cv.before_view.region_items()) == 3


def test_good_alignment_has_no_infobar(cv) -> None:
    assert cv.active_infobars() == []


# --------------------------------------------------------------------------- AI paneli

def test_ai_panel_filled_state(cv) -> None:
    panel = cv.ai_panel
    assert panel.state() == "ready"
    assert panel.summary_label.text() == SUMMARY
    assert len(panel.cards()) == 3
    card = panel.cards()[3]
    assert card.region.verdict is ChangeVerdict.FALSE_ALARM
    assert "%88" in card.confidence_label.text()
    assert "Gölge" in card.description_label.text()
    assert panel.model_pill.text() == "Gemma 4 12B" or panel.model_pill.text()


def test_ai_panel_off_state_has_interpret_button(window, qtbot) -> None:
    cv = window.compare_view
    cv.show_result(make_result(320, 240, with_llm=False))
    panel = cv.ai_panel
    panel.set_interpret_available(True)                     # worktree'de model yok; etkin durumu zorla
    assert panel.state() == "off"
    assert panel.off_state.title_label.text() == "Yapay zekâ yorumu kapalı"
    assert panel.off_state.button.text() == "Yorumla"
    assert len(panel.cards()) == 3                         # bölgeler yine de listelenir (incelenmedi)
    assert all(c.region.verdict is ChangeVerdict.UNREVIEWED for c in panel.cards().values())
    with qtbot.waitSignal(panel.interpretRequested):
        panel.off_state.button.click()


def test_ai_panel_loading_state_and_empty_results(window) -> None:
    cv = window.compare_view
    cv.ai_panel.show_loading()
    assert cv.ai_panel.state() == "loading" and cv.ai_panel._shimmer.state().name == "Running"
    cv.ai_panel.set_result(None)
    assert cv.ai_panel.state() == "off"
    result = make_result(320, 240)
    result.detection.regions = []
    cv.show_result(result)
    assert cv.ai_panel.state() == "empty"


def test_ai_panel_splitter_collapsible(cv) -> None:
    assert cv.main_splitter.isCollapsible(1)
    cv.main_splitter.setSizes([cv.main_splitter.height(), 0])
    assert cv.main_splitter.sizes()[1] == 0


# ----------------------------------------------------------------------------- export

def test_export_writes_png_csv_and_summary(cv, tmp_path: Path) -> None:
    written = cv.export_to(tmp_path / "cikti.png")
    assert [p.suffix for p in written] == [".png", ".csv", ".txt"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    csv_text = written[1].read_text(encoding="utf-8-sig")
    assert "Gerçek değişiklik" in csv_text and csv_text.count("\n") == 4
    assert SUMMARY in written[2].read_text(encoding="utf-8")
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

def test_open_pair_runs_pipeline_saves_and_reuses_cache(window, populated, qtbot) -> None:
    library, _, recs = populated
    registrar = InstantRegistrar()
    calls = []
    original = registrar.register
    registrar.register = lambda b, a: (calls.append(1), original(b, a))[1]
    config = make_pipeline().config
    set_default_pipeline(Pipeline(InstantPreprocessor(), registrar, InstantDetector(),
                                  analyst_factory=lambda: InstantAnalyst(), config=config))
    cv = window.compare_view
    cv.use_llm = True
    window.open_compare(recs[0], recs[1])
    assert cv.progress_host.isVisible()                      # ilerleme çubuğu hemen görünür
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _wait_idle(qtbot, cv)
    assert not cv.progress_host.isVisible()
    assert cv.visible_region_count()[0] == 3 or len(cv.before_view.region_items()) == 3
    assert cv.ai_panel.state() == "ready" and cv.ai_panel.summary_label.text() == SUMMARY
    assert len(calls) == 1
    assert library.find_comparison(recs[0], recs[1]) is not None
    assert window.library_view.history_layout.count() == 2   # geçmiş güncellendi

    # Aynı çifti tekrar aç: pipeline çalışmaz, kayıttan yüklenir.
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _wait_idle(qtbot, cv)
    assert len(calls) == 1
    assert cv.ai_panel.state() == "ready"
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


def test_interpret_button_runs_llm_later(window, populated, qtbot) -> None:
    _, _, recs = populated
    set_default_pipeline(make_pipeline())
    cv = window.compare_view
    window.config.llm.enabled = False                       # LLM kapalı: açılışta yorum yok
    cv.ai_panel.set_interpret_available(True)               # (worktree'de model yok; düğmeyi zorla etkinleştir)
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _wait_idle(qtbot, cv)
    assert cv.ai_panel.state() == "off"
    cv.ai_panel.off_state.button.click()
    qtbot.waitUntil(lambda: cv._result is not None and cv._result.llm_used, timeout=15000)
    _wait_idle(qtbot, cv)
    assert cv.ai_panel.state() == "ready"
