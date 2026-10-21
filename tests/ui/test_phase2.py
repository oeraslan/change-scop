"""Faz 2: yok sayma maskesi, şema migration'ı, geçmiş sekmesi, kilit sızıntısı, atomik kayıt, yöntem ayarı."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from app.storage import Library
from app.storage.repository import LibraryRepository, _migrate_v1
from app.ui.library_view import HISTORY_SORTS
from app.ui.mask_editor import TOOL_BRUSH, TOOL_ERASER, MaskEditor
from app.ui.qt_utils import heatmap_to_qimage, suppress_small_hotspots
from core import pipeline as core_pipeline
from core.config import AppConfig
from core.pipeline import set_default_pipeline
from tests.ui.helpers import make_pipeline, make_result


def _idle(qtbot, cv, timeout: int = 15000) -> None:
    qtbot.waitUntil(lambda: not cv.has_running_workers() and cv._worker is None, timeout=timeout)


def _capturing_pipeline():
    """Sahte pipeline; `run`'a giden `ignore_mask` değerlerini toplar."""
    pipe = make_pipeline()
    seen: list[np.ndarray | None] = []
    real = pipe.run

    def run(before, after, progress=None, use_llm=False, cancel=None, ignore_mask=None):
        seen.append(None if ignore_mask is None else np.array(ignore_mask))
        return real(before, after, progress, use_llm, cancel, ignore_mask)

    pipe.run = run  # type: ignore[method-assign]
    return pipe, seen


# =========================================================== MaskEditor

def test_editor_brush_eraser_undo_redo_clear() -> None:
    ed = MaskEditor()
    ed.set_canvas(400, 300)
    ed.brush_px = 20
    ed.tool = TOOL_BRUSH
    ed.stroke([(50, 50), (150, 50)])
    m = ed.mask_bool()
    assert m[50, 100] and not m[200, 200] and m.shape == (300, 400)
    ed.tool = TOOL_ERASER
    ed.stroke([(95, 50), (105, 50)])
    assert not ed.mask_bool()[50, 100] and ed.mask_bool()[50, 60]
    assert ed.undo() and ed.mask_bool()[50, 100]          # silgi geri alındı
    assert ed.undo() and not ed.has_content()             # fırça geri alındı
    assert not ed.undo()
    assert ed.redo() and ed.has_content()
    assert ed.redo() and not ed.mask_bool()[50, 100]
    ed.clear()
    assert not ed.has_content() and ed.can_undo()
    ed.undo()
    assert ed.has_content()


def test_editor_maps_scene_to_pixels_on_large_images() -> None:
    ed = MaskEditor()
    ed.set_canvas(6000, 4000)                             # çalışma çözünürlüğü 2048 genişlik
    assert ed.mask_size == (2048, 1365) and abs(ed.scale - 2048 / 6000) < 1e-9
    ed.brush_px = 300
    ed.stroke([(3000, 2000)])
    m = ed.mask_bool()
    assert m[683, 1024] and not m[100, 100]
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - 1024) < 3 and abs(ys.mean() - 683) < 3


def test_editor_modified_tracking_and_load() -> None:
    ed = MaskEditor()
    ed.set_canvas(200, 100)
    base = np.zeros((50, 100), bool)
    base[10:20, 10:30] = True                             # yarı çözünürlükte bir maske
    ed.load(base)
    assert ed.has_content() and not ed.is_modified()
    assert ed.mask_bool()[30, 40] and not ed.mask_bool()[80, 150]
    ed.tool = TOOL_BRUSH
    ed.stroke([(150, 80)])
    assert ed.is_modified()
    ed.undo()
    assert not ed.is_modified()


def test_drawing_with_mouse_follows_zoom_and_pan(window, populated, qtbot) -> None:
    _, _, recs = populated
    set_default_pipeline(make_pipeline())
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    cv.enter_mask_mode()
    _idle(qtbot, cv)
    assert cv.mask_mode_active() and cv.before_view.mask_item_visible() and cv.after_view.mask_item_visible()
    cv.mask_editor.brush_px = 16
    view = cv.before_view
    view.zoom_at(3.0, QPointF(view.viewport().rect().center()))    # yakınlaştırılmış halde çiz
    target = view.center_point() + QPointF(10, 5)
    pos = view.mapFromScene(target)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    QTest.mouseMove(view.viewport(), view.mapFromScene(target + QPointF(6, 0)))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    m = cv.mask_editor.mask_bool()
    assert m[int(round(target.y())), int(round(target.x()))]
    assert m.sum() < m.size * 0.05
    # Zoom, çizim modunda da diğer görünümü senkron tutar.
    assert abs(cv.after_view.zoom_factor() - view.zoom_factor()) < 1e-6
    # Orta tuşla kaydırma çalışır (çizim yapmaz)
    before_center = view.center_point()
    QTest.mousePress(view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, pos)
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + 20)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, pos)
    assert view.center_point() != before_center
    cv.cancel_mask_mode()
    assert not cv.before_view.mask_item_visible()


# =========================================================== depolama / migration

def test_mask_save_load_roundtrip_is_persistent(populated) -> None:
    library, _, recs = populated
    mask = np.zeros((360, 480), bool)
    mask[50:100, 60:200] = True
    rec = library.save_mask(recs[0], mask)
    assert rec.mask_path and rec.mask_hash
    with Image.open(library.abs_path(rec.mask_path)) as png:
        assert png.mode == "L" and png.size == (480, 360)
    loaded = library.load_mask(recs[0])
    np.testing.assert_array_equal(loaded, mask)
    library.close()
    reopened = Library(library.root)
    again = reopened.repo.get_image(recs[0].id)
    assert again.mask_hash == rec.mask_hash and again.mask_path == rec.mask_path
    np.testing.assert_array_equal(reopened.load_mask(again), mask)
    # Değişen maske eski dosyayı siler; boş maske kaydı kaldırır.
    mask[0:10, 0:10] = True
    rec2 = reopened.save_mask(again, mask)
    assert rec2.mask_hash != rec.mask_hash
    assert not library.abs_path(rec.mask_path).exists() and reopened.abs_path(rec2.mask_path).exists()
    cleared = reopened.save_mask(rec2, np.zeros_like(mask))
    assert cleared.mask_path is None and cleared.mask_hash == "" and reopened.load_mask(cleared) is None
    assert not reopened.abs_path(rec2.mask_path).exists()
    reopened.close()


def test_mask_larger_than_limit_is_downscaled(populated) -> None:
    library, _, recs = populated
    big = np.zeros((3000, 4000), bool)
    big[1000:2000, 2000:3000] = True
    rec = library.save_mask(recs[0], big)
    small = library.load_mask(rec)
    assert max(small.shape) == 2048 and small.shape[1] / small.shape[0] == 4000 / 3000
    frac = small.mean()
    assert abs(frac - big.mean()) < 0.005


def test_migration_upgrades_old_library(tmp_path: Path) -> None:
    """Sürüm 1 şemalı (maskesiz) kütüphane açılışta sorunsuz yükseltilir; veriler korunur."""
    db = tmp_path / "old" / "library.sqlite3"
    db.parent.mkdir()
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    con.execute("INSERT INTO schema_version VALUES (1)")
    _migrate_v1(con)
    con.execute("INSERT INTO folders(name, created_at) VALUES ('Eski', '2026-01-01T00:00:00')")
    con.execute("""INSERT INTO images(folder_id, file_name, original_name, captured_at, source, width,
                   height, thumb_path, created_at) VALUES (1,'images/a.png','a.png',NULL,'drone',10,10,NULL,'t')""")
    con.execute("""INSERT INTO images(folder_id, file_name, original_name, captured_at, source, width,
                   height, thumb_path, created_at) VALUES (1,'images/b.png','b.png',NULL,'drone',10,10,NULL,'t')""")
    con.execute("""INSERT INTO comparisons(folder_id, before_image_id, after_image_id, result_dir, quality,
                   region_count, summary, created_at) VALUES (1,1,2,'results/x','good',2,'özet','2026-01-02T00:00:00')""")
    con.commit()
    con.close()

    repo = LibraryRepository(db)
    assert repo.schema_version() == 2
    img = repo.get_image(1)
    assert img.mask_path is None and img.mask_hash == ""
    rec = repo.find_comparison(1, 2, "")
    assert rec is not None and rec.mask_hash == "" and rec.method == "" and rec.summary == "özet"
    assert repo.find_comparison(1, 2, "abc") is None
    repo.set_image_mask(1, "masks/1_x.png", "abc")
    assert repo.get_image(1).mask_hash == "abc"
    repo.close()
    assert LibraryRepository(db).schema_version() == 2      # ikinci açılış: değişiklik yok


def test_deleting_image_cascades_comparisons_and_mask(populated) -> None:
    """Kural: görüntü silinince onu kullanan karşılaştırmalar (kayıt + dosyalar) ve maske de silinir."""
    library, _, recs = populated
    res = make_result(recs[0].width, recs[0].height)
    c1 = library.save_comparison(recs[0], recs[1], res)
    c2 = library.save_comparison(recs[2], recs[0], res)           # recs[0] "sonra" rolünde
    c3 = library.save_comparison(recs[2], recs[3], res)           # ilişkisiz
    m = np.zeros((360, 480), bool)
    m[5:50, 5:50] = True
    withmask = library.save_mask(recs[0], m)
    assert library.repo.count_comparisons_for_image(recs[0].id) == 2
    dirs = [library.abs_path(c.result_dir) for c in (c1, c2, c3)]
    maskfile = library.abs_path(withmask.mask_path)
    assert all(d.exists() for d in dirs) and maskfile.exists()
    library.delete_image(recs[0].id)
    assert [c.id for c in library.repo.list_comparisons()] == [c3.id]
    assert not dirs[0].exists() and not dirs[1].exists() and dirs[2].exists()
    assert not maskfile.exists()


def test_tx_begin_failure_does_not_leak_lock(tmp_path: Path) -> None:
    repo = LibraryRepository(tmp_path / "db.sqlite3")
    repo.create_folder("x")
    real = repo._con

    class Boom:
        def execute(self, sql, *a):
            if sql == "BEGIN":
                raise sqlite3.OperationalError("disk I/O error")
            return real.execute(sql, *a)

    repo._con = Boom()      # type: ignore[assignment]
    try:
        repo.create_folder("y")
    except sqlite3.OperationalError:
        pass
    repo._con = real
    out: list[object] = []
    t = threading.Thread(target=lambda: out.append(repo.get_folder(1)), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive() and out
    assert repo.create_folder("z").name == "z"             # kilit ve transaction derinliği sağlam
    repo.close()


def test_concurrent_save_comparison_yields_single_row(populated) -> None:
    library, _, recs = populated
    res = make_result(recs[0].width, recs[0].height)
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def go() -> None:
        try:
            barrier.wait(5)
            library.save_comparison(recs[0], recs[1], res)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=go) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not errors, errors[:2]
    rows = library.repo.list_comparisons()
    assert len(rows) == 1
    assert library.load_comparison(rows[0]).detection is not None


# =========================================================== maskeli karşılaştırma

def test_masked_compare_sends_mask_and_invalidates_cache(window, populated, qtbot) -> None:
    library, _, recs = populated
    pipe, seen = _capturing_pipeline()
    set_default_pipeline(pipe)
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    assert seen == [None]                                   # maskesiz ilk koşu
    assert library.find_comparison(recs[0], recs[1]) is not None
    assert not cv.mask_badge.isVisibleTo(cv)

    # Maske çiz → kaydet ve yeniden karşılaştır
    cv.enter_mask_mode()
    _idle(qtbot, cv)
    qtbot.waitUntil(lambda: not cv._mask_workers, timeout=5000)
    cv.mask_editor.brush_px = 40
    cv.mask_editor.stroke([(100, 80), (160, 80)])
    cv.save_mask_and_recompare()
    qtbot.waitUntil(lambda: len(seen) == 2, timeout=15000)
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    m = seen[1]
    assert m is not None and m.dtype == bool and m.ndim == 2
    h, w = recs[0].height, recs[0].width
    assert abs(m.shape[1] / m.shape[0] - w / h) < 0.01      # görüntü en-boy oranı korunur
    ys, xs = np.nonzero(m)
    assert 0 < m.mean() < 0.2
    assert abs(xs.mean() / m.shape[1] - 130 / w) < 0.03 and abs(ys.mean() / m.shape[0] - 80 / h) < 0.03
    vm = cv._result.registration.valid_mask
    assert not vm[80, 130] and vm[300, 400]                 # maskeli alan "geçerli" değil
    assert cv.mask_badge.isVisibleTo(cv)
    rec = library.repo.find_comparison(recs[0].id, recs[1].id)
    assert rec.mask_hash == library.repo.get_image(recs[0].id).mask_hash != ""
    assert len(library.repo.list_comparisons()) == 1        # aynı çift: tek kayıt, maskeye geçti

    # Aynı maskeyle yeniden aç: önbellek (pipeline çalışmaz)
    fresh = library.repo.get_image(recs[0].id)
    window.open_compare(fresh, recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    assert len(seen) == 2

    # Maske değişince eski sonuç KULLANILMAZ
    changed = m.copy()
    changed[0:20, 0:20] = True
    library.save_mask(fresh, changed)
    assert library.find_comparison(library.repo.get_image(recs[0].id), recs[1]) is None
    window.open_compare(library.repo.get_image(recs[0].id), recs[1])
    qtbot.waitUntil(lambda: len(seen) == 3, timeout=15000)
    _idle(qtbot, cv)


def test_mask_mode_overlay_visible_even_when_diffs_hidden(window, populated, qtbot) -> None:
    _, _, recs = populated
    set_default_pipeline(make_pipeline())
    cv = window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    cv.diff_switch.setChecked(False)
    qtbot.wait(300)
    assert cv.visible_region_count() == (0, 0) and not cv.before_view.mask_item_visible()
    cv.enter_mask_mode()
    assert cv.before_view.mask_item_visible() and cv.visible_region_count() == (0, 0)
    cv.mask_toolbar.set_tool("eraser")
    assert cv.mask_editor.tool == "eraser"
    cv.mask_editor.tool = TOOL_BRUSH
    cv.mask_editor.stroke([(40, 40)])
    QTest.keyClick(cv, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(cv, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    cv.mask_undo()
    assert not cv.mask_editor.has_content()
    cv.save_mask_and_recompare()                           # değişiklik yok → yeniden hesaplamadan çık
    assert not cv.mask_mode_active() and not cv.is_busy()


# =========================================================== geçmiş sekmesi

def _fill_history(library, recs):
    res = make_result(recs[0].width, recs[0].height, with_llm=True)
    a = library.save_comparison(recs[0], recs[1], res)
    b = library.save_comparison(recs[2], recs[3], res)
    library.repo.add_comparison(recs[3].folder_id, recs[3].id, recs[2].id, "results/none", "weak", 9, "liman",
                                "abc", "dinov2")
    with library.repo._tx():
        library.repo._con.execute("UPDATE comparisons SET created_at='2026-10-01T10:00:00' WHERE id=?", (a.id,))
        library.repo._con.execute("UPDATE comparisons SET created_at='2026-10-03T10:00:00' WHERE id=?", (b.id,))
    return a, b


def test_history_cards_show_badges_and_filters(window, populated) -> None:
    library, folders, recs = populated
    a, b = _fill_history(library, recs)
    lv = window.library_view
    lv.tree.select_folder(None)
    lv.show_tab("history")
    rows = [lv.history_layout.itemAt(i).widget() for i in range(lv.history_layout.count() - 1)]
    assert len(rows) == 3
    assert [r.rec.id for r in rows][:2] == [b.id, a.id] or rows[0].rec.created_at >= rows[1].rec.created_at
    texts = {p.text() for r in rows for p in r.pills}
    assert {"Maskeli", "DINOv2", "3 bölge", "9 bölge", "Hizalama iyi", "Hizalama zayıf"} <= texts
    # Sıralama
    lv.sort_combo.setCurrentIndex([k for _, k in HISTORY_SORTS].index("regions"))
    assert lv.history_layout.itemAt(0).widget().rec.region_count == 9
    lv.sort_combo.setCurrentIndex(1)
    assert lv.history_layout.itemAt(0).widget().rec.created_at <= lv.history_layout.itemAt(1).widget().rec.created_at
    # Arama
    lv.search.setText("bir.png")
    lv.refresh_history()
    assert lv.history_layout.count() - 1 == 1 and lv.history_layout.itemAt(0).widget().rec.id == a.id
    lv.search.setText("yokboyle")
    lv.refresh_history()
    assert lv.history_page.currentWidget() is lv.history_filtered
    lv.search.clear()
    # Klasör filtresi: yalnız Y Dağı (alt klasör) → recs[0],[1] karşılaştırması
    lv.tree.select_folder(folders["b"].id)
    assert lv.history_layout.count() - 1 == 1


def test_history_open_from_cache_recompare_delete_reveal(window, populated, qtbot) -> None:
    library, folders, recs = populated
    pipe, seen = _capturing_pipeline()
    set_default_pipeline(pipe)
    lv, cv = window.library_view, window.compare_view
    window.open_compare(recs[0], recs[1])
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    rec = library.repo.find_comparison(recs[0].id, recs[1].id)
    assert len(seen) == 1

    lv.refresh_history()
    lv.show_tab("history")
    row = lv.history_layout.itemAt(0).widget()
    row.opened.emit(row.rec)                                # kayıttan aç → pipeline çalışmaz
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    assert len(seen) == 1

    row.recompareRequested.emit(row.rec)                    # önbelleği yok say
    qtbot.waitUntil(lambda: len(seen) == 2, timeout=15000)
    qtbot.waitUntil(lambda: cv._result is not None, timeout=15000)
    _idle(qtbot, cv)
    assert len(library.repo.list_comparisons()) == 1

    window.switchTo(lv)
    lv.show_tab("history")
    row = lv.history_layout.itemAt(0).widget()
    row.revealRequested.emit(row.rec)
    assert lv.tab_stack.currentIndex() == 0
    assert lv.current_folder_id() == folders["b"].id
    assert set(lv.gallery.selected_ids) == {recs[0].id, recs[1].id}

    lv.show_tab("history")
    rec = library.repo.find_comparison(recs[0].id, recs[1].id)   # yeniden karşılaştırma sonuç dizinini yeniler
    result_dir = library.abs_path(rec.result_dir)
    assert result_dir.exists()
    lv.delete_comparison(rec)
    assert not result_dir.exists() and library.repo.list_comparisons() == []
    assert lv.history_page.currentWidget() is lv.history_empty


# =========================================================== ayarlar / InfoBar / ısı haritası

def test_method_setting_reconfigures_pipeline(window, qtbot, monkeypatch) -> None:
    calls: list[AppConfig] = []
    monkeypatch.setattr(core_pipeline, "configure_default_pipeline", lambda cfg: calls.append(cfg))
    sv = window.settings_view
    sv.method_combo.setCurrentIndex(1)
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=3000)
    assert calls[0].detection.method == "dinov2"
    sv.method_combo.setCurrentIndex(2)
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=3000)
    assert calls[1].detection.method == "ssim+dinov2"
    assert "DINOv2" in sv.method_card.contentLabel.text()


def test_method_description_mentions_missing_model(window, monkeypatch) -> None:
    sv = window.settings_view
    monkeypatch.setattr(sv, "dinov2_available", lambda: False)
    sv.config.detection.method = "dinov2"
    assert "models/dinov2" in sv.method_description() and "SSIM" in sv.method_description()


def test_unreliable_and_fallback_infobars(window) -> None:
    cv = window.compare_view
    res = make_result(320, 240)
    res.detection.params = {"unreliable": 1, "unreliable_reason": "Otsu çok düşük",
                            "fallback_reason": "dinov2 modeli yok; SSIM kullanıldı"}
    cv.show_result(res)
    bars = {str(b.property("key")): b for b in cv._infobars}
    assert bars["unreliable"].property("kind") == "warning"
    assert "Sonuç güvenilmez olabilir: Otsu çok düşük" in bars["unreliable"].content
    assert bars["fallback"].property("kind") == "info" and "SSIM" in bars["fallback"].content
    res.detection.params = {"unreliable": 0}
    cv.show_result(res)
    assert not [b for b in cv._infobars if b.property("key") in ("unreliable", "fallback")]


def test_heatmap_suppresses_tiny_hotspots() -> None:
    cm = np.zeros((400, 600), np.float32)
    cm[100:104, 100:104] = 1.0              # 16 px: ağaç kenarı gibi küçük nokta
    cm[200:260, 300:380] = 1.0              # gerçek bölge
    img = heatmap_to_qimage(cm)
    assert img.pixelColor(101, 101).alpha() == 0
    assert img.pixelColor(340, 230).alpha() > 200
    out = suppress_small_hotspots(np.where(cm > 0, 1.0, 0.0), 0.08, 25)
    assert out[101, 101] == 0 and out[230, 340] == 1.0
    # Büyük görüntüde alan, ölçeğe göre değerlendirilir (küçültülünce de aynı sonuç)
    big = np.zeros((4096, 4096), np.float32)
    big[100:104, 100:104] = 1.0
    big[2000:2100, 2000:2100] = 1.0
    bi = heatmap_to_qimage(big, max_side=1024)
    assert bi.pixelColor(25, 25).alpha() == 0 and bi.pixelColor(512, 512).alpha() > 200
