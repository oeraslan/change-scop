"""Kütüphane ekranı: klasör ağacı, galeri, seçim, yüzen çubuk, içe aktarma, filtreler."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent

from app.ui.folder_tree import FOLDER_MIME
from app.ui.gallery import card_rect, circle_rect
from core.contracts import ImageSource
from tests.ui.conftest import make_image_file


def _click_card(qtbot, gallery, row: int, modifier=Qt.KeyboardModifier.NoModifier, circle: bool = False) -> None:
    index = gallery.model().index(row, 0)
    rect = card_rect(gallery.visualRect(index))
    pos = circle_rect(rect).center() if circle else rect.center()
    qtbot.mouseClick(gallery.viewport(), Qt.MouseButton.LeftButton, modifier, pos)


def test_tree_shows_nested_folders_and_count_badges(window, populated) -> None:
    _, folders, _ = populated
    tree = window.library_view.tree
    assert tree.item_count(None) == 4                  # Tüm görüntüler
    assert tree.item_count(folders["b"].id) == 2       # Y Dağı: doğrudan 2
    assert tree.item_count(folders["a"].id) == 4       # X Bölgesi: kendi 2 + alt klasör 2


def test_create_nested_subfolder_unlimited_depth(window) -> None:
    lv = window.library_view
    parent = None
    for i in range(6):
        parent = lv.create_folder(f"Derin {i}", parent.id if parent else None)
    assert parent.id in lv.tree.folder_ids()
    assert lv.current_folder_id() == parent.id
    assert len(lv.library.repo.folder_path(parent.id)) == 6


def test_rename_and_delete_folder_via_view(window, populated) -> None:
    library, folders, recs = populated
    lv = window.library_view
    lv.rename_folder(folders["b"].id, "Yeni Ad")
    assert library.repo.get_folder(folders["b"].id).name == "Yeni Ad"
    lv.delete_folder(folders["a"].id)                  # cascade: alt klasör + 4 görüntü
    assert library.repo.list_folders() == []
    assert library.repo.total_image_count() == 0
    assert not library.image_path(recs[0]).exists()
    assert lv.gallery_stack.currentWidget() is lv.empty_no_folder


def test_folder_drag_drop_moves_folder(window, populated, qtbot) -> None:
    library, folders, _ = populated
    lv = window.library_view
    other = lv.create_folder("Hedef")
    mime = QMimeData()
    mime.setData(FOLDER_MIME, QByteArray(str(folders["b"].id).encode()))
    tree = lv.tree
    pos = QPointF(tree.visualRect(tree._item_by_id[other.id].index()).center())
    event = QDropEvent(pos, Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    tree.dropEvent(event)
    assert library.repo.get_folder(folders["b"].id).parent_id == other.id
    assert tree.item_count(other.id) == 2


def test_cannot_move_folder_into_own_child(window, populated) -> None:
    library, folders, _ = populated
    assert window.library_view.move_folder(folders["a"].id, folders["b"].id) is False
    assert library.repo.get_folder(folders["a"].id).parent_id is None


def test_gallery_lists_images_sorted_by_date(window, populated) -> None:
    _, _, recs = populated
    lv = window.library_view
    lv.tree.select_folder(None)
    dates = [r.captured_at for r in lv.gallery_model.records()]
    assert dates == sorted(dates) and len(dates) == 4
    assert lv.count_label.text() == "4 görüntü"


def test_selecting_two_images_shows_compare_bar(window, qtbot) -> None:
    lv = window.library_view
    lv.tree.select_folder(None)
    assert not lv.compare_bar.isVisible()
    _click_card(qtbot, lv.gallery, 0)
    assert not lv.compare_bar.isVisible()               # tek seçim: çubuk yok
    assert "ikinci" in lv.hint_label.text()
    _click_card(qtbot, lv.gallery, 2, Qt.KeyboardModifier.ControlModifier)
    assert len(lv.gallery.selected_ids) == 2
    qtbot.waitUntil(lv.compare_bar.isVisible)
    pair = lv.selected_pair()
    assert pair[0].captured_at < pair[1].captured_at     # eski tarih otomatik "önce"
    assert lv.compare_bar.before_label.text() != lv.compare_bar.after_label.text()


def test_selection_circle_toggles_and_bar_hides(window, qtbot) -> None:
    lv = window.library_view
    lv.tree.select_folder(None)
    _click_card(qtbot, lv.gallery, 0, circle=True)
    _click_card(qtbot, lv.gallery, 1, circle=True)
    qtbot.waitUntil(lv.compare_bar.isVisible)
    _click_card(qtbot, lv.gallery, 1, circle=True)       # tekrar tıkla: seçimi kaldır
    assert len(lv.gallery.selected_ids) == 1
    qtbot.waitUntil(lambda: not lv.compare_bar.isVisible())


def test_third_selection_replaces_oldest(window, qtbot) -> None:
    lv = window.library_view
    lv.tree.select_folder(None)
    for row in (0, 1, 2):
        _click_card(qtbot, lv.gallery, row, circle=True)
    ids = [r.id for r in lv.gallery_model.records()]
    assert lv.gallery.selected_ids == [ids[1], ids[2]]


def test_swap_and_compare_signal(window, qtbot) -> None:
    lv = window.library_view
    lv.tree.select_folder(None)
    _click_card(qtbot, lv.gallery, 0, circle=True)
    _click_card(qtbot, lv.gallery, 3, circle=True)
    qtbot.waitUntil(lv.compare_bar.isVisible)
    first = lv.selected_pair()
    lv.compare_bar.swap_button.click()
    swapped = lv.selected_pair()
    assert swapped == (first[1], first[0])
    with qtbot.waitSignal(lv.compareRequested) as blocker:
        lv.compare_bar.compare_button.click()
    assert blocker.args == [swapped[0], swapped[1]]


def test_selection_works_across_folders(window, populated, qtbot) -> None:
    _, folders, recs = populated
    lv = window.library_view
    lv.tree.select_folder(folders["b"].id)               # Y Dağı: iki görüntü
    _click_card(qtbot, lv.gallery, 0, circle=True)
    lv.tree.select_folder(folders["a"].id)
    lv.subfolder_check.setChecked(False)                 # yalnızca X'in kendi görüntüleri
    _click_card(qtbot, lv.gallery, 0, circle=True)
    assert len(lv.gallery.selected_ids) == 2
    qtbot.waitUntil(lv.compare_bar.isVisible)
    pair = lv.selected_pair()
    assert {p.folder_id for p in pair} == {folders["a"].id, folders["b"].id}


def test_import_files_copies_and_updates_ui(window, tmp_path: Path) -> None:
    lv = window.library_view
    folder = lv.create_folder("Yeni")
    src = tmp_path / "ext"
    src.mkdir()
    files = [make_image_file(src, f"x{i}.png", seed=i) for i in range(3)]
    recs = lv.import_files(files, folder.id, ImageSource.SATELLITE, datetime(2026, 1, 2), sync=True)
    assert len(recs) == 3 and all(files[0].exists() for _ in recs)
    assert lv.tree.item_count(folder.id) == 3
    assert lv.count_label.text() == "3 görüntü"
    assert all(r.source == "satellite" for r in recs)
    assert lv.library.image_path(recs[0]).parent == lv.library.root / "images"


def test_import_creates_default_folder_when_none_selected(window) -> None:
    lv = window.library_view
    lv.tree.select_folder(None)
    paths = []
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        paths.append(make_image_file(Path(d), "z.png"))
        recs = lv.import_files(paths, None, sync=True)
    assert recs and lv.library.repo.get_folder(recs[0].folder_id).name == "Genel"


def test_import_in_background_worker(window, tmp_path: Path, qtbot) -> None:
    lv = window.library_view
    folder = lv.create_folder("Arka plan")
    files = [make_image_file(tmp_path, f"bg{i}.png", seed=i) for i in range(2)]
    lv.import_files(files, folder.id)
    qtbot.waitUntil(lambda: lv.library.repo.image_counts().get(folder.id) == 2, timeout=10000)
    lv.wait_for_import()


def test_file_drop_on_window_requests_import(window, tmp_path: Path, monkeypatch) -> None:
    lv = window.library_view
    captured = []
    monkeypatch.setattr(lv, "request_import", lambda paths, folder=None: captured.append((paths, folder)))
    f = make_image_file(tmp_path, "drop.png")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(f))])
    event = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    lv.dropEvent(event)
    assert captured == [([f], None)]


def test_images_drag_to_folder_moves_them(window, populated) -> None:
    library, folders, recs = populated
    lv = window.library_view
    target = lv.create_folder("Taşınan")
    lv.tree.imagesDropped.emit([recs[0].id, recs[1].id], target.id)
    assert library.repo.get_image(recs[0].id).folder_id == target.id
    assert lv.tree.item_count(target.id) == 2


def test_search_and_source_filters(window, qtbot) -> None:
    lv = window.library_view
    lv.tree.select_folder(None)
    lv.search.setText("iki")
    qtbot.waitUntil(lambda: lv.gallery_model.rowCount() == 1)
    lv.search.clear()
    qtbot.waitUntil(lambda: lv.gallery_model.rowCount() == 4)
    lv.source_combo.setCurrentIndex(3)                    # Uydu
    assert [r.source for r in lv.gallery_model.records()] == ["satellite"]
    lv.search.setText("yok-boyle-bir-dosya")
    qtbot.waitUntil(lambda: lv.gallery_stack.currentWidget() is lv.empty_filtered)
    lv.clear_filters()
    assert lv.gallery_model.rowCount() == 4


def test_empty_states(window, library) -> None:
    lv = window.library_view
    empty = lv.create_folder("Boş")
    assert lv.gallery_stack.currentWidget() is lv.empty_folder
    assert lv.empty_folder.button is not None and lv.empty_folder.button.text() == "Görüntü içe aktar"


def test_history_tab_lists_saved_comparisons(window, populated) -> None:
    from tests.ui.helpers import make_result

    library, _, recs = populated
    library.save_comparison(recs[0], recs[1], make_result(recs[0].width, recs[0].height, with_llm=True))
    lv = window.library_view
    lv.refresh_history()
    lv.show_tab("history")
    assert lv.history_page.currentWidget() is lv.history_scroll
    assert lv.history_layout.count() == 2                 # 1 satır + stretch
    got = []
    lv.comparisonOpened.connect(got.append)
    row = lv.history_layout.itemAt(0).widget()
    row.opened.emit(row.rec)
    assert got and got[0].before_image_id == recs[0].id
