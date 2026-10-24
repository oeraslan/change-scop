"""Kütüphane ekranı: klasör ağacı + galeri + geçmiş karşılaştırmalar + yüzen karşılaştırma çubuğu."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    InfoBar,
    InfoBarPosition,
    MessageBox,
    PrimaryPushButton,
    RoundMenu,
    SearchLineEdit,
    SegmentedWidget,
    SingleDirectionScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
    ToolButton,
)
from qfluentwidgets import FluentIcon as FIF

from app.storage import ComparisonRecord, Folder, ImageRecord, Library
from app.storage.library import is_image_file, read_exif_date
from app.ui.compare_view import format_date
from app.ui.dialogs import DateDialog, ImportDialog, NameDialog
from app.ui.folder_tree import FolderTree
from app.ui.gallery import FloatingBar, GalleryModel, GalleryView
from app.ui.theme import (
    QUALITY_COLORS,
    QUALITY_LABELS,
    SOURCE_LABELS,
    EmptyState,
    Pill,
    Surface,
    dim_label,
)
from app.ui.workers import ImportWorker, run_import
from core.contracts import ImageSource

DEFAULT_FOLDER_NAME = "Genel"
DATE_FILTERS = ["Tüm tarihler", "Son 7 gün", "Son 30 gün", "Bu yıl"]
HISTORY_SORTS: list[tuple[str, str]] = [
    ("Yeni → eski", "newest"), ("Eski → yeni", "oldest"), ("Bölge sayısı", "regions"),
]
SOURCE_FILTERS: list[tuple[str, str | None]] = [
    ("Tüm kaynaklar", None), ("Drone", "drone"), ("İHA", "uav"), ("Uydu", "satellite"),
    ("Bilinmiyor", "unknown"),
]


def expand_paths(paths: list[Path]) -> list[Path]:
    """Klasörleri özyinelemeli açar; yalnızca desteklenen görüntü dosyalarını döndürür."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.is_file() and is_image_file(q))
        elif p.is_file() and is_image_file(p):
            out.append(p)
    return out


METHOD_LABELS = {"ssim": "SSIM", "dinov2": "DINOv2", "ssim+dinov2": "SSIM + DINOv2"}


def method_label(method: str) -> str:
    """Fark yöntemi için okunur etiket (bilinmeyen değerler olduğu gibi)."""
    return METHOD_LABELS.get(method, method)


class HistoryRow(Surface):
    """Geçmiş karşılaştırma kartı: önce/sonra küçük resimleri, tarih çifti, klasör yolu, kalite, bölge sayısı,
    maske ve yöntem rozetleri."""

    opened = Signal(object)
    recompareRequested = Signal(object)
    deleteRequested = Signal(object)
    revealRequested = Signal(object)

    def __init__(self, rec: ComparisonRecord, library: Library, parent: QWidget | None = None) -> None:
        super().__init__(parent, hoverable=True)
        self.rec = rec
        self.setFixedHeight(96)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        before = library.repo.get_image(rec.before_image_id)
        after = library.repo.get_image(rec.after_image_id)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 16, 8)
        lay.setSpacing(12)
        for img in (before, after):
            thumb = QLabel()
            thumb.setFixedSize(96, 64)
            thumb.setStyleSheet("border-radius:6px;")
            tp = library.thumb_path(img) if img is not None else None
            if tp is not None and tp.exists():
                pix = QPixmap(str(tp)).scaled(
                    96, 64, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                thumb.setPixmap(pix.copy(0, 0, 96, 64))
            lay.addWidget(thumb)
            if img is before:
                lay.addWidget(BodyLabel("→"))
        info = QVBoxLayout()
        info.setSpacing(2)
        chain = library.repo.folder_path(rec.folder_id) if rec.folder_id else []
        self.folder_text = " / ".join(f.name for f in chain) or "Karşılaştırma"
        info.addWidget(StrongBodyLabel(self.folder_text))
        dates = f"{format_date(before.captured_at if before else None)}  →  {format_date(after.captured_at if after else None)}"
        cap = dim_label(f"Hesaplandı: {format_date(rec.created_at)}")
        cap.setMinimumWidth(0)
        self.pills: list[Pill] = []
        if rec.mask_hash:
            self.pills.append(Pill("Maskeli", "#A78BFA"))
        if rec.method:
            self.pills.append(Pill(method_label(rec.method), "#94A3B8"))
        self.pills.append(Pill(f"{rec.region_count} bölge", "#60A5FA"))
        self.pills.append(Pill(QUALITY_LABELS.get(rec.quality, rec.quality), QUALITY_COLORS.get(rec.quality, "#94A3B8")))
        line = QHBoxLayout()
        line.setSpacing(6)
        line.addWidget(BodyLabel(dates))
        line.addSpacing(6)
        for pill in self.pills:
            line.addWidget(pill)
        line.addStretch(1)
        info.addLayout(line)
        info.addWidget(cap)
        lay.addLayout(info, 1)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.opened.emit(self.rec)
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = RoundMenu(parent=self)
        menu.addAction(Action(FIF.VIEW, "Aç", triggered=lambda: self.opened.emit(self.rec)))
        menu.addAction(Action(FIF.SYNC, "Yeniden karşılaştır", triggered=lambda: self.recompareRequested.emit(self.rec)))
        menu.addAction(Action(FIF.FOLDER, "Klasörde göster", triggered=lambda: self.revealRequested.emit(self.rec)))
        menu.addSeparator()
        menu.addAction(Action(FIF.DELETE, "Geçmişten sil", triggered=lambda: self.deleteRequested.emit(self.rec)))
        menu.exec(event.globalPos())


class LibraryView(QWidget):
    """Ana kütüphane sayfası."""

    compareRequested = Signal(object, object)       # (before ImageRecord, after ImageRecord)
    comparisonOpened = Signal(object)               # ComparisonRecord (kayıttan aç)
    comparisonRecompareRequested = Signal(object)   # ComparisonRecord (önbelleği yok say)
    libraryChanged = Signal()

    def __init__(self, library: Library, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("libraryView")
        self.library = library
        self._swapped = False
        self._import_worker: ImportWorker | None = None
        self._pair: tuple[ImageRecord, ImageRecord] | None = None
        self.setAcceptDrops(True)

        # --- Sol panel ---
        self.left_panel = Surface(self)
        self.tree = FolderTree(self.left_panel)
        self.add_folder_button = ToolButton(FIF.FOLDER_ADD, self.left_panel)
        self.add_folder_button.setToolTip("Yeni klasör")
        self.add_folder_button.clicked.connect(lambda: self.prompt_new_folder(None))
        lhead = QHBoxLayout()
        lhead.addWidget(StrongBodyLabel("Klasörler", self.left_panel))
        lhead.addStretch(1)
        lhead.addWidget(self.add_folder_button)
        llay = QVBoxLayout(self.left_panel)
        llay.setContentsMargins(12, 12, 8, 8)
        llay.setSpacing(8)
        llay.addLayout(lhead)
        llay.addWidget(self.tree, 1)

        # --- Sağ panel ---
        self.content = QWidget(self)
        self.title_label = SubtitleLabel("Tüm görüntüler", self.content)
        self.count_label = dim_label("", self.content)
        self.hint_label = CaptionLabel("", self.content)
        self.hint_label.setStyleSheet("color: #22D3EE;")
        self.import_button = PrimaryPushButton(FIF.ADD, "Görüntü içe aktar", self.content)
        self.import_button.clicked.connect(lambda: self.prompt_import_dialog())

        self.search = SearchLineEdit(self.content)
        self.search.setPlaceholderText("Dosya adı ara…")
        self.search.setFixedWidth(240)
        self.source_combo = ComboBox(self.content)
        self.source_combo.addItems([t for t, _ in SOURCE_FILTERS])
        self.source_combo.setFixedWidth(150)
        self.date_combo = ComboBox(self.content)
        self.date_combo.addItems(DATE_FILTERS)
        self.date_combo.setFixedWidth(150)
        self.sort_combo = ComboBox(self.content)
        self.sort_combo.addItems([t for t, _ in HISTORY_SORTS])
        self.sort_combo.setFixedWidth(170)
        self.sort_combo.setVisible(False)
        self.sort_combo.currentIndexChanged.connect(lambda _: self.refresh_history())
        self.subfolder_check = CheckBox("Alt klasörler dahil", self.content)
        self.subfolder_check.setChecked(True)
        self.pivot = SegmentedWidget(self.content)
        self.pivot.addItem("images", "Görüntüler", lambda: self.show_tab("images"))
        self.pivot.addItem("history", "Geçmiş karşılaştırmalar", lambda: self.show_tab("history"))
        self.pivot.setCurrentItem("images")

        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(150)
        self._filter_timer.timeout.connect(self.refresh_gallery)
        self._filter_timer.timeout.connect(self.refresh_history)
        self.search.textChanged.connect(lambda _: self._filter_timer.start())
        self.source_combo.currentIndexChanged.connect(lambda _: self.refresh_gallery())
        self.date_combo.currentIndexChanged.connect(lambda _: self.refresh_gallery())
        self.subfolder_check.toggled.connect(lambda _: self.refresh_all())

        titles = QVBoxLayout()
        titles.setSpacing(0)
        trow = QHBoxLayout()
        trow.setSpacing(12)
        trow.addWidget(self.title_label)
        trow.addWidget(self.count_label)
        trow.addWidget(self.hint_label)
        trow.addStretch(1)
        titles.addLayout(trow)
        header = QHBoxLayout()
        header.addLayout(titles, 1)
        header.addWidget(self.import_button)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        for w in (self.search, self.source_combo, self.date_combo, self.sort_combo, self.subfolder_check):
            tools.addWidget(w)
        tools.addStretch(1)
        tools.addWidget(self.pivot)

        # Galeri sayfaları
        self.gallery_model = GalleryModel(self.library, self)
        self.gallery = GalleryView(self.content)
        self.gallery.setModel(self.gallery_model)
        self.gallery.selectionIdsChanged.connect(self._on_selection_changed)
        self.gallery.contextRequested.connect(self._on_gallery_context)

        self.empty_no_folder = EmptyState(
            FIF.FOLDER_ADD, "Henüz klasör yok",
            "Görüntülerini bölge ve zamana göre düzenlemek için ilk klasörünü oluştur.", "Klasör oluştur")
        self.empty_no_folder.button.clicked.connect(lambda: self.prompt_new_folder(None))
        self.empty_folder = EmptyState(
            FIF.PHOTO, "Bu klasörde görüntü yok",
            "Dosyaları buraya sürükleyip bırak ya da bilgisayarından seç. Dosyalar kütüphaneye kopyalanır.",
            "Görüntü içe aktar")
        self.empty_folder.button.clicked.connect(lambda: self.prompt_import_dialog())
        self.empty_filtered = EmptyState(
            FIF.SEARCH, "Sonuç bulunamadı", "Arama veya filtrelerle eşleşen görüntü yok.", "Filtreleri temizle")
        self.empty_filtered.button.clicked.connect(self.clear_filters)

        self.gallery_stack = QStackedWidget(self.content)
        for w in (self.gallery, self.empty_no_folder, self.empty_folder, self.empty_filtered):
            self.gallery_stack.addWidget(w)

        # Geçmiş
        self.history_scroll = SingleDirectionScrollArea(self.content, Qt.Orientation.Vertical)
        self.history_scroll.setWidgetResizable(True)
        self.history_host = QWidget()
        self.history_layout = QVBoxLayout(self.history_host)
        self.history_layout.setContentsMargins(0, 0, 8, 0)
        self.history_layout.setSpacing(8)
        self.history_layout.addStretch(1)
        self.history_scroll.setWidget(self.history_host)
        self.history_scroll.enableTransparentBackground()
        self.history_empty = EmptyState(
            FIF.HISTORY, "Henüz karşılaştırma yok",
            "İki görüntü seçip Karşılaştır'a bastığında sonuçlar burada saklanır; tekrar açınca yeniden hesaplanmaz.")
        self.history_filtered = EmptyState(
            FIF.SEARCH, "Sonuç bulunamadı", "Aramayla eşleşen karşılaştırma yok.", "Aramayı temizle")
        self.history_filtered.button.clicked.connect(self.search.clear)  # type: ignore[union-attr]
        self.tab_stack = QStackedWidget(self.content)
        self.history_page = QStackedWidget(self.content)
        self.history_page.addWidget(self.history_scroll)
        self.history_page.addWidget(self.history_empty)
        self.history_page.addWidget(self.history_filtered)
        self.tab_stack.addWidget(self.gallery_stack)
        self.tab_stack.addWidget(self.history_page)

        clay = QVBoxLayout(self.content)
        clay.setContentsMargins(16, 4, 0, 0)
        clay.setSpacing(12)
        clay.addLayout(header)
        clay.addLayout(tools)
        clay.addWidget(self.tab_stack, 1)

        # Yüzen çubuk
        self.compare_bar = FloatingBar(self.content)
        self.compare_bar.swapRequested.connect(self._swap)
        self.compare_bar.compareRequested.connect(self._emit_compare)
        self.compare_bar.clearRequested.connect(self.gallery.clear_selection)
        self.content.installEventFilter(self)

        # --- Ana yerleşim ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        self.splitter.setStyleSheet("QSplitter::handle{background:transparent;}")
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.content)
        self.splitter.setSizes([300, 1100])
        self.splitter.setStretchFactor(1, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 8, 24, 16)
        root.addWidget(self.splitter)

        # Ağaç sinyalleri
        self.tree.folderSelected.connect(self._on_folder_selected)
        self.tree.contextRequested.connect(self._on_tree_context)
        self.tree.folderMoveRequested.connect(self._on_folder_move)
        self.tree.imagesDropped.connect(self.move_images)
        self.tree.filesDropped.connect(lambda paths, target: self.request_import(paths, target))

        self.refresh_all()

    # ============================================================ yenileme

    def set_library(self, library: Library) -> None:
        """Kütüphane konumu değişince yeni kütüphaneye geçer."""
        self.library = library
        self.gallery_model.set_library(library)
        self.gallery.set_selection([], emit=False)
        self.compare_bar.hide()
        self.refresh_all()

    def refresh_all(self, select_id: int | None | object = ...) -> None:
        """Ağaç + galeri + geçmiş."""
        repo = self.library.repo
        self.tree.rebuild(repo.list_folders(), repo.image_counts(), repo.total_image_count(), select_id)
        self.refresh_gallery()
        self.refresh_history()

    def current_folder_id(self) -> int | None:
        return self.tree.current_folder_id()

    def _filters_active(self) -> bool:
        return bool(self.search.text().strip()) or self.source_combo.currentIndex() > 0 or self.date_combo.currentIndex() > 0

    def clear_filters(self) -> None:
        self.search.clear()
        self.source_combo.setCurrentIndex(0)
        self.date_combo.setCurrentIndex(0)
        self.refresh_gallery()

    def _date_from(self) -> str | None:
        now = datetime.now()
        idx = self.date_combo.currentIndex()
        if idx == 1:
            return (now - timedelta(days=7)).isoformat(timespec="seconds")
        if idx == 2:
            return (now - timedelta(days=30)).isoformat(timespec="seconds")
        if idx == 3:
            return datetime(now.year, 1, 1).isoformat(timespec="seconds")
        return None

    def refresh_gallery(self) -> None:
        """Seçili klasör + arama/filtrelere göre galeriyi doldurur."""
        repo = self.library.repo
        fid = self.current_folder_id()
        records = repo.list_images(
            fid, include_subfolders=self.subfolder_check.isChecked() or fid is None,
            search=self.search.text(), source=SOURCE_FILTERS[self.source_combo.currentIndex()][1],
            date_from=self._date_from(),
        )
        self.gallery_model.set_records(records)
        self.gallery.remember_records(records)
        alive = [i for i in self.gallery.selected_ids if repo.get_image(i) is not None]
        if alive != self.gallery.selected_ids:
            self.gallery.set_selection(alive)
        chain = repo.folder_path(fid) if fid is not None else []
        self.title_label.setText(" / ".join(f.name for f in chain) if chain else "Tüm görüntüler")
        self.count_label.setText(f"{len(records)} görüntü")
        if not repo.list_folders():
            self.gallery_stack.setCurrentWidget(self.empty_no_folder)
        elif not records:
            self.gallery_stack.setCurrentWidget(self.empty_filtered if self._filters_active() else self.empty_folder)
        else:
            self.gallery_stack.setCurrentWidget(self.gallery)
        self._update_selection_ui()

    def refresh_history(self) -> None:
        """Geçmiş karşılaştırmalar listesi (seçili klasör ve alt klasörleri)."""
        while self.history_layout.count() > 1:
            item = self.history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        search = self.search.text() if self.tab_stack.currentIndex() == 1 else ""
        sort = HISTORY_SORTS[self.sort_combo.currentIndex()][1]
        recs = self.library.repo.list_comparisons(self.current_folder_id(), search=search, sort=sort)
        for rec in recs:
            row = HistoryRow(rec, self.library, self.history_host)
            row.opened.connect(self.comparisonOpened)
            row.recompareRequested.connect(self.comparisonRecompareRequested)
            row.revealRequested.connect(self.reveal_comparison)
            row.deleteRequested.connect(self.confirm_delete_comparison)
            self.history_layout.insertWidget(self.history_layout.count() - 1, row)
        if recs:
            self.history_page.setCurrentWidget(self.history_scroll)
        elif search.strip():
            self.history_page.setCurrentWidget(self.history_filtered)
        else:
            self.history_page.setCurrentWidget(self.history_empty)
        total = len(recs)
        self.pivot.setItemText("history", f"Geçmiş karşılaştırmalar ({total})" if total else "Geçmiş karşılaştırmalar")

    def show_tab(self, key: str) -> None:
        """`images` veya `history` sekmesine geç."""
        history = key == "history"
        self.tab_stack.setCurrentIndex(1 if history else 0)
        self.pivot.setCurrentItem(key)
        self.source_combo.setVisible(not history)
        self.date_combo.setVisible(not history)
        self.sort_combo.setVisible(history)
        self.search.setPlaceholderText("Geçmişte ara…" if history else "Dosya adı ara…")
        if history:
            self.refresh_history()
        self._update_selection_ui()

    def reveal_comparison(self, rec: ComparisonRecord) -> None:
        """"Klasörde göster": görüntüler sekmesine geçer, klasörü seçer ve çiftin iki görüntüsünü işaretler."""
        before = self.library.repo.get_image(rec.before_image_id)
        if before is None:
            return
        self.search.clear()
        self.show_tab("images")
        self.tree.select_folder(before.folder_id)
        self.refresh_gallery()
        ids = [i for i in (rec.before_image_id, rec.after_image_id) if self.library.repo.get_image(i)]
        known = {r.id for r in self.gallery_model.records()}
        self.gallery.set_selection([i for i in ids if i in known])

    def _on_folder_selected(self, _fid) -> None:
        self.refresh_gallery()
        self.refresh_history()

    # ===================================================== klasör işlemleri

    def create_folder(self, name: str, parent_id: int | None = None) -> Folder:
        """Klasör (veya alt klasör) oluşturur ve ağaçta seçer."""
        folder = self.library.repo.create_folder(name, parent_id)
        self.refresh_all(select_id=folder.id)
        self.libraryChanged.emit()
        return folder

    def rename_folder(self, folder_id: int, name: str) -> None:
        self.library.repo.rename_folder(folder_id, name)
        self.refresh_all()
        self.libraryChanged.emit()

    def delete_folder(self, folder_id: int) -> None:
        """Klasörü içeriğiyle (alt klasörler, görüntüler) siler. Onay `confirm_delete_folder`'dadır."""
        parent = self.library.repo.get_folder(folder_id)
        self.library.delete_folder(folder_id)
        self.refresh_all(select_id=parent.parent_id if parent else None)
        self.libraryChanged.emit()

    def move_folder(self, folder_id: int, new_parent_id: int | None) -> bool:
        """Klasörü taşır; geçersizse (kendi alt klasörü) InfoBar gösterir ve False döner."""
        try:
            self.library.repo.move_folder(folder_id, new_parent_id)
        except (ValueError, KeyError) as exc:
            self.notify("error", "Klasör taşınamadı", str(exc))
            return False
        self.refresh_all(select_id=folder_id)
        self.libraryChanged.emit()
        return True

    def _on_folder_move(self, folder_id: int, target) -> None:
        folder = self.library.repo.get_folder(folder_id)
        if folder is not None and folder.parent_id == target:
            return
        self.move_folder(folder_id, target)

    def move_images(self, image_ids: list[int], folder_id: int) -> None:
        """Görüntüleri başka klasöre taşır."""
        for i in image_ids:
            self.library.repo.move_image(i, folder_id)
        self.refresh_all()
        self.libraryChanged.emit()

    def delete_images(self, image_ids: list[int]) -> None:
        for i in image_ids:
            self.library.delete_image(i)
        self.refresh_all()
        self.libraryChanged.emit()

    def delete_comparison(self, rec: ComparisonRecord) -> None:
        """Kaydı ve `result_dir` dosyalarını siler (onaysız çekirdek; onay `confirm_delete_comparison`'da)."""
        self.library.delete_comparison(rec.id)
        self.refresh_history()

    def confirm_delete_comparison(self, rec: ComparisonRecord) -> None:
        box = MessageBox(
            "Karşılaştırma silinsin mi?",
            "Kayıtlı sonuç (fark haritası, bölgeler) kalıcı olarak silinir. "
            "Görüntülere dokunulmaz; tekrar karşılaştırabilirsin.", self.window())
        box.yesButton.setText("Sil")
        box.cancelButton.setText("Vazgeç")
        if box.exec():
            self.delete_comparison(rec)

    # ----- diyalog sarmalayıcıları (etkileşimli)

    def prompt_new_folder(self, parent_id: int | None) -> None:
        dlg = NameDialog("Yeni alt klasör" if parent_id is not None else "Yeni klasör", "", self.window())
        if dlg.exec() and dlg.value():
            self.create_folder(dlg.value(), parent_id)

    def prompt_rename(self, folder_id: int) -> None:
        folder = self.library.repo.get_folder(folder_id)
        if folder is None:
            return
        dlg = NameDialog("Klasörü yeniden adlandır", folder.name, self.window())
        if dlg.exec() and dlg.value():
            self.rename_folder(folder_id, dlg.value())

    def confirm_delete_folder(self, folder_id: int) -> None:
        folder = self.library.repo.get_folder(folder_id)
        if folder is None:
            return
        n_img = len(self.library.repo.list_images(folder_id, include_subfolders=True))
        n_sub = len(self.library.repo.descendant_ids(folder_id)) - 1
        box = MessageBox(
            "Klasör silinsin mi?",
            f"“{folder.name}” klasörü, {n_sub} alt klasörü ve {n_img} görüntüsüyle birlikte kalıcı olarak silinecek. "
            "Orijinal dosyalarınıza dokunulmaz; yalnızca kütüphane kopyaları silinir.",
            self.window())
        box.yesButton.setText("Sil")
        box.cancelButton.setText("Vazgeç")
        if box.exec():
            self.delete_folder(folder_id)

    def _on_tree_context(self, folder_id, pos: QPoint) -> None:
        menu = RoundMenu(parent=self)
        parent_of = self.library.repo.get_folder(folder_id).parent_id if folder_id is not None else None
        menu.addAction(Action(FIF.FOLDER_ADD, "Yeni klasör", triggered=lambda: self.prompt_new_folder(parent_of)))
        if folder_id is not None:
            menu.addAction(Action(FIF.ADD, "Yeni alt klasör", triggered=lambda: self.prompt_new_folder(folder_id)))
            menu.addAction(Action(FIF.EDIT, "Yeniden adlandır", triggered=lambda: self.prompt_rename(folder_id)))
            menu.addSeparator()
            menu.addAction(Action(FIF.PHOTO, "Görüntü içe aktar", triggered=lambda: self.prompt_import_dialog(folder_id)))
            menu.addSeparator()
            menu.addAction(Action(FIF.DELETE, "Sil", triggered=lambda: self.confirm_delete_folder(folder_id)))
        menu.exec(pos)

    # ===================================================== içe aktarma

    def ensure_target_folder(self, folder_id: int | None) -> int:
        """Hedef klasör; yoksa seçili klasör, o da yoksa "Genel" klasörü (gerekirse oluşturulur)."""
        if folder_id is not None:
            return folder_id
        current = self.current_folder_id()
        if current is not None:
            return current
        for f in self.library.repo.list_folders():
            if f.parent_id is None and f.name == DEFAULT_FOLDER_NAME:
                return f.id
        return self.library.repo.create_folder(DEFAULT_FOLDER_NAME).id

    def prompt_import_dialog(self, folder_id: int | None = None) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Görüntü seç", "", "Görüntüler (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")
        if files:
            self.request_import([Path(f) for f in files], folder_id)

    def request_import(self, paths: list[Path], folder_id: int | None = None) -> None:
        """Seçenek diyaloğunu (kaynak + eksik EXIF tarihi) gösterip içe aktarır."""
        files = expand_paths([Path(p) for p in paths])
        if not files:
            self.notify("warning", "Görüntü bulunamadı", "Desteklenen biçimler: PNG, JPG, TIFF, BMP, WEBP.")
            return
        missing = sum(1 for f in files if read_exif_date(f) is None)
        dlg = ImportDialog(len(files), missing, self.window())
        if dlg.exec():
            self.import_files(files, folder_id, dlg.source(), dlg.fallback_date())

    def import_files(
        self, paths: list[Path], folder_id: int | None = None,
        source: ImageSource | str = ImageSource.UNKNOWN, fallback_date: datetime | None = None,
        sync: bool = False,
    ) -> list[ImageRecord] | None:
        """Dosyaları kütüphaneye kopyalar. `sync=True` ise bekler ve kayıtları döndürür."""
        files = expand_paths([Path(p) for p in paths])
        target = self.ensure_target_folder(folder_id)
        if sync:
            records, errors = run_import(self.library, files, target, source, fallback_date)
            self._after_import(records, errors, target)
            return records
        if self._import_worker is not None and self._import_worker.isRunning():
            self.notify("warning", "İçe aktarma sürüyor", "Önceki işlem bitince tekrar deneyin.")
            return None
        worker = ImportWorker(self.library, files, target, source, fallback_date, self)
        worker.imported.connect(lambda recs, errs, t=target: self._after_import(recs, errs, t))
        worker.finished.connect(self._on_import_finished)
        self._import_worker = worker
        self.notify("info", "İçe aktarılıyor…", f"{len(files)} görüntü kütüphaneye kopyalanıyor.", duration=2000)
        worker.start()
        return None

    def _on_import_finished(self) -> None:
        worker, self._import_worker = self._import_worker, None
        if worker is not None:
            worker.deleteLater()

    def _after_import(self, records: list, errors: list, target: int) -> None:
        self.refresh_all(select_id=target)
        self.libraryChanged.emit()
        if errors:
            self.notify("warning", f"{len(errors)} dosya aktarılamadı", errors[0], duration=6000)
        if records:
            self.notify("success", "İçe aktarıldı", f"{len(records)} görüntü eklendi.", duration=3000)

    def shutdown(self) -> None:
        """Kapanırken bekleyen zamanlayıcıları durdurur ve içe aktarmayı bekler."""
        self._filter_timer.stop()
        self.wait_for_import(5000)

    def wait_for_import(self, timeout_ms: int = 10000) -> None:
        """Çalışan içe aktarma işini bekler (testler)."""
        worker = self._import_worker
        if worker is not None:
            worker.wait(timeout_ms)

    # ---- dosya sürükle-bırak (pencerenin tamamı)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.request_import(paths, None)

    # ===================================================== galeri etkileşimi

    def _on_gallery_context(self, rec: ImageRecord | None, pos: QPoint) -> None:
        if rec is None:
            return
        menu = RoundMenu(parent=self)
        menu.addAction(Action(FIF.CALENDAR, "Çekim tarihini düzenle", triggered=lambda: self.prompt_edit_date(rec)))
        src_menu = RoundMenu("Kaynak türü", self)
        src_menu.setIcon(FIF.CAMERA)
        for label, value in (("Drone", "drone"), ("İHA", "uav"), ("Uydu", "satellite"), ("Bilinmiyor", "unknown")):
            src_menu.addAction(Action(label, triggered=lambda _=False, v=value: self.set_source(rec.id, v)))
        menu.addMenu(src_menu)
        menu.addSeparator()
        menu.addAction(Action(FIF.DELETE, "Kütüphaneden sil", triggered=lambda: self.confirm_delete_image(rec)))
        menu.exec(pos)

    def prompt_edit_date(self, rec: ImageRecord) -> None:
        dlg = DateDialog(rec.captured_at, self.window())
        if dlg.exec():
            self.set_captured_at(rec.id, dlg.value())

    def set_captured_at(self, image_id: int, when: datetime) -> None:
        self.library.repo.update_image(image_id, captured_at=when.replace(microsecond=0).isoformat())
        self.refresh_gallery()

    def set_source(self, image_id: int, source: str) -> None:
        self.library.repo.update_image(image_id, source=source)
        self.refresh_gallery()

    def confirm_delete_image(self, rec: ImageRecord) -> None:
        n = self.library.repo.count_comparisons_for_image(rec.id)
        extra = f" Bu görüntüyü kullanan {n} karşılaştırma kaydı da silinir." if n else ""
        box = MessageBox("Görüntü silinsin mi?",
                         f"“{rec.original_name}” kütüphaneden silinecek (orijinal dosyaya dokunulmaz).{extra}",
                         self.window())
        box.yesButton.setText("Sil")
        box.cancelButton.setText("Vazgeç")
        if box.exec():
            self.delete_images([rec.id])

    # ===================================================== seçim + çubuk

    def _on_selection_changed(self, ids: list) -> None:
        self._swapped = False
        self._update_selection_ui()

    def selected_pair(self) -> tuple[ImageRecord, ImageRecord] | None:
        """İki görüntü seçiliyse (önce, sonra) — eski tarihli otomatik "önce"; takas uygulanır."""
        ids = self.gallery.selected_ids
        if len(ids) != 2:
            return None
        recs = [self.library.repo.get_image(i) for i in ids]
        if any(r is None for r in recs):
            return None
        a, b = sorted(recs, key=lambda r: (r.captured_at or r.created_at, r.id))  # type: ignore[union-attr]
        pair = (a, b)  # type: ignore[assignment]
        return (pair[1], pair[0]) if self._swapped else pair  # type: ignore[return-value]

    def _update_selection_ui(self) -> None:
        n = len(self.gallery.selected_ids)
        self.hint_label.setText("1 görüntü seçildi — karşılaştırmak için ikinci görüntüyü seç" if n == 1 else "")
        pair = self.selected_pair()
        self._pair = pair
        on_images = self.tab_stack.currentIndex() == 0
        if pair is not None and on_images:
            self.compare_bar.set_dates(format_date(pair[0].captured_at), format_date(pair[1].captured_at))
            self.compare_bar.show_animated(self.content)
        else:
            self.compare_bar.hide_animated()

    def _swap(self) -> None:
        self._swapped = not self._swapped
        self._update_selection_ui()

    def _emit_compare(self) -> None:
        pair = self.selected_pair()
        if pair is not None:
            self.compareRequested.emit(*pair)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        from PySide6.QtCore import QEvent

        if obj is self.content and event.type() == QEvent.Type.Resize and self.compare_bar.isVisible():
            self.compare_bar.move(self.compare_bar.place(self.content))
        return super().eventFilter(obj, event)

    # ===================================================== yardımcılar

    def notify(self, kind: str, title: str, content: str, duration: int = 4000) -> None:
        """Pencerenin sağ üstünde kısa InfoBar."""
        factory = {"info": InfoBar.info, "warning": InfoBar.warning, "error": InfoBar.error,
                   "success": InfoBar.success}[kind]
        factory(title, content, orient=Qt.Orientation.Horizontal, isClosable=True, duration=duration,
                position=InfoBarPosition.TOP_RIGHT, parent=self.window())

