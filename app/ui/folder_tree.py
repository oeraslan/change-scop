"""Sol panel: sürükle-bırak destekli klasör ağacı (TreeView) + görüntü sayısı rozetleri."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QMimeData, QModelIndex, QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QFontMetrics, QPainter, QPen, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QAbstractItemView, QWidget
from qfluentwidgets import TreeView, isDarkTheme
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets.components.widgets.tree_view import TreeItemDelegate

from app.storage import Folder
from app.ui.gallery import IMAGES_MIME, decode_ids
from app.ui.theme import ACCENT

FOLDER_MIME = "application/x-changescope-folder"
ALL_ID = -1                         # "Tüm görüntüler" sanal düğümü
ID_ROLE = Qt.ItemDataRole.UserRole + 1
COUNT_ROLE = Qt.ItemDataRole.UserRole + 2


class FolderModel(QStandardItemModel):
    """Klasör ağacı modeli; sürüklemede klasör kimliğini mime olarak taşır."""

    def mimeTypes(self) -> list[str]:  # noqa: N802
        return [FOLDER_MIME]

    def mimeData(self, indexes) -> QMimeData:  # noqa: N802
        mime = QMimeData()
        for idx in indexes:
            fid = idx.data(ID_ROLE)
            if fid is not None and fid != ALL_ID:
                mime.setData(FOLDER_MIME, QByteArray(str(fid).encode()))
                break
        return mime


class FolderDelegate(TreeItemDelegate):
    """Fluent satır çizimi + sağda görüntü sayısı rozeti + sürükleme hedef vurgusu."""

    def __init__(self, view: "FolderTree") -> None:
        super().__init__(view)
        self._view = view

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        super().paint(painter, option, index)
        count = index.data(COUNT_ROLE)
        if count is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        text = str(count)
        w = max(22, QFontMetrics(font).horizontalAdvance(text) + 14)
        rect = QRectF(option.rect.right() - w - 8, option.rect.center().y() - 9, w, 18)
        dark = isDarkTheme()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 26) if dark else QColor(0, 0, 0, 20))
        painter.drawRoundedRect(rect, 9, 9)
        painter.setPen(QColor(255, 255, 255, 190) if dark else QColor(0, 0, 0, 160))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class FolderTree(TreeView):
    """Klasör ağacı. Klasörler sürüklenerek taşınır; görüntü ve dosya bırakmayı da kabul eder.

    Sinyaller:
        folderSelected(int | None)           None = "Tüm görüntüler"
        contextRequested(object, QPoint)     (klasör id | None, küresel konum)
        folderMoveRequested(int, object)     (taşınan, yeni üst id | None)
        imagesDropped(list, int)             (görüntü id'leri, hedef klasör)
        filesDropped(list, object)           (dosya yolları, hedef klasör id | None)
    """

    folderSelected = Signal(object)
    contextRequested = Signal(object, QPoint)
    folderMoveRequested = Signal(int, object)
    imagesDropped = Signal(list, int)
    filesDropped = Signal(list, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        self._silent = False
        self._drop_index = QModelIndex()
        self._item_by_id: dict[int, QStandardItem] = {}
        super().__init__(parent)
        self._model = FolderModel(self)
        self.setModel(self._model)
        self.setHeaderHidden(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(False)
        self.setIndentation(18)
        self.setIconSize(self.iconSize() * 1.0)
        self.setItemDelegate(FolderDelegate(self))
        self.setBorderVisible(False)
        self.setBorderRadius(8)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self._drop_index = QModelIndex()
        self._silent = False
        self._item_by_id: dict[int, QStandardItem] = {}

    # -- veri ----------------------------------------------------------------

    def rebuild(self, folders: list[Folder], counts: dict[int, int], total: int,
                select_id: int | None | object = ...) -> None:
        """Ağacı yeniden kurar; genişleme ve seçim durumunu korur.

        Args:
            folders: Tüm klasörler.
            counts: Klasör başına doğrudan görüntü sayısı.
            total: Kütüphane toplam görüntü sayısı.
            select_id: Seçilecek klasör (None = Tümü). Verilmezse mevcut seçim korunur.
        """
        expanded = {i for i, item in self._item_by_id.items() if self.isExpanded(item.index())}
        current = self.current_folder_id() if select_id is ... else select_id
        self.blockSignals(True)
        self._model.clear()
        self._item_by_id.clear()

        all_item = QStandardItem(FIF.PHOTO.icon(), "Tüm görüntüler")
        all_item.setData(ALL_ID, ID_ROLE)
        all_item.setData(total, COUNT_ROLE)
        all_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDropEnabled)
        self._model.appendRow(all_item)

        children: dict[int | None, list[Folder]] = {}
        for f in folders:
            children.setdefault(f.parent_id, []).append(f)

        def subtotal(fid: int) -> int:
            return counts.get(fid, 0) + sum(subtotal(c.id) for c in children.get(fid, []))

        def add(parent_item: QStandardItem | None, folder: Folder) -> None:
            item = QStandardItem(FIF.FOLDER.icon(), folder.name)
            item.setData(folder.id, ID_ROLE)
            item.setData(subtotal(folder.id), COUNT_ROLE)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                          | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled)
            (parent_item or self._model).appendRow(item)  # type: ignore[arg-type]
            self._item_by_id[folder.id] = item
            for child in children.get(folder.id, []):
                add(item, child)

        for root in children.get(None, []):
            add(None, root)
        self.blockSignals(False)

        for fid, item in self._item_by_id.items():
            if fid in expanded:
                self.expand(item.index())
        if current is not None and current in self._item_by_id:
            self._expand_ancestors(current)
        self.select_folder(current if current in self._item_by_id else None, emit=False)

    def _expand_ancestors(self, folder_id: int) -> None:
        idx = self._item_by_id[folder_id].index().parent()
        while idx.isValid():
            self.expand(idx)
            idx = idx.parent()

    def select_folder(self, folder_id: int | None, emit: bool = True) -> None:
        """Klasörü seçer (None = Tüm görüntüler)."""
        item = self._item_by_id.get(folder_id) if folder_id is not None else self._model.item(0)
        if item is None:
            item = self._model.item(0)
        if folder_id is not None and folder_id in self._item_by_id:
            self._expand_ancestors(folder_id)
        self._silent = True
        try:
            self.setCurrentIndex(item.index())
        finally:
            self._silent = False
        if emit:
            self.folderSelected.emit(folder_id)

    def current_folder_id(self) -> int | None:
        idx = self.currentIndex()
        fid = idx.data(ID_ROLE) if idx.isValid() else None
        return None if fid in (None, ALL_ID) else int(fid)

    def folder_ids(self) -> list[int]:
        return list(self._item_by_id)

    def item_count(self, folder_id: int | None) -> int | None:
        """Rozet sayısı (testler için)."""
        item = self._item_by_id.get(folder_id) if folder_id is not None else self._model.item(0)
        return None if item is None else item.data(COUNT_ROLE)

    def current_changed_slot(self, current: QModelIndex, previous: QModelIndex) -> None:
        fid = current.data(ID_ROLE) if current.isValid() else None
        self.folderSelected.emit(None if fid in (None, ALL_ID) else int(fid))

    def currentChanged(self, current: QModelIndex, previous: QModelIndex) -> None:  # noqa: N802
        super().currentChanged(current, previous)
        if not self.signalsBlocked() and not self._silent:
            self.current_changed_slot(current, previous)

    # -- bağlam menüsü -------------------------------------------------------

    def _on_context_menu(self, pos: QPoint) -> None:
        idx = self.indexAt(pos)
        fid = idx.data(ID_ROLE) if idx.isValid() else None
        if idx.isValid():
            self.setCurrentIndex(idx)
        self.contextRequested.emit(None if fid in (None, ALL_ID) else int(fid), self.viewport().mapToGlobal(pos))

    # -- sürükle & bırak -----------------------------------------------------

    def startDrag(self, supportedActions) -> None:  # noqa: N802
        idx = self.currentIndex()
        if not idx.isValid() or idx.data(ID_ROLE) in (None, ALL_ID):
            return
        mime = self._model.mimeData([idx])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)

    @staticmethod
    def _accepts(mime: QMimeData) -> bool:
        return mime.hasFormat(FOLDER_MIME) or mime.hasFormat(IMAGES_MIME) or mime.hasUrls()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._accepts(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if not self._accepts(event.mimeData()):
            event.ignore()
            return
        idx = self.indexAt(event.position().toPoint())
        if event.mimeData().hasFormat(IMAGES_MIME) and (not idx.isValid() or idx.data(ID_ROLE) == ALL_ID):
            idx = QModelIndex()
        if idx != self._drop_index:
            self._drop_index = idx
            self.viewport().update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._drop_index = QModelIndex()
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        idx = self.indexAt(event.position().toPoint())
        raw = idx.data(ID_ROLE) if idx.isValid() else None
        target: int | None = None if raw in (None, ALL_ID) else int(raw)
        mime = event.mimeData()
        self._drop_index = QModelIndex()
        self.viewport().update()
        if mime.hasFormat(FOLDER_MIME):
            moving = int(bytes(mime.data(FOLDER_MIME)).decode())
            self.folderMoveRequested.emit(moving, target)       # null = köke
        elif mime.hasFormat(IMAGES_MIME):
            if target is not None:
                self.imagesDropped.emit(decode_ids(mime.data(IMAGES_MIME)), target)
        elif mime.hasUrls():
            paths = [Path(u.toLocalFile()) for u in mime.urls() if u.isLocalFile()]
            if paths:
                self.filesDropped.emit(paths, target)
        event.setDropAction(Qt.DropAction.IgnoreAction)   # model satırları silinmesin
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._drop_index.isValid():
            p = QPainter(self.viewport())
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(ACCENT), 2))
            fill = QColor(ACCENT)
            fill.setAlpha(36)
            p.setBrush(fill)
            p.drawRoundedRect(QRectF(self.visualRect(self._drop_index)).adjusted(3, 2, -3, -2), 6, 6)
