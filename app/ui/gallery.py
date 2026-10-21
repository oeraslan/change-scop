"""Galeri bileşenleri: kart modeli, çizen delegate, seçim mantığı ve yüzen karşılaştırma çubuğu."""

from __future__ import annotations

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QEasingCurve,
    QMimeData,
    QModelIndex,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDrag,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QListView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)
from qfluentwidgets import BodyLabel, CaptionLabel, PrimaryPushButton, StrongBodyLabel, ToolButton, TransparentToolButton
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import isDarkTheme

from app.storage import ImageRecord, Library
from app.ui.compare_view import format_date
from app.ui.theme import ACCENT, SOURCE_LABELS, palette

IMAGES_MIME = "application/x-changescope-images"
CARD_W, CARD_H = 208, 196
PAD = 8
FOOTER_H = 48


def encode_ids(ids: list[int]) -> QByteArray:
    return QByteArray(",".join(str(i) for i in ids).encode())


def decode_ids(data: QByteArray | bytes) -> list[int]:
    raw = bytes(data).decode() if data else ""
    return [int(x) for x in raw.split(",") if x]


class GalleryModel(QAbstractListModel):
    """Seçili klasörün görüntüleri (çekim tarihine göre sıralı)."""

    RecordRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, library: Library | None = None, parent=None) -> None:
        super().__init__(parent)
        self._library = library
        self._records: list[ImageRecord] = []
        self._pixmaps: dict[int, QPixmap] = {}

    def set_library(self, library: Library | None) -> None:
        self._library = library
        self._pixmaps.clear()

    def set_records(self, records: list[ImageRecord]) -> None:
        self.beginResetModel()
        self._records = list(records)
        self.endResetModel()

    def records(self) -> list[ImageRecord]:
        return list(self._records)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._records)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        rec = self._records[index.row()]
        if role == self.RecordRole:
            return rec
        if role == Qt.ItemDataRole.DisplayRole:
            return rec.original_name
        return None

    def flags(self, index: QModelIndex):
        base = Qt.ItemFlag.ItemIsEnabled
        return base | Qt.ItemFlag.ItemIsDragEnabled if index.isValid() else base

    def pixmap(self, rec: ImageRecord) -> QPixmap:
        pix = self._pixmaps.get(rec.id)
        if pix is None:
            pix = QPixmap()
            if self._library is not None:
                path = self._library.thumb_path(rec)
                if path is not None:
                    pix.load(str(path))
            self._pixmaps[rec.id] = pix
        return pix


def circle_rect(card: QRect) -> QRect:
    """Seçim dairesinin (sol üst) dikdörtgeni."""
    return QRect(card.left() + PAD + 8, card.top() + PAD + 8, 24, 24)


def card_rect(cell: QRect) -> QRect:
    return cell.adjusted(PAD, PAD, -PAD, -PAD)


class GalleryDelegate(QStyledItemDelegate):
    """Thumbnail kartı çizer: seçim dairesi, kaynak etiketi, tarih, dosya adı."""

    def __init__(self, view: "GalleryView") -> None:
        super().__init__(view)
        self._view = view

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(CARD_W, CARD_H)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        rec: ImageRecord = index.data(GalleryModel.RecordRole)
        model: GalleryModel = index.model()  # type: ignore[assignment]
        pal = palette()
        dark = isDarkTheme()
        painter.save()
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
                               | QPainter.RenderHint.TextAntialiasing)
        card = card_rect(option.rect)
        hover = index == self._view.hover_index
        selected = rec.id in self._view.selected_ids

        # Kart zemini
        body = QPainterPath()
        body.addRoundedRect(QRectF(card), 8, 8)
        bg = pal.surface_hover if hover else pal.surface
        painter.fillPath(body, bg)
        if selected:
            sel = QColor(ACCENT)
            sel.setAlpha(34)
            painter.fillPath(body, sel)
        painter.setPen(QPen(QColor(ACCENT) if selected else pal.border, 2 if selected else 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(card).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        # Thumbnail (kırparak doldur)
        thumb = QRect(card.left() + 6, card.top() + 6, card.width() - 12, card.height() - FOOTER_H - 12)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(thumb), 6, 6)
        painter.setClipPath(clip)
        painter.fillRect(thumb, QColor(0, 0, 0, 60) if dark else QColor(0, 0, 0, 18))
        pix = model.pixmap(rec)
        if not pix.isNull():
            scaled = pix.scaled(thumb.size() * 1.0, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                Qt.TransformationMode.SmoothTransformation)
            sx = thumb.left() + (thumb.width() - scaled.width()) // 2
            sy = thumb.top() + (thumb.height() - scaled.height()) // 2
            painter.drawPixmap(sx, sy, scaled)
        painter.setClipping(False)

        # Kaynak etiketi (sağ üst)
        label = SOURCE_LABELS.get(rec.source, rec.source)
        font = QFont(painter.font())
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        tw = QFontMetrics(font).horizontalAdvance(label)
        tag = QRectF(thumb.right() - tw - 24, thumb.top() + 8, tw + 16, 20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 170))
        painter.drawRoundedRect(tag, 10, 10)
        painter.setPen(QColor(255, 255, 255, 235))
        painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)

        # Seçim dairesi (sol üst)
        circle = QRectF(circle_rect(card))
        any_selected = bool(self._view.selected_ids)
        if selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(ACCENT))
            painter.drawEllipse(circle)
            order = self._view.order_of(rec.id)
            painter.setPen(QColor(8, 20, 24))
            f2 = QFont(font)
            f2.setPixelSize(12)
            f2.setWeight(QFont.Weight.Bold)
            painter.setFont(f2)
            painter.drawText(circle, Qt.AlignmentFlag.AlignCenter, str(order) if order else "✓")
        elif hover or any_selected:
            painter.setPen(QPen(QColor(255, 255, 255, 235), 1.6))
            painter.setBrush(QColor(0, 0, 0, 90))
            painter.drawEllipse(circle.adjusted(1, 1, -1, -1))

        # Alt bilgi
        foot = QRect(card.left() + 12, thumb.bottom() + 6, card.width() - 24, FOOTER_H - 8)
        f3 = QFont(font)
        f3.setPixelSize(13)
        f3.setWeight(QFont.Weight.DemiBold)
        painter.setFont(f3)
        painter.setPen(pal.text)
        painter.drawText(foot.adjusted(0, 0, 0, -18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         format_date(rec.captured_at))
        f4 = QFont(f3)
        f4.setPixelSize(11)
        f4.setWeight(QFont.Weight.Normal)
        painter.setFont(f4)
        painter.setPen(pal.text_dim)
        name = QFontMetrics(f4).elidedText(rec.original_name, Qt.TextElideMode.ElideMiddle, foot.width())
        painter.drawText(foot.adjusted(0, 18, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
        painter.restore()


class GalleryView(QListView):
    """Kart ızgarası. En fazla iki görüntü seçilir (sıra: seçim sırası)."""

    selectionIdsChanged = Signal(list)
    contextRequested = Signal(object, QPoint)      # (ImageRecord | None, global pos)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.selected_ids: list[int] = []
        self.hover_index = QModelIndex()
        self._records_by_id: dict[int, ImageRecord] = {}
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setUniformItemSizes(True)
        self.setSpacing(0)
        self.setGridSize(QSize(CARD_W, CARD_H))
        self.setMouseTracking(True)
        self.setFrameShape(QListView.Shape.NoFrame)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(32)
        self.setDragEnabled(True)
        self.setStyleSheet("QListView{background:transparent;border:none;}")
        self.setItemDelegate(GalleryDelegate(self))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    # -- seçim ---------------------------------------------------------------

    def order_of(self, image_id: int) -> int:
        """Seçili görüntünün kronolojik sırası (1 = önce, 2 = sonra); seçili değilse 0."""
        if image_id not in self.selected_ids:
            return 0
        if len(self.selected_ids) < 2:
            return 1
        recs = [self._records_by_id.get(i) for i in self.selected_ids]
        if any(r is None for r in recs):
            return self.selected_ids.index(image_id) + 1
        keyed = sorted(recs, key=lambda r: (r.captured_at or r.created_at, r.id))  # type: ignore[union-attr]
        return [r.id for r in keyed].index(image_id) + 1  # type: ignore[union-attr]

    def remember_records(self, records: list[ImageRecord]) -> None:
        """Seçim sırası hesabı için kayıtları önbelleğe alır (diğer klasörlerden seçilenler korunur)."""
        for r in records:
            self._records_by_id[r.id] = r

    def set_selection(self, ids: list[int], emit: bool = True) -> None:
        self.selected_ids = list(ids)[-2:]
        self.viewport().update()
        if emit:
            self.selectionIdsChanged.emit(list(self.selected_ids))

    def toggle(self, image_id: int) -> None:
        ids = list(self.selected_ids)
        if image_id in ids:
            ids.remove(image_id)
        else:
            ids.append(image_id)
        self.set_selection(ids)

    def clear_selection(self) -> None:
        if self.selected_ids:
            self.set_selection([])

    # -- olaylar -------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and event.button() == Qt.MouseButton.LeftButton:
            rec: ImageRecord = index.data(GalleryModel.RecordRole)
            rect = card_rect(self.visualRect(index))
            on_circle = circle_rect(rect).adjusted(-4, -4, 4, 4).contains(event.position().toPoint())
            self._records_by_id[rec.id] = rec
            if on_circle or event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.toggle(rec.id)
            elif self.selected_ids == [rec.id]:
                self.set_selection([])
            else:
                self.set_selection([rec.id])
        elif not index.isValid() and event.button() == Qt.MouseButton.LeftButton:
            self.clear_selection()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index = self.indexAt(event.position().toPoint())
        if index != self.hover_index:
            old = self.hover_index
            self.hover_index = index
            for idx in (old, index):
                if idx.isValid():
                    self.update(idx)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self.hover_index.isValid():
            old, self.hover_index = self.hover_index, QModelIndex()
            self.update(old)
        super().leaveEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        index = self.indexAt(event.pos())
        rec = index.data(GalleryModel.RecordRole) if index.isValid() else None
        self.contextRequested.emit(rec, event.globalPos())

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.clear_selection()
            return
        super().keyPressEvent(event)

    def startDrag(self, supported) -> None:  # noqa: N802
        index = self.indexAt(self.viewport().mapFromGlobal(self.cursor().pos()))
        rec = index.data(GalleryModel.RecordRole) if index.isValid() else None
        ids = list(self.selected_ids) if rec is not None and rec.id in self.selected_ids else (
            [rec.id] if rec is not None else [])
        if not ids:
            return
        mime = QMimeData()
        mime.setData(IMAGES_MIME, encode_ids(ids))
        drag = QDrag(self)
        drag.setMimeData(mime)
        pix = self.model().pixmap(rec).scaledToWidth(96, Qt.TransformationMode.SmoothTransformation)  # type: ignore[union-attr]
        drag.setPixmap(pix)
        drag.exec(Qt.DropAction.MoveAction)


class FloatingBar(QWidget):
    """"Önce: [tarih] → Sonra: [tarih]  [⇄]  [Karşılaştır]" yüzen çubuğu (yumuşak gölgeli)."""

    swapRequested = Signal()
    compareRequested = Signal()
    clearRequested = Signal()
    M = 20   # gölge payı

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.before_label = StrongBodyLabel("", self)
        self.after_label = StrongBodyLabel("", self)
        self.swap_button = ToolButton(FIF.SYNC, self)
        self.swap_button.setToolTip("Önce / Sonra yer değiştir")
        self.swap_button.clicked.connect(self.swapRequested)
        self.compare_button = PrimaryPushButton(FIF.VIEW, "Karşılaştır", self)
        self.compare_button.clicked.connect(self.compareRequested)
        self.clear_button = TransparentToolButton(FIF.CLOSE, self)
        self.clear_button.setToolTip("Seçimi temizle")
        self.clear_button.clicked.connect(self.clearRequested)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(self.M + 16, self.M + 10, self.M + 10, self.M + 10)
        lay.setSpacing(10)
        lay.addWidget(CaptionLabel("Önce", self))
        lay.addWidget(self.before_label)
        arrow = BodyLabel("→", self)
        lay.addWidget(arrow)
        lay.addWidget(CaptionLabel("Sonra", self))
        lay.addWidget(self.after_label)
        lay.addSpacing(4)
        lay.addWidget(self.swap_button)
        lay.addWidget(self.compare_button)
        lay.addWidget(self.clear_button)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._anims: list[QPropertyAnimation] = []
        self.hide()

    def set_dates(self, before_text: str, after_text: str) -> None:
        self.before_label.setText(before_text)
        self.after_label.setText(after_text)
        self.adjustSize()

    def place(self, container: QWidget) -> QPoint:
        """Konteynerin altında ortalı hedef konum."""
        self.adjustSize()
        x = (container.width() - self.width()) // 2
        y = container.height() - self.height() - 4
        return QPoint(max(0, x), max(0, y))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(self.rect()).adjusted(self.M, self.M, -self.M, -self.M)
        for i in range(10, 0, -1):         # yumuşak gölge
            c = QColor(0, 0, 0, 7 if isDarkTheme() else 5)
            p.setBrush(c)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(body.adjusted(-i, -i + 4, i, i + 4), 14 + i, 14 + i)
        p.setBrush(QColor(38, 42, 52) if isDarkTheme() else QColor(255, 255, 255))
        p.setPen(QPen(QColor(255, 255, 255, 34) if isDarkTheme() else QColor(0, 0, 0, 30), 1))
        p.drawRoundedRect(body, 14, 14)

    def show_animated(self, container: QWidget) -> None:
        """Alttan kayarak ve solarak görünür."""
        for a in self._anims:
            a.stop()
        self._anims.clear()
        target = self.place(container)
        if self.isVisible() and self._effect.opacity() > 0.95:
            self.move(target)
            return
        self.show()
        self.raise_()
        start = QPoint(target.x(), target.y() + 24)
        self.move(start)
        move = QPropertyAnimation(self, b"pos", self)
        move.setDuration(220)
        move.setEasingCurve(QEasingCurve.Type.OutCubic)
        move.setStartValue(start)
        move.setEndValue(target)
        fade = QPropertyAnimation(self._effect, b"opacity", self)
        fade.setDuration(200)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        self._anims = [move, fade]
        move.start()
        fade.start()

    def hide_animated(self) -> None:
        """Solarak kaybolur."""
        if not self.isVisible():
            return
        for a in self._anims:
            a.stop()
        fade = QPropertyAnimation(self._effect, b"opacity", self)
        fade.setDuration(160)
        fade.setStartValue(self._effect.opacity())
        fade.setEndValue(0.0)
        fade.finished.connect(self.hide)
        self._anims = [fade]
        fade.start()

