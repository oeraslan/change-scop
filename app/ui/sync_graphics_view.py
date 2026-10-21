"""Senkron zoom/pan yapan `QGraphicsView` çifti ve bölge kutusu öğeleri."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPointF,
    Property,
    QPropertyAnimation,
    QRectF,
    Qt,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)

from app.ui.theme import VERDICT_COLORS, VERDICT_LABELS, palette
from core.contracts import ChangeVerdict

MAX_SCALE = 40.0
ZOOM_STEP = 1.25


@dataclass
class RegionSpec:
    """Bir bölge kutusunun çizim bilgisi (before koordinatları)."""

    region_id: int
    rect: QRectF
    verdict: ChangeVerdict = ChangeVerdict.UNREVIEWED
    score: float = 0.0
    description: str | None = None

    @property
    def tooltip(self) -> str:
        """Hover açıklaması: numara, verdict, açıklama ve skor."""
        lines = [f"#{self.region_id} · {VERDICT_LABELS[self.verdict]}"]
        if self.description:
            lines.append(self.description)
        lines.append(f"Skor: {self.score:.2f}")
        return "\n".join(lines)


class _Badge(QGraphicsItem):
    """Kutunun sol üstündeki numara rozeti; yakınlaştırmadan etkilenmez (sabit piksel boyutu)."""

    H = 20

    def __init__(self, text: str, color: QColor, parent: QGraphicsItem) -> None:
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._text, self._color = text, color
        self._w = 14 + 7 * len(text)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-1, -self.H - 1, self._w + 2, self.H + 2)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawRoundedRect(QRectF(0, -self.H, self._w, self.H), 6, 6)
        font = QFont(painter.font())
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(0, -self.H, self._w, self.H), Qt.AlignmentFlag.AlignCenter, self._text)


class RegionItem(QGraphicsRectItem):
    """Yarı saydam dolgulu, net kenarlı bölge kutusu (verdict rengi)."""

    def __init__(self, spec: RegionSpec, group: "SyncGroup | None", parent: QGraphicsItem) -> None:
        super().__init__(spec.rect, parent)
        self.spec = spec
        self._group = group
        self._highlight = False
        self._hover = False
        self._color = QColor(VERDICT_COLORS[spec.verdict])
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # sürükleyerek pan'i engellemesin
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(spec.tooltip)
        self.badge = _Badge(str(spec.region_id), self._color, self)
        self.badge.setPos(spec.rect.topLeft())
        self.badge.setToolTip(spec.tooltip)
        self._restyle()

    @property
    def region_id(self) -> int:
        return self.spec.region_id

    def set_highlight(self, on: bool) -> None:
        """Kart seçildiğinde / hover edildiğinde vurgu."""
        self._highlight = on
        self._restyle()

    def set_hover(self, on: bool) -> None:
        self._hover = on
        self._restyle()

    def _restyle(self) -> None:
        strong = self._highlight or self._hover
        fill = QColor(self._color)
        fill.setAlpha(110 if strong else 48)
        pen = QPen(self._color, 3.5 if self._highlight else 2.0)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.setPen(pen)
        self.setBrush(QBrush(fill))
        self.setZValue(5 if strong else 2)
        self.update()

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        if self._group:
            self._group.regionHovered.emit(self.region_id, True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        if self._group:
            self._group.regionHovered.emit(self.region_id, False)
        super().hoverLeaveEvent(event)


class _Layer(QGraphicsObject):
    """Çocuklarını gruplayan, saydamlığı animasyonlanabilen boş katman."""

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF()

    def paint(self, painter, option, widget=None) -> None:  # noqa: D401
        return None


class SyncGraphicsView(QGraphicsView):
    """Tek görüntüyü gösteren, wheel zoom + sürükle pan + çift tık "sığdır" yapan görünüm.

    Kullanıcı etkileşimi `viewChanged` sinyalini yayar; `SyncGroup` bunu eşlerine uygular.
    Görünüm durumu (ölçek + sahne merkezi) `view_state()` / `apply_state()` ile taşınır.
    """

    viewChanged = Signal()
    regionClicked = Signal(int)

    def __init__(self, group: "SyncGroup | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._group = group
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._applying = False
        self._scale = 1.0
        self._fit_scale = 1.0
        self._image_rect = QRectF()
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._heat_item: QGraphicsPixmapItem | None = None
        self._region_layer = _Layer()
        self._region_layer.setZValue(2)
        self._scene.addItem(self._region_layer)
        self._region_items: dict[int, RegionItem] = {}
        self._regions_visible = True
        self._hidden_verdicts: set[ChangeVerdict] = set()
        self._anims: dict[str, QPropertyAnimation] = {}
        self._zoom_anim: QVariantAnimation | None = None
        self._press_pos: QPointF | None = None

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMouseTracking(True)
        self.horizontalScrollBar().valueChanged.connect(self._on_scroll)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self._update_background()

    # -- tema ----------------------------------------------------------------

    def _update_background(self) -> None:
        self.setBackgroundBrush(QBrush(palette().canvas))

    def refresh_theme(self) -> None:
        """Tema değişince arka plan rengini yeniler."""
        self._update_background()

    # -- içerik --------------------------------------------------------------

    @property
    def image_rect(self) -> QRectF:
        return QRectF(self._image_rect)

    def set_image(self, image: QImage | QPixmap | None, keep_view: bool = False) -> None:
        """Ana görüntüyü ayarlar. `keep_view=False` ise sığdırır."""
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        if image is None or image.isNull():
            self._image_rect = QRectF()
            return
        pix = image if isinstance(image, QPixmap) else QPixmap.fromImage(image)
        item = QGraphicsPixmapItem(pix)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setZValue(0)
        self._scene.addItem(item)
        self._pixmap_item = item
        old = self._image_rect
        self._image_rect = QRectF(0, 0, pix.width(), pix.height())
        pad_x, pad_y = pix.width() * 0.5, pix.height() * 0.5
        self._scene.setSceneRect(self._image_rect.adjusted(-pad_x, -pad_y, pad_x, pad_y))
        if self._heat_item is not None:
            self._heat_item.setPos(0, 0)
        if not keep_view or old != self._image_rect:
            self.fit(emit=False)

    def set_heatmap(self, image: QImage | None) -> None:
        """Isı haritası katmanını (RGBA) ayarlar; görünürlüğü `set_heatmap_visible` yönetir."""
        if self._heat_item is not None:
            self._scene.removeItem(self._heat_item)
            self._heat_item = None
        if image is None:
            return
        pix = QPixmap.fromImage(image)
        item = QGraphicsPixmapItem(pix)
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        if not self._image_rect.isEmpty() and pix.width():
            k = self._image_rect.width() / pix.width()
            item.setScale(k)
        item.setZValue(1)
        item.setOpacity(0.0)
        item.setVisible(False)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(item)
        self._heat_item = item

    def has_heatmap(self) -> bool:
        return self._heat_item is not None

    def heatmap_visible(self) -> bool:
        """Isı haritası şu an (animasyon sonunda) görünür mü?"""
        return self._heat_item is not None and self._heat_item.isVisible() and self._heat_item.opacity() > 0

    def set_heatmap_visible(self, visible: bool, animate: bool = True) -> None:
        """Isı haritası overlay'ini yumuşak geçişle açar/kapatır."""
        if self._heat_item is None:
            return
        self._fade_item("heat", self._heat_item, 0.6 if visible else 0.0, animate)

    def set_regions(self, specs: list[RegionSpec]) -> None:
        """Bölge kutularını yeniden kurar (eskileri temizler)."""
        for item in list(self._region_items.values()):
            self._scene.removeItem(item)
        self._region_items.clear()
        for spec in specs:
            self._region_items[spec.region_id] = RegionItem(spec, self._group, self._region_layer)
        self._apply_verdict_filter()
        self._region_layer.setVisible(self._regions_visible and bool(specs))

    def region_items(self) -> list[RegionItem]:
        return list(self._region_items.values())

    def visible_region_items(self) -> list[RegionItem]:
        """Şu an gerçekten görünen (katman + öğe) kutular."""
        if not self._region_layer.isVisible() or self._region_layer.opacity() <= 0:
            return []
        return [i for i in self._region_items.values() if i.isVisible()]

    def regions_visible(self) -> bool:
        return self._regions_visible

    def set_regions_visible(self, visible: bool, animate: bool = True) -> None:
        """"Farkları göster": kapalıyken hiçbir kutu/numara görünmez."""
        self._regions_visible = visible
        if not self._region_items:
            self._region_layer.setVisible(False)
            return
        self._fade_item("regions", self._region_layer, 1.0 if visible else 0.0, animate)

    def set_hidden_verdicts(self, verdicts: set[ChangeVerdict]) -> None:
        """Belirtilen verdict'lere sahip kutuları gizler (ör. yanlış alarmlar)."""
        self._hidden_verdicts = set(verdicts)
        self._apply_verdict_filter()

    def _apply_verdict_filter(self) -> None:
        for item in self._region_items.values():
            item.setVisible(item.spec.verdict not in self._hidden_verdicts)

    def set_highlight(self, region_id: int | None) -> None:
        for rid, item in self._region_items.items():
            item.set_highlight(rid == region_id)

    def set_hover(self, region_id: int, on: bool) -> None:
        item = self._region_items.get(region_id)
        if item:
            item.set_hover(on)

    def _fade_item(self, key: str, item: QGraphicsItem, target: float, animate: bool) -> None:
        old = self._anims.pop(key, None)
        if old is not None:
            old.stop()
        if not animate:
            item.setOpacity(target)
            item.setVisible(target > 0)
            return
        item.setVisible(True)
        target_obj = item.toGraphicsObject() or _OpacityProxy(item, self)
        anim = QPropertyAnimation(target_obj, b"opacity", self)
        anim.setDuration(200)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(item.opacity())
        anim.setEndValue(target)
        anim.finished.connect(lambda i=item, t=target: i.setVisible(t > 0))
        self._anims[key] = anim
        anim.start()

    # -- görünüm durumu ------------------------------------------------------

    def zoom_factor(self) -> float:
        """Mevcut ölçek (piksel/sahne birimi)."""
        return self._scale

    def fit_scale(self) -> float:
        return self._fit_scale

    def center_point(self) -> QPointF:
        """Görünür alanın sahne koordinatındaki merkezi."""
        return self.mapToScene(self.viewport().rect().center())

    def view_state(self) -> tuple[float, QPointF]:
        return self._scale, self.center_point()

    def _set_scale(self, s: float) -> None:
        s = max(self._fit_scale * 0.5, min(MAX_SCALE, s)) if self._fit_scale else s
        self._scale = s
        self.setTransform(QTransform.fromScale(s, s))

    def apply_state(self, scale: float, center: QPointF) -> None:
        """Eşten gelen durumu uygular (sinyal YAYMAZ)."""
        if self._image_rect.isEmpty():
            return
        self._applying = True
        try:
            self._set_scale(scale)
            self.centerOn(center)
        finally:
            self._applying = False

    def fit(self, emit: bool = True) -> None:
        """Görüntüyü pencereye sığdırır."""
        if self._image_rect.isEmpty() or self.viewport().width() < 4:
            self._fit_scale = max(self._fit_scale, 1e-6)
            return
        vw, vh = self.viewport().width(), self.viewport().height()
        self._fit_scale = min(vw / self._image_rect.width(), vh / self._image_rect.height()) * 0.98
        self._applying = True
        try:
            self._set_scale(self._fit_scale)
            self.centerOn(self._image_rect.center())
        finally:
            self._applying = False
        if emit:
            self.viewChanged.emit()

    def zoom_at(self, factor: float, anchor: QPointF | None = None, emit: bool = True) -> None:
        """`anchor` (viewport pikseli) sabit kalacak şekilde yakınlaştırır."""
        if self._image_rect.isEmpty():
            return
        anchor = anchor if anchor is not None else QPointF(self.viewport().rect().center())
        before = self.mapToScene(anchor.toPoint())
        center = self.center_point()
        self._applying = True
        try:
            self._set_scale(self._scale * factor)
            after = self.mapToScene(anchor.toPoint())
            self.centerOn(center + (before - after))
        finally:
            self._applying = False
        if emit:
            self.viewChanged.emit()

    def animate_to_rect(self, rect: QRectF, context: float = 3.0, duration: int = 280) -> None:
        """Verilen bölgeye (çevresiyle birlikte) animasyonla yakınlaşır; eşler senkron izler."""
        if self._image_rect.isEmpty() or rect.isEmpty():
            return
        vw, vh = self.viewport().width(), self.viewport().height()
        target_scale = min(vw / (rect.width() * context), vh / (rect.height() * context))
        target_scale = max(self._fit_scale, min(target_scale, MAX_SCALE))
        start_scale, start_center = self._scale, self.center_point()
        end_center = rect.center()
        if self._zoom_anim is not None:
            self._zoom_anim.stop()
        anim = QVariantAnimation(self)
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def step(t: float) -> None:
            # Ölçek logaritmik, merkez doğrusal enterpolasyon.
            s = math.exp(math.log(start_scale) * (1 - t) + math.log(target_scale) * t)
            c = start_center + (end_center - start_center) * t
            self.apply_state(s, c)
            self.viewChanged.emit()

        anim.valueChanged.connect(step)
        self._zoom_anim = anim
        anim.start()

    # -- olaylar -------------------------------------------------------------

    def _on_scroll(self) -> None:
        if not self._applying:
            self.viewChanged.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.zoom_at(ZOOM_STEP ** steps, event.position())
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._press_pos = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        if self._press_pos is not None and event.button() == Qt.MouseButton.LeftButton:
            moved = (event.position() - self._press_pos).manhattanLength()
            if moved < 5:
                for item in self.items(event.position().toPoint()):
                    if isinstance(item, RegionItem) and item.isVisible():
                        self.regionClicked.emit(item.region_id)
                        break
                    if isinstance(item, _Badge) and item.parentItem() is not None:
                        parent = item.parentItem()
                        if isinstance(parent, RegionItem):
                            self.regionClicked.emit(parent.region_id)
                            break
        self._press_pos = None

    def resizeEvent(self, event) -> None:  # noqa: N802
        old_center = self.center_point() if not self._image_rect.isEmpty() else None
        was_fit = abs(self._scale - self._fit_scale) < 1e-6
        super().resizeEvent(event)
        if self._image_rect.isEmpty():
            return
        if was_fit or self._fit_scale <= 1e-6:
            self.fit(emit=False)
        elif old_center is not None:
            self._applying = True
            try:
                self.centerOn(old_center)
            finally:
                self._applying = False


class _OpacityProxy(QObject):
    """QGraphicsItem'ın opacity'sini QPropertyAnimation ile sürmek için."""

    def __init__(self, item: QGraphicsItem, parent: QObject) -> None:
        super().__init__(parent)
        self._item = item

    def _get(self) -> float:
        return self._item.opacity()

    def _set(self, value: float) -> None:
        self._item.setOpacity(value)

    opacity = Property(float, _get, _set)


class SyncGroup(QObject):
    """Birden çok `SyncGraphicsView`'i senkronlar ve bölge olaylarını yayar."""

    regionHovered = Signal(int, bool)
    regionClicked = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._views: list[SyncGraphicsView] = []
        self.regionHovered.connect(self._on_hover)

    def create_view(self, parent: QWidget | None = None) -> SyncGraphicsView:
        """Gruba bağlı yeni bir görünüm üretir."""
        view = SyncGraphicsView(self, parent)
        self.add(view)
        return view

    def add(self, view: SyncGraphicsView) -> None:
        self._views.append(view)
        view.viewChanged.connect(lambda v=view: self._propagate(v))
        view.regionClicked.connect(self.regionClicked)

    @property
    def views(self) -> list[SyncGraphicsView]:
        return list(self._views)

    def _propagate(self, source: SyncGraphicsView) -> None:
        scale, center = source.view_state()
        for other in self._views:
            if other is not source:
                other.apply_state(scale, center)

    def _on_hover(self, region_id: int, on: bool) -> None:
        for v in self._views:
            v.set_hover(region_id, on)

    def fit_all(self) -> None:
        """Hepsini sığdır (ilk görünümden eşleri takip eder)."""
        if self._views:
            for v in self._views:
                v.fit(emit=False)
            self._propagate(self._views[0])

    def highlight(self, region_id: int | None) -> None:
        for v in self._views:
            v.set_highlight(region_id)

    def zoom_to_region(self, rect: QRectF) -> None:
        """Tüm görünümleri bölgeye animasyonla yakınlaştırır (lider: ilk görünüm)."""
        if self._views:
            self._views[0].animate_to_rect(rect)
