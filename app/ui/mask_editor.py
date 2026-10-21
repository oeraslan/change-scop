"""Yok sayma maskesi düzenleyicisi: model (`MaskEditor`), sahne katmanı (`MaskOverlayItem`) ve araç çubuğu.

Maske, "önce" görüntüsünün EKRANDA gösterilen halinin piksel koordinatlarında çizilir (sahne koordinatı =
görüntü pikseli). Bellek için çalışma çözünürlüğü uzun kenarda en çok `WORK_MAX_SIDE` pikseldir; çizim
sahne koordinatından bu çözünürlüğe doğru ölçeklenerek eşlenir. Kaydedilen/pipeline'a verilen maske bu
çözünürlüktedir (pipeline herhangi bir çözünürlüğü kabul eder ve `valid_mask` boyutuna ölçekler).
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QHBoxLayout,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    Slider,
    StrongBodyLabel,
)
from qfluentwidgets import FluentIcon as FIF

from app.ui.theme import Surface

WORK_MAX_SIDE = 2048
UNDO_LIMIT = 60
MASK_RGB = (167, 139, 250)          # mor (#A78BFA)
ALPHA_FILL, ALPHA_HATCH = 96, 215   # dolgu / çizgi saydamlığı
TOOL_BRUSH, TOOL_ERASER, TOOL_PAN = "brush", "eraser", "pan"


class MaskEditor(QObject):
    """Fırça / silgi / geri al / yinele destekli maske modeli (Qt'den bağımsız numpy tabanı).

    Sinyaller:
        dirty(x, y, w, h)  maske uzayında değişen dikdörtgen (overlay yenileme için)
        historyChanged()   geri al / yinele / değişti durumu etkilenmiş olabilir
    """

    dirty = Signal(int, int, int, int)
    historyChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tool = TOOL_BRUSH
        self.brush_px = 48.0        # görüntü pikseli cinsinden çap
        self._img_w = self._img_h = 0
        self._w = self._h = 0
        self._k = 1.0               # görüntü pikseli -> maske pikseli
        self._mask = np.zeros((0, 0), np.uint8)
        self._overlay = np.zeros((0, 0, 4), np.uint8)
        self._hatch = np.zeros((0, 0), bool)
        self._qimage = QImage()
        self._undo: list[bytes] = []
        self._redo: list[bytes] = []
        self._baseline = b""
        self._stroke_start: bytes | None = None
        self._last: tuple[int, int] | None = None

    # -- tuval ---------------------------------------------------------------

    def set_canvas(self, img_w: int, img_h: int) -> None:
        """Görüntü boyutuna göre boş bir tuval kurar (geçmiş sıfırlanır)."""
        self._img_w, self._img_h = max(1, int(img_w)), max(1, int(img_h))
        self._k = min(1.0, WORK_MAX_SIDE / max(self._img_w, self._img_h))
        self._w = max(1, round(self._img_w * self._k))
        self._h = max(1, round(self._img_h * self._k))
        self._mask = np.zeros((self._h, self._w), np.uint8)
        yy, xx = np.mgrid[0:self._h, 0:self._w]
        self._hatch = ((xx + yy) % 10) < 2
        self._overlay = np.zeros((self._h, self._w, 4), np.uint8)
        self._overlay[..., 0], self._overlay[..., 1], self._overlay[..., 2] = MASK_RGB
        # QImage, `self._overlay` belleğini paylaşır; dizi bu nesnede yaşadığı sürece geçerlidir.
        self._qimage = QImage(self._overlay.data, self._w, self._h, self._w * 4, QImage.Format.Format_RGBA8888)
        self._undo.clear()
        self._redo.clear()
        self._baseline = self._pack()
        self._refresh_overlay(0, 0, self._w, self._h)
        self.historyChanged.emit()

    @property
    def image_size(self) -> tuple[int, int]:
        """(genişlik, yükseklik) – maskenin çizildiği görüntü boyutu."""
        return self._img_w, self._img_h

    @property
    def mask_size(self) -> tuple[int, int]:
        """(genişlik, yükseklik) – çalışma çözünürlüğü."""
        return self._w, self._h

    @property
    def scale(self) -> float:
        """Görüntü pikseli -> maske pikseli oranı."""
        return self._k

    def overlay_image(self) -> QImage:
        """Renkli yarı saydam overlay (maske çözünürlüğünde, RGBA8888)."""
        return self._qimage

    # -- içerik --------------------------------------------------------------

    def load(self, mask: np.ndarray | None) -> None:
        """Mevcut maskeyi yükler (herhangi bir çözünürlükte bool/uint8); geçmiş sıfırlanır."""
        self._mask[:] = 0
        if mask is not None and np.asarray(mask).size:
            m = (np.asarray(mask) > 0).astype(np.uint8) * 255
            if m.shape != (self._h, self._w):
                m = cv2.resize(m, (self._w, self._h), interpolation=cv2.INTER_NEAREST)
            self._mask[:] = m
        self._undo.clear()
        self._redo.clear()
        self._baseline = self._pack()
        self._refresh_overlay(0, 0, self._w, self._h)
        self.dirty.emit(0, 0, self._w, self._h)
        self.historyChanged.emit()

    def mask_bool(self) -> np.ndarray:
        """Maske kopyası (HxW bool, True = yok say), çalışma çözünürlüğünde."""
        return self._mask > 0

    def has_content(self) -> bool:
        """Hiç işaretli piksel var mı?"""
        return bool(self._mask.any())

    def coverage(self) -> float:
        """İşaretli alanın oranı (0..1)."""
        return float((self._mask > 0).mean()) if self._mask.size else 0.0

    def is_modified(self) -> bool:
        """Yüklenen/kaydedilen maskeye göre değişiklik var mı?"""
        return self._pack() != self._baseline

    def mark_saved(self) -> None:
        """Mevcut hali "temiz" referans yapar."""
        self._baseline = self._pack()
        self.historyChanged.emit()

    # -- çizim ---------------------------------------------------------------

    def _to_mask(self, x: float, y: float) -> tuple[int, int]:
        return (int(round(min(max(x * self._k, 0), self._w - 1))),
                int(round(min(max(y * self._k, 0), self._h - 1))))

    def begin_stroke(self, scene_x: float, scene_y: float) -> None:
        """Fırça/silgi vuruşunu başlatır (sahne koordinatı = görüntü pikseli)."""
        if self._w == 0 or self.tool not in (TOOL_BRUSH, TOOL_ERASER):
            return
        self._stroke_start = self._pack()
        self._last = None
        self.continue_stroke(scene_x, scene_y)

    def continue_stroke(self, scene_x: float, scene_y: float) -> None:
        """Vuruşu yeni noktaya uzatır."""
        if self._stroke_start is None:
            return
        p = self._to_mask(scene_x, scene_y)
        value = 255 if self.tool == TOOL_BRUSH else 0
        radius = max(1, int(round(self.brush_px * self._k / 2)))
        if self._last is None:
            cv2.circle(self._mask, p, radius, value, -1, cv2.LINE_8)
            x0, y0, x1, y1 = p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius
        else:
            cv2.line(self._mask, self._last, p, value, 2 * radius, cv2.LINE_8)
            cv2.circle(self._mask, p, radius, value, -1, cv2.LINE_8)
            x0, y0 = min(p[0], self._last[0]) - radius, min(p[1], self._last[1]) - radius
            x1, y1 = max(p[0], self._last[0]) + radius, max(p[1], self._last[1]) + radius
        self._last = p
        self._refresh_overlay(x0, y0, x1 + 1, y1 + 1)

    def end_stroke(self) -> None:
        """Vuruşu bitirir; maske değiştiyse geri alma yığınına işler."""
        if self._stroke_start is None:
            return
        before, self._stroke_start, self._last = self._stroke_start, None, None
        if self._pack() != before:
            self._undo.append(before)
            del self._undo[:-UNDO_LIMIT]
            self._redo.clear()
        self.historyChanged.emit()

    def stroke(self, points: list[tuple[float, float]]) -> None:
        """Tek vuruşta bir nokta dizisi çizer (testler / programatik kullanım)."""
        if not points:
            return
        self.begin_stroke(*points[0])
        for pt in points[1:]:
            self.continue_stroke(*pt)
        self.end_stroke()

    # -- geçmiş --------------------------------------------------------------

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        """Son vuruşu geri alır."""
        if not self._undo:
            return False
        self._redo.append(self._pack())
        self._unpack(self._undo.pop())
        return True

    def redo(self) -> bool:
        """Geri alınan vuruşu yineler."""
        if not self._redo:
            return False
        self._undo.append(self._pack())
        self._unpack(self._redo.pop())
        return True

    def clear(self) -> None:
        """Tüm maskeyi siler (geri alınabilir)."""
        if not self._mask.any():
            return
        self._undo.append(self._pack())
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self._mask[:] = 0
        self._refresh_overlay(0, 0, self._w, self._h)
        self.dirty.emit(0, 0, self._w, self._h)
        self.historyChanged.emit()

    # -- iç ------------------------------------------------------------------

    def _pack(self) -> bytes:
        return np.packbits(self._mask > 0).tobytes()

    def _unpack(self, data: bytes) -> None:
        bits = np.unpackbits(np.frombuffer(data, np.uint8))[: self._w * self._h]
        self._mask[:] = bits.reshape(self._h, self._w).astype(np.uint8) * 255
        self._refresh_overlay(0, 0, self._w, self._h)
        self.dirty.emit(0, 0, self._w, self._h)
        self.historyChanged.emit()

    def _refresh_overlay(self, x0: int, y0: int, x1: int, y1: int) -> None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self._w, x1), min(self._h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        sub = self._mask[y0:y1, x0:x1] > 0
        alpha = np.where(sub, np.where(self._hatch[y0:y1, x0:x1], ALPHA_HATCH, ALPHA_FILL), 0)
        self._overlay[y0:y1, x0:x1, 3] = alpha.astype(np.uint8)
        self.dirty.emit(x0, y0, x1 - x0, y1 - y0)


class MaskOverlayItem(QGraphicsItem):
    """Maske overlay'ini sahnede görüntü koordinatlarında çizer (görüntüyle birebir hizalı)."""

    def __init__(self, editor: MaskEditor) -> None:
        super().__init__()
        self._editor = editor
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(3)
        editor.dirty.connect(self._on_dirty)

    def boundingRect(self) -> QRectF:  # noqa: N802
        w, h = self._editor.image_size
        return QRectF(0, 0, w, h)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        img = self._editor.overlay_image()
        if not img.isNull():
            painter.drawImage(self.boundingRect(), img)

    def _on_dirty(self, x: int, y: int, w: int, h: int) -> None:
        k = self._editor.scale or 1.0
        self.update(QRectF(x / k, y / k, w / k, h / k).adjusted(-2, -2, 2, 2))

    def sync_geometry(self) -> None:
        """Tuval boyutu değişince sınırları yeniler."""
        self.prepareGeometryChange()


class BrushCursorItem(QGraphicsEllipseItem):
    """Fırça boyutunu gösteren imleç halkası (sahne biriminde çap = fırça px; kenar çizgisi sabit kalınlık)."""

    def __init__(self) -> None:
        super().__init__()
        pen = QPen(QColor(255, 255, 255, 235), 1.5)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QColor(MASK_RGB[0], MASK_RGB[1], MASK_RGB[2], 40))
        self.setZValue(9)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setVisible(False)

    def place(self, center: QPointF, diameter: float) -> None:
        r = max(0.5, diameter / 2)
        self.setRect(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r))


class MaskToolbar(Surface):
    """Maske modu araç çubuğu: fırça/silgi/kaydır, boyut kaydırıcısı, geri al/yinele/temizle, kaydet/iptal."""

    toolChanged = Signal(str)
    sizeChanged = Signal(int)
    undoRequested = Signal()
    redoRequested = Signal()
    clearRequested = Signal()
    cancelRequested = Signal()
    saveRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        title = StrongBodyLabel("Yok sayma maskesi", self)
        hint = CaptionLabel("Karşılaştırmaya katılmayacak alanları boya (bulut, su yüzeyi, park alanı…). "
                            "Tekerlek: yakınlaştır · Orta tuş / Boşluk + sürükle: kaydır", self)
        hint.setTextColor(QColor(0, 0, 0, 140), QColor(255, 255, 255, 150))
        self.tools = SegmentedWidget(self)
        self.tools.addItem(TOOL_BRUSH, "Fırça", lambda: self.set_tool(TOOL_BRUSH), FIF.BRUSH)
        self.tools.addItem(TOOL_ERASER, "Silgi", lambda: self.set_tool(TOOL_ERASER), FIF.ERASE_TOOL)
        self.tools.addItem(TOOL_PAN, "Kaydır", lambda: self.set_tool(TOOL_PAN), FIF.MOVE)
        self.tools.setCurrentItem(TOOL_BRUSH)

        self.size_slider = Slider(Qt.Orientation.Horizontal, self)
        self.size_slider.setRange(4, 400)
        self.size_slider.setValue(48)
        self.size_slider.setFixedWidth(180)
        self.size_label = BodyLabel("48 px", self)
        self.size_label.setFixedWidth(52)
        self.size_slider.valueChanged.connect(self._on_size)

        self.undo_button = PushButton(FIF.RETURN, "Geri al", self)
        self.undo_button.setToolTip("Geri al (Ctrl+Z)")
        self.redo_button = PushButton(FIF.ROTATE, "Yinele", self)
        self.redo_button.setToolTip("Yinele (Ctrl+Y)")
        self.clear_button = PushButton(FIF.DELETE, "Temizle", self)
        self.cancel_button = PushButton(FIF.CLOSE, "İptal", self)
        self.save_button = PrimaryPushButton(FIF.SAVE, "Kaydet ve yeniden karşılaştır", self)
        self.undo_button.clicked.connect(self.undoRequested)
        self.redo_button.clicked.connect(self.redoRequested)
        self.clear_button.clicked.connect(self.clearRequested)
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.save_button.clicked.connect(self.saveRequested)
        self.badge = CaptionLabel("", self)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(title)
        top.addWidget(hint, 1)
        top.addWidget(self.badge)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.tools)
        row.addSpacing(8)
        row.addWidget(BodyLabel("Boyut", self))
        row.addWidget(self.size_slider)
        row.addWidget(self.size_label)
        row.addSpacing(8)
        for b in (self.undo_button, self.redo_button, self.clear_button):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.cancel_button)
        row.addWidget(self.save_button)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 12)
        lay.setSpacing(8)
        lay.addLayout(top)
        lay.addLayout(row)

    def set_tool(self, tool: str) -> None:
        """Seçili aracı ayarlar (arayüz + sinyal)."""
        self.tools.setCurrentItem(tool)
        self.toolChanged.emit(tool)

    def set_brush_range(self, maximum: int, value: int) -> None:
        """Görüntü boyutuna göre fırça aralığını ayarlar."""
        self.size_slider.blockSignals(True)
        self.size_slider.setRange(4, max(32, maximum))
        self.size_slider.setValue(min(max(4, value), max(32, maximum)))
        self.size_slider.blockSignals(False)
        self.size_label.setText(f"{self.size_slider.value()} px")

    def _on_size(self, value: int) -> None:
        self.size_label.setText(f"{value} px")
        self.sizeChanged.emit(value)

    def update_state(self, can_undo: bool, can_redo: bool, has_content: bool, coverage: float) -> None:
        """Düğme etkinliklerini ve kapsama bilgisini günceller."""
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)
        self.clear_button.setEnabled(has_content)
        self.badge.setText(f"Kapsama: %{coverage * 100:.1f}" if has_content else "Maske boş")
