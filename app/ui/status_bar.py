"""Alt durum çubuğu: Gemma durumu, kütüphane özeti, GPU/CPU bilgisi."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, IconWidget
from qfluentwidgets import FluentIcon as FIF

from app.ui.theme import palette

LLM_STATES = {
    "off": ("kapalı", "#94A3B8"),
    "loading": ("yükleniyor", "#F59E0B"),
    "ready": ("hazır", "#34D399"),
}


class _Dot(QWidget):
    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(10, 10)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(QRectF(1, 1, 8, 8))


class StatusBar(QWidget):
    """İnce, saydam durum çubuğu."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        self._dot = _Dot("#94A3B8", self)
        self.llm_label = CaptionLabel("Gemma: kapalı", self)
        self.library_label = CaptionLabel("", self)
        self.device_icon = IconWidget(FIF.SPEED_HIGH, self)
        self.device_icon.setFixedSize(14, 14)
        self.device_label = CaptionLabel("Aygıt: denetleniyor…", self)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(8)
        lay.addWidget(self._dot)
        lay.addWidget(self.llm_label)
        lay.addSpacing(16)
        lay.addWidget(self.library_label)
        lay.addStretch(1)
        lay.addWidget(self.device_icon)
        lay.addWidget(self.device_label)

    def set_llm_state(self, state: str, detail: str = "") -> None:
        """`off` / `loading` / `ready` durumunu gösterir."""
        text, color = LLM_STATES[state]
        self._dot.set_color(color)
        self.llm_label.setText(f"Gemma: {text}" + (f" ({detail})" if detail else ""))

    def set_device(self, text: str) -> None:
        self.device_label.setText(text)

    def set_library_info(self, text: str) -> None:
        self.library_label.setText(text)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setPen(palette().border)
        p.drawLine(0, 0, self.width(), 0)
