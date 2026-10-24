"""Alt durum çubuğu: kütüphane özeti, GPU/CPU bilgisi."""

from __future__ import annotations

from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, IconWidget
from qfluentwidgets import FluentIcon as FIF

from app.ui.theme import palette

class StatusBar(QWidget):
    """İnce, saydam durum çubuğu."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        self.library_icon = IconWidget(FIF.PHOTO, self)
        self.library_icon.setFixedSize(14, 14)
        self.library_label = CaptionLabel("", self)
        self.device_icon = IconWidget(FIF.SPEED_HIGH, self)
        self.device_icon.setFixedSize(14, 14)
        self.device_label = CaptionLabel("Aygıt: denetleniyor…", self)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(8)
        lay.addWidget(self.library_icon)
        lay.addWidget(self.library_label)
        lay.addStretch(1)
        lay.addWidget(self.device_icon)
        lay.addWidget(self.device_label)

    def set_device(self, text: str) -> None:
        self.device_label.setText(text)

    def set_library_info(self, text: str) -> None:
        self.library_label.setText(text)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setPen(palette().border)
        p.drawLine(0, 0, self.width(), 0)
