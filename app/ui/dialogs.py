"""Küçük Fluent diyalogları: ad girme, içe aktarma seçenekleri, tarih seçme."""

from __future__ import annotations

from datetime import date, datetime

from PySide6.QtCore import QDate
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    BodyLabel,
    CalendarPicker,
    ComboBox,
    LineEdit,
    MessageBoxBase,
    SubtitleLabel,
)

from core.contracts import ImageSource

SOURCE_CHOICES: list[tuple[str, ImageSource]] = [
    ("Drone", ImageSource.DRONE),
    ("İHA", ImageSource.UAV),
    ("Uydu", ImageSource.SATELLITE),
    ("Bilinmiyor", ImageSource.UNKNOWN),
]


def _localize(box: MessageBoxBase, yes: str, no: str = "Vazgeç") -> None:
    box.yesButton.setText(yes)
    box.cancelButton.setText(no)
    box.widget.setMinimumWidth(420)


class NameDialog(MessageBoxBase):
    """Klasör adı girdirir."""

    def __init__(self, title: str, initial: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.titleLabel = SubtitleLabel(title, self)
        self.edit = LineEdit(self)
        self.edit.setText(initial)
        self.edit.setPlaceholderText("Klasör adı")
        self.edit.setClearButtonEnabled(True)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.edit)
        _localize(self, "Tamam")
        self.edit.returnPressed.connect(self.yesButton.click)
        self.edit.selectAll()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.edit.setFocus()

    def value(self) -> str:
        return self.edit.text().strip()


class ImportDialog(MessageBoxBase):
    """İçe aktarma seçenekleri: kaynak etiketi ve (EXIF yoksa) çekim tarihi."""

    def __init__(self, count: int, missing_exif: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.titleLabel = SubtitleLabel(f"{count} görüntü içe aktarılacak", self)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(BodyLabel("Kaynak türü", self))
        self.source_combo = ComboBox(self)
        for text, _ in SOURCE_CHOICES:
            self.source_combo.addItem(text)
        self.viewLayout.addWidget(self.source_combo)

        self.date_combo: ComboBox | None = None
        self.calendar: CalendarPicker | None = None
        if missing_exif:
            self.viewLayout.addSpacing(8)
            self.viewLayout.addWidget(
                BodyLabel(f"{missing_exif} görüntüde EXIF çekim tarihi yok. Hangi tarih kullanılsın?", self))
            self.date_combo = ComboBox(self)
            self.date_combo.addItems(["Dosya tarihini kullan", "Tarihi ben seçeceğim"])
            self.calendar = CalendarPicker(self)
            self.calendar.setDate(QDate.currentDate())
            self.calendar.setEnabled(False)
            self.date_combo.currentIndexChanged.connect(lambda i: self.calendar.setEnabled(i == 1))
            self.viewLayout.addWidget(self.date_combo)
            self.viewLayout.addWidget(self.calendar)
        _localize(self, "İçe aktar")

    def source(self) -> ImageSource:
        return SOURCE_CHOICES[self.source_combo.currentIndex()][1]

    def fallback_date(self) -> datetime | None:
        """Kullanıcının seçtiği tarih; None ise dosya tarihi kullanılır."""
        if self.date_combo is not None and self.date_combo.currentIndex() == 1 and self.calendar:
            d = self.calendar.getDate()
            return datetime(d.year(), d.month(), d.day(), 12, 0, 0)
        return None


class DateDialog(MessageBoxBase):
    """Tek bir görüntünün çekim tarihini düzenler."""

    def __init__(self, current_iso: str | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.titleLabel = SubtitleLabel("Çekim tarihini düzenle", self)
        self.calendar = CalendarPicker(self)
        try:
            d = datetime.fromisoformat(current_iso).date() if current_iso else date.today()
        except ValueError:
            d = date.today()
        self.calendar.setDate(QDate(d.year, d.month, d.day))
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.calendar)
        _localize(self, "Kaydet")

    def value(self) -> datetime:
        d = self.calendar.getDate()
        return datetime(d.year(), d.month(), d.day(), 12, 0, 0)
