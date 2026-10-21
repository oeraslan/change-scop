"""Görsel dil: renkler, yazı tipi, tema uygulama ve paylaşılan küçük bileşenler.

Tek vurgu rengi camgöbeği (#22D3EE); durum renkleri verdict'lere bağlıdır. 8 px grid, 8 px köşe yarıçapı.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIconBase,
    IconWidget,
    PrimaryPushButton,
    StrongBodyLabel,
    Theme,
    isDarkTheme,
    setTheme,
    setThemeColor,
)

from core.contracts import AlignmentQuality, ChangeVerdict, ImageSource

ACCENT = "#22D3EE"
RADIUS = 8
GRID = 8

COLOR_REAL = "#F43F5E"
COLOR_FALSE = "#94A3B8"
COLOR_UNCERTAIN = "#F59E0B"
COLOR_UNREVIEWED = "#60A5FA"

VERDICT_COLORS: dict[ChangeVerdict, str] = {
    ChangeVerdict.REAL_CHANGE: COLOR_REAL,
    ChangeVerdict.FALSE_ALARM: COLOR_FALSE,
    ChangeVerdict.UNCERTAIN: COLOR_UNCERTAIN,
    ChangeVerdict.UNREVIEWED: COLOR_UNREVIEWED,
}
VERDICT_LABELS: dict[ChangeVerdict, str] = {
    ChangeVerdict.REAL_CHANGE: "Gerçek değişiklik",
    ChangeVerdict.FALSE_ALARM: "Yanlış alarm",
    ChangeVerdict.UNCERTAIN: "Belirsiz",
    ChangeVerdict.UNREVIEWED: "İncelenmedi",
}
SOURCE_LABELS: dict[str, str] = {
    ImageSource.DRONE.value: "Drone",
    ImageSource.UAV.value: "İHA",
    ImageSource.SATELLITE.value: "Uydu",
    ImageSource.UNKNOWN.value: "Bilinmiyor",
}
QUALITY_LABELS: dict[str, str] = {
    AlignmentQuality.GOOD.value: "Hizalama iyi",
    AlignmentQuality.WEAK.value: "Hizalama zayıf",
    AlignmentQuality.FAILED.value: "Hizalama başarısız",
}
QUALITY_COLORS: dict[str, str] = {
    AlignmentQuality.GOOD.value: "#34D399",
    AlignmentQuality.WEAK.value: COLOR_UNCERTAIN,
    AlignmentQuality.FAILED.value: COLOR_REAL,
}
FONT_FAMILIES = ["Segoe UI Variable", "Segoe UI", "Inter", "Arial"]


def verdict_color(verdict: ChangeVerdict) -> QColor:
    """Verdict'in durum rengi."""
    return QColor(VERDICT_COLORS[verdict])


def _alpha(color: str | QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


@dataclass(frozen=True)
class Palette:
    """Tema duyarlı yüzey renkleri (özel çizilen bileşenler için)."""

    surface: QColor
    surface_hover: QColor
    border: QColor
    text: QColor
    text_dim: QColor
    canvas: QColor


def palette() -> Palette:
    """Geçerli temaya göre renk paleti."""
    if isDarkTheme():
        return Palette(
            surface=QColor(255, 255, 255, 11), surface_hover=QColor(255, 255, 255, 20),
            border=QColor(255, 255, 255, 22), text=QColor(244, 246, 250),
            text_dim=QColor(255, 255, 255, 140), canvas=QColor(14, 16, 20),
        )
    return Palette(
        surface=QColor(255, 255, 255, 235), surface_hover=QColor(255, 255, 255, 255),
        border=QColor(0, 0, 0, 24), text=QColor(24, 28, 36),
        text_dim=QColor(0, 0, 0, 130), canvas=QColor(232, 235, 240),
    )


def apply_theme(name: str = "dark") -> None:
    """Fluent temasını (`dark` / `light` / `auto`) ve vurgu rengini uygular. Ayarı diske yazmaz."""
    setTheme({"dark": Theme.DARK, "light": Theme.LIGHT}.get(name, Theme.AUTO), save=False)
    setThemeColor(ACCENT, save=False)


def apply_app_font(app: QApplication) -> None:
    """Uygulama yazı tipini Segoe UI Variable (yoksa Segoe UI) yapar."""
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSizeF(9.5)
    app.setFont(font)


# ---------------------------------------------------------------------------
# Paylaşılan bileşenler
# ---------------------------------------------------------------------------

class Surface(QFrame):
    """Yuvarlatılmış, hafif sınırlı kart yüzeyi (tema duyarlı)."""

    def __init__(self, parent: QWidget | None = None, radius: int = RADIUS, hoverable: bool = False) -> None:
        super().__init__(parent)
        self._radius = radius
        self._hoverable = hoverable
        self._hover = False
        self._selected = False
        if hoverable:
            self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def set_selected(self, selected: bool) -> None:
        """Vurgulu (accent kenarlıklı) durum."""
        self._selected = selected
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = palette()
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(pal.surface_hover if (self._hover and self._hoverable) else pal.surface)
        p.setPen(QPen(QColor(ACCENT) if self._selected else pal.border, 1.5 if self._selected else 1))
        p.drawRoundedRect(rect, self._radius, self._radius)


class Pill(QWidget):
    """Küçük renkli rozet (verdict, kaynak etiketi, sayaç)."""

    def __init__(self, text: str = "", color: str | QColor = ACCENT, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._color = QColor(color)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._update_size()

    def set(self, text: str, color: str | QColor | None = None) -> None:
        """Metni (ve rengi) günceller."""
        self._text = text
        if color is not None:
            self._color = QColor(color)
        self._update_size()
        self.update()

    def text(self) -> str:
        return self._text

    def color(self) -> QColor:
        return QColor(self._color)

    def _font(self) -> QFont:
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        f.setWeight(QFont.Weight.DemiBold)
        return f

    def _update_size(self) -> None:
        from PySide6.QtGui import QFontMetrics

        w = QFontMetrics(self._font()).horizontalAdvance(self._text) + 16
        self.setFixedSize(max(w, 22), 20)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_alpha(self._color, 46 if isDarkTheme() else 38))
        p.drawRoundedRect(QRectF(self.rect()), 10, 10)
        text_color = self._color.lighter(125) if isDarkTheme() else self._color.darker(130)
        p.setPen(text_color)
        p.setFont(self._font())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)


class EmptyState(QWidget):
    """Boş durum: ikon + başlık + açıklama + isteğe bağlı aksiyon butonu."""

    def __init__(
        self, icon: FluentIconBase, title: str, text: str = "", button_text: str | None = None,
        parent: QWidget | None = None, compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.icon_widget = IconWidget(icon, self)
        self.icon_widget.setFixedSize(32 if compact else 56, 32 if compact else 56)
        self.title_label = StrongBodyLabel(title, self)
        self.text_label = CaptionLabel(text, self)
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setFixedWidth(480 if compact else 420)
        self.text_label.setTextColor(QColor(0, 0, 0, 140), QColor(255, 255, 255, 150))
        self.button: PrimaryPushButton | None = None

        self.text_label.setVisible(bool(text))
        self._fit_text()
        if button_text:
            self.button = PrimaryPushButton(button_text, self)
        if compact:
            # Yatay: ikon | başlık + açıklama | buton
            self.text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            texts = QVBoxLayout()
            texts.setSpacing(2)
            texts.addWidget(self.title_label)
            texts.addWidget(self.text_label)
            lay = QHBoxLayout(self)
            lay.setContentsMargins(8, 4, 8, 4)
            lay.setSpacing(16)
            lay.addStretch(1)
            lay.addWidget(self.icon_widget)
            lay.addLayout(texts)
            if self.button is not None:
                lay.addWidget(self.button)
            lay.addStretch(1)
        else:
            lay = QVBoxLayout(self)
            lay.setContentsMargins(24, 16, 24, 16)
            lay.setSpacing(8)
            lay.addStretch(1)
            lay.addWidget(self.icon_widget, 0, Qt.AlignmentFlag.AlignHCenter)
            lay.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignHCenter)
            lay.addWidget(self.text_label, 0, Qt.AlignmentFlag.AlignHCenter)
            if self.button is not None:
                lay.addSpacing(8)
                lay.addWidget(self.button, 0, Qt.AlignmentFlag.AlignHCenter)
            lay.addStretch(1)

    def set_texts(self, title: str, text: str = "") -> None:
        """Başlık ve açıklamayı değiştirir."""
        self.title_label.setText(title)
        self.text_label.setText(text)
        self.text_label.setVisible(bool(text))
        self._fit_text()

    def _fit_text(self) -> None:
        """Hizalamalı layout'larda heightForWidth çalışmadığından kaydırılmış metnin yüksekliğini elle hesaplar."""
        from PySide6.QtGui import QFontMetrics

        font = self.text_label.font()
        font.setPixelSize(12)
        rect = QFontMetrics(font).boundingRect(
            0, 0, self.text_label.width(), 2000, int(Qt.TextFlag.TextWordWrap), self.text_label.text())
        self.text_label.setFixedHeight(rect.height() + 6)


def add_shadow(widget: QWidget, blur: int = 32, dy: int = 8, alpha: int = 110) -> QGraphicsDropShadowEffect:
    """Widget'a yumuşak gölge ekler."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, dy)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)
    return effect


def fade(widget: QWidget, show: bool, duration: int = 180) -> QPropertyAnimation:
    """Widget'ı kısa süreli solma animasyonuyla gösterir/gizler.

    `show=True` hemen görünür yapar; `show=False` animasyon bitince `hide()` çağırır.
    """
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
    old = getattr(widget, "_fade_anim", None)
    if old is not None:
        old.stop()
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.setStartValue(effect.opacity() if widget.isVisible() else 0.0)
    anim.setEndValue(1.0 if show else 0.0)
    if show:
        widget.show()
    else:
        anim.finished.connect(widget.hide)
    widget._fade_anim = anim  # type: ignore[attr-defined]
    anim.start()
    return anim


def hbox(*widgets: QWidget, spacing: int = 8, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QHBoxLayout:
    """Kısayol: verilen widget'larla yatay layout."""
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(*margins)
    for w in widgets:
        lay.addWidget(w)
    return lay


def dim_label(text: str, parent: QWidget | None = None) -> CaptionLabel:
    """Soluk renkli küçük açıklama etiketi."""
    label = CaptionLabel(text, parent)
    label.setTextColor(QColor(0, 0, 0, 140), QColor(255, 255, 255, 150))
    return label


def body_label(text: str, parent: QWidget | None = None) -> BodyLabel:
    """Normal gövde etiketi."""
    return BodyLabel(text, parent)
