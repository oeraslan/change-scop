"""Alt panel: hizalama eşleşme yüzdesi (halka gösterge) + özet istatistikler + bölge kartları."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QStackedLayout, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, IconWidget, StrongBodyLabel, isDarkTheme
from qfluentwidgets import FluentIcon as FIF

from app.ui.qt_utils import crop_rgb, rgb_to_pixmap
from app.ui.theme import (
    QUALITY_COLORS,
    QUALITY_LABELS,
    REGION_COLOR,
    EmptyState,
    Pill,
    Surface,
    dim_label,
    match_color,
    palette,
)
from core.contracts import AlignmentQuality, ChangeRegion, ComparisonResult

CARD_W, CARD_H = 248, 150
THUMB_W, THUMB_H = 108, 72
GAUGE_SIZE = 112


def _fmt_int(n: int) -> str:
    """Binlik ayırıcı nokta: 12.345"""
    return f"{n:,}".replace(",", ".")


def changed_area_percent(result: ComparisonResult) -> float:
    """Karşılaştırılabilen (örtüşen) alanın yüzde kaçı değişiklik bölgesi olarak işaretlendi."""
    det = result.detection
    if det is None:
        return 0.0
    valid = int(result.registration.valid_mask.sum())
    if valid <= 0:
        return 0.0
    return min(100.0, 100.0 * sum(r.area_px for r in det.regions) / valid)


class MatchGauge(QWidget):
    """Hizalama eşleşme yüzdesini gösteren halka gösterge."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._percent: float | None = None
        self.setFixedSize(GAUGE_SIZE, GAUGE_SIZE)

    def set_percent(self, percent: float | None) -> None:
        """Yüzde (0..100) ayarlar; None → '—'."""
        self._percent = None if percent is None else max(0.0, min(100.0, percent))
        self.update()

    def percent(self) -> float | None:
        return self._percent

    def text(self) -> str:
        """Ortadaki metin: `%87` ya da `—`."""
        return "—" if self._percent is None else f"%{round(self._percent)}"

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        thickness = 9
        rect = QRectF(self.rect()).adjusted(thickness, thickness, -thickness, -thickness)
        track = QColor(255, 255, 255, 26) if isDarkTheme() else QColor(0, 0, 0, 22)
        pen = QPen(track, thickness)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 0, 360 * 16)
        if self._percent:
            pen.setColor(QColor(match_color(self._percent)))
            p.setPen(pen)
            p.drawArc(rect, 90 * 16, -round(self._percent / 100 * 360 * 16))
        font = QFont(self.font())
        font.setPixelSize(28)
        font.setWeight(QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(palette().text)
        p.drawText(rect.adjusted(0, -8, 0, -8), Qt.AlignmentFlag.AlignCenter, self.text())
        small = QFont(self.font())
        small.setPixelSize(11)
        p.setFont(small)
        p.setPen(palette().text_dim)
        p.drawText(rect.adjusted(0, 34, 0, 0), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                   "eşleşme")


class _Stat(QWidget):
    """Küçük istatistik: değer (kalın) + açıklama (soluk)."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.value_label = StrongBodyLabel("—", self)
        self.caption_label = dim_label(caption, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.value_label)
        lay.addWidget(self.caption_label)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)

    def value(self) -> str:
        return self.value_label.text()


class _HScrollArea(QScrollArea):
    """Yalnızca yatay kayan alan: normal tekerlek ve Shift+tekerlek yatay kaydırır."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        bar = self.horizontalScrollBar()
        if bar.maximum() <= 0:
            event.ignore()
            return
        delta = event.angleDelta()
        d = delta.x() or delta.y()
        bar.setValue(bar.value() - d)
        event.accept()


class RegionCard(Surface):
    """Tek bir bölge için kart: numara, fark skoru, alan, önce/sonra kırpmaları."""

    clicked = Signal(int)

    def __init__(self, region: ChangeRegion, result: ComparisonResult, parent: QWidget | None = None) -> None:
        super().__init__(parent, hoverable=True)
        self.region = region
        self.setFixedSize(CARD_W, CARD_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        number = Pill(f"#{region.region_id}", REGION_COLOR)
        self.score_label = CaptionLabel(f"Skor {region.score:.2f}")
        self.score_label.setTextColor(QColor(0, 0, 0, 140), QColor(255, 255, 255, 150))
        self.area_label = CaptionLabel(f"{_fmt_int(region.area_px)} px")
        self.area_label.setTextColor(QColor(0, 0, 0, 120), QColor(255, 255, 255, 120))

        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(number)
        top.addWidget(self.area_label)
        top.addStretch(1)
        top.addWidget(self.score_label)

        crops = QHBoxLayout()
        crops.setSpacing(8)
        reg = result.registration
        b = region.bbox
        for pixels, caption in ((reg.before.pixels, "Önce"), (reg.after_warped, "Sonra")):
            crops.addWidget(self._crop_widget(pixels, b.x, b.y, b.width, b.height, caption))
        crops.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        lay.addLayout(top)
        lay.addLayout(crops)

    @staticmethod
    def _crop_widget(pixels, x: int, y: int, w: int, h: int, caption: str) -> QWidget:
        label = QLabel()
        label.setFixedSize(THUMB_W, THUMB_H)
        pix = rgb_to_pixmap(crop_rgb(pixels, x, y, w, h)).scaled(
            THUMB_W * 2, THUMB_H * 2, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Ortadan kırp
        left, top = (pix.width() - THUMB_W * 2) // 2, (pix.height() - THUMB_H * 2) // 2
        label.setPixmap(pix.copy(left, top, THUMB_W * 2, THUMB_H * 2).scaled(
            THUMB_W, THUMB_H, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))
        label.setStyleSheet("border-radius: 6px;")
        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.addWidget(label)
        cap = CaptionLabel(caption)
        cap.setTextColor(QColor(0, 0, 0, 120), QColor(255, 255, 255, 130))
        v.addWidget(cap)
        return holder

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit(self.region.region_id)
        super().mouseReleaseEvent(event)


class ResultPanel(Surface):
    """Karşılaştırma sonucu paneli.

    Durumlar: `idle` (sonuç yok), `loading` (hesaplanıyor), `ready` (bölge kartları), `empty`
    (değişiklik yok), `failed` (hizalama başarısız). `idle` dışındaki durumlarda solda eşleşme göstergesi
    ve istatistikler görünür.
    """

    regionActivated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, radius=8)
        self._state = "idle"
        self._cards: dict[int, RegionCard] = {}
        self._result: ComparisonResult | None = None

        # Başlık
        self.title_icon = IconWidget(FIF.ZOOM, self)
        self.title_icon.setFixedSize(18, 18)
        self.title_label = StrongBodyLabel("Karşılaştırma sonucu", self)
        self.quality_pill = Pill("", QUALITY_COLORS[AlignmentQuality.GOOD.value], self)
        self.quality_pill.setVisible(False)
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self.title_icon)
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.quality_pill)

        # Sol: eşleşme göstergesi + istatistikler
        self.gauge = MatchGauge(self)
        self.gauge_caption = dim_label("Hizalama eşleşmesi", self)
        self.gauge_caption.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        gauge_col = QVBoxLayout()
        gauge_col.setSpacing(4)
        gauge_col.addWidget(self.gauge, 0, Qt.AlignmentFlag.AlignHCenter)
        gauge_col.addWidget(self.gauge_caption, 0, Qt.AlignmentFlag.AlignHCenter)

        self.points_stat = _Stat("tutarlı / bulunan eşleşme", self)
        self.changed_stat = _Stat("değişen alan", self)
        self.regions_stat = _Stat("değişiklik bölgesi", self)
        self.overlap_stat = _Stat("örtüşen alan", self)
        self.time_stat = _Stat("süre", self)
        stats = QVBoxLayout()
        stats.setSpacing(6)
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        for row, items in ((row1, (self.points_stat, self.changed_stat)),
                           (row2, (self.regions_stat, self.overlap_stat, self.time_stat))):
            row.setSpacing(20)
            for w in items:
                row.addWidget(w)
            row.addStretch(1)
            stats.addLayout(row)
        stats.addStretch(1)

        self.summary_host = QWidget(self)
        sh = QHBoxLayout(self.summary_host)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(20)
        sh.addLayout(gauge_col)
        sh.addLayout(stats)
        self.summary_host.setFixedWidth(430)

        # Sağ: kartlar ya da mesaj
        self.right = QWidget(self)
        self.right_stack = QStackedLayout(self.right)
        self.right_stack.setContentsMargins(0, 0, 0, 0)

        self.scroll = _HScrollArea(self.right)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.cards_host = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_host)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(12)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.cards_host)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollArea>QWidget>QWidget{background:transparent;}"
            "QScrollBar:horizontal{height:8px;background:transparent;margin:0 2px;}"
            "QScrollBar::handle:horizontal{background:rgba(148,163,184,150);border-radius:4px;min-width:40px;}"
            "QScrollBar::handle:horizontal:hover{background:rgba(148,163,184,220);}"
            "QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{width:0;}"
            "QScrollBar::add-page:horizontal,QScrollBar::sub-page:horizontal{background:transparent;}")
        self.scroll.setFixedHeight(CARD_H + 8 + 12)

        self.message_state = EmptyState(FIF.INFO, "", "", None, self.right, compact=True)
        self.right_stack.addWidget(self.scroll)
        self.right_stack.addWidget(self.message_state)

        body = QHBoxLayout()
        body.setSpacing(24)
        body.addWidget(self.summary_host)
        body.addWidget(self.right, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)
        root.addLayout(header)
        root.addLayout(body)
        root.addStretch(1)

        self.show_idle()

    # -- durumlar ------------------------------------------------------------

    def state(self) -> str:
        """`idle` / `loading` / `ready` / `empty` / `failed`."""
        return self._state

    def _clear_cards(self) -> None:
        self._cards.clear()
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_message(self, title: str, text: str, icon=FIF.INFO) -> None:
        self._clear_cards()
        self.message_state.icon_widget.setIcon(icon)
        self.message_state.set_texts(title, text)
        self.right_stack.setCurrentWidget(self.message_state)

    def show_idle(self) -> None:
        """Henüz sonuç yok."""
        self._result = None
        self._state = "idle"
        self.summary_host.setVisible(False)
        self.quality_pill.setVisible(False)
        self._show_message("Sonuç bekleniyor", "Karşılaştırma bitince eşleşme oranı ve değişen bölgeler burada "
                                               "görünür.", FIF.VIEW)

    def show_loading(self, text: str = "") -> None:
        """Hesaplama sürüyor."""
        self._result = None
        self._state = "loading"
        self.summary_host.setVisible(False)
        self.quality_pill.setVisible(False)
        self._show_message("Karşılaştırılıyor…", text or "Görüntüler eşleştiriliyor ve farklar aranıyor.",
                           FIF.SYNC)

    def set_stage_text(self, text: str) -> None:
        """Yükleme sırasında aşama metni."""
        if self._state == "loading":
            self.message_state.set_texts("Karşılaştırılıyor…", text)

    def _fill_summary(self, result: ComparisonResult) -> None:
        reg = result.registration
        quality = reg.quality.value
        self.quality_pill.set(QUALITY_LABELS.get(quality, quality), QUALITY_COLORS.get(quality))
        self.quality_pill.setVisible(True)
        self.gauge.set_percent(reg.match_percent if reg.num_matches > 0 else None)
        self.points_stat.set_value(f"{_fmt_int(reg.num_inliers)} / {_fmt_int(reg.num_matches)}")
        failed = reg.quality is AlignmentQuality.FAILED or result.detection is None
        if failed:
            for stat in (self.changed_stat, self.regions_stat, self.overlap_stat):
                stat.set_value("—")
        else:
            valid = reg.valid_mask
            overlap = float(valid.mean()) * 100 if valid.size else 100.0
            self.changed_stat.set_value(f"%{changed_area_percent(result):.1f}")
            self.regions_stat.set_value(str(len(result.detection.regions)))  # type: ignore[union-attr]
            self.overlap_stat.set_value(f"%{round(overlap)}")
        self.time_stat.set_value(f"{result.elapsed_s:.1f} sn")
        self.summary_host.setVisible(True)

    def set_result(self, result: ComparisonResult | None) -> None:
        """Sonuca göre paneli doldurur (eşleşme + istatistik + kartlar ya da uygun mesaj)."""
        if result is None:
            self.show_idle()
            return
        self._result = result
        self._fill_summary(result)
        if result.registration.quality is AlignmentQuality.FAILED or result.detection is None:
            self._state = "failed"
            self._show_message("Hizalama başarısız",
                               "Görüntüler arasında yeterli ortak nokta bulunamadı; fark işaretlenmedi.", FIF.CLOSE)
            return
        regions = result.detection.regions
        if not regions:
            self._state = "empty"
            self._show_message("Değişiklik bulunamadı",
                               "Seçilen iki görüntü arasında anlamlı bir fark tespit edilmedi.", FIF.ACCEPT)
            return
        self._clear_cards()
        for region in regions:
            card = RegionCard(region, result, self.cards_host)
            card.clicked.connect(self.regionActivated)
            self._cards[region.region_id] = card
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self.right_stack.setCurrentWidget(self.scroll)
        self._state = "ready"

    def cards(self) -> dict[int, RegionCard]:
        return dict(self._cards)

    def visible_card_ids(self) -> list[int]:
        return [rid for rid, c in self._cards.items() if not c.isHidden()]

    def highlight(self, region_id: int | None) -> None:
        """Kartı vurgular ve görünür alana kaydırır."""
        for rid, card in self._cards.items():
            card.set_selected(rid == region_id)
        card = self._cards.get(region_id) if region_id is not None else None
        if card is not None and not card.isHidden():
            self.scroll.ensureWidgetVisible(card, 24, 0)

    def selected_card_id(self) -> int | None:
        for rid, card in self._cards.items():
            if card._selected:
                return rid
        return None
