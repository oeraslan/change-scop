"""Alt panel: yapay zekâ genel özeti + yatay kaydırılabilir bölge kartları."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QStackedLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    IconWidget,
    PushButton,
    SingleDirectionScrollArea,
    StrongBodyLabel,
)
from qfluentwidgets import FluentIcon as FIF

from app.ui.qt_utils import crop_rgb, rgb_to_pixmap
from app.ui.theme import (
    ACCENT,
    VERDICT_COLORS,
    VERDICT_LABELS,
    EmptyState,
    Pill,
    Surface,
    dim_label,
    palette,
)
from core.contracts import AlignmentQuality, ChangeRegion, ChangeVerdict, ComparisonResult

LOADING_TEXT = "Yapay zekâ yorumlanıyor…"
MODEL_LOADING_HINT = " İlk seferde ~2 dk sürebilir."
CARD_W, CARD_H = 248, 198
THUMB_W, THUMB_H = 108, 72


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


class SkeletonBar(QWidget):
    """Yükleme sırasında parlayan (shimmer) iskelet çubuğu."""

    def __init__(self, width: int | None = None, height: int = 12, radius: int = 6, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._radius = radius
        self._phase = 0.0
        self.setFixedHeight(height)
        if width:
            self.setFixedWidth(width)

    def set_phase(self, phase: float) -> None:
        self._phase = phase
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        base = QColor(255, 255, 255, 18) if palette().text.lightness() > 128 else QColor(0, 0, 0, 14)
        shine = QColor(255, 255, 255, 46) if palette().text.lightness() > 128 else QColor(255, 255, 255, 160)
        w = self.width()
        grad = QLinearGradient((self._phase * 2 - 0.5) * w, 0, (self._phase * 2 + 0.1) * w, 0)
        grad.setColorAt(0.0, base)
        grad.setColorAt(0.5, shine)
        grad.setColorAt(1.0, base)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(base)
        p.drawRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(self.rect()), self._radius, self._radius)


class _SkeletonCard(Surface):
    """Yükleme sırasında bölge kartı yer tutucusu."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(CARD_W, CARD_H)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        self.bars = [SkeletonBar(60, 16, 8), SkeletonBar(None, 10), SkeletonBar(150, 10)]
        for b in self.bars:
            lay.addWidget(b)
        lay.addStretch(1)
        row = QHBoxLayout()
        for _ in range(2):
            bar = SkeletonBar(THUMB_W, THUMB_H, 6)
            self.bars.append(bar)
            row.addWidget(bar)
        lay.addLayout(row)


class RegionCard(Surface):
    """Tek bir bölge için kart: numara, verdict rozeti, açıklama, güven yüzdesi, önce/sonra kırpmaları."""

    clicked = Signal(int)

    def __init__(self, region: ChangeRegion, result: ComparisonResult, parent: QWidget | None = None,
                 pending: bool = False) -> None:
        super().__init__(parent, hoverable=True)
        self.region = region
        self.setFixedSize(CARD_W, CARD_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        color = VERDICT_COLORS[region.verdict]

        number = Pill(f"#{region.region_id}", color)
        verdict = Pill(VERDICT_LABELS[region.verdict], color)
        if region.llm_confidence is not None:
            right = CaptionLabel(f"Güven %{round(region.llm_confidence * 100)}")
        else:
            right = CaptionLabel(f"Skor {region.score:.2f}")
        right.setTextColor(QColor(0, 0, 0, 140), QColor(255, 255, 255, 150))
        self.confidence_label = right

        desc_text = region.description or (
            "Yorum bekleniyor…" if pending else "Açıklama için “Yorumla” düğmesini kullanın.")
        desc = CaptionLabel(desc_text)
        desc.setWordWrap(True)
        desc.setFixedHeight(36)
        desc.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        if region.description is None:
            desc.setTextColor(QColor(0, 0, 0, 120), QColor(255, 255, 255, 120))
        self.description_label = desc

        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(number)
        top.addWidget(verdict)
        top.addStretch(1)
        top.addWidget(right)

        crops = QHBoxLayout()
        crops.setSpacing(8)
        reg = result.registration
        b = region.bbox
        for pixels, caption in ((reg.before.pixels, "Önce"), (reg.after_warped, "Sonra")):
            crops.addWidget(self._crop_widget(pixels, b.x, b.y, b.width, b.height, caption))
        crops.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        lay.addLayout(top)
        lay.addWidget(desc)
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


class AiPanel(Surface):
    """Yapay zekâ yorumu paneli.

    Durumlar: `off` (LLM kapalı → boş durum + "Yorumla"), `loading` (shimmer), `ready` (özet + kartlar),
    `empty` (değişiklik yok), `failed` (hizalama başarısız).
    """

    regionActivated = Signal(int)
    interpretRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, radius=8)
        self._state = "off"
        self._cards: dict[int, RegionCard] = {}
        self._hide_false = False
        self._result: ComparisonResult | None = None

        # Başlık
        self.title_icon = IconWidget(FIF.ROBOT, self)
        self.title_icon.setFixedSize(18, 18)
        self.title_label = StrongBodyLabel("Yapay zekâ yorumu", self)
        self.model_pill = Pill("Gemma 4 12B", ACCENT, self)
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self.title_icon)
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.model_pill)

        # Üst alan: durum sayfaları
        self.top = QWidget(self)
        self.top_stack = QStackedLayout(self.top)
        self.top_stack.setContentsMargins(0, 0, 0, 0)

        self.summary_label = BodyLabel("", self.top)
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        summary_page = QWidget(self.top)
        sl = QVBoxLayout(summary_page)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        self.summary_caption = dim_label("Genel özet", summary_page)
        sl.addWidget(self.summary_caption)
        sl.addWidget(self.summary_label)
        sl.addStretch(1)

        self.off_state = EmptyState(
            FIF.ROBOT, "Yapay zekâ yorumu kapalı",
            "Bölgeler yalnızca görüntü analiziyle bulundu. Gemma ile açıklama ve yanlış alarm ayıklaması isteyebilirsiniz.",
            "Yorumla", self.top, compact=True,
        )
        self.off_state.button.setIcon(FIF.SEND)  # type: ignore[union-attr]
        self.off_state.button.clicked.connect(self.interpretRequested)  # type: ignore[union-attr]
        self._off_default = (self.off_state.title_label.text(), self.off_state.text_label.text())
        self._interpret_enabled = True

        self.message_state = EmptyState(FIF.INFO, "", "", None, self.top, compact=True)

        loading_page = QWidget(self.top)
        ll = QVBoxLayout(loading_page)
        ll.setContentsMargins(0, 4, 0, 0)
        ll.setSpacing(8)
        self._top_bars = [SkeletonBar(None, 12), SkeletonBar(None, 12), SkeletonBar(360, 12)]
        for bar in self._top_bars:
            ll.addWidget(bar)
        self.stage_label = CaptionLabel(LOADING_TEXT, loading_page)
        self.stage_label.setTextColor(QColor(0, 0, 0, 140), QColor(255, 255, 255, 150))
        ll.addWidget(self.stage_label)
        ll.addStretch(1)

        self.page_summary, self.page_off, self.page_message, self.page_loading = (
            summary_page, self.off_state, self.message_state, loading_page)
        for page in (summary_page, self.off_state, self.message_state, loading_page):
            self.top_stack.addWidget(page)

        # Kartlar
        self.scroll = _HScrollArea(self)
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

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)
        root.addLayout(header)
        root.addWidget(self.top)
        root.addWidget(self.scroll)
        root.addStretch(1)

        # Shimmer
        self._shimmer = QVariantAnimation(self)
        self._shimmer.setStartValue(0.0)
        self._shimmer.setEndValue(1.0)
        self._shimmer.setDuration(1300)
        self._shimmer.setLoopCount(-1)
        self._shimmer.setEasingCurve(QEasingCurve.Type.Linear)
        self._shimmer.valueChanged.connect(self._on_shimmer)
        self._skeleton_cards: list[_SkeletonCard] = []

        self.show_off()

    # -- durumlar ------------------------------------------------------------

    def state(self) -> str:
        """`off` / `loading` / `ready` / `empty` / `failed`."""
        return self._state

    def set_model_name(self, name: str) -> None:
        self.model_pill.set(name)

    def interpret_enabled(self) -> bool:
        """"Yorumla" kullanılabilir mi (LLM açık ve model kurulu)?"""
        return self._interpret_enabled

    def set_interpret_available(self, available: bool, reason: str = "", title: str = "") -> None:
        """"Yorumla" düğmesini etkinleştirir/devre dışı bırakır; devre dışıyken nedenini açıklar."""
        self._interpret_enabled = available
        button = self.off_state.button
        if button is not None:
            button.setEnabled(available)
            button.setToolTip("" if available else reason)
        if available:
            self.off_state.set_texts(*self._off_default)
        else:
            self.off_state.set_texts(title or self._off_default[0], reason)

    def _on_shimmer(self, value: float) -> None:
        for bar in self._top_bars:
            bar.set_phase(value)
        for card in self._skeleton_cards:
            for bar in card.bars:
                bar.set_phase(value)

    def _clear_cards(self) -> None:
        self._cards.clear()
        self._skeleton_cards.clear()
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _set_state(self, state: str) -> None:
        self._state = state
        if state != "loading":
            self._shimmer.stop()

    def show_off(self) -> None:
        """LLM kapalı boş durumu (kartlar yoksa)."""
        self._clear_cards()
        self._set_state("off")
        self._show_top(self.page_off)
        self.scroll.setVisible(False)

    def _show_top(self, page: QWidget) -> None:
        """Üst sayfayı gösterir ve yüksekliğini içeriğe göre ayarlar."""
        heights = {id(self.page_summary): 84, id(self.page_off): 96, id(self.page_message): 96,
                   id(self.page_loading): 84}
        self.top_stack.setCurrentWidget(page)
        self.top.setFixedHeight(heights[id(page)])

    def set_stage_text(self, text: str) -> None:
        """Yükleme sırasında aşama metni ("Bölge 3/8 yorumlandı"); model yüklenirken süre ipucu eklenir."""
        text = text or LOADING_TEXT
        if "yükleniyor" in text and MODEL_LOADING_HINT.strip() not in text:
            text += MODEL_LOADING_HINT
        self.stage_label.setText(text)

    def is_interpreting(self) -> bool:
        """Kartlar ekrandayken AI yorumu sürüyor mu?"""
        return self._state == "loading" and bool(self._cards)

    def show_loading(self, text: str = "") -> None:
        """Skeleton/shimmer yükleme animasyonu."""
        self._clear_cards()
        self._set_state("loading")
        self.set_stage_text(text)
        self._show_top(self.page_loading)
        for _ in range(3):
            card = _SkeletonCard(self.cards_host)
            self._skeleton_cards.append(card)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self.scroll.setVisible(True)
        self._shimmer.start()

    def _show_message(self, state: str, title: str, text: str, icon=FIF.INFO) -> None:
        self._clear_cards()
        self._set_state(state)
        self.message_state.icon_widget.setIcon(icon)
        self.message_state.set_texts(title, text)
        self._show_top(self.page_message)
        self.scroll.setVisible(False)

    def set_result(self, result: ComparisonResult | None, interpreting: bool = False, text: str = "") -> None:
        """Sonuca göre paneli doldurur (özet + kartlar ya da uygun boş durum).

        `interpreting=True`: CV kartları hemen gösterilir, üst alan skeleton/shimmer + aşama metni olur.
        """
        self._result = result
        if result is None:
            self.show_off()
            return
        if result.registration.quality is AlignmentQuality.FAILED or result.detection is None:
            self._show_message("failed", "Yorum yapılamadı",
                               "Görüntüler hizalanamadığı için bölge bulunamadı.", FIF.CLOSE)
            return
        regions = result.detection.regions
        if not regions:
            self._show_message("empty", "Değişiklik bulunamadı",
                               "Seçilen iki görüntü arasında anlamlı bir fark tespit edilmedi.", FIF.ACCEPT)
            return

        self._clear_cards()
        for region in regions:
            card = RegionCard(region, result, self.cards_host, pending=interpreting)
            card.clicked.connect(self.regionActivated)
            self._cards[region.region_id] = card
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self.scroll.setVisible(True)
        self._apply_card_filter()

        interpreted = result.llm_used or bool(result.summary)
        if interpreting:
            self._set_state("loading")
            self.set_stage_text(text)
            self._show_top(self.page_loading)
            self._shimmer.start()
        elif interpreted:
            self._set_state("ready")
            self.summary_label.setText(result.summary or "Bölgeler yorumlandı; genel özet üretilemedi.")
            self._show_top(self.page_summary)
        else:
            self._set_state("off")
            self._show_top(self.page_off)

    def cards(self) -> dict[int, RegionCard]:
        return dict(self._cards)

    def visible_card_ids(self) -> list[int]:
        return [rid for rid, c in self._cards.items() if not c.isHidden()]

    def set_hide_false_alarms(self, hide: bool) -> None:
        """"Yanlış alarmları gizle": yanlış alarm kartlarını saklar."""
        self._hide_false = hide
        self._apply_card_filter()

    def _apply_card_filter(self) -> None:
        for card in self._cards.values():
            card.setVisible(not (self._hide_false and card.region.verdict is ChangeVerdict.FALSE_ALARM))

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
