"""Karşılaştırma ekranı: yan yana senkron görüntüler, anahtarlar, ilerleme, InfoBar'lar, AI paneli."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QImageReader
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    InfoBar,
    InfoBarPosition,
    ProgressBar,
    PushButton,
    SubtitleLabel,
    SwitchButton,
    ToolButton,
    TransparentToolButton,
)
from qfluentwidgets import FluentIcon as FIF

from app.storage import ComparisonRecord, ImageRecord, Library
from app.ui.ai_panel import AiPanel
from app.ui.exporter import export_comparison
from app.ui.qt_utils import heatmap_to_qimage, rgb_to_qimage
from app.ui.sync_graphics_view import RegionSpec, SyncGraphicsView, SyncGroup
from app.ui.theme import ACCENT, EmptyState, Pill, Surface
from app.ui.workers import ComparisonWorker
from core.contracts import AlignmentQuality, ChangeVerdict, ComparisonResult

PREVIEW_MAX_SIDE = 3000


def format_date(iso: str | None) -> str:
    """ISO tarihini `gg.aa.yyyy` biçimine çevirir; yoksa 'Tarih yok'."""
    if not iso:
        return "Tarih yok"
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.%Y")
    except ValueError:
        return iso[:10]


def load_preview(path: Path, max_side: int = PREVIEW_MAX_SIDE) -> QImage | None:
    """Büyük dosyaları ölçekleyerek okuyan hızlı önizleme yükleyici."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(False)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_side:
        k = max_side / max(size.width(), size.height())
        reader.setScaledSize(size * k)
    img = reader.read()
    return None if img.isNull() else img


class _Pane(Surface):
    """Bir görüntü görünümü + köşe etiketi (ÖNCE / SONRA)."""

    def __init__(self, group: SyncGroup, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = group.create_view(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.view)
        self.chip = Pill(label, ACCENT, self)
        self.chip.move(16, 16)
        self.chip.raise_()

    def set_label(self, text: str) -> None:
        self.chip.set(text)
        self.chip.raise_()


class CompareView(QWidget):
    """En önemli ekran: iki görüntü yan yana, fark kutuları, AI yorumu."""

    backRequested = Signal()
    resultSaved = Signal(int)       # kaydedilen karşılaştırma id'si (geçmiş listesini yenilemek için)

    def __init__(self, library: Library | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("compareView")
        self.library = library
        self.use_llm = False
        self._before_rec: ImageRecord | None = None
        self._after_rec: ImageRecord | None = None
        self._comparison: ComparisonRecord | None = None
        self._result: ComparisonResult | None = None
        self._worker: ComparisonWorker | None = None
        self._infobars: list[InfoBar] = []
        self._model_name = "Gemma 4 12B"
        self._sized = False

        self.group = SyncGroup(self)
        self.before_pane = _Pane(self.group, "ÖNCE", self)
        self.after_pane = _Pane(self.group, "SONRA", self)
        self.before_view: SyncGraphicsView = self.before_pane.view
        self.after_view: SyncGraphicsView = self.after_pane.view
        self.group.regionClicked.connect(self._on_region_clicked_in_view)

        # --- Üst çubuk ---
        self.back_button = TransparentToolButton(FIF.RETURN, self)
        self.back_button.setToolTip("Galeriye dön")
        self.back_button.clicked.connect(self.backRequested)
        self.title_label = SubtitleLabel("Karşılaştırma", self)
        self.dates_label = BodyLabel("", self)
        self.dates_label.setTextColor("#6b7280", "#9aa3b2")

        self.diff_switch = SwitchButton(self)
        self.diff_switch.setOnText("Farkları göster")
        self.diff_switch.setOffText("Farkları göster")
        self.diff_switch.setChecked(True)
        self.diff_switch.checkedChanged.connect(self._on_diffs_toggled)

        self.heat_switch = SwitchButton(self)
        self.heat_switch.setOnText("Isı haritası")
        self.heat_switch.setOffText("Isı haritası")
        self.heat_switch.checkedChanged.connect(self._on_heat_toggled)

        self.hide_false_check = CheckBox("Yanlış alarmları gizle", self)
        self.hide_false_check.toggled.connect(self._on_hide_false_toggled)

        self.fit_button = ToolButton(FIF.FIT_PAGE, self)
        self.fit_button.setToolTip("Görüntüleri sığdır (çift tık)")
        self.fit_button.clicked.connect(self.group.fit_all)
        self.export_button = ToolButton(FIF.DOWNLOAD, self)
        self.export_button.setToolTip("Dışa aktar (PNG + CSV + özet)")
        self.export_button.clicked.connect(self.export_dialog)
        self.export_button.setEnabled(False)

        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(self.back_button)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(self.title_label)
        titles.addWidget(self.dates_label)
        head.addLayout(titles)
        head.addStretch(1)
        head.addWidget(self.diff_switch)
        head.addSpacing(8)
        head.addWidget(self.heat_switch)
        head.addSpacing(8)
        head.addWidget(self.hide_false_check)
        head.addSpacing(8)
        head.addWidget(self.fit_button)
        head.addWidget(self.export_button)

        # --- İlerleme ---
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setFixedHeight(4)
        self.stage_label = CaptionLabel("", self)
        self.cancel_button = PushButton(FIF.CLOSE, "İptal", self)
        self.cancel_button.clicked.connect(self.cancel)
        prow = QHBoxLayout()
        prow.setContentsMargins(0, 0, 0, 0)
        prow.setSpacing(12)
        prow.addWidget(self.stage_label)
        prow.addStretch(1)
        prow.addWidget(self.cancel_button)
        self.progress_host = QWidget(self)
        pv = QVBoxLayout(self.progress_host)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)
        pv.addWidget(self.progress_bar)
        pv.addLayout(prow)
        self.progress_host.setVisible(False)

        # --- InfoBar alanı ---
        self.infobar_host = QWidget(self)
        self.infobar_layout = QVBoxLayout(self.infobar_host)
        self.infobar_layout.setContentsMargins(0, 0, 0, 0)
        self.infobar_layout.setSpacing(8)

        # --- Görüntüler + AI paneli ---
        self.image_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.image_splitter.setChildrenCollapsible(False)
        self.image_splitter.setHandleWidth(10)
        self.image_splitter.addWidget(self.before_pane)
        self.image_splitter.addWidget(self.after_pane)
        self.image_splitter.setSizes([1, 1])

        self.ai_panel = AiPanel(self)
        self.ai_panel.set_model_name(self._model_name)
        self.ai_panel.regionActivated.connect(self._on_card_clicked)
        self.ai_panel.interpretRequested.connect(self.interpret)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.addWidget(self.image_splitter)
        self.main_splitter.addWidget(self.ai_panel)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, True)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        for sp in (self.image_splitter, self.main_splitter):
            sp.setStyleSheet("QSplitter::handle{background:transparent;}")

        self.body = QWidget(self)
        root = QVBoxLayout(self.body)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(10)
        root.addLayout(head)
        root.addWidget(self.progress_host)
        root.addWidget(self.infobar_host)
        root.addWidget(self.main_splitter, 1)

        self.empty_state = EmptyState(
            FIF.VIEW, "Henüz bir karşılaştırma açık değil",
            "Kütüphanede iki görüntü seç ve Karşılaştır'a bas; sonuç burada yan yana görünür.",
            "Kütüphaneye dön", self)
        self.empty_state.button.clicked.connect(self.backRequested)  # type: ignore[union-attr]
        self._outer = QStackedLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.addWidget(self.empty_state)
        self._outer.addWidget(self.body)
        self._outer.setCurrentWidget(self.empty_state)

    def has_content(self) -> bool:
        """Bir karşılaştırma açık mı (boş durum değil)?"""
        return self._outer.currentWidget() is self.body

    # ------------------------------------------------------------------ yaşam

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._sized:
            QTimer.singleShot(0, self._init_sizes)
        QTimer.singleShot(0, self.group.fit_all)

    def _init_sizes(self) -> None:
        """İlk gösterimde AI paneline makul bir yükseklik verir (sonra kullanıcı sürükler)."""
        total = self.main_splitter.height()
        if total < 300:
            return
        self._sized = True
        ai_h = max(280, min(340, total // 3))
        self.main_splitter.setSizes([total - ai_h, ai_h])

    def set_library(self, library: Library | None) -> None:
        """Kütüphane değiştiğinde (ayarlardan) çağrılır."""
        self.library = library

    def set_model_name(self, name: str) -> None:
        self._model_name = name
        self.ai_panel.set_model_name(name)

    def is_busy(self) -> bool:
        """Pipeline şu an çalışıyor mu?"""
        return self._worker is not None and self._worker.isRunning()

    def shutdown(self) -> None:
        """Pencere kapanırken çalışan işi iptal edip bekler."""
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(5000)

    # ------------------------------------------------------------- dışarıdan API

    def open_pair(
        self, before: ImageRecord, after: ImageRecord, force: bool = False, use_llm: bool | None = None
    ) -> None:
        """İki görüntüyü karşılaştırma ekranında açar.

        Daha önce hesaplanmışsa kayıttan okur (yeniden hesaplamaz); değilse ilerleme çubuğuyla başlatır.
        """
        assert self.library is not None
        self._abort_worker()
        self._outer.setCurrentWidget(self.body)
        self._before_rec, self._after_rec = before, after
        self._comparison = None
        self._result = None
        self._set_header(before, after)
        self._clear_infobars()
        self.ai_panel.show_off()
        self.export_button.setEnabled(False)

        cached = None if force else self.library.find_comparison(before, after)
        if cached is not None:
            try:
                result = self.library.load_comparison(cached)
            except Exception as exc:  # noqa: BLE001 - bozuk kayıt: yeniden hesapla
                self._comparison = None
                self._info("warning", "Kayıtlı sonuç okunamadı", f"Yeniden hesaplanacak ({exc}).")
            else:
                self._comparison = cached
                self.show_result(result)
                return
        self._show_previews(before, after)
        self._start_worker(self.use_llm if use_llm is None else use_llm, loading_ai=False)

    def open_comparison(self, rec: ComparisonRecord) -> None:
        """Geçmişten kaydedilmiş karşılaştırmayı aç."""
        assert self.library is not None
        before = self.library.repo.get_image(rec.before_image_id)
        after = self.library.repo.get_image(rec.after_image_id)
        if before is None or after is None:
            self._info("error", "Görüntü bulunamadı", "Bu karşılaştırmanın görüntülerinden biri silinmiş.")
            return
        self.open_pair(before, after)

    def show_result(self, result: ComparisonResult, title: str | None = None, dates: str | None = None) -> None:
        """Hazır bir sonucu gösterir (kayıttan açma, demo ve testler için de kullanılır)."""
        self._result = result
        self._outer.setCurrentWidget(self.body)
        if title is not None:
            self.title_label.setText(title)
        if dates is not None:
            self.dates_label.setText(dates)
        reg = result.registration
        self.before_view.set_image(rgb_to_qimage(reg.before.pixels))
        self.after_view.set_image(rgb_to_qimage(reg.after_warped))
        failed = reg.quality is AlignmentQuality.FAILED or result.detection is None

        self._clear_infobars()
        if reg.quality is AlignmentQuality.WEAK:
            note = (" " + reg.notes[0]) if reg.notes else ""
            self._info("warning", "Hizalama zayıf",
                       f"{reg.num_inliers}/{reg.num_matches} eşleşme kullanıldı; sonuçlar dikkatle yorumlanmalı.{note}",
                       key="weak")
        elif failed:
            self._info("error", "Hizalama başarısız",
                       "Görüntüler yan yana gösteriliyor ancak fark işaretlenmedi.", key="failed")

        if failed:
            self.before_view.set_regions([])
            self.after_view.set_regions([])
            self.before_view.set_heatmap(None)
            self.after_view.set_heatmap(None)
        else:
            specs = [self._spec(r) for r in result.detection.regions]  # type: ignore[union-attr]
            self.before_view.set_regions(specs)
            self.after_view.set_regions(specs)
            heat = heatmap_to_qimage(result.detection.change_map)  # type: ignore[union-attr]
            self.before_view.set_heatmap(heat)
            self.after_view.set_heatmap(heat)
        self.diff_switch.setEnabled(not failed)
        self.heat_switch.setEnabled(not failed)
        self.hide_false_check.setEnabled(not failed)
        self._apply_toggles(animate=False)
        self.ai_panel.set_hide_false_alarms(self.hide_false_check.isChecked())
        self.ai_panel.set_result(result)
        self.export_button.setEnabled(True)
        self.group.highlight(None)
        QTimer.singleShot(0, self.group.fit_all)

    @staticmethod
    def _spec(region) -> RegionSpec:
        b = region.bbox
        return RegionSpec(
            region.region_id, QRectF(b.x, b.y, b.width, b.height), region.verdict, region.score,
            region.description,
        )

    # --------------------------------------------------------------- başlık/önizleme

    def _set_header(self, before: ImageRecord, after: ImageRecord) -> None:
        name = "Karşılaştırma"
        if self.library is not None:
            chain = self.library.repo.folder_path(before.folder_id)
            if chain:
                name = " / ".join(f.name for f in chain)
        self.title_label.setText(name)
        self.dates_label.setText(f"{format_date(before.captured_at)}  →  {format_date(after.captured_at)}")
        self.before_pane.set_label(f"ÖNCE · {format_date(before.captured_at)}")
        self.after_pane.set_label(f"SONRA · {format_date(after.captured_at)}")

    def _show_previews(self, before: ImageRecord, after: ImageRecord) -> None:
        assert self.library is not None
        for view, rec in ((self.before_view, before), (self.after_view, after)):
            view.set_regions([])
            view.set_heatmap(None)
            view.set_image(load_preview(self.library.image_path(rec)))
        QTimer.singleShot(0, self.group.fit_all)

    # ---------------------------------------------------------------- işlem

    def _start_worker(self, use_llm: bool, loading_ai: bool) -> None:
        assert self._before_rec and self._after_rec and self.library
        worker = ComparisonWorker(
            self.library.to_image_input(self._before_rec),
            self.library.to_image_input(self._after_rec), use_llm, self,
        )
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._on_success)
        worker.cancelled.connect(self._on_cancelled)
        worker.unavailable.connect(self._on_unavailable)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._set_progress_visible(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Başlatılıyor…")
        self.export_button.setEnabled(False)
        if loading_ai:
            self.ai_panel.show_loading()
        worker.start()

    def interpret(self) -> None:
        """"Yorumla": AI yorumunu (LLM açık) sonradan çalıştırır."""
        if self._before_rec is None or self._after_rec is None or self.is_busy():
            return
        self._clear_infobars(keep=("weak", "failed"))
        self._start_worker(True, loading_ai=True)

    def cancel(self) -> None:
        """Çalışan karşılaştırmayı iptal eder."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.stage_label.setText("İptal ediliyor…")
            self.cancel_button.setEnabled(False)

    def _abort_worker(self) -> None:
        if self._worker is not None:
            for sig in ("progress", "succeeded", "cancelled", "unavailable", "failed"):
                try:
                    getattr(self._worker, sig).disconnect()
                except (TypeError, RuntimeError):
                    pass
            self._worker.cancel()
            self._worker.wait(5000)
            self._worker = None
        self._set_progress_visible(False)

    def _set_progress_visible(self, visible: bool) -> None:
        self.progress_host.setVisible(visible)
        self.cancel_button.setEnabled(True)

    def _on_progress(self, stage: str, fraction: float, message: str) -> None:
        self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self.stage_label.setText(message or stage)
        if stage.startswith("llm") and self.ai_panel.state() != "loading":
            self.ai_panel.show_loading()

    def _on_success(self, result: ComparisonResult) -> None:
        self._set_progress_visible(False)
        self.show_result(result)
        if self.library and self._before_rec and self._after_rec:
            try:
                self._comparison = self.library.save_comparison(self._before_rec, self._after_rec, result)
                self.resultSaved.emit(self._comparison.id)
            except Exception as exc:  # noqa: BLE001
                self._info("warning", "Sonuç kaydedilemedi", str(exc))

    def _on_cancelled(self) -> None:
        self._set_progress_visible(False)
        self._info("info", "İptal edildi", "Karşılaştırma iptal edildi.", duration=4000)
        if self._result is not None:
            self.ai_panel.set_result(self._result)
            self.export_button.setEnabled(True)
        else:
            self.ai_panel.show_off()

    def _on_unavailable(self, message: str) -> None:
        self._set_progress_visible(False)
        self._info("error", "Analiz modülleri henüz kurulu değil",
                   "Hizalama ve fark tespiti modülleri bu sürümde bağlı değil; görüntüler yalnızca önizleniyor.",
                   key="unavailable")
        self.ai_panel.show_off()

    def _on_failed(self, message: str) -> None:
        self._set_progress_visible(False)
        self._info("error", "Karşılaştırma başarısız", message, key="error")
        if self._result is not None:
            self.ai_panel.set_result(self._result)
        else:
            self.ai_panel.show_off()

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        if worker is not None:
            worker.deleteLater()

    # ----------------------------------------------------------------- anahtarlar

    def _apply_toggles(self, animate: bool) -> None:
        for view in (self.before_view, self.after_view):
            view.set_regions_visible(self.diff_switch.isChecked(), animate)
            view.set_heatmap_visible(self.heat_switch.isChecked(), animate)
            view.set_hidden_verdicts(
                {ChangeVerdict.FALSE_ALARM} if self.hide_false_check.isChecked() else set())

    def _on_diffs_toggled(self, on: bool) -> None:
        for view in (self.before_view, self.after_view):
            view.set_regions_visible(on, animate=True)

    def _on_heat_toggled(self, on: bool) -> None:
        for view in (self.before_view, self.after_view):
            view.set_heatmap_visible(on, animate=True)

    def _on_hide_false_toggled(self, on: bool) -> None:
        for view in (self.before_view, self.after_view):
            view.set_hidden_verdicts({ChangeVerdict.FALSE_ALARM} if on else set())
        self.ai_panel.set_hide_false_alarms(on)

    # ---------------------------------------------------------- bölge etkileşimi

    def _on_region_clicked_in_view(self, region_id: int) -> None:
        self.group.highlight(region_id)
        self.ai_panel.highlight(region_id)

    def _on_card_clicked(self, region_id: int) -> None:
        self.group.highlight(region_id)
        self.ai_panel.highlight(region_id)
        item = self.before_view._region_items.get(region_id)
        if item is not None:
            self.group.zoom_to_region(item.spec.rect)

    # ------------------------------------------------------------- InfoBar'lar

    def _info(self, kind: str, title: str, content: str, key: str | None = None, duration: int = -1) -> InfoBar:
        factory = {"info": InfoBar.info, "warning": InfoBar.warning, "error": InfoBar.error,
                   "success": InfoBar.success}[kind]
        bar = factory(title, content, orient=Qt.Orientation.Horizontal, isClosable=True, duration=duration,
                      position=InfoBarPosition.NONE, parent=self.infobar_host)
        bar.setProperty("kind", kind)
        bar.setProperty("key", key or "")
        self.infobar_layout.addWidget(bar)
        self._infobars.append(bar)
        bar.closedSignal.connect(lambda b=bar: self._forget_infobar(b))
        return bar

    def _forget_infobar(self, bar: InfoBar) -> None:
        if bar in self._infobars:
            self._infobars.remove(bar)

    def _clear_infobars(self, keep: tuple[str, ...] = ()) -> None:
        for bar in list(self._infobars):
            if bar.property("key") in keep:
                continue
            self._infobars.remove(bar)
            self.infobar_layout.removeWidget(bar)
            bar.hide()
            bar.deleteLater()

    def active_infobars(self) -> list[tuple[str, str]]:
        """Görünen InfoBar'lar: (tür, başlık). Testler için."""
        return [(str(b.property("kind")), b.title) for b in self._infobars if not b.isHidden()]

    # ------------------------------------------------------------------ dışa aktarma

    def export_dialog(self) -> None:
        """Dosya seçtirip PNG + CSV + özet yazar."""
        if self._result is None:
            return
        default = "karsilastirma.png"
        path, _ = QFileDialog.getSaveFileName(self, "Dışa aktar", default, "PNG (*.png)")
        if path:
            self.export_to(Path(path))

    def export_to(self, png_path: Path) -> list[Path]:
        """Mevcut sonucu (anahtar durumlarına göre kutulu/kutusuz) dışa aktarır."""
        assert self._result is not None
        written = export_comparison(
            self._result, png_path, show_boxes=self.diff_switch.isChecked(),
            hide_false_alarms=self.hide_false_check.isChecked(), title=self.title_label.text(),
        )
        self._info("success", "Dışa aktarıldı", ", ".join(p.name for p in written), duration=5000)
        return written

    # --------------------------------------------------------------- test yardımcıları

    def visible_region_count(self) -> tuple[int, int]:
        """(önce, sonra) görünümlerinde görünen kutu sayısı."""
        return len(self.before_view.visible_region_items()), len(self.after_view.visible_region_items())

