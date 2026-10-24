"""Karşılaştırma ekranı: yan yana senkron görüntüler, anahtarlar, ilerleme, InfoBar'lar, sonuç paneli."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRectF, QSettings, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
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
from app.ui.exporter import export_comparison
from app.ui.mask_editor import TOOL_BRUSH, MaskEditor, MaskToolbar
from app.ui.qt_utils import heatmap_to_qimage, invalid_area_to_qimage, rgb_to_qimage
from app.ui.result_panel import ResultPanel
from app.ui.sync_graphics_view import RegionSpec, SyncGraphicsView, SyncGroup
from app.ui.theme import ACCENT, EmptyState, Pill, Surface, match_color
from app.ui.workers import ComparisonWorker, MaskWorker, load_preview  # noqa: F401 - load_preview geriye uyumluluk
from core.contracts import AlignmentQuality, ComparisonResult

_RETIRED: set[ComparisonWorker] = set()   # iptal edilmiş ama henüz bitmemiş worker'lar (referans tutulur)


def format_date(iso: str | None) -> str:
    """ISO tarihini `gg.aa.yyyy` biçimine çevirir; yoksa 'Tarih yok'."""
    if not iso:
        return "Tarih yok"
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.%Y")
    except ValueError:
        return iso[:10]


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
    """En önemli ekran: iki görüntü yan yana, fark kutuları, eşleşme oranı ve bölge kartları."""

    backRequested = Signal()
    resultSaved = Signal(int)       # kaydedilen karşılaştırma id'si (geçmiş listesini yenilemek için)
    maskSaved = Signal(object)      # kaydedilen maskeli ImageRecord
    workersIdle = Signal()        # hiçbir arka plan işi kalmadı (ertelenen ayar değişiklikleri için)

    def __init__(self, library: Library | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("compareView")
        self.library = library
        self._before_rec: ImageRecord | None = None
        self._after_rec: ImageRecord | None = None
        self._comparison: ComparisonRecord | None = None
        self._result: ComparisonResult | None = None
        self._worker: ComparisonWorker | None = None
        self._deferred_start: tuple | None = None   # (force,)
        self._infobars: list[InfoBar] = []
        self._sized = False
        self._mask_workers: set[MaskWorker] = set()
        self._mask_saving = False
        self.mask_editor = MaskEditor(self)

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
        self.match_pill = Pill("", ACCENT, self)
        self.match_pill.setToolTip("Hizalama eşleşmesi: bulunan ortak noktaların ne kadarının tutarlı olduğu")
        self.match_pill.setVisible(False)

        self.diff_switch = SwitchButton(self)
        self.diff_switch.setOnText("Farkları göster")
        self.diff_switch.setOffText("Farkları göster")
        self.diff_switch.setChecked(True)
        self.diff_switch.checkedChanged.connect(self._on_diffs_toggled)

        self.heat_switch = SwitchButton(self)
        self.heat_switch.setOnText("Isı haritası")
        self.heat_switch.setOffText("Isı haritası")
        self.heat_switch.checkedChanged.connect(self._on_heat_toggled)

        self.mask_button = PushButton(FIF.BRUSH, "Yok sayma maskesi", self)
        self.mask_button.setToolTip("Karşılaştırmaya katılmayacak alanları boya (bulut, su, park alanı…)")
        self.mask_button.clicked.connect(self.toggle_mask_mode)
        self.mask_button.setEnabled(False)
        self.mask_badge = Pill("Maske aktif", "#A78BFA", self)
        self.mask_badge.setToolTip("Bu 'önce' görüntüsü için kayıtlı bir yok sayma maskesi uygulandı")
        self.mask_badge.setVisible(False)

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
        head.addWidget(self.match_pill)
        head.addWidget(self.mask_badge)
        head.addStretch(1)
        head.addWidget(self.mask_button)
        head.addSpacing(8)
        head.addWidget(self.diff_switch)
        head.addSpacing(8)
        head.addWidget(self.heat_switch)
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

        # --- Görüntüler + sonuç paneli ---
        self.image_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.image_splitter.setChildrenCollapsible(False)
        self.image_splitter.setHandleWidth(10)
        self.image_splitter.addWidget(self.before_pane)
        self.image_splitter.addWidget(self.after_pane)
        self.image_splitter.setSizes([1, 1])

        # --- Maske düzenleme ---
        self.mask_toolbar = MaskToolbar(self)
        self.mask_toolbar.setVisible(False)
        for view in (self.before_view, self.after_view):
            view.attach_mask_editor(self.mask_editor)
        tb = self.mask_toolbar
        tb.toolChanged.connect(self._on_mask_tool)
        tb.sizeChanged.connect(self._on_mask_size)
        tb.undoRequested.connect(self.mask_undo)
        tb.redoRequested.connect(self.mask_redo)
        tb.clearRequested.connect(self.mask_editor.clear)
        tb.cancelRequested.connect(self.cancel_mask_mode)
        tb.saveRequested.connect(self.save_mask_and_recompare)
        self.mask_editor.historyChanged.connect(self._refresh_mask_toolbar)
        for seq, slot in (("Ctrl+Z", self.mask_undo), ("Ctrl+Y", self.mask_redo), ("Ctrl+Shift+Z", self.mask_redo)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        self.result_panel = ResultPanel(self)
        self.result_panel.regionActivated.connect(self._on_card_clicked)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.addWidget(self.image_splitter)
        self.main_splitter.addWidget(self.result_panel)
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
        root.addWidget(self.mask_toolbar)
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
        """İlk gösterimde sonuç paneline makul bir yükseklik verir (sonra kullanıcı sürükler)."""
        total = self.main_splitter.height()
        if total < 300:
            return
        self._sized = True
        panel_h = max(230, min(260, total // 3))
        self.main_splitter.setSizes([total - panel_h, panel_h])

    def set_library(self, library: Library | None) -> None:
        """Kütüphane değiştiğinde (ayarlardan) çağrılır; eski kütüphaneyle çalışan iş bırakılır."""
        self._abort_worker()
        self.library = library

    def is_busy(self) -> bool:
        """Bu ekranın aktif işi var mı (pipeline / kayıt / önbellek okuma)?"""
        return self._deferred_start is not None or (self._worker is not None and self._worker.isRunning())

    def has_running_workers(self) -> bool:
        """İptal edilmiş olanlar dahil arka planda çalışan herhangi bir iş var mı?"""
        return (self.is_busy() or any(w.isRunning() for w in _RETIRED)
                or any(w.isRunning() for w in self._mask_workers))

    def shutdown(self, timeout_ms: int = 10000) -> None:
        """Pencere kapanırken: tüm işleri iptal eder, olay döngüsünü işleterek (UI donmadan) bitmelerini bekler."""
        import time

        self._deferred_start = None
        workers = [w for w in [self._worker, *_RETIRED, *self._mask_workers] if w is not None]
        for w in workers:
            if hasattr(w, "cancel"):
                w.cancel()
        deadline = time.monotonic() + timeout_ms / 1000
        while any(w.isRunning() for w in workers) and time.monotonic() < deadline:
            QApplication.processEvents()
            for w in workers:
                w.wait(20)

    # ------------------------------------------------------------- dışarıdan API

    def open_pair(self, before: ImageRecord, after: ImageRecord, force: bool = False) -> None:
        """İki görüntüyü karşılaştırma ekranında açar.

        Daha önce hesaplanmışsa kayıttan okur (yeniden hesaplamaz); değilse ilerleme çubuğuyla başlatır.
        """
        assert self.library is not None
        self._abort_worker()
        self._leave_mask_mode()
        self._outer.setCurrentWidget(self.body)
        self._before_rec, self._after_rec = before, after
        self.mask_button.setEnabled(False)
        self.mask_badge.setVisible(bool(before.mask_hash))
        self._comparison = None
        self._result = None
        self._set_header(before, after)
        self._clear_infobars()
        self.match_pill.setVisible(False)
        self.result_panel.show_loading()
        self.export_button.setEnabled(False)
        for view in (self.before_view, self.after_view):
            view.set_regions([])
            view.set_heatmap(None)
        # Önbellek araması/okuması, önizleme yükleme, pipeline ve kayıt: hepsi worker thread'inde.
        self._start_worker(force=force)

    def open_comparison(self, rec: ComparisonRecord, force: bool = False) -> None:
        """Geçmişten kaydedilmiş karşılaştırmayı aç (`force=True`: önbelleği yok say, yeniden hesapla)."""
        assert self.library is not None
        before = self.library.repo.get_image(rec.before_image_id)
        after = self.library.repo.get_image(rec.after_image_id)
        if before is None or after is None:
            self._outer.setCurrentWidget(self.body)
            self._info("error", "Görüntü bulunamadı", "Bu karşılaştırmanın görüntülerinden biri silinmiş.")
            return
        self.open_pair(before, after, force=force)

    def show_result(
        self, result: ComparisonResult, title: str | None = None, dates: str | None = None,
        keep_view: bool = False,
    ) -> None:
        """Hazır bir sonucu gösterir (kayıttan açma, demo ve testler için de kullanılır).

        `keep_view=True`: aynı görüntüler yeniden gösteriliyor; zoom/pan korunur.
        """
        self._result = result
        self._outer.setCurrentWidget(self.body)
        if title is not None:
            self.title_label.setText(title)
        if dates is not None:
            self.dates_label.setText(dates)
        reg = result.registration
        self.before_view.set_image(rgb_to_qimage(reg.before.pixels), keep_view=keep_view)
        self.after_view.set_image(rgb_to_qimage(reg.after_warped), keep_view=keep_view)
        failed = reg.quality is AlignmentQuality.FAILED or result.detection is None
        if reg.num_matches > 0:
            pct = reg.match_percent
            self.match_pill.set(f"Eşleşme %{round(pct)}", match_color(pct))
            self.match_pill.setVisible(True)
        else:
            self.match_pill.set("Eşleşme yok", match_color(0))
            self.match_pill.setVisible(True)

        self._clear_infobars()
        if reg.quality is AlignmentQuality.WEAK:
            note = (" " + reg.notes[0]) if reg.notes else ""
            self._info("warning", "Hizalama zayıf",
                       f"Eşleşme %{round(reg.match_percent)} ({reg.num_inliers}/{reg.num_matches} nokta); "
                       f"sonuçlar dikkatle yorumlanmalı.{note}",
                       key="weak")
        elif failed:
            self._info("error", "Hizalama başarısız",
                       "Görüntüler yan yana gösteriliyor ancak fark işaretlenmedi.", key="failed")

        if failed:
            self.before_view.set_regions([])
            self.after_view.set_regions([])
            self.before_view.set_heatmap(None)
            self.after_view.set_heatmap(None)
            self.before_view.set_invalid_overlay(None)
            self.after_view.set_invalid_overlay(None)
        else:
            valid = reg.valid_mask
            overlap = float(valid.mean()) if valid.size else 1.0
            invalid_img = invalid_area_to_qimage(valid)
            self.before_view.set_invalid_overlay(invalid_img)
            self.after_view.set_invalid_overlay(invalid_img)
            if overlap < 0.95:
                self._info("info", "Görüntüler kısmen örtüşüyor",
                           f"Görüntülerin %{round((1 - overlap) * 100)}'i örtüşmüyor; bu alanlarda fark aranmaz.",
                           key="overlap")
            specs = [self._spec(r) for r in result.detection.regions]  # type: ignore[union-attr]
            self.before_view.set_regions(specs)
            self.after_view.set_regions(specs)
            heat = heatmap_to_qimage(result.detection.change_map)  # type: ignore[union-attr]
            self.before_view.set_heatmap(heat)
            self.after_view.set_heatmap(heat)
        if result.detection is not None:
            self._detection_notices(result.detection)
        self.mask_button.setEnabled(self._before_rec is not None and self.library is not None
                                    and not self._mask_saving)
        self.mask_editor.set_canvas(reg.before.pixels.shape[1], reg.before.pixels.shape[0])
        self._leave_mask_mode()
        self.diff_switch.setEnabled(not failed)
        self.heat_switch.setEnabled(not failed)
        self._apply_toggles(animate=False)
        self.result_panel.set_result(result)
        self.export_button.setEnabled(True)
        self.group.highlight(None)
        if not keep_view:
            QTimer.singleShot(0, self.group.fit_all)

    @staticmethod
    def _spec(region) -> RegionSpec:
        b = region.bbox
        return RegionSpec(region.region_id, QRectF(b.x, b.y, b.width, b.height), region.score, region.area_px)

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

    def _detection_notices(self, detection) -> None:
        """Tespit sonucundaki `unreliable` / yöntem düşüşü (fallback) bilgilerini InfoBar olarak gösterir."""
        params = detection.params or {}
        if params.get("unreliable"):
            reason = params.get("unreliable_reason") or "bilinmeyen neden"
            self._info("warning", "Güvenilirlik uyarısı",
                       f"Sonuç güvenilmez olabilir: {reason}", key="unreliable")
        fallback = params.get("fallback_reason")
        # CPU'da varsayılan yöntemde DINOv2'nin kapatılması bilinçli bir hız tercihidir; uyarı değildir.
        if fallback and not str(fallback).startswith("CPU:"):
            self._info("info", "Fark yöntemi değiştirildi",
                       f"İstenen yöntem kullanılamadı ({detection.method or 'ssim'} ile devam edildi): {fallback}",
                       key="fallback")

    # ------------------------------------------------------------- yok sayma maskesi

    def mask_mode_active(self) -> bool:
        """Maske düzenleme modu açık mı?"""
        return self.mask_toolbar.isVisibleTo(self)

    def toggle_mask_mode(self) -> None:
        """Maske modunu aç/kapat (kapatma = değişiklikleri atma)."""
        if self.mask_mode_active():
            self.cancel_mask_mode()
        else:
            self.enter_mask_mode()

    def enter_mask_mode(self) -> None:
        """Maske moduna girer: mevcut maske worker'da okunur, sonra düzenlemeye açılır."""
        if self._result is None or self._before_rec is None or self.library is None or self.is_busy():
            return
        if self.mask_mode_active():
            return
        reg = self._result.registration
        self.mask_editor.set_canvas(reg.before.pixels.shape[1], reg.before.pixels.shape[0])
        longest = max(reg.before.pixels.shape[:2])
        self.mask_toolbar.set_brush_range(max(64, longest // 4), max(8, longest // 40))
        self.mask_editor.brush_px = float(self.mask_toolbar.size_slider.value())
        self.mask_editor.tool = TOOL_BRUSH
        self.mask_toolbar.tools.setCurrentItem(TOOL_BRUSH)
        self.mask_toolbar.setVisible(True)
        self.mask_button.setText("Maske modundan çık")
        for view in (self.before_view, self.after_view):
            view.set_mask_mode(True)
        self._refresh_mask_toolbar()
        w = MaskWorker(self.library, self._before_rec, "load", parent=None)
        w.loaded.connect(self._on_mask_loaded)
        self._track_mask_worker(w)

    def _on_mask_loaded(self, mask, record) -> None:
        if not self.mask_mode_active() or self._before_rec is None or record.id != self._before_rec.id:
            return
        self.mask_editor.load(mask)
        self._refresh_mask_toolbar()

    def _track_mask_worker(self, w: MaskWorker) -> None:
        self._mask_workers.add(w)
        w.finished.connect(lambda w=w: (self._mask_workers.discard(w), w.deleteLater()))
        w.start()

    def _leave_mask_mode(self) -> None:
        self.mask_toolbar.setVisible(False)
        self.mask_button.setText("Yok sayma maskesi")
        for view in (self.before_view, self.after_view):
            view.set_mask_mode(False)

    def cancel_mask_mode(self) -> None:
        """Değişiklikleri atıp maske modundan çıkar."""
        if self._mask_saving:
            return
        self._leave_mask_mode()

    def _on_mask_tool(self, tool: str) -> None:
        self.mask_editor.tool = tool
        for view in (self.before_view, self.after_view):
            view.refresh_mask_tool()

    def _on_mask_size(self, value: int) -> None:
        self.mask_editor.brush_px = float(value)

    def mask_undo(self) -> None:
        if self.mask_mode_active():
            self.mask_editor.undo()

    def mask_redo(self) -> None:
        if self.mask_mode_active():
            self.mask_editor.redo()

    def _refresh_mask_toolbar(self) -> None:
        ed = self.mask_editor
        self.mask_toolbar.update_state(ed.can_undo(), ed.can_redo(), ed.has_content(), ed.coverage())

    def save_mask_and_recompare(self) -> None:
        """Maskeyi (worker'da) kaydeder, sonra aynı çifti maskeyle yeniden karşılaştırır."""
        if not self.mask_mode_active() or self._mask_saving or self._before_rec is None or self.library is None:
            return
        if not self.mask_editor.is_modified():
            self._leave_mask_mode()          # değişiklik yok: yeniden hesaplamaya gerek yok
            return
        self._mask_saving = True
        self.mask_toolbar.save_button.setEnabled(False)
        w = MaskWorker(self.library, self._before_rec, "save", self.mask_editor.mask_bool())
        w.saved.connect(self._on_mask_saved)
        w.failed.connect(self._on_mask_failed)
        self._track_mask_worker(w)

    def _on_mask_saved(self, record) -> None:
        self._mask_saving = False
        self.mask_toolbar.save_button.setEnabled(True)
        if self._before_rec is None or self._after_rec is None:
            return
        self.mask_editor.mark_saved()
        self._leave_mask_mode()
        self.maskSaved.emit(record)
        self.open_pair(record, self._after_rec, force=False)   # maske özeti değişti: önbellek bulunmaz, yeniden hesaplanır

    def _on_mask_failed(self, message: str) -> None:
        self._mask_saving = False
        self.mask_toolbar.save_button.setEnabled(True)
        self._info("error", "Maske kaydedilemedi", message, key="mask-error")

    # ---------------------------------------------------------------- işlem

    def _start_worker(self, force: bool = True) -> None:
        assert self._before_rec and self._after_rec and self.library
        self._set_progress_visible(True)
        self.progress_bar.setValue(0)
        self.export_button.setEnabled(False)
        if any(w.isRunning() for w in _RETIRED):
            # Önceki (iptal edilmiş) iş modelleri bırakana kadar bekle; çakışma olmasın.
            self._deferred_start = (force,)
            self.stage_label.setText("Önceki işlem durduruluyor…")
            return
        self._launch(force)

    def _launch(self, force: bool) -> None:
        assert self._before_rec and self._after_rec and self.library
        worker = ComparisonWorker(self.library, self._before_rec, self._after_rec, force=force)
        worker.previews.connect(self._on_previews)
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._on_success)
        worker.cancelled.connect(self._on_cancelled)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self.stage_label.setText("Başlatılıyor…")
        worker.start()

    def cancel(self) -> None:
        """Çalışan karşılaştırmayı iptal eder; UI beklemeden "İptal edildi" durumuna geçer."""
        if self.is_busy():
            self._abort_worker()
            self._on_cancelled()

    def _abort_worker(self) -> None:
        """Worker'ı bloklamadan bırakır: sinyalleri keser, iptal ister, bitene kadar referansı tutar."""
        self._deferred_start = None
        worker, self._worker = self._worker, None
        if worker is not None:
            for sig in ("previews", "progress", "succeeded", "cancelled", "failed"):
                try:
                    getattr(worker, sig).disconnect()
                except (TypeError, RuntimeError):
                    pass
            worker.cancel()
            if worker.isRunning():
                _RETIRED.add(worker)       # finished -> _on_worker_finished -> deleteLater
            else:
                worker.deleteLater()
        self._set_progress_visible(False)

    def _set_progress_visible(self, visible: bool) -> None:
        self.progress_host.setVisible(visible)
        self.cancel_button.setEnabled(True)

    def _on_previews(self, before_img, after_img) -> None:
        for view, img in ((self.before_view, before_img), (self.after_view, after_img)):
            if img is not None and self._result is None:
                view.set_image(img)
        QTimer.singleShot(0, self.group.fit_all)

    def _on_progress(self, stage: str, fraction: float, message: str) -> None:
        self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self.stage_label.setText(message or stage)
        self.result_panel.set_stage_text(message)

    def _on_success(self, result: ComparisonResult, record, cached: bool, save_error: str) -> None:
        self._set_progress_visible(False)
        self._comparison = record
        keep = self._result is not None and self._result.registration is result.registration
        self.show_result(result, keep_view=keep)
        if save_error:
            self._info("warning", "Sonuç kaydedilemedi", save_error)
        elif record is not None and not cached:
            self.resultSaved.emit(record.id)

    def _on_cancelled(self) -> None:
        self._set_progress_visible(False)
        self._info("info", "İptal edildi", "Karşılaştırma iptal edildi.", duration=4000)
        if self._result is not None:
            self.result_panel.set_result(self._result)
            self.export_button.setEnabled(True)
        else:
            self.result_panel.show_idle()

    def _on_failed(self, message: str) -> None:
        self._set_progress_visible(False)
        self._info("error", "Karşılaştırma başarısız", message, key="error")
        if self._result is not None:
            self.result_panel.set_result(self._result)
        else:
            self.result_panel.show_idle()

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        if worker is not None:
            _RETIRED.discard(worker)        # type: ignore[arg-type]
            worker.deleteLater()
        if self._deferred_start is not None and not any(w.isRunning() for w in _RETIRED):
            args, self._deferred_start = self._deferred_start, None
            self._launch(*args)
        elif not self.has_running_workers():
            self.workersIdle.emit()

    # ----------------------------------------------------------------- anahtarlar

    def _apply_toggles(self, animate: bool) -> None:
        for view in (self.before_view, self.after_view):
            view.set_regions_visible(self.diff_switch.isChecked(), animate)
            view.set_heatmap_visible(self.heat_switch.isChecked(), animate)

    def _on_diffs_toggled(self, on: bool) -> None:
        for view in (self.before_view, self.after_view):
            view.set_regions_visible(on, animate=True)

    def _on_heat_toggled(self, on: bool) -> None:
        for view in (self.before_view, self.after_view):
            view.set_heatmap_visible(on, animate=True)

    # ---------------------------------------------------------- bölge etkileşimi

    def _on_region_clicked_in_view(self, region_id: int) -> None:
        self.group.highlight(region_id)
        self.result_panel.highlight(region_id)

    def _on_card_clicked(self, region_id: int) -> None:
        self.group.highlight(region_id)
        self.result_panel.highlight(region_id)
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
        qs = QSettings("ChangeScope", "ChangeScope")
        folder = str(qs.value("export_dir", "") or "")
        if not folder or not Path(folder).is_dir():
            folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)                 or str(Path.home())
        default = str(Path(folder).resolve() / "karsilastirma.png")
        path, _ = QFileDialog.getSaveFileName(self, "Dışa aktar", default, "PNG (*.png)")
        if path:
            qs.setValue("export_dir", str(Path(path).resolve().parent))
            self.export_to(Path(path))

    def export_to(self, png_path: Path) -> list[Path]:
        """Mevcut sonucu (anahtar durumlarına göre kutulu/kutusuz) dışa aktarır."""
        assert self._result is not None
        written = export_comparison(
            self._result, png_path, show_boxes=self.diff_switch.isChecked(), title=self.title_label.text(),
        )
        self._info("success", "Dışa aktarıldı", ", ".join(p.name for p in written), duration=5000)
        return written

    # --------------------------------------------------------------- test yardımcıları

    def visible_region_count(self) -> tuple[int, int]:
        """(önce, sonra) görünümlerinde görünen kutu sayısı."""
        return len(self.before_view.visible_region_items()), len(self.after_view.visible_region_items())

