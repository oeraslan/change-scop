"""Ana pencere: Fluent navigasyon + Kütüphane / Karşılaştırma / Ayarlar sayfaları + durum çubuğu."""

from __future__ import annotations

import copy
import logging
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QVBoxLayout
from qfluentwidgets import FluentWindow, InfoBar, InfoBarPosition, NavigationItemPosition
from qfluentwidgets import FluentIcon as FIF

from app.prefs import UiPrefs
from app.storage import Library
from app.ui.compare_view import CompareView
from app.ui.library_view import LibraryView
from app.ui.settings_view import SettingsView
from app.ui.status_bar import StatusBar
from app.ui.theme import apply_theme
from app.ui.workers import DeviceProbe
from core import pipeline as core_pipeline
from core.config import AppConfig, LLM_MODELS

GEMMA_DOWNLOAD_HINT = "Gemma modeli indirilmemiş: scripts/download_models.py --only gemma-12b"
RECONFIGURE_DEBOUNCE_MS = 400

log = logging.getLogger(__name__)


class MainWindow(FluentWindow):
    """ChangeScope ana penceresi."""

    def __init__(
        self,
        library: Library,
        config: AppConfig,
        prefs: UiPrefs | None = None,
        config_path: Path | None = None,
        prefs_path: Path | None = None,
        probe_device: bool = True,
        reconfigure_pipeline: bool = True,
    ) -> None:
        super().__init__()
        self.library = library
        self.config = config
        self.prefs = prefs or UiPrefs()
        self._probe: DeviceProbe | None = None
        self._reconfigure_enabled = reconfigure_pipeline
        self._reconfigure_pending = False
        self._closing = False
        self._reconfigure_timer = QTimer(self)
        self._reconfigure_timer.setSingleShot(True)
        self._reconfigure_timer.setInterval(RECONFIGURE_DEBOUNCE_MS)
        self._reconfigure_timer.timeout.connect(self._apply_pipeline_config)
        self.setWindowTitle("ChangeScope")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 640)
        if QGuiApplication.platformName() == "windows":
            self.setMicaEffectEnabled(True)

        # Durum çubuğu: içerik alanının altına
        self.status_bar = StatusBar(self)
        self.widgetLayout.removeWidget(self.stackedWidget)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.stackedWidget, 1)
        column.addWidget(self.status_bar)
        self.widgetLayout.addLayout(column)

        self.library_view = LibraryView(library, self)
        self.compare_view = CompareView(library, self)
        self.settings_view = SettingsView(config, self.prefs, config_path, prefs_path, self)
        self.compare_view.use_llm = config.llm.enabled

        self.addSubInterface(self.library_view, FIF.PHOTO, "Kütüphane")
        self.addSubInterface(self.compare_view, FIF.VIEW, "Karşılaştırma")
        self.addSubInterface(self.settings_view, FIF.SETTING, "Ayarlar", NavigationItemPosition.BOTTOM)
        self.navigationInterface.setExpandWidth(220)

        # Bağlantılar
        self.library_view.compareRequested.connect(self.open_compare)
        self.library_view.comparisonOpened.connect(self.open_saved_comparison)
        self.library_view.libraryChanged.connect(self._update_library_info)
        self.compare_view.backRequested.connect(lambda: self.switchTo(self.library_view))
        self.compare_view.resultSaved.connect(lambda _id: self.library_view.refresh_history())
        self.settings_view.themeChanged.connect(self.set_theme)
        self.settings_view.libraryDirChanged.connect(self.switch_library)
        self.settings_view.llmEnabledChanged.connect(self._on_llm_toggled)
        self.settings_view.pipelineSettingsChanged.connect(self.schedule_pipeline_reconfigure)
        self.settings_view.deviceChanged.connect(self._start_device_probe)
        self.compare_view.workersIdle.connect(self._on_workers_idle)

        self._update_library_info()
        self._refresh_llm_status()
        if probe_device:
            self._start_device_probe(config.runtime.device)
        else:
            self.status_bar.set_device("CPU")
        self._center()

    # ------------------------------------------------------------------ geçişler

    def open_compare(self, before, after) -> None:
        """Karşılaştırma ekranını açar ve (gerekirse) hesaplamayı başlatır."""
        self.compare_view.use_llm = self.config.llm.enabled
        self.switchTo(self.compare_view)
        self.compare_view.open_pair(before, after)

    def open_saved_comparison(self, rec) -> None:
        """Geçmişten kaydedilmiş bir karşılaştırmayı açar (yeniden hesaplamaz)."""
        self.switchTo(self.compare_view)
        self.compare_view.open_comparison(rec)

    # ------------------------------------------------------------------- ayarlar

    def set_theme(self, name: str) -> None:
        """Temayı uygular ve özel çizimli bileşenleri yeniler."""
        apply_theme(name)
        self.compare_view.before_view.refresh_theme()
        self.compare_view.after_view.refresh_theme()
        self.library_view.refresh_all()
        self.update()

    def switch_library(self, new_dir: Path) -> None:
        """Kütüphane konumunu değiştirir; yeni konumda kütüphane yoksa mevcut içerik kopyalanır."""
        new_dir = Path(new_dir)
        old_root = self.library.root
        if new_dir.resolve() == old_root.resolve():
            return
        try:
            if not (new_dir / "library.sqlite3").exists() and (old_root / "library.sqlite3").exists():
                self.library.close()
                shutil.copytree(old_root, new_dir, dirs_exist_ok=True)
            else:
                self.library.close()
            new_lib = Library(new_dir)
        except OSError as exc:
            self.library = Library(old_root)
            InfoBar.error("Kütüphane taşınamadı", str(exc), duration=6000,
                          position=InfoBarPosition.TOP_RIGHT, parent=self)
            self.library_view.set_library(self.library)
            self.compare_view.set_library(self.library)
            return
        self.library = new_lib
        self.library_view.set_library(new_lib)
        self.compare_view.set_library(new_lib)
        self._update_library_info()
        InfoBar.success("Kütüphane konumu değişti", str(new_dir), duration=4000,
                        position=InfoBarPosition.TOP_RIGHT, parent=self)

    def _start_device_probe(self, preference: str) -> None:
        """GPU/CPU bilgisini arka planda çözer (aynı thread torch/lightglue'ı da ön ısıtır)."""
        self._probe = DeviceProbe(preference, self)
        self._probe.resolved.connect(self.status_bar.set_device)
        self._probe.start()

    def schedule_pipeline_reconfigure(self) -> None:
        """Pipeline'ı etkileyen bir ayar değişti: kısa gecikmeyle (kaydırıcı için) yeniden yapılandırılır."""
        if self._reconfigure_enabled:
            self._reconfigure_timer.start()

    def _apply_pipeline_config(self) -> None:
        """`configure_default_pipeline`'ı çağırır; karşılaştırma sürüyorsa worker'lar bitene kadar ertelenir."""
        if not self._reconfigure_enabled:
            return
        if self.compare_view.has_running_workers():
            self._reconfigure_pending = True
            return
        self._reconfigure_pending = False
        core_pipeline.configure_default_pipeline(copy.deepcopy(self.config))

    def _on_workers_idle(self) -> None:
        if self._reconfigure_pending and not self._closing:
            self._apply_pipeline_config()

    def _on_llm_toggled(self, on: bool) -> None:
        self.compare_view.use_llm = on
        self._refresh_llm_status()

    def _refresh_llm_status(self) -> None:
        """Gemma durumu + "Yorumla" düğmesinin kullanılabilirliği (ayar kapalı / model yok / hazır)."""
        cfg = self.config
        spec = cfg.llm.model_spec
        self.compare_view.set_model_name(spec.display_name)
        panel = self.compare_view.ai_panel
        if not cfg.llm.enabled:
            self.status_bar.set_llm_state("off")
            panel.set_interpret_available(
                False, "Yapay zekâ yorumu Ayarlar sayfasından kapatılmış. Açmak için Ayarlar > Yapay zekâ yorumu.",
                "Yapay zekâ yorumu ayarlarda kapalı")
        elif not spec.gguf_path(cfg.paths.models_dir).exists():
            self.status_bar.set_llm_state("off", "model bulunamadı")
            panel.set_interpret_available(False, GEMMA_DOWNLOAD_HINT, "Yapay zekâ yorumu kullanılamıyor")
        elif not cfg.paths.llama_server_exe.exists():
            self.status_bar.set_llm_state("off", "llama-server bulunamadı")
            panel.set_interpret_available(
                False, f"llama-server bulunamadı: {cfg.paths.llama_server_exe}", "Yapay zekâ yorumu kullanılamıyor")
        else:
            self.status_bar.set_llm_state("ready", spec.display_name)
            panel.set_interpret_available(True)

    def _update_library_info(self) -> None:
        repo = self.library.repo
        self.status_bar.set_library_info(
            f"Kütüphane: {repo.total_image_count()} görüntü · {len(repo.list_folders())} klasör")

    def _center(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2, max(0, geo.center().y() - self.height() // 2))

    def closeEvent(self, event) -> None:  # noqa: N802
        """Kapanış: pencere hemen gizlenir; çalışan işler iptal edilip beklenir, sonra pipeline kapatılır."""
        if self._closing:
            super().closeEvent(event)
            return
        self._closing = True
        self._reconfigure_timer.stop()
        self.hide()                       # kullanıcı bekleme görmesin
        self.library_view.shutdown()
        self.compare_view.shutdown()      # iptal + olay döngüsünü işleterek kısa bekleme
        try:
            core_pipeline.shutdown_default_pipeline()   # kurulmamışsa no-op; llama-server'ı sonlandırır
        except Exception:  # noqa: BLE001
            log.exception("Pipeline kapatılamadı")
        super().closeEvent(event)
