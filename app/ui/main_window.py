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

DOWNLOAD_KEYS = {"gemma-4-12b": "gemma-12b", "gemma-4-e4b": "gemma-e4b"}


def download_hint(model_key: str) -> str:
    """Model dosyası yoksa gösterilen indirme ipucu."""
    return f"Gemma modeli indirilmemiş: scripts/download_models.py --only {DOWNLOAD_KEYS.get(model_key, 'gemma-12b')}"


GEMMA_DOWNLOAD_HINT = download_hint("gemma-4-12b")
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
        self.compare_view.use_llm = self.llm_ready()

        self.addSubInterface(self.library_view, FIF.PHOTO, "Kütüphane")
        self.addSubInterface(self.compare_view, FIF.VIEW, "Karşılaştırma")
        self.addSubInterface(self.settings_view, FIF.SETTING, "Ayarlar", NavigationItemPosition.BOTTOM)
        self.navigationInterface.setExpandWidth(220)

        # Bağlantılar
        self.library_view.compareRequested.connect(self.open_compare)
        self.library_view.comparisonOpened.connect(self.open_saved_comparison)
        self.library_view.comparisonRecompareRequested.connect(
            lambda rec: self.open_saved_comparison(rec, force=True))
        self.compare_view.maskSaved.connect(lambda _rec: self.library_view.refresh_history())
        self.library_view.libraryChanged.connect(self._update_library_info)
        self.compare_view.backRequested.connect(lambda: self.switchTo(self.library_view))
        self.compare_view.resultSaved.connect(lambda _id: self.library_view.refresh_history())
        self.settings_view.themeChanged.connect(self.set_theme)
        self.settings_view.libraryDirChanged.connect(self.switch_library)
        self.settings_view.llmEnabledChanged.connect(self._on_llm_toggled)
        self.settings_view.llmSettingsChanged.connect(self._on_llm_toggled)
        self.compare_view.llmActivity.connect(self._on_llm_activity)
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
        self.compare_view.use_llm = self.llm_ready()
        self._refresh_llm_status()
        self.switchTo(self.compare_view)
        self.compare_view.open_pair(before, after)

    def open_saved_comparison(self, rec, force: bool = False) -> None:
        """Geçmişten kaydedilmiş bir karşılaştırmayı açar (yeniden hesaplamaz; `force=True` hesaplar)."""
        self.compare_view.use_llm = self.llm_ready()
        self._refresh_llm_status()
        self.switchTo(self.compare_view)
        self.compare_view.open_comparison(rec, force=force)

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
        """`configure_default_pipeline`'ı çağırır. Çalışan koşuyu beklemez: eski pipeline koşu bitince bırakılır."""
        if not self._reconfigure_enabled or self._closing:
            return
        self._reconfigure_pending = False
        core_pipeline.configure_default_pipeline(copy.deepcopy(self.config))

    def _on_workers_idle(self) -> None:
        if self._reconfigure_pending and not self._closing:
            self._apply_pipeline_config()

    def llm_problem(self) -> tuple[str, str] | None:
        """LLM kullanılamıyorsa (kısa neden, ayrıntı); hazırsa None. Sunucuyu başlatmaz."""
        cfg = self.config
        spec = cfg.llm.model_spec
        if not cfg.llm.enabled:
            return "", "disabled"
        if not (spec.gguf_path(cfg.paths.models_dir).is_file() and spec.mmproj_path(cfg.paths.models_dir).is_file()):
            return "model bulunamadı", "model"
        if not cfg.paths.llama_server_exe.is_file():
            return "llama-server bulunamadı", "exe"
        return None

    def llm_ready(self) -> bool:
        """Ayarlarda açık VE model + llama-server mevcut mu (`llm_status()["available"]` ile aynı kural)."""
        return self.llm_problem() is None

    def _on_llm_toggled(self, *_args) -> None:
        self.compare_view.use_llm = self.llm_ready()
        self._refresh_llm_status()

    def _on_llm_activity(self, active: bool) -> None:
        """Karşılaştırma ekranındaki LLM aşaması başladı/bitti: durum çubuğu olay bazlı güncellenir."""
        if active:
            self.status_bar.set_llm_state("loading")
        else:
            self._refresh_llm_status()

    def _refresh_llm_status(self) -> None:
        """Gemma durumu + "Yorumla" düğmesinin kullanılabilirliği (ayar kapalı / model yok / hazır / çalışıyor)."""
        cfg = self.config
        spec = cfg.llm.model_spec
        self.compare_view.set_model_name(spec.display_name)
        panel = self.compare_view.ai_panel
        problem = self.llm_problem()
        if problem is None:
            info: dict = {}
            try:
                info = dict(core_pipeline.llm_status())
            except Exception:  # noqa: BLE001 - durum bilgisi isteğe bağlı
                log.debug("llm_status okunamadı", exc_info=True)
            if self.compare_view.is_llm_active():
                self.status_bar.set_llm_state("loading")
            elif info.get("running"):
                ngl = info.get("ngl")
                self.status_bar.set_llm_state(
                    "running", f"GPU katmanı {ngl}/{spec.n_layers}" if ngl is not None else "")
            else:
                self.status_bar.set_llm_state("ready", spec.display_name)
            panel.set_interpret_available(True)
            return
        reason, kind = problem
        if kind == "disabled":
            self.status_bar.set_llm_state("off")
            panel.set_interpret_available(
                False, "Yapay zekâ yorumu Ayarlar sayfasından kapatılmış. Açmak için Ayarlar > Yapay zekâ yorumu.",
                "Yapay zekâ yorumu ayarlarda kapalı")
        elif kind == "model":
            self.status_bar.set_llm_state("off", reason)
            panel.set_interpret_available(False, download_hint(cfg.llm.model), "Yapay zekâ yorumu kullanılamıyor")
        else:
            self.status_bar.set_llm_state("off", reason)
            panel.set_interpret_available(
                False, f"llama-server bulunamadı: {cfg.paths.llama_server_exe}", "Yapay zekâ yorumu kullanılamıyor")

    def _update_library_info(self) -> None:
        repo = self.library.repo
        self.status_bar.set_library_info(
            f"Kütüphane: {repo.total_image_count()} görüntü · {len(repo.list_folders())} klasör")

    def _center(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2, max(0, geo.center().y() - self.height() // 2))

    @staticmethod
    def _kill_llm_server() -> None:
        """Çalışan llama-server'ı hemen durdurur (süren LLM istekleri düşer, CV sonucu korunur)."""
        try:
            core_pipeline.stop_llm_server()
        except Exception:  # noqa: BLE001
            log.exception("llama-server durdurulamadı")

    def closeEvent(self, event) -> None:  # noqa: N802
        """Kapanış: pencere hemen gizlenir; çalışan işler iptal edilip beklenir, sonra pipeline kapatılır."""
        if self._closing:
            super().closeEvent(event)
            return
        self._closing = True
        self._reconfigure_timer.stop()
        self.hide()                       # kullanıcı bekleme görmesin
        self.library_view.shutdown()
        self.compare_view.shutdown(timeout_ms=3000)      # iptal + olay döngüsünü işleterek kısa bekleme
        if self.compare_view.has_running_workers():
            # LLM bölge isteği sürüyor olabilir (8-22 sn): sunucuyu öldürmek istekleri hemen düşürür.
            self._kill_llm_server()
            self.compare_view.shutdown(timeout_ms=8000)
        try:
            core_pipeline.shutdown_default_pipeline()   # kurulmamışsa no-op; llama-server'ı sonlandırır
        except Exception:  # noqa: BLE001
            log.exception("Pipeline kapatılamadı")
        super().closeEvent(event)
