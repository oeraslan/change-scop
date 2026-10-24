"""Ayarlar sayfası: tema, kütüphane konumu, hassasiyet, fark yöntemi, tile boyutu, cihaz."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    PushButton,
    SettingCard,
    SingleDirectionScrollArea,
    Slider,
    TitleLabel,
)
from qfluentwidgets import FluentIcon as FIF

from app.prefs import THEMES, UiPrefs, save_prefs
from core.config import AppConfig, save_config

THEME_TEXTS = ["Koyu", "Açık", "Sistem"]
TILE_SIZES = [512, 768, 1024, 1536, 2048]
METHODS = ["ssim", "dinov2", "ssim+dinov2"]
METHOD_TEXTS = ["SSIM", "DINOv2", "Birleşik (SSIM + DINOv2)"]
DEVICES = ["auto", "cuda", "cpu"]
DEVICE_TEXTS = ["Otomatik", "GPU (CUDA)", "CPU"]


class SettingsView(QWidget):
    """Ayarlar. Her değişiklik anında `config.json` / `ui_prefs.json` dosyasına yazılır."""

    themeChanged = Signal(str)
    libraryDirChanged = Signal(object)      # Path
    settingsSaved = Signal()
    pipelineSettingsChanged = Signal()      # hassasiyet / yöntem / tile / cihaz: pipeline yeniden kurulmalı
    deviceChanged = Signal(str)

    def __init__(
        self, config: AppConfig, prefs: UiPrefs, config_path: Path | None = None,
        prefs_path: Path | None = None, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsView")
        self.config, self.prefs = config, prefs
        self._config_path, self._prefs_path = config_path, prefs_path

        title = TitleLabel("Ayarlar", self)
        note = BodyLabel("Değişiklikler anında kaydedilir. Hassasiyet, fark yöntemi, tile boyutu ve cihaz "
                         "ayarları bir sonraki karşılaştırmada uygulanır (çalışan işlem eski ayarla biter).", self)
        note.setTextColor("#6b7280", "#9aa3b2")

        # Tema
        self.theme_combo = ComboBox(self)
        self.theme_combo.addItems(THEME_TEXTS)
        self.theme_combo.setCurrentIndex(THEMES.index(prefs.theme) if prefs.theme in THEMES else 0)
        self.theme_combo.setFixedWidth(160)
        self.theme_combo.currentIndexChanged.connect(self._on_theme)
        theme_card = self._card(FIF.PALETTE, "Tema", "Koyu tema varsayılandır; açık veya sistem temasına geçebilirsin.",
                                self.theme_combo)

        # Kütüphane
        self.library_button = PushButton(FIF.FOLDER, "Değiştir", self)
        self.library_button.clicked.connect(self._choose_library)
        self.library_card = self._card(FIF.LIBRARY, "Kütüphane konumu", str(config.paths.library_dir),
                                       self.library_button)

        # Hassasiyet
        self.sens_slider = Slider(Qt.Orientation.Horizontal, self)
        self.sens_slider.setRange(0, 100)
        self.sens_slider.setValue(round(config.detection.sensitivity * 100))
        self.sens_slider.setFixedWidth(220)
        self.sens_label = BodyLabel(f"%{self.sens_slider.value()}", self)
        self.sens_label.setFixedWidth(44)
        self.sens_slider.valueChanged.connect(self._on_sensitivity)
        sens_card = self._card(FIF.BRIGHTNESS, "Hassasiyet",
                               "Yüksek değer daha küçük farkları da işaretler (daha fazla yanlış alarm olabilir).",
                               self.sens_slider, self.sens_label)

        # Fark yöntemi
        self.method_combo = ComboBox(self)
        self.method_combo.addItems(METHOD_TEXTS)
        method = config.detection.method
        self.method_combo.setCurrentIndex(METHODS.index(method) if method in METHODS else 0)
        self.method_combo.setFixedWidth(200)
        self.method_combo.currentIndexChanged.connect(self._on_method)
        self.method_card = self._card(FIF.DEVELOPER_TOOLS, "Fark yöntemi", self.method_description(),
                                      self.method_combo)

        # Tile
        self.tile_combo = ComboBox(self)
        self.tile_combo.addItems([f"{n} px" for n in TILE_SIZES])
        tile = config.registration.tile_size_px
        self.tile_combo.setCurrentIndex(TILE_SIZES.index(tile) if tile in TILE_SIZES else 2)
        self.tile_combo.setFixedWidth(160)
        self.tile_combo.currentIndexChanged.connect(self._on_tile)
        tile_card = self._card(FIF.LAYOUT, "Tile boyutu",
                               "Büyük görüntüler bu boyuttaki parçalara bölünerek işlenir (%10 örtüşmeli).",
                               self.tile_combo)

        # Cihaz
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(DEVICE_TEXTS)
        dev = config.runtime.device
        self.device_combo.setCurrentIndex(DEVICES.index(dev) if dev in DEVICES else 0)
        self.device_combo.setFixedWidth(160)
        self.device_combo.currentIndexChanged.connect(self._on_device)
        device_card = self._card(FIF.SPEED_HIGH, "İşlem cihazı",
                                 "Eşleştirme ve fark modelleri için GPU/CPU seçimi (Otomatik: GPU varsa GPU).",
                                 self.device_combo)

        host = QWidget(self)
        lay = QVBoxLayout(host)
        lay.setContentsMargins(36, 16, 36, 24)
        lay.setSpacing(8)
        lay.addWidget(title)
        lay.addWidget(note)
        lay.addSpacing(16)
        for card in (theme_card, self.library_card, sens_card, self.method_card, tile_card, device_card):
            lay.addWidget(card)
        lay.addStretch(1)

        scroll = SingleDirectionScrollArea(self, Qt.Orientation.Vertical)
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.enableTransparentBackground()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------

    def _card(self, icon, title: str, content: str, *widgets: QWidget) -> SettingCard:
        card = SettingCard(icon, title, content, self)
        for w in widgets:
            card.hBoxLayout.addWidget(w, 0, Qt.AlignmentFlag.AlignRight)
        card.hBoxLayout.addSpacing(16)
        return card

    def _save(self) -> None:
        save_config(self.config, self._config_path)
        self.settingsSaved.emit()

    def _on_theme(self, index: int) -> None:
        self.prefs.theme = THEMES[index]
        save_prefs(self.prefs, self._prefs_path)
        self.themeChanged.emit(self.prefs.theme)

    def _choose_library(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Kütüphane klasörünü seç", str(self.config.paths.library_dir))
        if path:
            self.set_library_dir(Path(path))

    def set_library_dir(self, path: Path) -> None:
        """Kütüphane konumunu değiştirir ve kaydeder."""
        self.config.paths.library_dir = path
        self.library_card.contentLabel.setText(str(path))
        self.library_card.contentLabel.show()
        self._save()
        self.libraryDirChanged.emit(path)

    def _on_sensitivity(self, value: int) -> None:
        self.sens_label.setText(f"%{value}")
        self.config.detection.sensitivity = value / 100.0
        self._save()
        self.pipelineSettingsChanged.emit()

    def dinov2_available(self) -> bool:
        """DINOv2 model ağırlıkları `models/dinov2` altında var mı?"""
        try:
            return self.config.paths.dinov2_weights.exists()
        except OSError:
            return False

    def method_description(self) -> str:
        """Seçili yönteme göre açıklama (DINOv2 modeli yoksa uyarı içerir)."""
        text = ("SSIM: hızlı, yapısal fark. DINOv2: anlamsal öznitelik farkı (ışık/gölgeye daha dayanıklı). "
                "Birleşik: ikisinin ağırlıklı toplamı.")
        if not self.dinov2_available() and self.config.detection.method != "ssim":
            text += " Uyarı: models/dinov2 bulunamadı; DINOv2 yerine SSIM kullanılır."
        elif not self.dinov2_available():
            text += " Not: DINOv2 modeli (models/dinov2) kurulu değil; seçilirse SSIM'e düşülür."
        return text

    def _on_method(self, index: int) -> None:
        self.config.detection.method = METHODS[index]  # type: ignore[assignment]
        self.method_card.contentLabel.setText(self.method_description())
        self._save()
        self.pipelineSettingsChanged.emit()

    def _on_tile(self, index: int) -> None:
        self.config.registration.tile_size_px = TILE_SIZES[index]
        self._save()
        self.pipelineSettingsChanged.emit()

    def _on_device(self, index: int) -> None:
        self.config.runtime.device = DEVICES[index]  # type: ignore[assignment]
        self._save()
        self.deviceChanged.emit(DEVICES[index])
        self.pipelineSettingsChanged.emit()
