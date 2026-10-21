"""Ayarlar sayfası: tema, kütüphane konumu, LLM, hassasiyet, tile boyutu."""

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
    SwitchButton,
    TitleLabel,
)
from qfluentwidgets import FluentIcon as FIF

from app.prefs import THEMES, UiPrefs, save_prefs
from core.config import AppConfig, save_config

THEME_TEXTS = ["Koyu", "Açık", "Sistem"]
TILE_SIZES = [512, 768, 1024, 1536, 2048]


class SettingsView(QWidget):
    """Ayarlar. Her değişiklik anında `config.json` / `ui_prefs.json` dosyasına yazılır."""

    themeChanged = Signal(str)
    libraryDirChanged = Signal(object)      # Path
    llmEnabledChanged = Signal(bool)
    settingsSaved = Signal()

    def __init__(
        self, config: AppConfig, prefs: UiPrefs, config_path: Path | None = None,
        prefs_path: Path | None = None, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsView")
        self.config, self.prefs = config, prefs
        self._config_path, self._prefs_path = config_path, prefs_path

        title = TitleLabel("Ayarlar", self)
        note = BodyLabel("Değişiklikler anında kaydedilir. Hassasiyet ve tile boyutu bir sonraki "
                         "karşılaştırmalarda geçerli olur.", self)
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

        # LLM
        self.llm_switch = SwitchButton(self)
        self.llm_switch.setOnText("Açık")
        self.llm_switch.setOffText("Kapalı")
        self.llm_switch.setChecked(config.llm.enabled)
        self.llm_switch.checkedChanged.connect(self._on_llm)
        llm_card = self._card(FIF.ROBOT, "Yapay zekâ yorumu (Gemma)",
                              "Bölgeleri açıklar ve gölge/ışık kaynaklı yanlış alarmları ayıklar.", self.llm_switch)

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

        host = QWidget(self)
        lay = QVBoxLayout(host)
        lay.setContentsMargins(36, 16, 36, 24)
        lay.setSpacing(8)
        lay.addWidget(title)
        lay.addWidget(note)
        lay.addSpacing(16)
        for card in (theme_card, self.library_card, llm_card, sens_card, tile_card):
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

    def _on_llm(self, on: bool) -> None:
        self.config.llm.enabled = on
        self._save()
        self.llmEnabledChanged.emit(on)

    def _on_sensitivity(self, value: int) -> None:
        self.sens_label.setText(f"%{value}")
        self.config.detection.sensitivity = value / 100.0
        self._save()

    def _on_tile(self, index: int) -> None:
        self.config.registration.tile_size_px = TILE_SIZES[index]
        self._save()
