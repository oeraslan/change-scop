"""Ayarlar sayfası, durum çubuğu, tema ve pencere düzeyi davranışlar."""

from __future__ import annotations

import json
from pathlib import Path

from qfluentwidgets import isDarkTheme

from app.prefs import load_prefs
from core.config import load_config


def test_status_bar_texts(window) -> None:
    bar = window.status_bar
    assert bar.llm_label.text().startswith("Gemma: kapalı")        # worktree'de model/exe yok
    assert "9" not in bar.library_label.text() and "4 görüntü" in bar.library_label.text()
    assert bar.device_label.text() == "CPU"


def test_status_bar_llm_states(window) -> None:
    window.status_bar.set_llm_state("loading")
    assert window.status_bar.llm_label.text() == "Gemma: yükleniyor"
    window.status_bar.set_llm_state("ready")
    assert window.status_bar.llm_label.text() == "Gemma: hazır"


def test_navigation_between_pages(window) -> None:
    window.switchTo(window.settings_view)
    assert window.stackedWidget.currentWidget() is window.settings_view
    window.switchTo(window.compare_view)
    assert not window.compare_view.has_content()               # boş durum gösterilir
    window.switchTo(window.library_view)
    assert window.stackedWidget.currentWidget() is window.library_view


def test_settings_persist_to_config_file(window, tmp_path: Path) -> None:
    sv = window.settings_view
    sv.sens_slider.setValue(80)
    sv.tile_combo.setCurrentIndex(3)                            # 1536
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["detection"]["sensitivity"] == 0.8
    assert saved["registration"]["tile_size_px"] == 1536
    reloaded = load_config(path)
    assert reloaded.detection.sensitivity == 0.8 and reloaded.registration.tile_size_px == 1536


def test_llm_toggle_updates_status_and_compare_view(window, tmp_path: Path) -> None:
    sv = window.settings_view
    sv.llm_switch.setChecked(False)
    assert window.config.llm.enabled is False
    assert window.compare_view.use_llm is False
    assert window.status_bar.llm_label.text().startswith("Gemma: kapalı")
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["llm"]["enabled"] is False


def test_theme_switch_and_persistence(window, tmp_path: Path) -> None:
    sv = window.settings_view
    assert isDarkTheme()
    sv.theme_combo.setCurrentIndex(1)                           # Açık
    assert not isDarkTheme()
    assert load_prefs(tmp_path / "prefs.json").theme == "light"
    sv.theme_combo.setCurrentIndex(0)
    assert isDarkTheme()
    assert load_prefs(tmp_path / "prefs.json").theme == "dark"


def test_library_location_change_copies_library(window, tmp_path: Path) -> None:
    old_count = window.library.repo.total_image_count()
    new_dir = tmp_path / "yeni_konum"
    window.settings_view.set_library_dir(new_dir)
    assert (new_dir / "library.sqlite3").exists()
    assert window.library.root == new_dir
    assert window.library.repo.total_image_count() == old_count
    assert window.library_view.gallery_model.rowCount() == old_count
    assert window.config.paths.library_dir == new_dir
    rec = window.library.repo.list_images()[0]
    assert window.library.image_path(rec).exists()             # dosyalar da taşındı


def test_tests_never_touch_real_localappdata(window, tmp_path: Path) -> None:
    import os

    local = os.environ.get("LOCALAPPDATA", "")
    assert not str(window.library.root).startswith(str(Path(local) / "ChangeScope")) or not local
