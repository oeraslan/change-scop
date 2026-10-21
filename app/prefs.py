"""Arayüze özel kullanıcı tercihleri (tema vb.). `core.config.AppConfig` alanı olmayanlar burada saklanır."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from core.config import user_data_dir

THEMES = ("dark", "light", "auto")


@dataclass
class UiPrefs:
    """Kullanıcı arayüz tercihleri."""

    theme: str = "dark"


def default_prefs_path() -> Path:
    """`%LOCALAPPDATA%/ChangeScope/ui_prefs.json`."""
    return user_data_dir() / "ui_prefs.json"


def load_prefs(path: Path | None = None) -> UiPrefs:
    """Tercihleri okur; dosya yok/bozuksa varsayılanlar."""
    prefs = UiPrefs()
    path = path or default_prefs_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("theme") in THEMES:
            prefs.theme = data["theme"]
    except (OSError, ValueError, AttributeError):
        pass
    return prefs


def save_prefs(prefs: UiPrefs, path: Path | None = None) -> None:
    """Tercihleri yazar."""
    path = path or default_prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(prefs), indent=2), encoding="utf-8")
