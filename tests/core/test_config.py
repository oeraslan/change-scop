"""core/config.py testleri."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from core.config import (
    AppConfig,
    app_root,
    load_config,
    resolve_device,
    save_config,
)


def test_defaults_point_inside_project(monkeypatch):
    monkeypatch.delenv("CHANGESCOPE_MODELS_DIR", raising=False)
    cfg = AppConfig()
    assert cfg.paths.models_dir == app_root() / "models"
    assert cfg.registration.tile_size_px == 1024
    assert cfg.registration.tile_overlap == 0.10
    assert not hasattr(cfg, "llm")


def test_app_root_in_onefile_bundle_is_meipass(monkeypatch, tmp_path):
    """Tek dosya exe'de models/ pakete gömülüdür: kök, exe'nin açıldığı geçici klasördür."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert app_root() == tmp_path


def test_missing_or_corrupt_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "yok.json") == AppConfig()
    bad = tmp_path / "bad.json"
    bad.write_text("{bozuk", encoding="utf-8")
    assert load_config(bad) == AppConfig()


def test_save_writes_only_overrides_and_roundtrips(tmp_path):
    path = tmp_path / "config.json"
    cfg = AppConfig()
    cfg.detection.min_region_area_px = 900
    cfg.runtime.device = "cpu"
    cfg.paths.library_dir = tmp_path / "lib"
    save_config(cfg, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["detection"] == {"min_region_area_px": 900}
    assert data["runtime"] == {"device": "cpu"}
    assert "registration" not in data

    loaded = load_config(path)
    assert loaded == cfg
    assert isinstance(loaded.paths.library_dir, Path)


def test_unknown_keys_ignored(tmp_path):
    """Eski sürümün LLM ayarları (ve bilinmeyen bölümler) yok sayılır; uygulama açılmaya devam eder."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"model": "gemma-4-e4b"}, "detection": {"sensitivity": 0.7, "garip": 1},
                                "eski_bolum": {}}), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.detection.sensitivity == 0.7
    assert not hasattr(cfg, "llm")


def test_resolve_device(monkeypatch):
    import torch

    assert resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_env_overrides_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHANGESCOPE_MODELS_DIR", str(tmp_path / "m"))
    cfg = AppConfig()
    assert cfg.paths.models_dir == tmp_path / "m"
