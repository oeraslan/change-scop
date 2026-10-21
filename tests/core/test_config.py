"""core/config.py testleri."""

from __future__ import annotations

import json
from pathlib import Path

from core.config import (
    DEFAULT_LLM_MODEL,
    LLM_MODELS,
    AppConfig,
    app_root,
    load_config,
    resolve_device,
    save_config,
)


def test_defaults_point_inside_project():
    cfg = AppConfig()
    assert cfg.paths.models_dir == app_root() / "models"
    assert cfg.paths.llama_server_exe == app_root() / "bin" / "llama" / "llama-server.exe"
    assert cfg.registration.tile_size_px == 1024
    assert cfg.registration.tile_overlap == 0.10
    assert cfg.llm.model == DEFAULT_LLM_MODEL == "gemma-4-12b"
    assert cfg.llm.n_gpu_layers == "auto"


def test_both_llm_models_defined():
    assert set(LLM_MODELS) == {"gemma-4-12b", "gemma-4-e4b"}
    for spec in LLM_MODELS.values():
        assert spec.gguf_file.endswith(".gguf") and spec.mmproj_file.endswith(".gguf")
        assert spec.n_layers > 0
        assert spec.gguf_path(Path("m")).parent == Path("m") / "gemma" / spec.key


def test_unknown_model_falls_back_to_default():
    cfg = AppConfig()
    cfg.llm.model = "yok-boyle-model"
    assert cfg.llm.model_spec.key == DEFAULT_LLM_MODEL


def test_missing_or_corrupt_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "yok.json") == AppConfig()
    bad = tmp_path / "bad.json"
    bad.write_text("{bozuk", encoding="utf-8")
    assert load_config(bad) == AppConfig()


def test_save_writes_only_overrides_and_roundtrips(tmp_path):
    path = tmp_path / "config.json"
    cfg = AppConfig()
    cfg.llm.model = "gemma-4-e4b"
    cfg.llm.n_gpu_layers = 20
    cfg.detection.min_region_area_px = 900
    cfg.paths.library_dir = tmp_path / "lib"
    save_config(cfg, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["llm"] == {"model": "gemma-4-e4b", "n_gpu_layers": 20}
    assert data["detection"] == {"min_region_area_px": 900}
    assert "registration" not in data

    loaded = load_config(path)
    assert loaded == cfg
    assert isinstance(loaded.paths.library_dir, Path)


def test_unknown_keys_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"model": "gemma-4-e4b", "garip": 1}, "eski_bolum": {}}),
                    encoding="utf-8")
    cfg = load_config(path)
    assert cfg.llm.model == "gemma-4-e4b"


def test_resolve_device(monkeypatch):
    import torch

    assert resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_env_overrides_model_and_bin_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("CHANGESCOPE_MODELS_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("CHANGESCOPE_BIN_DIR", str(tmp_path / "b"))
    cfg = AppConfig()
    assert cfg.paths.models_dir == tmp_path / "m"
    assert cfg.paths.llama_server_exe == tmp_path / "b" / "llama" / "llama-server.exe"
