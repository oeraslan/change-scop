"""Sunucu yönetimi testleri: ngl hesabı, yokluk durumu, sahte süreçle başlat/öldür/fallback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.config import LLM_MODELS, AppConfig
from core.llm.server import LlamaServer, find_free_port, kv_cache_mb, plan_ngl

FAKE_SCRIPT = Path(__file__).parent / "fake_server_script.py"

GGUF_12B = 6.5 * 1024
MMPROJ_12B = 0.16 * 1024
GGUF_E4B = 4.8 * 1024
MMPROJ_E4B = 0.92 * 1024


def cfg_for(model: str = "gemma-4-12b", **kw) -> AppConfig:
    cfg = AppConfig()
    cfg.llm.model = model
    for k, v in kw.items():
        setattr(cfg.llm, k, v)
    return cfg


def test_ngl_auto_partial_offload_on_6gb():
    plan = plan_ngl(cfg_for(), 5000, GGUF_12B, MMPROJ_12B)
    assert 0 < plan.ngl < 48 and plan.mode == "auto"


def test_ngl_auto_monotonic_and_bounds():
    cfg = cfg_for()
    values = [plan_ngl(cfg, f, GGUF_12B, MMPROJ_12B).ngl for f in (500, 2000, 4000, 5000, 6000, 8000, 12000, 24000)]
    assert values == sorted(values)
    assert values[0] == 0
    assert values[-1] == 48                      # tam offload, katman sayısını aşmaz


def test_ngl_low_vram_is_cpu():
    assert plan_ngl(cfg_for(), 300, GGUF_12B, MMPROJ_12B).ngl == 0


def test_ngl_no_cuda_is_cpu():
    plan = plan_ngl(cfg_for(), None, GGUF_12B, MMPROJ_12B)
    assert plan.ngl == 0 and plan.mode == "cpu"


def test_ngl_manual_used_verbatim():
    assert plan_ngl(cfg_for(n_gpu_layers=17), 100, GGUF_12B, MMPROJ_12B).ngl == 17
    assert plan_ngl(cfg_for(n_gpu_layers=0), 99999, GGUF_12B, MMPROJ_12B).ngl == 0


def test_ngl_e4b_spec_and_mmproj_offload_flag():
    big = plan_ngl(cfg_for("gemma-4-e4b"), 9000, GGUF_E4B, MMPROJ_E4B)
    assert big.ngl == LLM_MODELS["gemma-4-e4b"].n_layers
    cfg = cfg_for("gemma-4-e4b", mmproj_offload=False)
    on = plan_ngl(cfg_for("gemma-4-e4b"), 5000, GGUF_E4B, MMPROJ_E4B)
    off = plan_ngl(cfg, 5000, GGUF_E4B, MMPROJ_E4B)
    assert off.ngl >= on.ngl and off.mmproj_mb == 0


def test_ngl_safety_margin_reduces_layers():
    low = plan_ngl(cfg_for(vram_safety_margin_mb=100), 5000, GGUF_12B, MMPROJ_12B).ngl
    high = plan_ngl(cfg_for(vram_safety_margin_mb=1500), 5000, GGUF_12B, MMPROJ_12B).ngl
    assert high < low


def test_kv_cache_scales_with_ctx():
    spec = LLM_MODELS["gemma-4-12b"]
    assert kv_cache_mb(spec, 16384) > kv_cache_mb(spec, 4096) > 0
    assert kv_cache_mb(LLM_MODELS["gemma-4-e4b"], 8192) > 0


def test_find_free_port():
    p = find_free_port()
    assert 1024 <= p <= 65535


def test_missing_files_not_available_and_no_process(tmp_path):
    cfg = AppConfig()
    cfg.paths.bin_dir = tmp_path / "bin"
    cfg.paths.models_dir = tmp_path / "models"
    server = LlamaServer(cfg, log_path=tmp_path / "log.txt")
    assert server.files_present() is False
    assert server.start() is None
    assert server.is_running is False and server.pid is None
    assert server.info.error
    server.stop()                                  # sessizce döner


# -- Sahte süreç ----------------------------------------------------------------

class FakeProcServer(LlamaServer):
    """llama-server.exe yerine küçük bir Python betiği çalıştırır."""

    extra: list[str]

    @property
    def exe_path(self) -> Path:
        return FAKE_SCRIPT

    def build_command(self, ngl: int, port: int) -> list[str]:
        return [sys.executable, str(FAKE_SCRIPT), "--port", str(port), "-ngl", str(ngl), *self.extra]


def fake_server(tmp_path: Path, **llm) -> FakeProcServer:
    cfg = AppConfig()
    cfg.paths.models_dir = tmp_path / "models"
    spec = cfg.llm.model_spec
    spec.model_dir(cfg.paths.models_dir).mkdir(parents=True)
    spec.gguf_path(cfg.paths.models_dir).write_bytes(b"x" * 1024)
    spec.mmproj_path(cfg.paths.models_dir).write_bytes(b"x" * 1024)
    for k, v in llm.items():
        setattr(cfg.llm, k, v)
    cfg.llm.startup_timeout_s = 30
    s = FakeProcServer(cfg, log_path=tmp_path / "logs" / "llama.log")
    s.extra = []
    return s


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def test_start_health_stop_kills_process(tmp_path):
    s = fake_server(tmp_path, n_gpu_layers=0)
    url = s.start()
    assert url and s.is_running
    pid = s.pid
    assert _alive(pid)
    assert s.info.ngl == 0 and s.info.startup_s is not None
    assert s.start() == url                        # ikinci çağrı yeniden başlatmaz
    s.stop()
    assert not s.is_running and s.pid is None
    assert not _alive(pid)
    assert (tmp_path / "logs" / "llama.log").exists()


def test_restart_after_release(tmp_path):
    s = fake_server(tmp_path, n_gpu_layers=0)
    u1 = s.start()
    s.stop()
    u2 = s.start()
    assert u1 and u2 and s.is_running
    s.stop()


def test_context_manager(tmp_path):
    with fake_server(tmp_path, n_gpu_layers=0) as s:
        assert s.is_running
        pid = s.pid
    assert not _alive(pid)


def test_failure_falls_back_to_lower_ngl_then_cpu(tmp_path):
    s = fake_server(tmp_path, n_gpu_layers=40)
    s.extra = ["--fail-above", "0"]               # ngl > 0 → hemen çıkar (OOM benzetimi)
    url = s.start()
    assert url is not None
    assert [a[0] for a in s.info.attempts] == [40, 24, 0]
    assert s.info.ngl == 0
    s.stop()


def test_all_attempts_fail_returns_none_without_raising(tmp_path):
    s = fake_server(tmp_path, n_gpu_layers=0)
    s.extra = ["--fail-above", "-1"]
    assert s.start() is None
    assert s.info.error and not s.is_running


def test_health_timeout(tmp_path):
    s = fake_server(tmp_path, n_gpu_layers=0)
    s.config.llm.startup_timeout_s = 1.0
    s.extra = ["--delay", "10"]
    assert s.start() is None
    assert "hazır olmadı" in s.info.error
    assert s.pid is None


def test_disabled_config_no_start(tmp_path):
    s = fake_server(tmp_path, enabled=False)
    assert s.start() is None and s.pid is None


def test_real_command_line_flags(tmp_path):
    cfg = AppConfig()
    cfg.llm.mmproj_offload = False
    cfg.llm.threads = 4
    cmd = LlamaServer(cfg, log_path=tmp_path / "l").build_command(20, 1234)
    assert cmd[cmd.index("-ngl") + 1] == "20"
    assert "--no-mmproj-offload" in cmd and cmd[cmd.index("-t") + 1] == "4"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1" and "--mmproj" in cmd


def test_invalid_argument_does_not_retry_lower_ngl(tmp_path):
    s = fake_server(tmp_path, n_gpu_layers=40)
    s.extra = ["--invalid"]
    assert s.start() is None
    assert len(s.info.attempts) == 1
