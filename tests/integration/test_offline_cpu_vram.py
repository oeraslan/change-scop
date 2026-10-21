"""Offline kuralı, CPU modu ve VRAM serbest bırakma."""

from __future__ import annotations

import socket
import time

import pytest

from core.config import AppConfig
from core.contracts import AlignmentQuality, ImageInput
from core.pipeline import build_default_pipeline
from scripts.make_synthetic_pair import PairParams, make_pair
from tests.integration.helpers import evaluate, record, write_pair


_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _host_of(value) -> str | None:  # noqa: ANN001
    if isinstance(value, tuple) and value:
        return str(value[0])
    if isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value
    return None


def _block_network(monkeypatch) -> list:
    """Dış ağa giden tüm soket bağlantı/DNS girişimlerini kaydedip reddeder; denemeleri döndürür.

    Yerel llama-server (127.0.0.1) bağlantısı CLAUDE.md'deki izinli istisnadır; ona izin verilir.
    """
    attempts: list = []

    def guard(original, host_arg_index: int):  # noqa: ANN001, ANN202
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
            target = args[host_arg_index] if len(args) > host_arg_index else None
            if _host_of(target) in _LOOPBACK:
                return original(*args, **kwargs)
            attempts.append(args)
            raise OSError("Ağ erişimi test tarafından kapatıldı")
        return wrapper

    monkeypatch.setattr(socket.socket, "connect", guard(socket.socket.connect, 1))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(socket.socket.connect_ex, 1))
    monkeypatch.setattr(socket, "create_connection", guard(socket.create_connection, 0))
    monkeypatch.setattr(socket, "getaddrinfo", guard(socket.getaddrinfo, 0))
    return attempts


def test_full_pipeline_with_network_blocked(monkeypatch, tmp_path) -> None:
    """Yeni pipeline (modeller sıfırdan yüklenir) ağ kapalıyken GOOD sonuç vermeli; hiçbir bağlantı denenmemeli."""
    import torch

    original_loader = torch.hub.load_state_dict_from_url
    attempts = _block_network(monkeypatch)
    pair = make_pair(PairParams(seed=3))
    before, after = write_pair(pair, tmp_path, "net")
    pipe = build_default_pipeline(AppConfig())
    try:
        result = pipe.run(ImageInput(before), ImageInput(after), use_llm=True)
    finally:
        pipe.shutdown()
    assert attempts == [], f"çalışma zamanında ağ denemesi: {attempts}"
    assert result.registration.quality is AlignmentQuality.GOOD, result.registration.notes
    assert any("lightglue" in n for n in result.registration.notes), "LightGlue yerine yedek kullanıldı"
    ev = evaluate(pair, result)
    record("ağ_kapalı", ev)
    assert all(ev.hits) and ev.n_large_extra == 0
    # lightglue yüklemesi için yapılan geçici yama geri alınmış olmalı
    assert torch.hub.load_state_dict_from_url is original_loader


def test_cpu_mode_runs(tmp_path) -> None:
    """`runtime.device="cpu"`: küçük görüntüyle çalışır, eşleştirme CPU'da, makul sürede."""
    cfg = AppConfig()
    cfg.runtime.device = "cpu"
    pair = make_pair(PairParams(width=800, height=600, seed=5, shift_px=(20.0, -12.0)))
    before, after = write_pair(pair, tmp_path, "cpu")
    pipe = build_default_pipeline(cfg)
    t0 = time.perf_counter()
    try:
        result = pipe.run(ImageInput(before), ImageInput(after))
    finally:
        pipe.shutdown()
    elapsed = time.perf_counter() - t0
    ev = evaluate(pair, result)
    record("cpu_modu(800x600)", ev)
    assert any("cpu" in n for n in result.registration.notes), result.registration.notes
    assert result.registration.quality is not AlignmentQuality.FAILED
    assert all(ev.hits)
    assert elapsed < 120, f"CPU modu çok yavaş: {elapsed:.1f}s"


def test_cpu_mode_full_size_is_reasonable(tmp_path) -> None:
    """CPU modu, varsayılan boyutta (1600x1200) 3 dakikayı aşmamalı."""
    cfg = AppConfig()
    cfg.runtime.device = "cpu"
    pair = make_pair(PairParams())
    before, after = write_pair(pair, tmp_path, "cpu2")
    pipe = build_default_pipeline(cfg)
    try:
        result = pipe.run(ImageInput(before), ImageInput(after))
    finally:
        pipe.shutdown()
    ev = evaluate(pair, result)
    record("cpu_modu(1600x1200)", ev)
    assert all(ev.hits)
    assert result.elapsed_s < 180


# --------------------------------------------------------------------------- VRAM

cuda = pytest.importorskip("torch").cuda
needs_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA yok")

MIB = 1024 * 1024


@needs_cuda
def test_cv_models_released_from_vram_after_run(tmp_path) -> None:
    """`Pipeline.run` bitince CV modelleri VRAM'den çıkmalı; art arda koşularda sızıntı olmamalı."""
    import torch

    pair = make_pair(PairParams(seed=11))
    before, after = write_pair(pair, tmp_path, "vram")
    pipe = build_default_pipeline(AppConfig())
    peak_in_run: list[int] = []

    def progress(stage: str, fraction: float, message: str = "") -> None:
        if stage == "release":  # tespit bitti, boşaltma henüz başlamadı
            peak_in_run.append(torch.cuda.memory_allocated())

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    after_runs = []
    for _ in range(3):
        result = pipe.run(ImageInput(before), ImageInput(after), progress)
        assert result.registration.quality is AlignmentQuality.GOOD
        after_runs.append((torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))

    # Test anlamlı mı: modeller koşu sırasında gerçekten VRAM'deydi
    assert min(peak_in_run) - baseline > 20 * MIB, peak_in_run
    for allocated, reserved in after_runs:
        assert allocated - baseline <= 16 * MIB, f"koşu sonrası {(allocated - baseline) / MIB:.1f} MiB VRAM tutuluyor"
        assert reserved <= 64 * MIB, f"önbellek {reserved / MIB:.1f} MiB"
    assert after_runs[2][0] <= after_runs[0][0] + 1 * MIB, "koşular arası VRAM büyüyor (sızıntı)"
    pipe.shutdown()
    assert torch.cuda.memory_allocated() - baseline <= 16 * MIB


@needs_cuda
def test_vram_released_when_alignment_fails(tmp_path) -> None:
    """Hizalama FAILED olsa da (erken çıkış) modeller bırakılmalı."""
    import torch

    a = make_pair(PairParams(seed=1)).before
    b = make_pair(PairParams(seed=2)).before
    from tests.integration.helpers import save_image
    pa, pb = save_image(tmp_path / "a.png", a), save_image(tmp_path / "b.png", b)
    pipe = build_default_pipeline(AppConfig())
    baseline = torch.cuda.memory_allocated()
    result = pipe.run(ImageInput(pa), ImageInput(pb))
    assert result.registration.quality is AlignmentQuality.FAILED
    assert torch.cuda.memory_allocated() - baseline <= 16 * MIB


@needs_cuda
def test_vram_released_when_preprocess_raises(tmp_path) -> None:
    """Ön işleme hata fırlatsa da `finally` ile boşaltma çalışır ve sonraki koşu sorunsuz olur."""
    import torch

    pipe = build_default_pipeline(AppConfig())
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"x")
    baseline = torch.cuda.memory_allocated()
    with pytest.raises(ValueError):
        pipe.run(ImageInput(bad), ImageInput(bad))
    assert torch.cuda.memory_allocated() - baseline <= 16 * MIB
