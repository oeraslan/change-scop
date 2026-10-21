"""Gerçek llama-server + Gemma testleri (`pytest -m llm`). exe/model yoksa skip edilir.

GPU'yu ve ~5 GB VRAM'i kullanır; CV modelleri pipeline tarafından önce bırakılır.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from core.config import AppConfig
from core.contracts import ChangeVerdict, ImageInput
from core.llm import create_analyst

pytestmark = pytest.mark.llm

PIKNIK = Path(__file__).resolve().parents[1] / "data" / "real" / "piknik"


def _config() -> AppConfig:
    return AppConfig()


@pytest.fixture(scope="module")
def analyst():
    a = create_analyst(_config())
    if not a.is_available():
        pytest.skip("llama-server.exe veya Gemma model dosyaları yok")
    yield a
    a.release()


def _cv(before: Path, after: Path, tmp_cfg: AppConfig):
    from core.pipeline import build_default_pipeline

    pipe = build_default_pipeline(tmp_cfg)
    try:
        r = pipe.run(ImageInput(before), ImageInput(after), use_llm=False)
    except Exception as exc:  # noqa: BLE001 - CV modelleri yoksa skip
        pytest.skip(f"CV pipeline çalışmadı: {exc}")
    finally:
        pipe.shutdown()
    return r.registration, r.detection


def _vram_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def test_synthetic_pair_end_to_end(analyst, tmp_path):
    from scripts.make_synthetic_pair import make_pair, save_pair

    save_pair(make_pair(), tmp_path)
    reg, det = _cv(tmp_path / "before.png", tmp_path / "after.jpg", analyst.config)
    assert det is not None and det.regions
    det.regions = sorted(det.regions, key=lambda r: r.score, reverse=True)[:2]

    assert not analyst.is_server_running           # lazy: henüz başlamadı
    used_before = _vram_used_mb()
    analyst.analyze(reg, det)
    assert analyst.is_server_running
    info = analyst.last_start_info
    assert info.ngl is not None and info.startup_s is not None

    for r in det.regions:
        assert r.verdict != ChangeVerdict.UNREVIEWED
        assert r.description and re.search(r"[çğıöşüÇĞİÖŞÜ]|\b(bir|ve|ile|yeni|alan)\b", r.description)
        assert r.llm_confidence is not None and 0.0 <= r.llm_confidence <= 1.0

    summary = analyst.summarize(reg, det)
    assert summary is not None
    n_sent = len([s for s in re.split(r"(?<=[.!?])\s+", summary.strip()) if s])
    assert 1 <= n_sent <= 4

    pid = analyst.server.pid
    analyst.release()
    assert not analyst.is_server_running and not _alive(pid)
    used_after = _vram_used_mb()
    if used_before is not None and used_after is not None:
        assert used_after <= used_before + 300     # VRAM geri bırakıldı (diğer süreçlere payla)


def test_piknik_pair_reports_verdicts(analyst):
    before, after = PIKNIK / "before.jpg", PIKNIK / "after.jpg"
    if not before.exists():
        pytest.skip("piknik verisi yok")
    reg, det = _cv(before, after, analyst.config)
    assert det is not None
    analyst.analyze(reg, det)
    for r in det.regions:
        print(r.region_id, r.bbox, r.verdict.value, r.llm_confidence, r.description)
        assert r.verdict != ChangeVerdict.UNREVIEWED
    print("ÖZET:", analyst.summarize(reg, det))
    analyst.release()
