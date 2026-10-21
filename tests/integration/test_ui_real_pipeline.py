"""Ana pencere + gerçek pipeline: içe aktarma → karşılaştırma → sonuç; UI donmamalı; ekran görüntüsü."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pytestqt")

from tests.integration import ui_scenario  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHOT = PROJECT_ROOT / "tests" / "integration" / "screenshots" / "real_pipeline_compare.png"

# Worker thread'inde model yükleme/torch import'u GIL'i kısa süre tutabilir; yine de ana thread
# saniyelerce takılmamalı.
MAX_UI_GAP_MS = 1500


def _check(out: dict) -> None:
    assert out["imported"] == 2
    assert out["result_ok"], out
    assert out["quality"] in ("good", "weak"), out
    assert out["n_regions"] >= 2
    assert all(out["hits"]), out
    assert out["worker_finished"] and not out["progress_visible_after"]
    n = out["n_regions"]
    assert out["visible_regions"] == [n, n], "bölge kutuları iki görünümde de görünmeli"
    assert out["visible_regions_when_off"] == [0, 0], "'Farkları göster' kapalıyken işaret görünmemeli"
    assert out["saved"], "sonuç kütüphaneye kaydedilmedi"
    assert out["reopen_is_cached"]


def test_main_window_runs_real_pipeline_without_freezing(qapp, tmp_path) -> None:
    """Offscreen: gerçek `run_comparison` ile sonuç gelir; olay döngüsü takılmaz."""
    out = ui_scenario.run_scenario(qapp, tmp_path)
    print("\nUI senaryo:", json.dumps(out, ensure_ascii=False))
    _check(out)
    assert out["ticks"] > 20, "olay döngüsü karşılaştırma sırasında neredeyse hiç çalışmadı"
    assert out["max_ui_gap_ms"] < MAX_UI_GAP_MS, f"UI {out['max_ui_gap_ms']} ms donmuş"
    assert any("Eşleştiriliyor" in s or "hizalan" in s.lower() for s in out["stage_texts"]), out["stage_texts"]


@pytest.mark.skipif(sys.platform != "win32", reason="windows Qt platformu gerekli")
def test_compare_screen_screenshot_with_windows_platform(tmp_path) -> None:
    """Ayrı süreçte (QT_QPA_PLATFORM=windows) senaryoyu çalıştırıp karşılaştırma ekranının görüntüsünü kaydeder."""
    SHOT.parent.mkdir(parents=True, exist_ok=True)
    SHOT.unlink(missing_ok=True)
    env = dict(os.environ, QT_QPA_PLATFORM="windows")
    json_path = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tests.integration.ui_scenario", "--shot", str(SHOT), "--json", str(json_path)],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=400,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(json_path.read_text(encoding="utf-8"))
    print("\nUI senaryo (windows):", json.dumps(out, ensure_ascii=False))
    assert out["platform"] == "windows"
    _check(out)
    assert out["max_ui_gap_ms"] < MAX_UI_GAP_MS
    assert SHOT.is_file() and SHOT.stat().st_size > 20_000
