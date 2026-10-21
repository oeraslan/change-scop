"""Gerçek pipeline + gerçek ana pencere senaryosu (pytest ve bağımsız çalıştırma ortak kullanır).

Bağımsız çalıştırma (ekran görüntüsü için `windows` platformu; offscreen'de font glifleri çizilmez):

    set QT_QPA_PLATFORM=windows
    .venv\\Scripts\\python.exe -m tests.integration.ui_scenario --shot tests/integration/screenshots/real_pipeline_compare.png

Akış: geçici kütüphane → iç içe klasör → sentetik çifti kütüphaneye içe aktar → iki görüntüyü seç →
`MainWindow.open_compare` → gerçek `core.pipeline.run_comparison` (varsayılan pipeline, lazy kurulum).
Karşılaştırma sürerken ana thread'in olay döngüsü ölçülür (UI donma testi).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.prefs import UiPrefs
from app.storage import Library
from app.ui.main_window import MainWindow
from app.ui.theme import apply_app_font, apply_theme
from core.config import AppConfig
from core.contracts import ImageSource
from core.pipeline import get_default_pipeline, set_default_pipeline
from scripts.make_synthetic_pair import PairParams, make_pair
from tests.integration.helpers import write_pair

TICK_MS = 20


def run_scenario(app: QApplication, workdir: Path, shot: Path | None = None, timeout_s: float = 180.0) -> dict:
    """Senaryoyu çalıştırır ve ölçümleri sözlük olarak döndürür. Hata durumunda alanlar eksik/None kalır."""
    out: dict = {"platform": app.platformName()}
    pair = make_pair(PairParams(seed=21))
    src = workdir / "src"
    before_file, after_file = write_pair(pair, src, "ui")

    apply_theme("dark")
    library = Library(workdir / "library")
    config = AppConfig()
    config.paths.library_dir = library.root
    win = MainWindow(library, config, UiPrefs(), workdir / "config.json", workdir / "prefs.json",
                     probe_device=False)
    win.resize(1440, 900)
    win.show()

    def pump(ms: int) -> None:
        end = time.perf_counter() + ms / 1000
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)

    set_default_pipeline(None)          # uygulamadaki gibi: ilk karşılaştırmada lazy kurulur (worker thread'inde)
    try:
        pump(300)
        root = library.repo.create_folder("X Bölgesi")
        leaf = library.repo.create_folder("Y Dağı", root.id)
        recs = win.library_view.import_files([before_file, after_file], leaf.id, ImageSource.DRONE, sync=True)
        out["imported"] = len(recs or [])
        by_name = {r.original_name: r for r in recs}
        rec_before, rec_after = by_name[before_file.name], by_name[after_file.name]
        pump(200)

        # UI donma ölçümü: ana thread'deki 20 ms'lik zamanlayıcının gerçek aralıkları
        gaps: list[float] = []
        last = [time.perf_counter()]

        def tick() -> None:
            now = time.perf_counter()
            gaps.append(now - last[0])
            last[0] = now

        timer = QTimer()
        timer.setInterval(TICK_MS)
        timer.timeout.connect(tick)
        timer.start()

        stages: list[str] = []
        cv = win.compare_view
        t0 = time.perf_counter()
        win.open_compare(rec_before, rec_after)
        out["busy_right_after_open"] = cv.is_busy()
        win.library_view  # noqa: B018  (erişim; ana pencere canlı)
        while time.perf_counter() - t0 < timeout_s:
            app.processEvents()
            text = cv.stage_label.text()
            if not stages or stages[-1] != text:
                stages.append(text)
            if not cv.is_busy() and cv._result is not None:
                break
            time.sleep(0.005)
        timer.stop()
        out["elapsed_s"] = round(time.perf_counter() - t0, 2)
        out["stage_texts"] = stages
        out["max_ui_gap_ms"] = round(max(gaps, default=0.0) * 1000)
        out["ticks"] = len(gaps)

        result = cv._result
        out["result_ok"] = result is not None
        if result is None:
            return out
        pump(600)
        out["quality"] = result.registration.quality.value
        regions = result.detection.regions if result.detection else []
        out["n_regions"] = len(regions)
        hits = []
        for x, y, w, h in pair.added_bboxes:
            cx, cy = x + w / 2, y + h / 2
            hits.append(any(r.bbox.x <= cx <= r.bbox.x + r.bbox.width and
                            r.bbox.y <= cy <= r.bbox.y + r.bbox.height for r in regions))
        out["hits"] = hits
        out["visible_regions"] = list(cv.visible_region_count())
        out["saved"] = library.find_comparison(rec_before, rec_after) is not None
        out["worker_finished"] = not cv.is_busy()
        out["progress_visible_after"] = cv.progress_host.isVisible()

        # Farkları göster anahtarı kapalıyken hiçbir işaret görünmemeli
        cv.diff_switch.setChecked(False)
        pump(500)
        out["visible_regions_when_off"] = list(cv.visible_region_count())
        cv.diff_switch.setChecked(True)
        pump(500)

        if shot is not None:
            shot.parent.mkdir(parents=True, exist_ok=True)
            try:
                win.setMicaEffectEnabled(False)   # grab() DWM arka planını yakalayamaz
            except Exception:  # noqa: BLE001
                pass
            pump(400)
            out["screenshot_saved"] = bool(win.grab().save(str(shot)))
            out["screenshot"] = str(shot)

        # Aynı çifti tekrar aç: kayıttan gelmeli, işçi başlamamalı
        cv.open_pair(rec_before, rec_after)
        pump(300)
        out["reopen_is_cached"] = not cv.is_busy() and cv._result is not None
        return out
    finally:
        win.close()
        pump(100)
        win.deleteLater()
        try:
            get_default_pipeline().shutdown()
        finally:
            set_default_pipeline(None)
        library.close()


def main(argv: list[str] | None = None) -> int:
    """Bağımsız çalıştırma: sonucu JSON olarak stdout'a yazar."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    apply_app_font(app)
    with tempfile.TemporaryDirectory(prefix="changescope_it_") as tmp:
        result = run_scenario(app, Path(tmp), args.shot)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print("SCENARIO_JSON_BEGIN\n" + text + "\nSCENARIO_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
