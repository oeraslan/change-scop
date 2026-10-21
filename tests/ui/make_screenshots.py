"""Ana ekranların ekran görüntülerini üretir (`QWidget.grab()`), demo pipeline + sentetik görüntülerle.

Kullanım (proje kökünden):
    .venv\\Scripts\\python.exe -m tests.ui.make_screenshots [--out tests/ui/screenshots]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "windows"

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.demo import install_demo_pipeline, seed_demo_library  # noqa: E402
from app.prefs import UiPrefs  # noqa: E402
from app.storage import Library  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.theme import apply_app_font, apply_theme  # noqa: E402
from core.config import AppConfig  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "screenshots"


def wait(ms: int) -> None:
    QTest.qWait(ms)


def wait_until(cond, timeout_ms: int = 20000) -> bool:
    waited = 0
    while waited < timeout_ms:
        if cond():
            return True
        QTest.qWait(50)
        waited += 50
    return False


def shot(widget, out: Path, name: str) -> Path:
    wait(350)
    path = out / name
    widget.grab().save(str(path))
    print("yazıldı:", path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    apply_app_font(app)
    apply_theme("dark")
    tmp = Path(tempfile.mkdtemp(prefix="changescope_shots_"))
    library = Library(tmp / "library")
    seed_demo_library(library, tmp / "seed")
    install_demo_pipeline()
    config = AppConfig()
    config.paths.library_dir = library.root
    win = MainWindow(library, config, UiPrefs(), tmp / "config.json", tmp / "prefs.json", probe_device=False)
    win.status_bar.set_device("GPU · NVIDIA GeForce RTX 2060")
    win.status_bar.set_llm_state("ready", "Gemma 4 12B")
    win.setMicaEffectEnabled(False)   # grab() DWM arka planını yakalayamaz; düz koyu zemin
    win.resize(1440, 900)
    win.show()
    wait(500)

    lv = win.library_view
    north = next(f for f in library.repo.list_folders() if f.name == "Kuzey Yamaç")
    lv.tree.select_folder(north.id)
    lv.tree.expandAll()
    wait(300)
    shot(win, out, "01_ana_pencere.png")

    # İki görüntü seç → yüzen çubuk
    recs = lv.gallery_model.records()
    gv = lv.gallery
    by_name = {r.original_name: r for r in recs}
    gv.set_selection([by_name['sahne42_once.png'].id, by_name['sahne42_sonra.jpg'].id])
    wait(500)
    shot(win, out, "02_secim_cubugu.png")

    # Karşılaştır (AI açık) – işlem sırasında ve sonrası
    pair = lv.selected_pair()
    win.open_compare(*pair)
    wait(900)
    shot(win, out, "03_karsilastirma_islem_suruyor.png")
    cv = win.compare_view
    wait_until(lambda: not cv.is_busy() and cv._result is not None)
    wait(800)
    shot(win, out, "04_karsilastirma_farklar_acik.png")

    # Karta tıkla → bölgeye zoom
    first_card = next(iter(cv.ai_panel.cards()))
    cv._on_card_clicked(first_card)
    wait(700)
    shot(win, out, "05_karsilastirma_bolgeye_zoom.png")
    cv.group.fit_all()

    cv.diff_switch.setChecked(False)
    wait(500)
    shot(win, out, "06_karsilastirma_farklar_kapali.png")
    cv.diff_switch.setChecked(True)
    cv.heat_switch.setChecked(True)
    wait(500)
    shot(win, out, "07_karsilastirma_isi_haritasi.png")
    cv.heat_switch.setChecked(False)

    # AI paneli boş durum (LLM yok): kayıtlı sonucu CV-only haliyle göster
    import copy

    result = copy.deepcopy(cv._result)
    result.summary, result.llm_used = None, False
    for r in result.detection.regions:
        r.verdict, r.description, r.llm_confidence = r.verdict.__class__("unreviewed"), None, None
    cv.show_result(result)
    cv.ai_panel.set_interpret_available(True)
    wait(400)
    shot(win, out, "08_ai_paneli_kapali.png")
    # Model indirilmemiş: "Yorumla" devre dışı + açıklama
    win.config.llm.enabled = True
    win._refresh_llm_status()
    wait(400)
    shot(win, out, "15_yorumla_devre_disi_model_yok.png")
    cv.ai_panel.set_interpret_available(
        False, "Yapay zekâ yorumu Ayarlar sayfasından kapatılmış. Açmak için Ayarlar > Yapay zekâ yorumu.",
        "Yapay zekâ yorumu ayarlarda kapalı")
    wait(300)
    shot(win, out, "16_yorumla_devre_disi_ayar_kapali.png")

    # Geçmiş sekmesi
    win.switchTo(lv)
    lv.show_tab("history")
    wait(400)
    shot(win, out, "09_gecmis_karsilastirmalar.png")
    lv.show_tab("images")

    # Boş durum: boş klasör
    empty = library.repo.create_folder("Yeni Bölge")
    lv.refresh_all(select_id=empty.id)
    wait(300)
    shot(win, out, "10_bos_klasor.png")

    # Hizalama WEAK / FAILED InfoBar'ları ve AI yükleniyor (skeleton)
    from core.contracts import AlignmentQuality
    from tests.ui.helpers import make_result

    win.switchTo(cv)
    cv.show_result(make_result(960, 720, quality=AlignmentQuality.WEAK, with_llm=True))
    cv.ai_panel.show_loading()
    wait(500)
    shot(win, out, "13_hizalama_zayif_ai_yukleniyor.png")
    cv.show_result(make_result(960, 720, quality=AlignmentQuality.FAILED))
    wait(300)
    shot(win, out, "14_hizalama_basarisiz.png")

    win.switchTo(win.settings_view)
    wait(300)
    shot(win, out, "11_ayarlar.png")

    # Açık tema örneği
    win.set_theme("light")
    win.switchTo(lv)
    lv.tree.select_folder(north.id)
    wait(500)
    shot(win, out, "12_acik_tema_galeri.png")

    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
