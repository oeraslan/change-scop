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
    parser.add_argument("--phase3", action="store_true", help="yalnızca Faz 3 ekranları (AI yükleniyor/dolu, ayarlar)")
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.phase3:
        return phase3(out)

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

    # Yok sayma maskesi: düzenleme modu (çizilmiş maskeyle)
    cv.group.fit_all()
    cv.enter_mask_mode()
    wait_until(lambda: not cv._mask_workers)
    items = sorted(cv.before_view.region_items(), key=lambda i: i.region_id)
    cv.mask_editor.brush_px = 60
    for item in items[-1:]:                                  # son bölge: "gölge/bulut" gibi yok sayılacak alan
        r = item.spec.rect
        cv.mask_editor.stroke([(r.left() + 10, r.center().y()), (r.center().x(), r.center().y() - 8),
                               (r.right() - 10, r.center().y() + 10)])
    cv.mask_editor.brush_px = 90
    cv.mask_editor.stroke([(60, 70), (180, 60), (300, 90)])
    cv.mask_editor.brush_px = 40
    cv.mask_editor.tool = "eraser"
    cv.mask_editor.stroke([(120, 65)])
    cv.mask_editor.tool = "brush"
    wait(300)
    shot(win, out, "17_maske_duzenleme.png")
    # Kaydet ve yeniden karşılaştır → maskeli sonuç
    cv.save_mask_and_recompare()
    wait_until(lambda: not cv.mask_mode_active())
    wait(300)
    wait_until(lambda: not cv.is_busy() and cv._result is not None)
    wait(900)
    shot(win, out, "18_maskeli_karsilastirma.png")

    # Geçmiş sekmesine birkaç kayıt daha ekle
    for fname in ("Güney Vadi", "Z Limanı"):
        folder = next(f for f in library.repo.list_folders() if f.name == fname)
        imgs = library.repo.list_images(folder.id)
        if len(imgs) >= 2:
            win.open_compare(imgs[0], imgs[1])
            wait(300)
            wait_until(lambda: not cv.is_busy() and cv._result is not None)
            wait(600)

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
    lv.tree.select_folder(None)
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
    win._reconfigure_enabled = False        # demo pipeline'ı ezilmesin
    win.settings_view.method_combo.setCurrentIndex(2)
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


def phase3(out: Path) -> int:
    """Faz 3 ekranları: AI paneli yükleniyor (kartlar + shimmer + aşama metni), dolu (verdict), ayarlar."""
    from tests.ui.helpers import make_result

    app = QApplication.instance() or QApplication(sys.argv)
    apply_app_font(app)
    apply_theme("dark")
    tmp = Path(tempfile.mkdtemp(prefix="changescope_shots_"))
    library = Library(tmp / "library")
    seed_demo_library(library, tmp / "seed")
    install_demo_pipeline()
    config = AppConfig()
    config.paths.library_dir = library.root
    config.paths.models_dir = tmp / "models"
    config.paths.bin_dir = tmp / "bin"
    spec = config.llm.model_spec
    for path in (spec.gguf_path(config.paths.models_dir), spec.mmproj_path(config.paths.models_dir),
                 config.paths.llama_server_exe):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    win = MainWindow(library, config, UiPrefs(), tmp / "config.json", tmp / "prefs.json",
                     probe_device=False, reconfigure_pipeline=False)
    win.status_bar.set_device("GPU · NVIDIA GeForce RTX 2060")
    win.setMicaEffectEnabled(False)
    win.resize(1440, 900)
    win.show()
    wait(500)
    cv = win.compare_view
    win.switchTo(cv)
    cv.show_result(make_result(960, 720), "X Bölgesi / Y Dağı", "12.10.2026  →  19.10.2026", interpreting=True)
    cv.ai_panel.set_stage_text("Bölge 2/3 yorumlandı")
    win._on_llm_activity(True)
    wait(500)
    shot(win, out, "20_ai_yukleniyor_kartlar_bekliyor.png")
    cv.ai_panel.set_stage_text("Yapay zekâ modeli yükleniyor…")
    wait(300)
    shot(win, out, "21_ai_model_yukleniyor.png")
    cv.show_result(make_result(960, 720, with_llm=True), keep_view=True)
    win._on_llm_activity(False)
    win.status_bar.set_llm_state("running", "GPU katmanı 28/48")
    wait(500)
    shot(win, out, "22_ai_dolu_verdict_rozetleri.png")
    win.switchTo(win.settings_view)
    win.settings_view.llm_model_combo.setCurrentIndex(1)
    win.settings_view.llm_model_combo.setCurrentIndex(0)
    win.settings_view.gpu_mode_combo.setCurrentIndex(1)
    win.settings_view.gpu_spin.setValue(28)
    wait(300)
    # LLM kartlarının görünmesi için kaydır
    from qfluentwidgets import SingleDirectionScrollArea
    area = win.settings_view.findChild(SingleDirectionScrollArea)
    if area is not None:
        area.verticalScrollBar().setValue(120)
    wait(300)
    shot(win, out, "23_ayarlar_llm.png")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
