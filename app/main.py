"""ChangeScope giriş noktası: `python -m app.main [--demo]`."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import tempfile
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.prefs import load_prefs
from app.storage import Library
from app.ui.main_window import MainWindow
from app.ui.theme import apply_app_font, apply_theme
from core import pipeline as core_pipeline
from core.config import load_config

ICON_PATH = Path(__file__).resolve().parent / "assets" / "changescope.ico"


def _set_windows_app_id() -> None:
    """Windows görev çubuğunun python.exe yerine uygulamanın kendi ikonunu göstermesi için."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ChangeScope.ChangeScope")
        except (AttributeError, OSError):
            pass


def build_window(
    library_dir: Path | None = None, config_path: Path | None = None, prefs_path: Path | None = None,
    probe_device: bool = True, demo: bool = False,
) -> MainWindow:
    """Kütüphane + ayarlarla ana pencereyi kurar (QApplication önceden var olmalı)."""
    config = load_config(config_path)
    if library_dir is not None:
        config.paths.library_dir = library_dir
    prefs = load_prefs(prefs_path)
    apply_theme(prefs.theme)
    library = Library(config.paths.library_dir)
    if demo:
        from app.demo import install_demo_pipeline, seed_demo_library

        install_demo_pipeline()          # demo kendi sahte pipeline'ını korur (configure çağrılmaz)
        if library.repo.total_image_count() == 0:
            seed_demo_library(library)
    else:
        core_pipeline.configure_default_pipeline(copy.deepcopy(config))   # --config / --library-dir dahil
    return MainWindow(library, config, prefs, config_path, prefs_path, probe_device=probe_device,
                      reconfigure_pipeline=not demo)


def main(argv: list[str] | None = None) -> int:
    """Uygulamayı başlatır."""
    parser = argparse.ArgumentParser(prog="python -m app.main", description="ChangeScope")
    parser.add_argument("--demo", action="store_true",
                        help="Sahte pipeline + geçici örnek kütüphane ile çalış (gerçek verilere dokunmaz)")
    parser.add_argument("--library-dir", type=Path, help="Kütüphane klasörünü geçersiz kıl")
    parser.add_argument("--config", type=Path, help="config.json yolu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _set_windows_app_id()
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("ChangeScope")
    if ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    apply_app_font(app)
    library_dir = args.library_dir
    prefs_path = None
    if args.demo and library_dir is None:
        library_dir = Path(tempfile.mkdtemp(prefix="changescope_demo_")) / "library"
        prefs_path = library_dir.parent / "ui_prefs.json"
    window = build_window(library_dir, args.config or (library_dir.parent / "config.json" if args.demo and library_dir else None),
                          prefs_path, demo=args.demo)
    window.show()
    _close_splash()
    return app.exec()


def _close_splash() -> None:
    """Tek dosya exe'nin açılış ekranını (PyInstaller splash) kapatır; geliştirmede hiçbir şey yapmaz."""
    try:
        import pyi_splash  # type: ignore[import-not-found]  # yalnızca paketlenmiş exe'de vardır
    except ImportError:
        return
    try:
        pyi_splash.close()
    except RuntimeError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
