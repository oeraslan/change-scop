"""ChangeScope giriş noktası: `python -m app.main [--demo]`."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.prefs import load_prefs
from app.storage import Library
from app.ui.main_window import MainWindow
from app.ui.theme import apply_app_font, apply_theme
from core.config import load_config


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

        install_demo_pipeline()
        if library.repo.total_image_count() == 0:
            seed_demo_library(library)
    return MainWindow(library, config, prefs, config_path, prefs_path, probe_device=probe_device)


def main(argv: list[str] | None = None) -> int:
    """Uygulamayı başlatır."""
    parser = argparse.ArgumentParser(prog="python -m app.main", description="ChangeScope")
    parser.add_argument("--demo", action="store_true",
                        help="Sahte pipeline + geçici örnek kütüphane ile çalış (gerçek verilere dokunmaz)")
    parser.add_argument("--library-dir", type=Path, help="Kütüphane klasörünü geçersiz kıl")
    parser.add_argument("--config", type=Path, help="config.json yolu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("ChangeScope")
    apply_app_font(app)
    library_dir = args.library_dir
    prefs_path = None
    if args.demo and library_dir is None:
        library_dir = Path(tempfile.mkdtemp(prefix="changescope_demo_")) / "library"
        prefs_path = library_dir.parent / "ui_prefs.json"
    window = build_window(library_dir, args.config or (library_dir.parent / "config.json" if args.demo and library_dir else None),
                          prefs_path, demo=args.demo)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
