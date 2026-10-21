"""UI test ortamı: offscreen Qt, geçici kütüphane (gerçek %LOCALAPPDATA% asla kullanılmaz)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from qfluentwidgets import isDarkTheme  # noqa: E402
from PIL import Image  # noqa: E402

from app.prefs import UiPrefs  # noqa: E402
from app.storage import Library  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.theme import apply_theme  # noqa: E402
from core.config import AppConfig  # noqa: E402
from core.contracts import ImageSource  # noqa: E402
from core.pipeline import set_default_pipeline  # noqa: E402
from tests.ui.helpers import scene  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_pipeline():
    """Her testten sonra varsayılan pipeline sıfırlanır."""
    if not isDarkTheme():
        apply_theme("dark")
    yield
    set_default_pipeline(None)


@pytest.fixture()
def library(tmp_path: Path) -> Library:
    lib = Library(tmp_path / "library")
    yield lib
    lib.close()


def make_image_file(directory: Path, name: str, seed: int = 1, size=(480, 360)) -> Path:
    """Test için küçük bir görüntü dosyası yazar."""
    path = directory / name
    Image.fromarray(scene(size[0], size[1], seed=seed)).save(path)
    return path


@pytest.fixture()
def populated(library: Library, tmp_path: Path):
    """İki klasörde dört görüntü: (library, {'a': folder, 'b': folder}, [records])."""
    src = tmp_path / "src"
    src.mkdir()
    a = library.repo.create_folder("X Bölgesi")
    b = library.repo.create_folder("Y Dağı", a.id)
    recs = [
        library.import_image(make_image_file(src, "bir.png", 1), b.id, datetime(2026, 10, 12), ImageSource.DRONE),
        library.import_image(make_image_file(src, "iki.png", 1), b.id, datetime(2026, 10, 19), ImageSource.UAV),
        library.import_image(make_image_file(src, "uc.png", 2), a.id, datetime(2026, 10, 5), ImageSource.SATELLITE),
        library.import_image(make_image_file(src, "dort.png", 3), a.id, datetime(2026, 10, 26), ImageSource.DRONE),
    ]
    return library, {"a": a, "b": b}, recs


@pytest.fixture()
def window(qtbot, populated, tmp_path: Path):
    """Örnek verili ana pencere (sahte cihaz bilgisi, tmp ayar dosyaları)."""
    library, _, _ = populated
    config = AppConfig()
    config.paths.library_dir = library.root
    # Gerçek models/ ve bin/ (Gemma indirili olabilir) testleri etkilemesin: boş klasörler.
    config.paths.models_dir = tmp_path / "models_yok"
    config.paths.bin_dir = tmp_path / "bin_yok"
    win = MainWindow(
        library, config, UiPrefs(), tmp_path / "config.json", tmp_path / "prefs.json", probe_device=False
    )
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()          # zamanlayıcılar/iş parçacıkları kütüphane kapanmadan durur
    win.deleteLater()
