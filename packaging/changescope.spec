# PyInstaller TEK DOSYA (onefile) paketi: dist/ChangeScope.exe
# Derleme: build.ps1 (CPU PyTorch'lu .venv-cpu ile; doğrudan: pyinstaller packaging/changescope.spec)
#
# models/ (SuperPoint, LightGlue, DINOv2) exe'nin İÇİNE gömülür; exe açılırken geçici klasöre (sys._MEIPASS)
# açılır ve core.config.app_root() orayı döndürür. Açılma birkaç saniye sürdüğü için açılış ekranı (splash)
# gösterilir; ana pencere açılınca app.main kapatır.
# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

ROOT = Path(SPECPATH).parent  # noqa: F821  (SPECPATH PyInstaller tarafından tanımlanır)
ASSETS = ROOT / "app" / "assets"
VERSION = (1, 0, 0, 0)

# torchvision'ın derlenmiş op'ları (_C*.pyd: nms, deform_conv2d...) hazır hook'ta YOK;
# eksik kalınca LightGlue "torchvision::nms does not exist" ile yüklenemiyor ve SIFT'e düşülüyor.
_tv_dir = Path(importlib.util.find_spec("torchvision").origin).parent
binaries = [(str(p), "torchvision") for p in _tv_dir.glob("*.pyd")] + collect_dynamic_libs("torchvision")

hiddenimports = (
    # Pipeline modülleri lazy import ediliyor; analiz bunları göremez.
    collect_submodules("core")
    + collect_submodules("app")
    + collect_submodules("lightglue")
    + collect_submodules("qfluentwidgets")
    + ["skimage.metrics", "skimage.exposure",
       "scripts.make_synthetic_pair"]   # --demo örnek kütüphanesi için
)


def _model_files() -> list[tuple[str, str]]:
    """models/ altındaki CV model dosyaları (indirme artığı ve eski LLM dosyaları hariç)."""
    out = []
    models = ROOT / "models"
    for sub in ("superpoint", "lightglue", "dinov2"):
        for f in (models / sub).rglob("*"):
            if not f.is_file() or f.suffix == ".part" or "_downloads" in f.parts or "__pycache__" in f.parts:
                continue
            out.append((str(f), str(f.parent.relative_to(ROOT))))
    if not out:
        raise SystemExit("models/ boş: önce scripts/download_models.py çalıştırın.")
    return out


datas = (
    collect_data_files("qfluentwidgets") + collect_data_files("lightglue")
    + [(str(ASSETS / "changescope.ico"), "app/assets"), (str(ASSETS / "changescope.png"), "app/assets")]
    + _model_files()
)

excludes = [
    # Kullanılmayan büyük Qt parçaları (paketi yüzlerce MB küçültür)
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick3D", "PySide6.QtPdf",
    "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtDesigner",
    # Geliştirme araçları ve kullanılmayan büyük paketler
    "pytest", "pytestqt", "PyInstaller", "matplotlib", "IPython",   # sympy GEREKLİ (torch.jit / lightglue)
    "tests", "scripts.download_models", "scripts.make_icon",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    # TorchScript (@torch.jit.script) derleme sırasında .py kaynağını okur; kornia/lightglue/torchvision
    # yalnızca bytecode olarak paketlenirse "Can't get source ... TorchScript requires source access"
    # hatasıyla LightGlue yüklenemez. Bu paketlerin .py dosyaları da yanına konur.
    module_collection_mode={"kornia": "pyz+py", "lightglue": "pyz+py", "torchvision": "pyz+py"},
)
pyz = PYZ(a.pure)  # noqa: F821

splash = Splash(  # noqa: F821
    str(ASSETS / "splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(194, 262),
    text_size=10,
    text_color="#94A3B8",
    minify_script=True,
    always_on_top=False,
)

version = VSVersionInfo(
    ffi=FixedFileInfo(filevers=VERSION, prodvers=VERSION),
    kids=[
        StringFileInfo([StringTable("041F04B0", [
            StringStruct("CompanyName", "ChangeScope"),
            StringStruct("FileDescription", "ChangeScope - Görüntü değişiklik tespiti"),
            StringStruct("FileVersion", ".".join(map(str, VERSION))),
            StringStruct("InternalName", "ChangeScope"),
            StringStruct("OriginalFilename", "ChangeScope.exe"),
            StringStruct("ProductName", "ChangeScope"),
            StringStruct("ProductVersion", ".".join(map(str, VERSION[:3]))),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x041F, 1200])]),
    ],
)

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.datas,
    [],
    name="ChangeScope",
    icon=str(ASSETS / "changescope.ico"),
    version=version,
    console=False,          # pencereli uygulama; çıktı %LOCALAPPDATA%/ChangeScope/logs/app.log
    disable_windowed_traceback=False,
    upx=False,              # torch DLL'lerinde UPX sorun çıkarır
    runtime_tmpdir=None,
)
