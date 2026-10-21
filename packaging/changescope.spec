# PyInstaller onedir paketi: dist/ChangeScope/ChangeScope.exe
# Derleme: build.ps1 (bu dosyayı doğrudan da kullanabilirsiniz: pyinstaller packaging/changescope.spec)
#
# models/ ve bin/ pakete GÖMÜLMEZ; build.ps1 bunları exe'nin yanına ayrı klasör olarak kopyalar
# (core.config.app_root() paketlenmişken exe klasörünü döndürür).
# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821  (SPECPATH PyInstaller tarafından tanımlanır)

# torchvision 0.29'un derlenmiş op'ları (_C_stable.pyd: nms, deform_conv2d...) hazır hook'ta YOK;
# eksik kalınca LightGlue "torchvision::nms does not exist" ile yüklenemiyor ve SIFT'e düşülüyor.
_tv_dir = Path(importlib.util.find_spec("torchvision").origin).parent
binaries = [(str(p), "torchvision") for p in _tv_dir.glob("*.pyd")] + collect_dynamic_libs("torchvision")

hiddenimports = (
    # Pipeline modülleri lazy import ediliyor; analiz bunları göremez.
    collect_submodules("core")
    + collect_submodules("app")
    + collect_submodules("lightglue")
    + collect_submodules("qfluentwidgets")
    + ["skimage.metrics", "skimage.exposure", "httpx",
       "scripts.make_synthetic_pair"]   # --demo örnek kütüphanesi için
)

datas = collect_data_files("qfluentwidgets") + collect_data_files("lightglue")

excludes = [
    # Kullanılmayan büyük Qt parçaları (paketi yüzlerce MB küçültür)
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick3D", "PySide6.QtPdf",
    "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    # Geliştirme araçları
    "pytest", "pytestqt", "PyInstaller", "tkinter", "matplotlib", "IPython",
    "tests", "scripts.download_models",
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

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChangeScope",
    console=False,          # pencereli uygulama; çıktı %LOCALAPPDATA%/ChangeScope/logs/app.log
    disable_windowed_traceback=False,
    upx=False,              # torch/CUDA DLL'lerinde UPX sorun çıkarır
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ChangeScope",
)
