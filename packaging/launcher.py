"""PyInstaller giriş noktası: paketlenmiş ChangeScope.exe buradan başlar.

Pencereli (konsolsuz) exe'de `sys.stdout/stderr` None olur; logging ve kütüphanelerin uyarıları çökmesin
diye ikisi de `%LOCALAPPDATA%/ChangeScope/logs/app.log` dosyasına yönlendirilir.
"""

from __future__ import annotations

import sys


def _redirect_std_streams() -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    from core.config import user_data_dir

    log_dir = user_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = open(log_dir / "app.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _selftest(pair_dir: str) -> int:
    """Paketlenmiş exe'de gerçek pipeline'ı (torch/LightGlue/DINOv2) bir çift üzerinde çalıştırır.

    `ChangeScope.exe --selftest <klasör>`: klasörde before.* ve after.* olmalı. Sonuç günlüğe yazılır;
    çıkış kodu 0 = hizalama FAILED değil ve en az bir bölge bulundu.
    """
    from pathlib import Path

    from core.contracts import AlignmentQuality, ImageInput
    from core.pipeline import build_default_pipeline

    d = Path(pair_dir)
    before = next(d.glob("before.*"))
    after = next(d.glob("after.*"))
    pipe = build_default_pipeline()
    try:
        result = pipe.run(ImageInput(before), ImageInput(after))
    finally:
        pipe.shutdown()
    det = result.detection
    n = len(det.regions) if det else 0
    print(f"SELFTEST quality={result.registration.quality.value} method={det.method if det else '-'} "
          f"regions={n} notes={result.registration.notes}", flush=True)
    return 0 if result.registration.quality is not AlignmentQuality.FAILED and n > 0 else 1


def _splash_text(text: str) -> None:
    """Açılış ekranındaki durum metnini günceller (splash yoksa sessizce geçer)."""
    try:
        import pyi_splash  # type: ignore[import-not-found]

        pyi_splash.update_text(text)
    except (ImportError, RuntimeError):
        pass


if __name__ == "__main__":
    _redirect_std_streams()
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        from app.main import _close_splash

        _close_splash()
        try:
            sys.exit(_selftest(sys.argv[2]))
        except Exception:
            import traceback

            traceback.print_exc()
            sys.exit(2)
    _splash_text("Arayüz yükleniyor…")
    from app.main import main

    sys.exit(main())
