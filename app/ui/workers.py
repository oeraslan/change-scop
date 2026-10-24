"""Arka plan iş parçacıkları: pipeline çalıştırma ve aygıt bilgisi. UI thread'i asla bloklanmaz."""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage, QImageReader

from core import pipeline as core_pipeline
from core.contracts import ComparisonResult

log = logging.getLogger(__name__)


PREVIEW_MAX_SIDE = 3000


def load_preview(path: Path, max_side: int = PREVIEW_MAX_SIDE) -> QImage | None:
    """Büyük dosyaları ölçekleyerek okuyan önizleme yükleyici (worker thread'inde çağrılır)."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)       # EXIF yönü (küçük resim/ön işlemeyle tutarlı)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_side:
        k = max_side / max(size.width(), size.height())
        reader.setScaledSize(size * k)
    img = reader.read()
    return None if img.isNull() else img


class ComparisonWorker(QThread):
    """Karşılaştırmanın TÜM ağır işini UI dışında yapar.

    Sırayla: (1) önbellek araması + kayıttan okuma, (2) büyük önizleme yükleme, (3) pipeline,
    (4) sonucu diske kaydetme. Kütüphane/SQLite erişimi güvenlidir: `LibraryRepository` tek bağlantı
    + RLock ile thread'ler arası paylaşılır (`check_same_thread=False`).

    Sinyaller (UI thread'ine kuyruklanır):
        previews(before_qimage, after_qimage)   pipeline başlamadan önce (önbellekte yoksa)
        progress(stage, fraction, message)
        succeeded(result, record, cached, save_error)
        cancelled()
        failed(message)
    """

    previews = Signal(object, object)
    progress = Signal(str, float, str)
    succeeded = Signal(object, object, bool, str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self, library, before_rec, after_rec, force: bool = False,
        load_previews: bool = True, use_cache: bool = True, parent=None,
    ) -> None:
        super().__init__(parent)
        self._library, self._before_rec, self._after_rec = library, before_rec, after_rec
        self._force = force
        self._load_previews, self._use_cache = load_previews, use_cache
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        """İptal ister; pipeline aşamalar arasında `ComparisonCancelled` fırlatır."""
        self.cancel_event.set()

    def _report(self, stage: str, fraction: float, message: str = "") -> None:
        self.progress.emit(stage, float(fraction), message)

    def _try_cached(self):
        if self._force or not self._use_cache:
            return None
        # Önbellek anahtarı maske özetini içerir: maske değiştiyse eski sonuç bulunmaz.
        # ... ve İSTENEN fark yöntemini (fallback'te sonsuz yeniden hesaplama olmasın diye istenen yöntem).
        try:
            method = core_pipeline.get_default_pipeline().config.detection.method
        except Exception:  # noqa: BLE001
            method = None
        cached = self._library.find_comparison(self._before_rec, self._after_rec, method)
        if cached is None:
            return None
        try:
            return cached, self._library.load_comparison(cached)
        except Exception as exc:  # noqa: BLE001 - bozuk kayıt: yeniden hesapla
            log.warning("Kayıtlı sonuç okunamadı (%s); yeniden hesaplanacak", exc)
            return None

    def _save(self, result: ComparisonResult, mask_hash: str | None):
        """Sonucu kaydeder; (kayıt, hata metni) döndürür."""
        try:
            return self._library.save_comparison(
                self._before_rec, self._after_rec, result, mask_hash=mask_hash), ""
        except Exception as exc:  # noqa: BLE001
            log.exception("Sonuç kaydedilemedi")
            return None, str(exc)

    def run(self) -> None:
        try:
            hit = self._try_cached()
            if hit is not None:
                if self.cancel_event.is_set():
                    self.cancelled.emit()
                else:
                    self.succeeded.emit(hit[1], hit[0], True, "")
                return
            if self._load_previews:
                self.previews.emit(load_preview(self._library.image_path(self._before_rec)),
                                   load_preview(self._library.image_path(self._after_rec)))
            # "Önce" görüntüsünün güncel maskesi (PNG okuma burada, worker thread'inde).
            fresh = self._library.repo.get_image(self._before_rec.id) or self._before_rec
            mask = self._library.load_mask(fresh)
            mask_hash = fresh.mask_hash if mask is not None else ""
            result: ComparisonResult = core_pipeline.run_comparison(
                self._library.to_image_input(self._before_rec),
                self._library.to_image_input(self._after_rec),
                self._report, cancel=self.cancel_event, ignore_mask=mask,
            )
            if self.cancel_event.is_set():
                self.cancelled.emit()
                return
            record, save_error = self._save(result, mask_hash)
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.succeeded.emit(result, record, False, save_error)
        except core_pipeline.ComparisonCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - kullanıcıya gösterilir
            log.exception("Karşılaştırma başarısız")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MaskWorker(QThread):
    """Maske PNG okuma/yazma işini UI dışında yapar.

    `mode="load"`: loaded(mask_or_None, record) · `mode="save"`: saved(record) · hata: failed(msg)
    """

    loaded = Signal(object, object)
    saved = Signal(object)
    failed = Signal(str)

    def __init__(self, library, image_rec, mode: str, mask=None, parent=None) -> None:
        super().__init__(parent)
        self._library, self._rec, self._mode, self._mask = library, image_rec, mode, mask

    def run(self) -> None:
        try:
            if self._mode == "load":
                fresh = self._library.repo.get_image(self._rec.id) or self._rec
                self.loaded.emit(self._library.load_mask(fresh), fresh)
            else:
                self.saved.emit(self._library.save_mask(self._rec, self._mask))
        except Exception as exc:  # noqa: BLE001
            log.exception("Maske işlemi başarısız")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DeviceProbe(QObject):
    """GPU/CPU bilgisini üretir, sonra ağır modülleri arka planda ön ısıtır (ilk karşılaştırma takılmasın).

    Bir QThread DEĞİL, daemon Python thread'i kullanır: torch içe aktarması saniyeler sürebilir ve
    uygulama kapanırken çalışan bir QThread süreci çökertirdi.
    """

    resolved = Signal(str)

    def __init__(self, preference: str = "auto", parent=None) -> None:
        super().__init__(parent)
        self._preference = preference
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Arka plan thread'ini başlatır."""
        self._thread = threading.Thread(target=self.run, name="device-probe", daemon=True)
        self._thread.start()

    def isRunning(self) -> bool:  # noqa: N802 - QThread uyumlu arayüz
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout_ms: int = 0) -> bool:
        """Thread'in bitmesini (en çok `timeout_ms`) bekler; bitti mi?"""
        if self._thread is not None:
            self._thread.join(timeout_ms / 1000)
        return not self.isRunning()

    def run(self) -> None:
        from core.config import resolve_device

        device = resolve_device(self._preference)  # type: ignore[arg-type]
        text = "CPU"
        if device == "cuda":
            try:
                import torch

                text = f"GPU · {torch.cuda.get_device_name(0)}"
            except Exception:  # noqa: BLE001
                text = "GPU"
        try:
            self.resolved.emit(text)
        except RuntimeError:        # pencere kapanmış
            return
        warm_up_imports()


def warm_up_imports() -> None:
    """torch, lightglue ve çekirdek modülleri önceden içe aktarır (model YÜKLEMEZ)."""
    for name in ("torch", "lightglue", "core.registration", "core.detection"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - kurulu olmayabilir
            log.debug("Ön ısıtma atlandı (%s): %s", name, exc)


class ImportWorker(QThread):
    """Dosyaları kütüphaneye kopyalar + thumbnail üretir (UI donmasın diye arka planda)."""

    progress = Signal(int, int)             # (tamamlanan, toplam)
    imported = Signal(list, list)           # (ImageRecord listesi, hata mesajları)

    def __init__(self, library, paths, folder_id: int, source, fallback_date, parent=None) -> None:
        super().__init__(parent)
        self._library, self._paths = library, list(paths)
        self._folder_id, self._source, self._fallback = folder_id, source, fallback_date

    def run(self) -> None:
        run_import(self._library, self._paths, self._folder_id, self._source, self._fallback,
                   on_progress=lambda i, n: self.progress.emit(i, n), emit=self.imported.emit)


def run_import(library, paths, folder_id: int, source, fallback_date, on_progress=None, emit=None):
    """Eşzamanlı içe aktarma çekirdeği (worker ve testler paylaşır).

    EXIF tarihi olan dosyalar o tarihi, olmayanlar `fallback_date` (None → dosya tarihi) kullanır.
    """
    from app.storage.library import read_exif_date

    records, errors = [], []
    total = len(paths)
    for i, path in enumerate(paths, start=1):
        try:
            captured = read_exif_date(path) or fallback_date
            records.append(library.import_image(path, folder_id, captured, source))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")
        if on_progress:
            on_progress(i, total)
    if emit:
        emit(records, errors)
    return records, errors
