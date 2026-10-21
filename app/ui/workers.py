"""Arka plan iş parçacıkları: pipeline çalıştırma ve aygıt bilgisi. UI thread'i asla bloklanmaz."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QThread, Signal

from core import pipeline as core_pipeline
from core.contracts import ComparisonResult, ImageInput

log = logging.getLogger(__name__)


class ComparisonWorker(QThread):
    """`core.pipeline.run_comparison`'ı ayrı thread'de çalıştırır.

    Sinyaller (UI thread'ine kuyruklanır):
        progress(stage, fraction, message)
        succeeded(result)
        cancelled()
        unavailable(message)   analiz modülleri kurulu değil (NotImplementedError)
        failed(message)        beklenmeyen hata
    """

    progress = Signal(str, float, str)
    succeeded = Signal(object)
    cancelled = Signal()
    unavailable = Signal(str)
    failed = Signal(str)

    def __init__(self, before: ImageInput, after: ImageInput, use_llm: bool, parent=None) -> None:
        super().__init__(parent)
        self._before, self._after, self._use_llm = before, after, use_llm
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        """İptal ister; pipeline aşamalar arasında `ComparisonCancelled` fırlatır."""
        self.cancel_event.set()

    def _report(self, stage: str, fraction: float, message: str = "") -> None:
        self.progress.emit(stage, float(fraction), message)

    def run(self) -> None:
        try:
            result: ComparisonResult = core_pipeline.run_comparison(
                self._before, self._after, self._report, self._use_llm, cancel=self.cancel_event
            )
        except core_pipeline.ComparisonCancelled:
            self.cancelled.emit()
        except NotImplementedError as exc:
            log.warning("Analiz modülleri kurulu değil: %s", exc)
            self.unavailable.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - kullanıcıya gösterilir
            log.exception("Karşılaştırma başarısız")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.succeeded.emit(result)


class DeviceProbe(QThread):
    """torch'u UI'ı dondurmadan yükleyip GPU/CPU bilgisini üretir (`core.config.resolve_device`)."""

    resolved = Signal(str)

    def __init__(self, preference: str = "auto", parent=None) -> None:
        super().__init__(parent)
        self._preference = preference

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
        self.resolved.emit(text)


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
