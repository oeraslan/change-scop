"""Ana orkestratör: modülleri `core/contracts.py` Protocol'leri üzerinden birleştirir.

Akış:
    ön işleme → hizalama → değişiklik tespiti → CV modellerini bellekten bırak

CV adımları bitince bileşenlerin `release()` metodu (varsa) çağrılır ve CUDA varsa
`torch.cuda.empty_cache()` ile VRAM boşaltılır.

Bu dosya ana oturuma aittir. Modüller burayı import etmez; UI sadece `run_comparison` kullanır.
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from core.config import AppConfig, load_config
from core.contracts import (
    AlignmentQuality,
    ChangeDetector,
    ComparisonResult,
    DetectionResult,
    ImageInput,
    Preprocessor,
    ProgressCallback,
    Registrar,
    RegistrationResult,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aşamalar (UI'daki ilerleme metni için)
# ---------------------------------------------------------------------------

STAGE_PREPROCESS = "preprocess"
STAGE_REGISTER = "register"
STAGE_DETECT = "detect"
STAGE_RELEASE = "release"
STAGE_DONE = "done"

STAGE_MESSAGES: dict[str, str] = {
    STAGE_PREPROCESS: "Görüntüler hazırlanıyor…",
    STAGE_REGISTER: "Eşleştiriliyor ve hizalanıyor…",
    STAGE_DETECT: "Farklar bulunuyor…",
    STAGE_RELEASE: "Bellek boşaltılıyor…",
    STAGE_DONE: "Tamamlandı",
}

# Her aşamanın başladığı ilerleme oranı.
STAGE_FRACTIONS: dict[str, float] = {
    STAGE_PREPROCESS: 0.00,
    STAGE_REGISTER: 0.15,
    STAGE_DETECT: 0.65,
    STAGE_RELEASE: 0.95,
    STAGE_DONE: 1.00,
}


class ComparisonCancelled(Exception):
    """Kullanıcı karşılaştırmayı iptal etti."""


@runtime_checkable
class ReleasesResources(Protocol):
    """Opsiyonel: modellerini bellekten (özellikle VRAM'den) bırakabilen bileşen."""

    def release(self) -> None: ...


def free_gpu_memory() -> None:
    """Python çöpünü toplar ve CUDA varsa PyTorch önbellek VRAM'ini işletim sistemine iade eder."""
    gc.collect()
    # torch hiç yüklenmediyse boşaltılacak VRAM yoktur; import'u (worker thread'inde saniyeler
    # sürebilir) tetikleme.
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def apply_ignore_mask(registration: RegistrationResult, ignore_mask: np.ndarray) -> RegistrationResult:
    """`ignore_mask` (True = yok say) alanını `valid_mask`'tan çıkarılmış yeni bir sonuç döndürür.

    Maske herhangi bir çözünürlükte olabilir; `valid_mask` boyutuna en yakın komşu ile ölçeklenir.
    Orijinal `registration` değiştirilmez.
    """
    mask = np.asarray(ignore_mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"ignore_mask 2 boyutlu olmalı, gelen şekil: {mask.shape}")
    h, w = registration.valid_mask.shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return replace(registration, valid_mask=registration.valid_mask & ~mask.astype(bool))


def _noop_progress(stage: str, fraction: float, message: str = "") -> None:
    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    """Bileşenleri enjekte edilen karşılaştırma pipeline'ı.

    Args:
        preprocessor: `Preprocessor` uygulaması.
        registrar: `Registrar` uygulaması.
        detector: `ChangeDetector` uygulaması.
        config: Uygulama ayarları (None → `load_config()`).
    """

    preprocessor: Preprocessor
    registrar: Registrar
    detector: ChangeDetector
    config: AppConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_config()
        # Emekliye ayırma: configure/shutdown çağıranı çalışan koşuyu beklemez (bkz. `retire`).
        self._state_lock = threading.Lock()
        self._active_runs = 0
        self._retired = False

    def release_cv_models(self) -> None:
        """CV bileşenlerinin modellerini bırakır ve VRAM'i boşaltır."""
        for component in (self.preprocessor, self.registrar, self.detector):
            if isinstance(component, ReleasesResources):
                try:
                    component.release()
                except Exception:
                    log.exception("%s.release() başarısız", type(component).__name__)
        free_gpu_memory()

    def shutdown(self) -> None:
        """Uygulama kapanırken: CV modellerini serbest bırakır."""
        self.release_cv_models()

    def retire(self) -> bool:
        """Pipeline'ı kullanımdan kaldırır; çağıranı ASLA çalışan bir koşu kadar bekletmez.

        Çalışan koşu yoksa kaynaklar hemen bırakılır (`True`). Varsa son koşu bittiğinde, o koşunun
        thread'inde bırakılır (`False`). Emekli bir pipeline'da yeni `run` yine çalışır (modeller lazy
        yeniden yüklenir) ve bittiğinde tekrar bırakılır.
        """
        with self._state_lock:
            self._retired = True
            idle = self._active_runs == 0
        if idle:
            self.shutdown()
        return idle

    # -- Ana akış ----------------------------------------------------------

    def run(
        self,
        before: ImageInput,
        after: ImageInput,
        progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
        ignore_mask: np.ndarray | None = None,
    ) -> ComparisonResult:
        """İki görüntüyü karşılaştırır.

        Args:
            before: Referans (önceki) görüntü; tüm koordinatlar bunun sisteminde.
            after: Sonraki görüntü; before'a hizalanır.
            progress: `(stage, fraction, message)` geri çağrısı. Pipeline'ın çalıştığı thread'den
                çağrılır; UI tarafı sinyal ile ana thread'e aktarmalıdır.
            cancel: Set edilirse aşamalar arasında `ComparisonCancelled` fırlatılır.
            ignore_mask: Opsiyonel HxW bool maske, True = "bu alanı yok say". before görüntüsünün
                ekranda gösterilen (EXIF yönü düzeltilmiş) hali üzerinde, herhangi bir çözünürlükte
                çizilir; hizalama çözünürlüğüne yeniden ölçeklenir ve `valid_mask`'tan çıkarılır, yani
                o pikseller asla değişiklik sayılmaz. Hizalamayı etkilemez.

        Returns:
            `ComparisonResult`. Hizalama FAILED ise `detection=None`.

        Raises:
            ComparisonCancelled: `cancel` set edildiğinde.
            Exception: Ön işleme/hizalama/tespit hataları (ör. bozuk dosya) çağırana iletilir.
        """
        with self._state_lock:
            self._active_runs += 1
        try:
            return self._run(before, after, progress, cancel, ignore_mask)
        finally:
            with self._state_lock:
                self._active_runs -= 1
                release_now = self._retired and self._active_runs == 0
            if release_now:
                self.shutdown()

    def _run(
        self,
        before: ImageInput,
        after: ImageInput,
        progress: ProgressCallback | None,
        cancel: threading.Event | None,
        ignore_mask: np.ndarray | None,
    ) -> ComparisonResult:
        assert self.config is not None
        report = progress or _noop_progress
        t0 = time.perf_counter()

        def stage(name: str) -> None:
            if cancel is not None and cancel.is_set():
                raise ComparisonCancelled()
            report(name, STAGE_FRACTIONS[name], STAGE_MESSAGES[name])

        try:
            stage(STAGE_PREPROCESS)
            prepared_before = self.preprocessor.prepare(before)
            prepared_after = self.preprocessor.prepare(after)

            stage(STAGE_REGISTER)
            registration = self.registrar.register(prepared_before, prepared_after)

            detection: DetectionResult | None = None
            if registration.quality is AlignmentQuality.FAILED:
                log.warning("Hizalama başarısız; tespit atlandı. Notlar: %s", registration.notes)
            else:
                if ignore_mask is not None:
                    registration = apply_ignore_mask(registration, ignore_mask)
                stage(STAGE_DETECT)
                detection = self.detector.detect(registration)
        finally:
            # Başarı/hata/iptal fark etmeksizin CV modelleri bellekten bırakılır.
            if self.config.runtime.release_cv_models_after_run:
                report(STAGE_RELEASE, STAGE_FRACTIONS[STAGE_RELEASE], STAGE_MESSAGES[STAGE_RELEASE])
                self.release_cv_models()

        report(STAGE_DONE, 1.0, STAGE_MESSAGES[STAGE_DONE])
        return ComparisonResult(
            registration=registration,
            detection=detection,
            elapsed_s=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# UI giriş noktası
# ---------------------------------------------------------------------------

_default_pipeline: Pipeline | None = None
_default_config: AppConfig | None = None
_default_lock = threading.Lock()


def build_default_pipeline(config: AppConfig | None = None) -> Pipeline:
    """Gerçek modüllerle pipeline kurar. Modeller burada YÜKLENMEZ; her bileşen ilk kullanımda lazy yükler."""
    from core.detection import create_detector
    from core.registration import create_preprocessor, create_registrar

    config = config or load_config()
    return Pipeline(
        preprocessor=create_preprocessor(config),
        registrar=create_registrar(config),
        detector=create_detector(config),
        config=config,
    )


def set_default_pipeline(pipeline: Pipeline | None) -> None:
    """`run_comparison`'ın kullanacağı pipeline'ı ayarlar (test / sahte pipeline için).

    Önceki pipeline KAPATILMAZ; kapatmak için `shutdown_default_pipeline()` kullanın.
    """
    global _default_pipeline
    with _default_lock:
        _default_pipeline = pipeline


def configure_default_pipeline(config: AppConfig) -> None:
    """Ayarlar değişince çağrılır: eski pipeline'ı kapatır, yenisini `config` ile kurar.

    Modeller lazy yüklendiği için bu çağrı hızlıdır; bir sonraki karşılaştırma modelleri yeniden yükler.
    Karşılaştırma çalışırken de güvenle çağrılabilir ve onu BEKLEMEZ: çalışan koşu eski ayarla biter,
    eski pipeline'ın kaynakları o koşu bitince bırakılır.
    """
    global _default_pipeline, _default_config
    with _default_lock:
        old, _default_pipeline = _default_pipeline, None
        _default_config = config
    if old is not None:
        old.retire()   # çalışan koşu varsa onu beklemez; koşu bitince bırakılır


def shutdown_default_pipeline() -> None:
    """Uygulama kapanırken çağrılır: modelleri bırakır.

    Pipeline hiç kurulmadıysa hiçbir şey yapmaz (kapanışta model yüklemez). Çalışan karşılaştırma
    varsa onu beklemez; kaynaklar o koşu bitince bırakılır (çağıran iptal + kısa bekleme yapmalıdır).
    """
    global _default_pipeline
    with _default_lock:
        old, _default_pipeline = _default_pipeline, None
    if old is not None:
        old.retire()   # çalışan koşu varsa onu beklemez; koşu bitince bırakılır


def get_default_pipeline() -> Pipeline:
    """Varsayılan pipeline'ı döndürür; yoksa son `configure_default_pipeline` ayarıyla (yoksa
    `load_config()`) bir kez kurar."""
    global _default_pipeline
    with _default_lock:
        if _default_pipeline is None:
            _default_pipeline = build_default_pipeline(_default_config)
        return _default_pipeline


def run_comparison(
    before: ImageInput,
    after: ImageInput,
    progress: ProgressCallback,
    cancel: threading.Event | None = None,
    ignore_mask: np.ndarray | None = None,
) -> ComparisonResult:
    """UI'ın çağırdığı ana fonksiyon. Arka plan thread'inde çağrılmalıdır (UI thread'inde DEĞİL).

    Ayrıntılar için `Pipeline.run`.
    """
    return get_default_pipeline().run(before, after, progress, cancel, ignore_mask)
