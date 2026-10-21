"""Ana orkestratör: modülleri `core/contracts.py` Protocol'leri üzerinden birleştirir.

Akış:
    ön işleme → hizalama → değişiklik tespiti → CV modellerini VRAM'den boşalt → (opsiyonel) LLM yorumu

VRAM kuralı (RTX 2060, 6 GB): CV modelleri (SuperPoint/LightGlue/DINOv2) ile llama-server aynı anda
VRAM'de olmaz. CV adımları bitince bileşenlerin `release()` metodu (varsa) çağrılır ve
`torch.cuda.empty_cache()` ile VRAM boşaltılır; LLM adımı ancak ondan sonra başlar.

LLM lazy: `RegionAnalyst` bir fabrika ile ilk ihtiyaçta oluşturulur; llama-server da (llm modülünün
sorumluluğunda) ilk `analyze()` çağrısında başlar. LLM hatası karşılaştırmayı asla bozmaz.

Bu dosya ana oturuma aittir. Modüller burayı import etmez; UI sadece `run_comparison` kullanır.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.config import AppConfig, load_config
from core.contracts import (
    AlignmentQuality,
    ChangeDetector,
    ComparisonResult,
    DetectionResult,
    ImageInput,
    Preprocessor,
    ProgressCallback,
    RegionAnalyst,
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
STAGE_LLM_ANALYZE = "llm_analyze"
STAGE_LLM_SUMMARIZE = "llm_summarize"
STAGE_DONE = "done"

STAGE_MESSAGES: dict[str, str] = {
    STAGE_PREPROCESS: "Görüntüler hazırlanıyor…",
    STAGE_REGISTER: "Eşleştiriliyor ve hizalanıyor…",
    STAGE_DETECT: "Farklar bulunuyor…",
    STAGE_RELEASE: "Bellek boşaltılıyor…",
    STAGE_LLM_ANALYZE: "Yapay zekâ yorumluyor…",
    STAGE_LLM_SUMMARIZE: "Yapay zekâ özetliyor…",
    STAGE_DONE: "Tamamlandı",
}

# Her aşamanın başladığı ilerleme oranı. LLM kullanılmazsa CV aşamaları tüm çubuğu doldurur.
_FRACTIONS_WITH_LLM = {
    STAGE_PREPROCESS: 0.00,
    STAGE_REGISTER: 0.08,
    STAGE_DETECT: 0.35,
    STAGE_RELEASE: 0.50,
    STAGE_LLM_ANALYZE: 0.52,
    STAGE_LLM_SUMMARIZE: 0.92,
    STAGE_DONE: 1.00,
}
_FRACTIONS_CV_ONLY = {
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


AnalystFactory = Callable[[], RegionAnalyst]


def free_gpu_memory() -> None:
    """Python çöpünü toplar ve CUDA varsa PyTorch önbellek VRAM'ini işletim sistemine iade eder."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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
        analyst_factory: İlk LLM ihtiyacında `RegionAnalyst` üreten fabrika (None → LLM yok).
        config: Uygulama ayarları (None → `load_config()`).
    """

    preprocessor: Preprocessor
    registrar: Registrar
    detector: ChangeDetector
    analyst_factory: AnalystFactory | None = None
    config: AppConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_config()
        self._analyst: RegionAnalyst | None = None
        self._analyst_lock = threading.Lock()

    # -- LLM ---------------------------------------------------------------

    def get_analyst(self) -> RegionAnalyst | None:
        """Analist örneğini ilk çağrıda oluşturur ve önbelleğe alır. Fabrika hatası → None."""
        if self.analyst_factory is None:
            return None
        with self._analyst_lock:
            if self._analyst is None:
                try:
                    self._analyst = self.analyst_factory()
                except Exception:
                    log.exception("RegionAnalyst oluşturulamadı; LLM'siz devam ediliyor.")
                    return None
            return self._analyst

    def release_cv_models(self) -> None:
        """CV bileşenlerinin modellerini bırakır ve VRAM'i boşaltır (LLM'den önce çağrılır)."""
        for component in (self.preprocessor, self.registrar, self.detector):
            if isinstance(component, ReleasesResources):
                try:
                    component.release()
                except Exception:
                    log.exception("%s.release() başarısız", type(component).__name__)
        free_gpu_memory()

    def shutdown(self) -> None:
        """Uygulama kapanırken: CV modellerini ve (oluşturulduysa) analisti serbest bırakır."""
        self.release_cv_models()
        analyst = self._analyst
        if isinstance(analyst, ReleasesResources):
            try:
                analyst.release()
            except Exception:
                log.exception("Analist kapatılamadı")

    # -- Ana akış ----------------------------------------------------------

    def run(
        self,
        before: ImageInput,
        after: ImageInput,
        progress: ProgressCallback | None = None,
        use_llm: bool = False,
        cancel: threading.Event | None = None,
    ) -> ComparisonResult:
        """İki görüntüyü karşılaştırır.

        Args:
            before: Referans (önceki) görüntü; tüm koordinatlar bunun sisteminde.
            after: Sonraki görüntü; before'a hizalanır.
            progress: `(stage, fraction, message)` geri çağrısı. Pipeline'ın çalıştığı thread'den
                çağrılır; UI tarafı sinyal ile ana thread'e aktarmalıdır.
            use_llm: True ise ve LLM ayarlarda açıksa bölgeler yorumlanır ve özet üretilir.
            cancel: Set edilirse aşamalar arasında `ComparisonCancelled` fırlatılır.

        Returns:
            `ComparisonResult`. Hizalama FAILED ise `detection=None`.

        Raises:
            ComparisonCancelled: `cancel` set edildiğinde.
            Exception: Ön işleme/hizalama/tespit hataları (ör. bozuk dosya) çağırana iletilir.
        """
        assert self.config is not None
        report = progress or _noop_progress
        llm_wanted = use_llm and self.config.llm.enabled and self.analyst_factory is not None
        fractions = _FRACTIONS_WITH_LLM if llm_wanted else _FRACTIONS_CV_ONLY
        t0 = time.perf_counter()

        def stage(name: str, message: str | None = None) -> None:
            if cancel is not None and cancel.is_set():
                raise ComparisonCancelled()
            report(name, fractions[name], message or STAGE_MESSAGES[name])

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
                stage(STAGE_DETECT)
                detection = self.detector.detect(registration)
        finally:
            # Başarı/hata/iptal fark etmeksizin CV modelleri LLM'den önce VRAM'den çıkar.
            if self.config.runtime.release_cv_models_before_llm:
                report(STAGE_RELEASE, fractions[STAGE_RELEASE], STAGE_MESSAGES[STAGE_RELEASE])
                self.release_cv_models()

        summary: str | None = None
        llm_used = False
        if llm_wanted and detection is not None and detection.regions:
            detection, summary, llm_used = self._run_llm(registration, detection, stage)

        report(STAGE_DONE, 1.0, STAGE_MESSAGES[STAGE_DONE])
        return ComparisonResult(
            registration=registration,
            detection=detection,
            llm_used=llm_used,
            elapsed_s=time.perf_counter() - t0,
            summary=summary,
        )

    def _run_llm(
        self,
        registration: RegistrationResult,
        detection: DetectionResult,
        stage: Callable[..., None],
    ) -> tuple[DetectionResult, str | None, bool]:
        """LLM adımı. Hata olursa CV sonucu aynen döner (`llm_used=False`); iptal yukarı iletilir."""
        analyst = self.get_analyst()
        if analyst is None:
            return detection, None, False
        try:
            if not analyst.is_available():
                log.info("LLM kullanılamıyor (exe/model yok); yorum atlandı.")
                return detection, None, False
            stage(STAGE_LLM_ANALYZE)
            analyzed = analyst.analyze(registration, detection)
            stage(STAGE_LLM_SUMMARIZE)
            summary = analyst.summarize(registration, analyzed)
            return analyzed, summary, True
        except ComparisonCancelled:
            raise
        except Exception:
            log.exception("LLM adımı başarısız; CV sonuçlarıyla devam ediliyor.")
            return detection, None, False


# ---------------------------------------------------------------------------
# UI giriş noktası
# ---------------------------------------------------------------------------

_default_pipeline: Pipeline | None = None
_default_lock = threading.Lock()


def build_default_pipeline(config: AppConfig | None = None) -> Pipeline:
    """Gerçek modüllerle pipeline kurar.

    Modeller burada YÜKLENMEZ; her bileşen ilk kullanımda lazy yükler. LLM (llm-analyst) Faz 3'te
    `analyst_factory` olarak bağlanacak; şimdilik yok.
    """
    from core.detection import create_detector
    from core.registration import create_preprocessor, create_registrar

    config = config or load_config()
    return Pipeline(
        preprocessor=create_preprocessor(config),
        registrar=create_registrar(config),
        detector=create_detector(config),
        analyst_factory=None,
        config=config,
    )


def set_default_pipeline(pipeline: Pipeline | None) -> None:
    """`run_comparison`'ın kullanacağı pipeline'ı ayarlar (test / sahte pipeline için)."""
    global _default_pipeline
    with _default_lock:
        _default_pipeline = pipeline


def get_default_pipeline() -> Pipeline:
    """Varsayılan pipeline'ı döndürür; yoksa `build_default_pipeline()` ile bir kez kurar."""
    global _default_pipeline
    with _default_lock:
        if _default_pipeline is None:
            _default_pipeline = build_default_pipeline()
        return _default_pipeline


def run_comparison(
    before: ImageInput,
    after: ImageInput,
    progress: ProgressCallback,
    use_llm: bool,
    cancel: threading.Event | None = None,
) -> ComparisonResult:
    """UI'ın çağırdığı tek fonksiyon. Arka plan thread'inde çağrılmalıdır (UI thread'inde DEĞİL).

    Ayrıntılar için `Pipeline.run`.
    """
    return get_default_pipeline().run(before, after, progress, use_llm, cancel)
