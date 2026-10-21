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
import sys
import threading
import time
from collections.abc import Callable
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
PartialResultCallback = Callable[[ComparisonResult], None]


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
        # Emekliye ayırma: configure/shutdown çağıranı çalışan koşuyu beklemez (bkz. `retire`).
        self._state_lock = threading.Lock()
        self._active_runs = 0
        self._retired = False

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

    def release_llm(self) -> None:
        """Çalışan LLM sunucusunu durdurur (VRAM kuralı: CV adımlarından ÖNCE çağrılır)."""
        analyst = self._analyst
        if analyst is None or not getattr(analyst, "is_server_running", False):
            return
        if isinstance(analyst, ReleasesResources):
            try:
                analyst.release()
            except Exception:
                log.exception("LLM sunucusu durdurulamadı")
        free_gpu_memory()

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
        use_llm: bool = False,
        cancel: threading.Event | None = None,
        ignore_mask: np.ndarray | None = None,
        on_cv_result: PartialResultCallback | None = None,
    ) -> ComparisonResult:
        """İki görüntüyü karşılaştırır.

        Args:
            before: Referans (önceki) görüntü; tüm koordinatlar bunun sisteminde.
            after: Sonraki görüntü; before'a hizalanır.
            progress: `(stage, fraction, message)` geri çağrısı. Pipeline'ın çalıştığı thread'den
                çağrılır; UI tarafı sinyal ile ana thread'e aktarmalıdır.
            use_llm: True ise ve LLM ayarlarda açıksa bölgeler yorumlanır ve özet üretilir.
            cancel: Set edilirse aşamalar arasında `ComparisonCancelled` fırlatılır.
            ignore_mask: Opsiyonel HxW bool maske, True = "bu alanı yok say". before görüntüsünün
                ekranda gösterilen (EXIF yönü düzeltilmiş) hali üzerinde, herhangi bir çözünürlükte
                çizilir; hizalama çözünürlüğüne yeniden ölçeklenir ve `valid_mask`'tan çıkarılır, yani
                o pikseller asla değişiklik sayılmaz. Hizalamayı etkilemez.
            on_cv_result: LLM adımı yapılacaksa, CV sonucu hazır olur olmaz (LLM'den ÖNCE) bir kez
                `llm_used=False` sonuçla çağrılır; UI kutuları hemen çizip yorumu sonra doldurabilir.
                Pipeline thread'inden çağrılır. LLM adımı yoksa çağrılmaz.

        Returns:
            `ComparisonResult`. Hizalama FAILED ise `detection=None`.

        Raises:
            ComparisonCancelled: `cancel` set edildiğinde.
            Exception: Ön işleme/hizalama/tespit hataları (ör. bozuk dosya) çağırana iletilir.
        """
        with self._state_lock:
            self._active_runs += 1
        try:
            return self._run(before, after, progress, use_llm, cancel, ignore_mask, on_cv_result)
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
        use_llm: bool,
        cancel: threading.Event | None,
        ignore_mask: np.ndarray | None,
        on_cv_result: PartialResultCallback | None = None,
    ) -> ComparisonResult:
        assert self.config is not None
        report = progress or _noop_progress
        llm_wanted = use_llm and self.config.llm.enabled and self.analyst_factory is not None
        fractions = _FRACTIONS_WITH_LLM if llm_wanted else _FRACTIONS_CV_ONLY
        t0 = time.perf_counter()

        def stage(name: str, message: str | None = None) -> None:
            if cancel is not None and cancel.is_set():
                raise ComparisonCancelled()
            report(name, fractions[name], message or STAGE_MESSAGES[name])

        # VRAM kuralı: önceki karşılaştırmadan açık kalan llama-server CV modelleriyle çakışmasın.
        self.release_llm()
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
            # Başarı/hata/iptal fark etmeksizin CV modelleri LLM'den önce VRAM'den çıkar.
            if self.config.runtime.release_cv_models_before_llm:
                report(STAGE_RELEASE, fractions[STAGE_RELEASE], STAGE_MESSAGES[STAGE_RELEASE])
                self.release_cv_models()

        summary: str | None = None
        llm_used = False
        if llm_wanted and detection is not None and detection.regions:
            if on_cv_result is not None:
                try:
                    on_cv_result(ComparisonResult(registration, detection, False,
                                                  time.perf_counter() - t0, None))
                except Exception:
                    log.exception("on_cv_result geri çağrısı başarısız")
            detection, summary, llm_used = self._run_llm(registration, detection, stage, report, fractions)

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
        report: ProgressCallback = _noop_progress,
        fractions: dict[str, float] = _FRACTIONS_WITH_LLM,
    ) -> tuple[DetectionResult, str | None, bool]:
        """LLM adımı. Hata olursa CV sonucu aynen döner (`llm_used=False`); iptal yukarı iletilir.

        Analist bölgeleri yerinde doldurabildiği için KOPYA üzerinde çalışılır; `on_cv_result` ile UI'a
        verilmiş CV sonucu başka thread'den değiştirilmez.
        """
        analyst = self.get_analyst()
        if analyst is None:
            return detection, None, False
        work = replace(detection, regions=[replace(r) for r in detection.regions])
        f0 = fractions.get(STAGE_LLM_ANALYZE, 0.5)
        f1 = fractions.get(STAGE_LLM_SUMMARIZE, 0.9)

        def on_region(done: int, total: int, message: str = "") -> None:
            frac = f0 + (f1 - f0) * (done / total if total else 0.0)
            report(STAGE_LLM_ANALYZE, frac, message or STAGE_MESSAGES[STAGE_LLM_ANALYZE])

        has_hook = hasattr(analyst, "progress_callback")
        try:
            if not analyst.is_available():
                log.info("LLM kullanılamıyor (exe/model yok); yorum atlandı.")
                return detection, None, False
            stage(STAGE_LLM_ANALYZE)
            if has_hook:
                analyst.progress_callback = on_region
            analyzed = analyst.analyze(registration, work)
            stage(STAGE_LLM_SUMMARIZE)
            summary = analyst.summarize(registration, analyzed)
            return analyzed, summary, True
        except ComparisonCancelled:
            raise
        except Exception:
            log.exception("LLM adımı başarısız; CV sonuçlarıyla devam ediliyor.")
            return detection, None, False
        finally:
            if has_hook:
                analyst.progress_callback = None

    def interpret(
        self,
        result: ComparisonResult,
        progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> ComparisonResult:
        """Var olan bir CV sonucunu ("Yorumla" düğmesi) hizalama/tespit tekrarlanmadan LLM ile yorumlar.

        Yüklü CV modelleri önce bırakılır. LLM yoksa/başarısızsa sonuç `llm_used=False` ile aynen döner.
        """
        assert self.config is not None
        report = progress or _noop_progress
        if (result.detection is None or not result.detection.regions or not self.config.llm.enabled
                or self.analyst_factory is None):
            return result
        with self._state_lock:
            self._active_runs += 1
        try:
            def stage(name: str, message: str | None = None) -> None:
                if cancel is not None and cancel.is_set():
                    raise ComparisonCancelled()
                report(name, _FRACTIONS_WITH_LLM[name], message or STAGE_MESSAGES[name])

            t0 = time.perf_counter()
            self.release_cv_models()
            detection, summary, used = self._run_llm(result.registration, result.detection, stage, report)
            report(STAGE_DONE, 1.0, STAGE_MESSAGES[STAGE_DONE])
            return ComparisonResult(result.registration, detection, used,
                                    result.elapsed_s + time.perf_counter() - t0, summary)
        finally:
            with self._state_lock:
                self._active_runs -= 1
                release_now = self._retired and self._active_runs == 0
            if release_now:
                self.shutdown()


# ---------------------------------------------------------------------------
# UI giriş noktası
# ---------------------------------------------------------------------------

_default_pipeline: Pipeline | None = None
_default_config: AppConfig | None = None
_default_lock = threading.Lock()


def build_default_pipeline(config: AppConfig | None = None) -> Pipeline:
    """Gerçek modüllerle pipeline kurar.

    Modeller burada YÜKLENMEZ; her bileşen ilk kullanımda lazy yükler. LLM analisti (ve llama-server)
    yalnızca `use_llm=True` ile ilk yorum istendiğinde oluşturulur/başlatılır.
    """
    from core.detection import create_detector
    from core.registration import create_preprocessor, create_registrar

    config = config or load_config()

    def analyst_factory() -> RegionAnalyst:
        from core.llm import create_analyst   # llm modülü yalnızca ilk yorum isteğinde yüklenir

        return create_analyst(config)

    return Pipeline(
        preprocessor=create_preprocessor(config),
        registrar=create_registrar(config),
        detector=create_detector(config),
        analyst_factory=analyst_factory,
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
    """Uygulama kapanırken çağrılır: modelleri bırakır ve (Faz 3) llama-server'ı sonlandırır.

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
    use_llm: bool,
    cancel: threading.Event | None = None,
    ignore_mask: np.ndarray | None = None,
    on_cv_result: PartialResultCallback | None = None,
) -> ComparisonResult:
    """UI'ın çağırdığı ana fonksiyon. Arka plan thread'inde çağrılmalıdır (UI thread'inde DEĞİL).

    Ayrıntılar için `Pipeline.run`.
    """
    return get_default_pipeline().run(before, after, progress, use_llm, cancel, ignore_mask, on_cv_result)


def interpret_comparison(
    result: ComparisonResult,
    progress: ProgressCallback,
    cancel: threading.Event | None = None,
) -> ComparisonResult:
    """"Yorumla" düğmesi: var olan CV sonucunu yalnızca LLM ile yorumlar (arka plan thread'inde çağır)."""
    return get_default_pipeline().interpret(result, progress, cancel)


def llm_status() -> dict[str, object]:
    """Durum çubuğu için LLM durumu: available / running / ngl / model. Sunucuyu BAŞLATMAZ."""
    cfg = _default_config or load_config()
    spec = cfg.llm.model_spec
    files_ok = (cfg.paths.llama_server_exe.is_file() and spec.gguf_path(cfg.paths.models_dir).is_file()
                and spec.mmproj_path(cfg.paths.models_dir).is_file())
    status: dict[str, object] = {"model": spec.display_name, "available": bool(cfg.llm.enabled and files_ok),
                                 "running": False, "ngl": None}
    pipe = _default_pipeline
    analyst = pipe._analyst if pipe is not None else None
    if analyst is not None:
        status["running"] = bool(getattr(analyst, "is_server_running", False))
        info = getattr(analyst, "last_start_info", None)
        status["ngl"] = getattr(info, "ngl", None) if info is not None else None
    return status
