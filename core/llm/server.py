"""Gömülü `llama-server.exe` yaşam döngüsü yönetimi.

Sorumluluklar:
    - Boş port bulma, süreci `CREATE_NO_WINDOW` ile başlatma, `/health` yoklama.
    - `-ngl` "auto" hesabı (boş VRAM - mmproj - KV cache - güvenlik payı).
    - Başlatma başarısız olursa ngl'yi azaltıp bir kez daha, sonra CPU'ya (ngl 0) düşme.
    - `atexit` + `stop()` ile sürecin kesin sonlandırılması (Windows Job Object ile ek güvence).

Hiçbir yöntem istisna fırlatmaz: başarısızlıkta `start()` None döner, `LlamaServer.info.error` doldurulur
ve uygulama LLM'siz çalışmaya devam eder.
"""

from __future__ import annotations

import atexit
import logging
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from core.config import AppConfig, LLMModelSpec, user_data_dir

log = logging.getLogger(__name__)

_MB = 1024 * 1024
# llama.cpp'nin hesaplama (compute) tamponları için kaba pay; vision encoder çalışırken ek yük.
COMPUTE_BUFFER_MB = 150
# Gemma SWA katmanlarında KV, pencere + ubatch kadar tutulur.
_UBATCH = 512


# ---------------------------------------------------------------------------
# VRAM / ngl hesabı
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVProfile:
    """Modelin KV cache bellek profili (f16 K+V).

    SWA (kayan pencere) katmanlar KV'yi sadece `window + ubatch` token için tutar; global katmanlar
    bağlamın tamamı için.
    """

    swa_layers: int
    swa_bytes_per_token: int        # katman başına, K+V, f16
    swa_window: int
    global_layers: int
    global_bytes_per_token: int


# Gemma 4 12B: 48 katman, 40 SWA (8 kv head x 256 dim) + 8 global (1 kv head x 512 dim). GGUF metaverisinden.
_KV_PROFILES: dict[str, KVProfile] = {
    "gemma-4-12b": KVProfile(40, 8 * 256 * 2 * 2, 1024, 8, 1 * 512 * 2 * 2),
}


def kv_cache_mb(spec: LLMModelSpec, ctx_size: int, parallel_slots: int = 1) -> float:
    """Seçili model için KV cache tahmini (MiB). Profili bilinmeyen modelde muhafazakar (tam bağlam) tahmin."""
    prof = _KV_PROFILES.get(spec.key)
    if prof is None:
        # E4B vb. (GGUF henüz doğrulanmadı): 2 kv head x 256 dim x K+V x f16 = 2048 B/token/katman, tam bağlam
        total = spec.n_layers * 2048 * ctx_size
    else:
        swa_tokens = min(ctx_size, prof.swa_window + _UBATCH)
        total = prof.swa_layers * prof.swa_bytes_per_token * swa_tokens
        total += prof.global_layers * prof.global_bytes_per_token * ctx_size
    return total * parallel_slots / _MB


@dataclass
class NglPlan:
    """`-ngl` hesabının sonucu ve gerekçesi (loglama/rapor için)."""

    ngl: int
    mode: str                       # "auto" | "manual" | "cpu"
    free_vram_mb: float | None
    mmproj_mb: float
    kv_mb: float
    compute_mb: float
    margin_mb: float
    per_layer_mb: float
    budget_mb: float                # katmanlara ayrılabilen VRAM
    note: str = ""

    def describe(self) -> str:
        return (
            f"-ngl {self.ngl} ({self.mode}): boş VRAM={self.free_vram_mb} MB, mmproj={self.mmproj_mb:.0f}, "
            f"KV={self.kv_mb:.0f}, compute={self.compute_mb:.0f}, pay={self.margin_mb:.0f} -> "
            f"bütçe={self.budget_mb:.0f} MB, katman başı={self.per_layer_mb:.0f} MB {self.note}"
        ).strip()


def free_vram_mb() -> float | None:
    """İlk NVIDIA GPU'nun boş VRAM'i (MiB). `nvidia-smi`, olmazsa `torch.cuda`; hiçbiri yoksa None."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        )
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()[0].strip()
            return float(first)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:  # pragma: no cover - torch ortamına bağlı
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / _MB
    except Exception:  # noqa: BLE001
        pass
    return None


def plan_ngl(config: AppConfig, free_mb: float | None, gguf_mb: float, mmproj_mb: float) -> NglPlan:
    """`config.llm.n_gpu_layers` ve boş VRAM'den `-ngl` değerini hesaplar.

    Args:
        config: Uygulama ayarları.
        free_mb: Başlatma anındaki boş VRAM (MiB); None → CUDA/NVIDIA yok → ngl 0 (auto modunda).
        gguf_mb: Model GGUF dosya boyutu (MiB).
        mmproj_mb: mmproj dosya boyutu (MiB).
    """
    llm = config.llm
    spec = llm.model_spec
    kv = kv_cache_mb(spec, llm.ctx_size)
    per_layer = gguf_mb / spec.n_layers
    mm = mmproj_mb if llm.mmproj_offload else 0.0
    margin = float(llm.vram_safety_margin_mb)
    common = dict(free_vram_mb=free_mb, mmproj_mb=mm, kv_mb=kv, compute_mb=COMPUTE_BUFFER_MB,
                  margin_mb=margin, per_layer_mb=per_layer)

    if llm.n_gpu_layers != "auto":
        return NglPlan(ngl=int(llm.n_gpu_layers), mode="manual", budget_mb=0.0, **common)
    if free_mb is None:
        return NglPlan(ngl=0, mode="cpu", budget_mb=0.0, note="(CUDA/NVIDIA bulunamadı)", **common)

    budget = free_mb - mm - kv - COMPUTE_BUFFER_MB - margin
    ngl = int(budget // per_layer) if budget > 0 else 0
    ngl = max(0, min(spec.n_layers, ngl))
    note = "(tam offload)" if ngl >= spec.n_layers else ("(kısmi offload)" if ngl else "(CPU)")
    return NglPlan(ngl=ngl, mode="auto", budget_mb=budget, note=note, **common)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    """127.0.0.1 üzerinde boş bir TCP portu döndürür."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def default_log_path() -> Path:
    """`%LOCALAPPDATA%/ChangeScope/logs/llama-server.log`."""
    return user_data_dir() / "logs" / "llama-server.log"


def _create_kill_on_close_job() -> object | None:
    """Windows'ta ana süreç ölürse alt süreçleri de öldüren Job Object (en iyi çaba). Diğer OS'lerde None."""
    if sys.platform != "win32":
        return None
    try:  # pragma: no cover - yalnızca Windows
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None

        class _BasicLimit(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in
                        ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class _ExtLimit(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = _ExtLimit()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        return job
    except Exception:  # noqa: BLE001
        return None


def _assign_to_job(job: object, proc: subprocess.Popen) -> None:
    """Süreci Job Object'e ekler (en iyi çaba)."""
    try:  # pragma: no cover - yalnızca Windows
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject(job, int(proc._handle))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


@dataclass
class StartInfo:
    """Son başlatma denemesinin bilgisi."""

    ngl: int | None = None
    plan: NglPlan | None = None
    startup_s: float | None = None
    attempts: list[tuple[int, bool, float]] = field(default_factory=list)   # (ngl, başarılı, süre)
    port: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Sunucu
# ---------------------------------------------------------------------------

class LlamaServer:
    """Gömülü llama-server sürecini yönetir (lazy başlatma, kesin sonlandırma).

    Kullanım:
        server = LlamaServer(config)
        url = server.start()       # None → kullanılamıyor
        ...
        server.stop()

    Context manager olarak da kullanılır (`with LlamaServer(cfg) as s:`).
    """

    def __init__(self, config: AppConfig, log_path: Path | None = None) -> None:
        self.config = config
        self.log_path = log_path or default_log_path()
        self.info = StartInfo()
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._base_url: str | None = None
        self._lock = threading.RLock()
        self._job = None
        self._last_launch_invalid_args = False
        atexit.register(self.stop)

    # -- Dosyalar ----------------------------------------------------------------

    @property
    def exe_path(self) -> Path:
        return self.config.paths.llama_server_exe

    @property
    def gguf_path(self) -> Path:
        return self.config.llm.model_spec.gguf_path(self.config.paths.models_dir)

    @property
    def mmproj_path(self) -> Path:
        return self.config.llm.model_spec.mmproj_path(self.config.paths.models_dir)

    def files_present(self) -> bool:
        """exe + gguf + mmproj var mı? Sunucuyu BAŞLATMAZ."""
        return self.exe_path.is_file() and self.gguf_path.is_file() and self.mmproj_path.is_file()

    # -- Durum -------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None and self._base_url is not None

    @property
    def base_url(self) -> str | None:
        return self._base_url if self.is_running else None

    # -- Başlatma ----------------------------------------------------------------

    def start(self) -> str | None:
        """Sunucu çalışmıyorsa başlatır; hazır olunca taban URL'yi (`http://127.0.0.1:<port>`) döndürür.

        Başarısızlıkta (dosya yok, OOM, zaman aşımı...) istisna fırlatmaz; None döner ve `info.error` dolar.
        """
        with self._lock:
            if self.is_running:
                return self._base_url
            self._cleanup_process()
            self.info = StartInfo()
            if not self.config.llm.enabled:
                self.info.error = "LLM devre dışı"
                return None
            if not self.files_present():
                self.info.error = "llama-server.exe veya model dosyaları bulunamadı"
                log.warning(self.info.error)
                return None
            try:
                return self._start_with_fallback()
            except Exception as exc:  # noqa: BLE001 - uygulama düşmemeli
                self.info.error = f"{type(exc).__name__}: {exc}"
                log.exception("llama-server başlatılamadı")
                self._cleanup_process()
                return None

    def _start_with_fallback(self) -> str | None:
        gguf_mb = self.gguf_path.stat().st_size / _MB
        mmproj_mb = self.mmproj_path.stat().st_size / _MB
        free = free_vram_mb() if self.config.llm.n_gpu_layers == "auto" else None
        plan = plan_ngl(self.config, free, gguf_mb, mmproj_mb)
        self.info.plan = plan
        log.info("llama-server %s", plan.describe())

        attempts = [plan.ngl]
        if plan.ngl > 8:
            attempts.append(max(0, int(plan.ngl * 0.6)))   # OOM vb.: daha az katman
        if attempts[-1] != 0:
            attempts.append(0)                              # son çare: tamamen CPU

        t_total = time.monotonic()
        for ngl in attempts:
            t0 = time.monotonic()
            url = self._launch(ngl)
            dt = time.monotonic() - t0
            self.info.attempts.append((ngl, url is not None, dt))
            if url is not None:
                self.info.ngl = ngl
                self.info.startup_s = time.monotonic() - t_total
                log.info("llama-server hazır: %s (ngl=%d, %.1f sn)", url, ngl, dt)
                return url
            log.warning("llama-server ngl=%d ile başlamadı (%.1f sn): %s", ngl, dt, self.info.error)
            if self._last_launch_invalid_args:
                break   # komut satırı hatası: ngl azaltmak işe yaramaz
        return None

    def build_command(self, ngl: int, port: int) -> list[str]:
        """llama-server komut satırı."""
        llm = self.config.llm
        cmd = [
            str(self.exe_path), "-m", str(self.gguf_path), "--mmproj", str(self.mmproj_path),
            "--host", "127.0.0.1", "--port", str(port),
            "-ngl", str(ngl), "-c", str(llm.ctx_size), "-np", "1",
            "--fit", "off",
        ]
        if not llm.mmproj_offload:
            cmd.append("--no-mmproj-offload")
        if llm.threads:
            cmd += ["-t", str(llm.threads)]
        cmd += list(llm.extra_server_args)
        return cmd

    def _launch(self, ngl: int) -> str | None:
        port = find_free_port()
        cmd = self.build_command(ngl, port)
        self.info.port = port
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_launch_invalid_args = False
        log_start = self.log_path.stat().st_size if self.log_path.exists() else 0
        self._log_fh = open(self.log_path, "ab")  # noqa: SIM115 - süreç boyunca açık kalır
        self._log_fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ngl={ngl} ===\n{' '.join(cmd)}\n".encode())
        self._log_fh.flush()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=self._log_fh, stderr=subprocess.STDOUT,
                creationflags=flags, cwd=str(self.exe_path.parent),
            )
        except OSError as exc:
            self.info.error = f"süreç başlatılamadı: {exc}"
            self._cleanup_process()
            return None
        if self._job is None:
            self._job = _create_kill_on_close_job()
        if self._job is not None:
            _assign_to_job(self._job, self._proc)

        url = f"http://127.0.0.1:{port}"
        if self._wait_healthy(url):
            self._base_url = url
            return url
        self._last_launch_invalid_args = self._log_has_invalid_argument(log_start)
        self._cleanup_process()
        return None

    def _log_has_invalid_argument(self, offset: int) -> bool:
        """Bu denemenin log çıktısında 'invalid argument' var mı (geçersiz komut satırı seçeneği)?"""
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(offset)
                return b"invalid argument" in fh.read()
        except OSError:
            return False

    def _wait_healthy(self, url: str) -> bool:
        """`/health` 200 dönene kadar yoklar; süreç ölürse ya da zaman aşımında False."""
        deadline = time.monotonic() + self.config.llm.startup_timeout_s
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                code = None if proc is None else proc.returncode
                self.info.error = f"llama-server beklenmedik şekilde çıktı (kod {code})"
                return False
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2.0) as resp:
                    if resp.status == 200:
                        return True
            except urllib.error.HTTPError:
                pass            # 503: model hâlâ yükleniyor
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(0.25)
        self.info.error = f"/health {self.config.llm.startup_timeout_s:.0f} sn içinde hazır olmadı"
        return False

    # -- Durdurma ----------------------------------------------------------------

    def stop(self) -> None:
        """Süreci kesin olarak sonlandırır (VRAM boşalır). Çalışmıyorsa sessizce döner."""
        with self._lock:
            self._cleanup_process()

    def _cleanup_process(self) -> None:
        proc, self._proc = self._proc, None
        self._base_url = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                log.exception("llama-server sonlandırılamadı")
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

