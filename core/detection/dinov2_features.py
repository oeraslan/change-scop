"""DINOv2-small (ViT-S/14) patch öznitelikleriyle çift karşılaştırma (Faz 2).

Ağırlıklar ve mimari kodu YALNIZCA yerel `models/dinov2/` altından yüklenir; ağa çıkılmaz
(`torch.hub.load` / `from_pretrained` kullanılmaz). Model ilk ihtiyaçta yüklenir, `release()` ile bırakılır.
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
from pathlib import Path

import numpy as np

from core.config import AppConfig, resolve_device

log = logging.getLogger(__name__)

PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TILE_OVERLAP = 0.10


class Dinov2Unavailable(RuntimeError):
    """Model/kaynak bulunamadı veya yüklenemedi (çağıran SSIM'e düşer)."""


class Dinov2Extractor:
    """Tembel yüklenen DINOv2-small; patch bazlı cosine distance haritası üretir.

    Thread-safe: yükleme, çıkarım ve `release()` tek bir kilitle serileşir. Dolayısıyla `release()`
    çalışmakta olan bir çıkarımı bozmaz; bitmesini bekler.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._model = None
        self._device = "cpu"

    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        """Model şu an bellekte mi."""
        return self._model is not None

    def available(self) -> tuple[bool, str]:
        """Ağırlık ve kaynak dosyaları mevcut mu (yüklemeden hızlı kontrol)."""
        p = self._config.paths
        if not Path(p.dinov2_weights).is_file():
            return False, f"DINOv2 ağırlığı yok: {p.dinov2_weights}"
        if not (Path(p.dinov2_source_dir) / "dinov2" / "hub" / "backbones.py").is_file():
            return False, f"DINOv2 kaynak kodu yok: {p.dinov2_source_dir}"
        return True, ""

    def _load(self, device: str):
        """Modeli yerel dosyalardan yükler (kilit altında çağrılır)."""
        ok, why = self.available()
        if not ok:
            raise Dinov2Unavailable(why)
        try:
            import torch
            src = str(Path(self._config.paths.dinov2_source_dir))
            added = src not in sys.path
            if added:
                sys.path.insert(0, src)
            try:
                from dinov2.hub.backbones import dinov2_vits14
            finally:
                if added and src in sys.path:
                    sys.path.remove(src)
            model = dinov2_vits14(pretrained=False)
            state = torch.load(str(self._config.paths.dinov2_weights), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval().requires_grad_(False)
            model.to(device)
        except Dinov2Unavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - her yükleme hatası SSIM'e düşmeyi gerektirir
            raise Dinov2Unavailable(f"DINOv2 yüklenemedi: {type(exc).__name__}: {exc}") from exc
        return model

    def release(self) -> None:
        """Modeli bırakır (`del`, `gc.collect()`, CUDA varsa `empty_cache()`). Çalışan çıkarımı bekler."""
        with self._lock:
            had = self._model is not None
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if had:
                log.info("DINOv2 serbest bırakıldı.")

    # ------------------------------------------------------------------
    def distance_map(self, before: np.ndarray, after: np.ndarray, tile_px: int, fp16: bool,
                     info: dict) -> np.ndarray:
        """İki RGB uint8 görüntü (aynı boyut) arasında patch bazlı cosine distance, görüntü boyutunda float32.

        Değerler ham `1 - cos` (genelde 0..1; negatif cos'ta >1 olabilir). `info` cihaz/OOM/VRAM
        bilgisiyle doldurulur.

        Raises:
            Dinov2Unavailable: model yüklenemezse.
        """
        import torch
        with self._lock:
            device = resolve_device(self._config.runtime.device)
            if self._model is not None and self._device != device and self._device != "cpu":
                self._model = None    # cihaz tercihi değişti
            if self._model is None:
                self._device = device
                self._model = self._load(device)
                info["dinov2_loaded_now"] = 1
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                try:
                    grid = self._run(before, after, tile_px, fp16)
                except torch.cuda.OutOfMemoryError:
                    log.warning("DINOv2 CUDA OOM; CPU'ya düşülüyor.")
                    info["dinov2_oom_fallback"] = 1
                    self._model = self._model.to("cpu")
                    self._device = "cpu"
                    gc.collect()
                    torch.cuda.empty_cache()
                    grid = self._run(before, after, tile_px, False)
            finally:
                if self._device == "cuda":
                    info["dinov2_peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            info["dinov2_device"] = self._device
        import cv2
        h, w = before.shape[:2]
        gh, gw = grid.shape
        full = cv2.resize(grid, (gw * PATCH, gh * PATCH), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(full[:h, :w])

    # ------------------------------------------------------------------
    def _run(self, before: np.ndarray, after: np.ndarray, tile_px: int, fp16: bool) -> np.ndarray:
        """Patch ızgarasında (gh, gw) mesafe haritası; tile'lar örtüşmeli, örtüşmede ortalama."""
        import torch
        h, w = before.shape[:2]
        tile = max(PATCH * 4, (int(tile_px) // PATCH) * PATCH)
        ph, pw = -(-h // PATCH), -(-w // PATCH)          # patch ızgarası boyutu (yukarı yuvarlı)
        pad_h, pad_w = ph * PATCH - h, pw * PATCH - w
        if pad_h or pad_w:
            before = np.pad(before, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            after = np.pad(after, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        tp = tile // PATCH
        step = max(tp - max(int(round(tp * TILE_OVERLAP)), 1), 1)

        def starts(n: int) -> list[int]:
            if n <= tp:
                return [0]
            s = list(range(0, n - tp, step))
            s.append(n - tp)
            return s

        acc = np.zeros((ph, pw), np.float32)
        cnt = np.zeros((ph, pw), np.float32)
        mean = torch.tensor(IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self._device).view(1, 3, 1, 1)
        use_amp = bool(fp16) and self._device == "cuda"
        with torch.inference_mode():
            for gy in starts(ph):
                for gx in starts(pw):
                    y0, x0 = gy * PATCH, gx * PATCH
                    y1, x1 = min(y0 + tile, ph * PATCH), min(x0 + tile, pw * PATCH)
                    pair = np.stack([before[y0:y1, x0:x1], after[y0:y1, x0:x1]])      # (2,h,w,3)
                    x = torch.from_numpy(pair).to(self._device).permute(0, 3, 1, 2).float().div_(255.0)
                    x = (x - mean) / std
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        feats = self._model.forward_features(x)["x_norm_patchtokens"]    # (2,N,384)
                    feats = feats.float()
                    cos = torch.nn.functional.cosine_similarity(feats[0], feats[1], dim=-1)
                    d = (1.0 - cos).reshape((y1 - y0) // PATCH, (x1 - x0) // PATCH).cpu().numpy()
                    acc[gy:gy + d.shape[0], gx:gx + d.shape[1]] += d
                    cnt[gy:gy + d.shape[0], gx:gx + d.shape[1]] += 1.0
        return acc / np.maximum(cnt, 1.0)
