"""SSIM tabanlı değişiklik tespiti (Faz 1) ve ortak son işleme."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from core.config import AppConfig, DetectionConfig
from core.contracts import (
    AlignmentQuality,
    BoundingBox,
    ChangeRegion,
    DetectionResult,
    RegistrationResult,
)
from core.detection.normalize import normalize_pair

log = logging.getLogger(__name__)


CHROMA_WEIGHT = 0.8      # LAB kroma farkı terimi azami katkısı (0 = kapalı)
CHROMA_DEADZONE = 12.0   # bu kadar a/b farkı sıkıştırma/ışık gürültüsü sayılır
CHROMA_SCALE = 30.0      # deadzone üstünde bu kadar fark -> terim azami


def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 1 else n + 1


class SsimChangeDetector:
    """SSIM + ışık normalizasyonu + eşikleme/morfoloji/connected components.

    Faz 1'de yalnızca SSIM vardır; `method` "dinov2" veya "ssim+dinov2" istenirse SSIM'e
    düşülür ve bu `params` içinde belirtilir.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()

    @property
    def cfg(self) -> DetectionConfig:
        """Aktif tespit ayarları."""
        return self._config.detection

    def release(self) -> None:
        """Model belleğini bırakır (Faz 1'de tutulan model yok; no-op)."""

    # ------------------------------------------------------------------
    def detect(self, registration: RegistrationResult) -> DetectionResult:
        """Hizalanmış çiftten değişiklik haritası ve bölge listesi üretir.

        Raises:
            ValueError: `registration.quality == FAILED` ise veya boyutlar uyuşmazsa.
        """
        if registration.quality == AlignmentQuality.FAILED:
            raise ValueError("Hizalama başarısız (FAILED); değişiklik tespiti çalıştırılamaz.")
        c = self.cfg
        before = registration.before.pixels
        after = registration.after_warped
        if before.shape != after.shape:
            raise ValueError(f"before {before.shape} ve after_warped {after.shape} boyutları farklı.")
        valid = registration.valid_mask.astype(bool)

        params: dict[str, float | int | str] = {
            "method_requested": c.method,
            "light_normalization": c.light_normalization,
            "ssim_window": _odd(c.ssim_window),
            "blur_sigma": c.blur_sigma,
            "min_threshold": c.min_threshold,
            "sensitivity": c.sensitivity,
            "morph_kernel_px": c.morph_kernel_px,
            "min_region_area_px": c.min_region_area_px,
            "edge_erode_px": c.edge_erode_px,
            "method_used": "ssim",
        }
        if c.method != "ssim":
            params["fallback_reason"] = "dinov2 henüz uygulanmadı (Faz 2); SSIM kullanıldı"

        # Geçerli alan: kenar erozyonu + SSIM penceresinin kenara taşan etkisi
        erode = int(c.edge_erode_px) + _odd(c.ssim_window) // 2
        safe = self._erode(valid, erode)
        params["effective_erode_px"] = erode

        change = self._ssim_change_map(before, after, valid, params)
        change[~safe] = 0.0

        mask = self._threshold(change, safe, params)
        regions = self._regions(mask, change, safe, float(params["threshold"]))
        params["num_regions"] = len(regions)
        return DetectionResult(change_map=change, regions=regions, method="ssim", params=params)

    # ------------------------------------------------------------------
    @staticmethod
    def _erode(mask: np.ndarray, px: int) -> np.ndarray:
        """Maskeyi `px` kadar aşındırır; görüntü sınırı da geçersiz sayılır (filtre kenar etkisi)."""
        if px <= 0:
            return mask.copy()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        padded = cv2.copyMakeBorder(mask.astype(np.uint8), px, px, px, px, cv2.BORDER_CONSTANT, value=0)
        return cv2.erode(padded, k)[px:-px, px:-px].astype(bool)

    def _ssim_change_map(self, before: np.ndarray, after: np.ndarray, valid: np.ndarray,
                         params: dict) -> np.ndarray:
        c = self.cfg
        b, a = normalize_pair(before, after, valid, c.light_normalization)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        if c.blur_sigma > 0:
            gb = cv2.GaussianBlur(gb, (0, 0), c.blur_sigma)
            ga = cv2.GaussianBlur(ga, (0, 0), c.blur_sigma)
        # Geçersiz alanı before ile doldur: sahte kenar yapısı SSIM'i bozmasın
        ga = np.where(valid, ga, gb)
        win = _odd(c.ssim_window)
        _, ssim_map = structural_similarity(gb, ga, win_size=win, data_range=255, full=True)
        change = np.clip(1.0 - ssim_map, 0.0, 1.0).astype(np.float32)

        # Renk (kroma) farkı: düz bir yüzey başka renk bir düz yüzeyle değiştiğinde SSIM
        # yapı farkı göremez. LAB a/b kanalları parlaklıktan bağımsızdır.
        if CHROMA_WEIGHT > 0:
            lb = cv2.cvtColor(cv2.GaussianBlur(b, (0, 0), max(c.blur_sigma, 2.5)), cv2.COLOR_RGB2LAB)
            la = cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), max(c.blur_sigma, 2.5)), cv2.COLOR_RGB2LAB)
            dab = np.linalg.norm(lb[..., 1:].astype(np.float32) - la[..., 1:].astype(np.float32), axis=2)
            chroma = np.clip((dab - CHROMA_DEADZONE) / CHROMA_SCALE, 0.0, 1.0) * CHROMA_WEIGHT
            chroma[~valid] = 0.0
            params["chroma_weight"] = CHROMA_WEIGHT
            params["chroma_scale"] = CHROMA_SCALE
            params["chroma_deadzone"] = CHROMA_DEADZONE
            change = np.maximum(change, chroma.astype(np.float32))
        return cv2.GaussianBlur(change, (0, 0), 1.5)

    def _threshold(self, change: np.ndarray, safe: np.ndarray, params: dict) -> np.ndarray:
        c = self.cfg
        if not safe.any():
            params["threshold"] = 1.0
            return np.zeros(change.shape, bool)
        if c.threshold == "otsu":
            vals = np.clip(change[safe] * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
            otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            base = max(float(otsu) / 255.0, c.min_threshold)
            params["threshold_mode"] = "otsu"
            params["otsu_threshold"] = round(float(otsu) / 255.0, 4)
        else:
            base = float(c.threshold)
            params["threshold_mode"] = "fixed"
        # sensitivity 0.5 -> x1.0, 1.0 -> x0.5, 0.0 -> x1.5
        sens = float(np.clip(c.sensitivity, 0.0, 1.0))
        thr = float(np.clip(base * (1.5 - sens), 0.0, 1.0))
        params["threshold"] = round(thr, 4)
        return (change >= thr) & safe

    def _regions(self, mask: np.ndarray, change: np.ndarray, safe: np.ndarray, thr: float) -> list[ChangeRegion]:
        c = self.cfg
        m = mask.astype(np.uint8)
        if c.morph_kernel_px > 1:
            k = _odd(c.morph_kernel_px)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
            m &= safe.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        found: list[tuple[float, BoundingBox, np.ndarray, int]] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < c.min_region_area_px:
                continue
            sub = labels[y:y + h, x:x + w] == i
            # Düz yüzeylerin içi (SSIM yapı farkı görmez) halka gibi kalabilir: delikleri doldur
            contours, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros(sub.shape, np.uint8)
            cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
            sub = (filled > 0) & safe[y:y + h, x:x + w]
            area = int(sub.sum())
            score = float(np.clip(change[y:y + h, x:x + w][sub].mean(), 0.0, 1.0))
            found.append((score, BoundingBox(x, y, w, h), sub, area))
        found.sort(key=lambda t: t[0], reverse=True)
        return [
            ChangeRegion(region_id=i, bbox=bb, mask=sub, score=s, area_px=a)
            for i, (s, bb, sub, a) in enumerate(found, start=1)
        ]
