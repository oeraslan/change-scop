"""Işık normalizasyonu: after görüntüsünü before'un ışık koşullarına yaklaştırır."""

from __future__ import annotations

import cv2
import numpy as np


MAX_SHIFT = 60.0   # histogram eşleştirmenin tek bir seviyeyi kaydırabileceği azami miktar


def match_histogram(source: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`source` (HxWx3 uint8) kanallarını `reference` histogramına eşler.

    Histogramlar yalnızca `mask` (HxW bool) içindeki piksellerden hesaplanır; böylece warp
    sonrası siyah kenarlar dağılımı bozmaz. Dönüş: uint8, source ile aynı boyut.
    """
    out = source.copy()
    if not mask.any():
        return out
    qs = np.linspace(1.0, 99.0, 33)
    x = np.arange(256, dtype=np.float64)
    for c in range(source.shape[2]):
        s_q = np.percentile(source[..., c][mask], qs)
        r_q = np.percentile(reference[..., c][mask], qs)
        # Kuantil tabanlı (seyrek histogram kutularında sıçrama yapmaz); uçlarda kaydırma sabit
        shift = np.interp(x, s_q, r_q - s_q)
        shift = np.clip(shift, -MAX_SHIFT, MAX_SHIFT)
        lut = np.clip(x + shift, 0, 255)
        lut = np.maximum.accumulate(lut)                   # monoton
        out[..., c] = np.round(lut).astype(np.uint8)[source[..., c]]
    return out


def apply_clahe(rgb: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """LAB uzayında L kanalına CLAHE uygular (RGB uint8 girdi/çıktı)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalize_pair(
    before: np.ndarray, after: np.ndarray, mask: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """`mode`'a göre (none/histogram/clahe/histogram+clahe) çifti normalize eder.

    Histogram eşleştirme after'ı before'a yaklaştırır; CLAHE her iki görüntüye aynı
    parametrelerle uygulanır (yerel kontrast/gölge farkını azaltır).
    """
    if "histogram" in mode:
        after = match_histogram(after, before, mask)
    if "clahe" in mode:
        before = apply_clahe(before)
        after = apply_clahe(after)
    return before, after
