"""Değişiklik tespiti modülü (change-detector).

Dışa açık API: `create_detector(config)`.
"""

from __future__ import annotations

from core.config import AppConfig
from core.contracts import ChangeDetector
from core.detection.ssim_detector import SsimChangeDetector


def create_detector(config: AppConfig | None = None) -> ChangeDetector:
    """Config'e göre bir `ChangeDetector` üretir (`release()` metodu vardır).

    `method`: "ssim" | "dinov2" | "ssim+dinov2". DINOv2 modeli/kaynağı yoksa SSIM'e düşülür ve
    bu durum `params["fallback_reason"]` ile `DetectionResult.method` içinde belirtilir.
    """
    return SsimChangeDetector(config)


__all__ = ["create_detector", "SsimChangeDetector"]
