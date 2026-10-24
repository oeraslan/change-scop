"""Modüller arası veri sözleşmesi.

Bu dosya tüm modüller için TEK doğruluk kaynağıdır. Ajanlar bu dosyayı DEĞİŞTİRMEZ;
değişiklik ihtiyacı ana oturuma gerekçesiyle raporlanır.

Koordinat kuralı: Tüm piksel koordinatları REFERANS görüntünün (before) koordinat
sisteminde, (x, y) sırasıyla, sol üst köşe orijinli verilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Girdi
# ---------------------------------------------------------------------------

class ImageSource(str, Enum):
    DRONE = "drone"
    UAV = "uav"
    SATELLITE = "satellite"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ImageInput:
    """Kullanıcının yüklediği tek bir görüntü."""

    path: Path
    captured_at: str | None = None          # ISO 8601, bilinmiyorsa None
    source: ImageSource = ImageSource.UNKNOWN


@dataclass
class PreparedImage:
    """Ön işlemeden geçmiş görüntü."""

    original: ImageInput
    pixels: np.ndarray                      # HxWx3, uint8, RGB
    scale_from_original: float              # pixels boyutu / orijinal boyut
    was_compressed: bool                    # JPEG/sohbet sıkıştırması tespit edildi mi


# ---------------------------------------------------------------------------
# Hizalama (image-registration)
# ---------------------------------------------------------------------------

class AlignmentQuality(str, Enum):
    GOOD = "good"           # Yeterli inlier, düşük reprojeksiyon hatası
    WEAK = "weak"           # Sonuç kullanılabilir ama güvenilirlik düşük
    FAILED = "failed"       # Hizalama yapılamadı; tespit adımı çalıştırılmamalı


@dataclass
class RegistrationResult:
    """İki görüntünün hizalanma sonucu. after görüntüsü before'un bakış açısına warp edilmiştir."""

    before: PreparedImage
    after_warped: np.ndarray                # HxWx3 uint8 RGB, before ile aynı boyut
    valid_mask: np.ndarray                  # HxW bool; iki görüntünün de bulunduğu pikseller True
    homography: np.ndarray | None           # 3x3, after -> before; FAILED ise None
    num_matches: int
    num_inliers: int
    reprojection_error_px: float
    quality: AlignmentQuality
    notes: list[str] = field(default_factory=list)

    @property
    def match_percent(self) -> float:
        """Hizalama eşleşmesi (%): homografiyle tutarlı (inlier) eşleşmelerin tüm eşleşmelere oranı, 0..100."""
        if self.num_matches <= 0:
            return 0.0
        return 100.0 * min(self.num_inliers, self.num_matches) / self.num_matches


# ---------------------------------------------------------------------------
# Değişiklik tespiti (change-detector)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class ChangeRegion:
    """Tek bir aday değişiklik bölgesi."""

    region_id: int
    bbox: BoundingBox
    mask: np.ndarray                    # bbox boyutunda bool maske
    score: float                        # 0..1, yüksek = daha güçlü değişiklik
    area_px: int


@dataclass
class DetectionResult:
    change_map: np.ndarray              # HxW float32, 0..1 ısı haritası
    regions: list[ChangeRegion]
    method: str                         # ör. "ssim", "dinov2", "ssim+dinov2"
    params: dict[str, float | int | str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Modül arayüzleri
# ---------------------------------------------------------------------------

class Preprocessor(Protocol):
    def prepare(self, image: ImageInput) -> PreparedImage: ...


class Registrar(Protocol):
    def register(self, before: PreparedImage, after: PreparedImage) -> RegistrationResult: ...


class ChangeDetector(Protocol):
    def detect(self, registration: RegistrationResult) -> DetectionResult: ...


# ---------------------------------------------------------------------------
# İlerleme bildirimi (UI için)
# ---------------------------------------------------------------------------

class ProgressCallback(Protocol):
    def __call__(self, stage: str, fraction: float, message: str = "") -> None: ...


@dataclass
class ComparisonResult:
    """Pipeline'ın UI'a döndürdüğü nihai sonuç."""

    registration: RegistrationResult
    detection: DetectionResult | None   # Hizalama FAILED ise None
    elapsed_s: float
