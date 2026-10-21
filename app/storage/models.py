"""Kütüphane veri modelleri (Qt'den bağımsız)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Folder:
    """Klasör ağacındaki bir düğüm. `parent_id=None` ise kök seviyededir."""

    id: int
    parent_id: int | None
    name: str
    created_at: str
    sort_order: int = 0


@dataclass(frozen=True)
class ImageRecord:
    """Kütüphaneye kopyalanmış bir görüntü. Yollar kütüphane köküne göredir."""

    id: int
    folder_id: int
    file_name: str                  # kütüphane köküne göre göreli yol
    original_name: str
    captured_at: str | None         # ISO 8601
    source: str                     # ImageSource değeri: drone / uav / satellite / unknown
    width: int
    height: int
    thumb_path: str | None
    created_at: str
    mask_path: str | None = None    # yok sayma maskesi (tek kanallı PNG, 255 = yok say); kütüphane köküne göre
    mask_hash: str = ""             # maske içeriğinin özeti ("" = maske yok); önbellek anahtarının parçası


@dataclass(frozen=True)
class ComparisonRecord:
    """Kaydedilmiş bir karşılaştırma. `result_dir` kütüphane köküne göredir."""

    id: int
    folder_id: int | None
    before_image_id: int
    after_image_id: int
    result_dir: str
    quality: str
    region_count: int
    summary: str | None
    created_at: str
    mask_hash: str = ""             # hesaplamada kullanılan maskenin özeti ("" = maskesiz)
    method: str = ""                # kullanılan fark yöntemi (ssim / dinov2 / ssim+dinov2 ...)
