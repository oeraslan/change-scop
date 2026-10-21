"""Ön işleme: yükleme, EXIF yönü, RGB, sıkıştırma tespiti, hafif denoise ve boyut sınırı.

Sıkıştırma tespiti (`was_compressed`):
    * JPEG dosyalarda kuantizasyon tablolarından tahmini kalite hesaplanır; kalite
      `PreprocessConfig.jpeg_quality_suspect` altındaysa sıkıştırılmış sayılır.
    * JPEG olmayan dosyalarda (ör. JPEG'den dönüştürülüp PNG yapılmış sohbet görselleri)
      8x8 blok sınırlarındaki süreksizlik oranı (`BLOCKINESS_RATIO_THRESHOLD`) ölçülür.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from core.config import PreprocessConfig
from core.contracts import ImageInput, PreparedImage

log = logging.getLogger(__name__)

# Meşru ama çok büyük uydu görüntüleri için Pillow sınırı (varsayılan ~89 MP) 400 MP'e çıkarılır;
# sınırsız bırakılmaz. Sınırın 2 katını aşan görüntüde Pillow DecompressionBombError fırlatır.
MAX_IMAGE_PIXELS = 400_000_000
if Image.MAX_IMAGE_PIXELS is not None and Image.MAX_IMAGE_PIXELS < MAX_IMAGE_PIXELS:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Yüksek bit derinlikli görüntüler için yüzdelik tabanlı normalizasyon aralığı.
NORMALIZE_PERCENTILES = (0.5, 99.5)
_HIGH_BIT_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")

# 8x8 blok sınırı gradyanı / blok içi gradyan oranı bunun üstündeyse blok artefaktı vardır.
BLOCKINESS_RATIO_THRESHOLD = 1.12

# JPEG Annex K standart parlaklık kuantizasyon tablosu (kalite 50 referansı).
_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def estimate_jpeg_quality(quant_table: list[int] | tuple[int, ...]) -> float:
    """Parlaklık kuantizasyon tablosundan IJG kalite değerini (1..100) tahmin eder."""
    table = np.asarray(quant_table, dtype=np.float64)
    if table.size != 64:
        return 100.0
    # Tablo zigzag sırasında olabilir; ortalama oran sıradan bağımsızdır.
    ratio = float(np.mean(np.clip(table, 1, 255)) / np.mean(_STD_LUMA_Q))
    quality = (2.0 - ratio) * 50.0 if ratio <= 1.0 else 50.0 / ratio
    return float(np.clip(quality, 1.0, 100.0))


def blockiness_ratio(rgb: np.ndarray) -> float:
    """8x8 blok sınırlarındaki ortalama gradyanın diğer konumlara oranı (~1 = blok yok)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    if h < 32 or w < 32:
        return 1.0
    dx = np.abs(np.diff(gray, axis=1)).mean(axis=0)   # w-1
    dy = np.abs(np.diff(gray, axis=0)).mean(axis=1)   # h-1
    ratios = []
    for d in (dx, dy):
        idx = np.arange(d.size)
        on = np.median(d[idx % 8 == 7])       # medyan: 8'in katına denk gelen gerçek kenarlara dayanıklı
        off = np.median(d[idx % 8 != 7])
        ratios.append(on / (off + 1e-6))
    return float(np.mean(ratios))


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Herhangi bir sayısal diziyi yüzdelik (0.5-99.5) tabanlı olarak uint8'e çevirir.

    Tek kanallı girdi 3 kanala çoğaltılır; çok kanallı girdide aralık tüm kanallar için
    ortaktır (renk dengesi korunur). Sabit görüntüde (hi <= lo) düz ölçekleme yapılır.
    """
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[2] == 2:          # gri + alfa
        a = a[..., :1]
    elif a.shape[2] > 3:
        a = a[..., :3]
    lo, hi = np.percentile(a, NORMALIZE_PERCENTILES)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        out = np.zeros(a.shape, np.uint8)
    else:
        out = np.clip((a - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    return np.ascontiguousarray(out)


def _png_bit_depth(path: Path) -> int:
    """PNG IHDR'dan bit derinliğini okur (PNG değilse 0)."""
    with open(path, "rb") as fh:
        head = fh.read(25)
    if len(head) < 25 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return 0
    return int(head[24])


def _load_16bit_png(path: Path) -> np.ndarray | None:
    """16-bit PNG'yi (gri/RGB/RGBA) cv2 ile kayıpsız okuyup uint8 RGB'ye normalize eder."""
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None or data.dtype != np.uint16:
        return None
    if data.ndim == 3:
        data = data[..., :3][..., ::-1]    # BGR(A) -> RGB
    return normalize_to_uint8(data)


class DefaultPreprocessor:
    """`Preprocessor` uygulaması."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def prepare(self, image: ImageInput) -> PreparedImage:
        """Görüntüyü yükler, EXIF yönünü düzeltir, RGB'ye çevirir ve gerekirse denoise/küçültme uygular.

        Raises:
            ValueError: dosya yok, okunamıyor ya da geçerli bir görüntü değil.
        """
        path = Path(image.path)
        if not path.is_file():
            raise ValueError(f"Görüntü dosyası bulunamadı: {path}")
        try:
            depth16 = _png_bit_depth(path) == 16
            with Image.open(path) as im:
                im.load()
                fmt = (im.format or "").upper()
                quant = getattr(im, "quantization", None) if fmt == "JPEG" else None
                im = ImageOps.exif_transpose(im)
                rgb = _load_16bit_png(path) if depth16 else None
                if rgb is not None:
                    pass
                elif im.mode in _HIGH_BIT_MODES:
                    rgb = normalize_to_uint8(np.asarray(im))
                else:
                    if im.mode in ("RGBA", "LA", "P", "PA"):
                        im = im.convert("RGBA")
                        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                        im = Image.alpha_composite(bg, im)
                    rgb = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
        except Image.DecompressionBombError as exc:
            raise ValueError(
                f"Görüntü çok büyük ({path.name}); güvenlik sınırı aşıldı. "
                f"Lütfen daha küçük bir görüntü kullanın. ({exc})") from exc
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise ValueError(f"Görüntü okunamadı ({path.name}): {exc}") from exc
        if rgb.ndim != 3 or rgb.shape[0] < 16 or rgb.shape[1] < 16:
            raise ValueError(f"Görüntü çok küçük ya da geçersiz: {path.name} {rgb.shape}")

        compressed = self._detect_compression(rgb, fmt, quant)

        scale = 1.0
        long_side = max(rgb.shape[:2])
        if long_side > self.config.max_side_px:
            ratio = self.config.max_side_px / long_side
            new_w = max(1, round(rgb.shape[1] * ratio))
            new_h = max(1, round(rgb.shape[0] * ratio))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale = max(new_w, new_h) / long_side

        if compressed and self.config.denoise_if_compressed:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h = float(self.config.denoise_h)
            bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        return PreparedImage(original=image, pixels=rgb, scale_from_original=float(scale),
                             was_compressed=bool(compressed))

    def _detect_compression(self, rgb: np.ndarray, fmt: str, quant: dict | None) -> bool:
        if fmt == "JPEG":
            if quant and 0 in quant:
                q = estimate_jpeg_quality(list(quant[0]))
                log.debug("JPEG tahmini kalite: %.0f", q)
                return q < self.config.jpeg_quality_suspect
            return True
        return blockiness_ratio(rgb) > BLOCKINESS_RATIO_THRESHOLD
