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
            with Image.open(path) as im:
                im.load()
                fmt = (im.format or "").upper()
                quant = getattr(im, "quantization", None) if fmt == "JPEG" else None
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    im = Image.alpha_composite(bg, im)
                rgb = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
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
