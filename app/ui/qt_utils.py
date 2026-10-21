"""numpy <-> Qt dönüşümleri. Veri HER ZAMAN kopyalanır (QImage numpy belleğine bağımlı kalmaz)."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def rgb_to_qimage(pixels: np.ndarray) -> QImage:
    """HxWx3 uint8 RGB (veya HxW gri) diziyi bağımsız bir `QImage`'e kopyalar."""
    arr = np.ascontiguousarray(pixels)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    if arr.shape[2] == 4:
        return QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    return QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def rgb_to_pixmap(pixels: np.ndarray) -> QPixmap:
    """`rgb_to_qimage` + `QPixmap`."""
    return QPixmap.fromImage(rgb_to_qimage(pixels))


HEAT_FLOOR = 0.08      # normalize değerin altı tamamen saydam
HEAT_FULL = 0.45       # bunun üstü tam opak (katman opaklığı ayrıca ~%50)


def heatmap_to_qimage(change_map: np.ndarray, max_side: int = 2048) -> QImage:
    """0..1 `change_map`'i renk haritalı (TURBO) RGBA ısı haritasına çevirir.

    Değerler, gerçek haritalar genelde düşük aralıkta kaldığından %99.5 yüzdeliğine göre
    normalize edilir (en az 0.25). Düşük değerler saydam, yüksekler opak ve sıcak renklidir.
    """
    cm = np.clip(np.nan_to_num(change_map.astype(np.float32)), 0.0, 1.0)
    h, w = cm.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        cm = cv2.resize(cm, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    top = max(float(np.percentile(cm, 99.5)), 0.25)
    norm = np.clip(cm / top, 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = np.clip((norm - HEAT_FLOOR) / (HEAT_FULL - HEAT_FLOOR), 0.0, 1.0)
    rgba = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alpha * 255]).astype(np.uint8)
    return rgb_to_qimage(rgba)


def crop_rgb(pixels: np.ndarray, x: int, y: int, w: int, h: int, margin: float = 0.25) -> np.ndarray:
    """Kutunun çevresinden (kenar payıyla) güvenli kırpma; görüntü sınırlarına kırpılır."""
    ih, iw = pixels.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(iw, x + w + mx), min(ih, y + h + my)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    return pixels[y0:y1, x0:x1]
