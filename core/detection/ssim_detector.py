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
from core.detection.dinov2_features import Dinov2Extractor, Dinov2Unavailable
from core.detection.normalize import normalize_pair

log = logging.getLogger(__name__)


CHROMA_WEIGHT = 0.8      # LAB kroma farkı terimi azami katkısı (0 = kapalı)
CHROMA_DEADZONE = 12.0   # bu kadar a/b farkı sıkıştırma/ışık gürültüsü sayılır
CHROMA_SCALE = 30.0      # deadzone üstünde bu kadar fark -> terim azami


MIN_AREA_FRACTION = 4e-4   # geçerli alanın bu oranı: min bölge alanı; YALNIZCA çalışma küçültmesi varken uygulanır
SCALE_COMPENSATE_FROM = 1.2   # etkin büyütme bu değerin üstündeyse çözünürlük eşitlenir
MAX_REGIONS_RELIABLE = 40  # bundan fazla bölge -> sonuç güvenilmez işaretlenir
THRESHOLD_SCALE_GAIN = 0.08  # çözünürlük farkı büyüdükçe taban eşik (1 + gain*(s-1)) kadar yükselir


def estimate_scale(homography: np.ndarray | None, shape: tuple[int, int]) -> float:
    """after->before homografisinin görüntü merkezindeki yerel ölçeği: sqrt(|det J|).

    1'den büyük değer, after'ın before'a büyütülerek (bulanıklaşarak) warp edildiğini gösterir.
    """
    if homography is None:
        return 1.0
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return 1.0
    h, w = shape
    # Merkezdeki before noktasına karşılık gelen after noktasında Jacobian (ters homografi)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 1.0
    x, y = w / 2.0, h / 2.0
    d = Hi[2, 0] * x + Hi[2, 1] * y + Hi[2, 2]
    if abs(d) < 1e-12:
        return 1.0
    u = (Hi[0, 0] * x + Hi[0, 1] * y + Hi[0, 2]) / d
    v = (Hi[1, 0] * x + Hi[1, 1] * y + Hi[1, 2]) / d
    J = np.array([
        [(Hi[0, 0] - u * Hi[2, 0]) / d, (Hi[0, 1] - u * Hi[2, 1]) / d],
        [(Hi[1, 0] - v * Hi[2, 0]) / d, (Hi[1, 1] - v * Hi[2, 1]) / d],
    ])
    det = abs(float(np.linalg.det(J)))     # before px -> after px alan oranı (= 1/s^2)
    if det <= 1e-12:
        return 1.0
    return float(np.clip(1.0 / np.sqrt(det), 1.0 / 16.0, 16.0))


FUSION_GATE = 0.4      # birleşimde DINOv2 skoru bu değere ulaşınca SSIM'e tam güven
DINO_EDGE_ERODE_PX = 14  # DINOv2 kullanılırken kenar erozyonuna eklenen px (çalışma çözünürlüğünde)
COMPACT_STRONG = 0.6  # kapı muafiyeti: SSIM(+kroma) skoru bu eşiğin üstünde
COMPACT_MAX_PX = 1000  # ... ve bileşen alanı (çalışma çözünürlüğünde) en fazla bu kadar (~5 DINOv2 patch'i)
COMPACT_MAX_COUNT = 25  # ... ve bu tür bileşen sayısı en fazla bu kadar (fazlası yaygın gürültü)
COMPACT_MIN_PX = 12   # ... ve en az bu kadar (tek piksel gürültü değil)
FUSION_FLOOR = 0.25   # DINOv2 skoru 0 iken birleşik skor çarpanı
DINO_PREBLUR = 1.0     # DINOv2 girdisine uygulanan Gaussian sigma (px, çalışma çözünürlüğünde)
DINO_SMOOTH = 7.0      # kalibrasyondan önce ham mesafe haritasına Gaussian sigma (px): yalıtılmış patch aykırılarını söndürür
DINO_DEADZONE = 0.15   # medyan üstü bu kadar mesafe doku/ton gürültüsü sayılır
DINO_SCALE = 0.35      # deadzone üstünde bu kadar mesafe -> 1.0


def calibrate_dino(raw: np.ndarray, region: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Ham cosine distance'ı 0..1 değişiklik skoruna çevirir.

    Taban `region` içindeki medyan (sahnenin genel doku/ton kayması çıkarılır); sabit ölçek
    kullanılır, böylece değişiklik olmayan çiftte gürültü büyütülmez.
    """
    if DINO_SMOOTH > 0:
        raw = cv2.GaussianBlur(raw, (0, 0), DINO_SMOOTH)
    base = float(np.median(raw[region])) if region.any() else 0.0
    out = np.clip((raw - base - DINO_DEADZONE) / DINO_SCALE, 0.0, 1.0).astype(np.float32)
    if params is not None:
        params["dinov2_base_distance"] = round(base, 4)
        params["dinov2_smooth_px"] = DINO_SMOOTH
        params["dinov2_preblur"] = DINO_PREBLUR
        params["dinov2_deadzone"] = DINO_DEADZONE
        params["dinov2_scale"] = DINO_SCALE
    return out


def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 1 else n + 1


class SsimChangeDetector:
    """SSIM / DINOv2 / birleşik değişiklik tespiti + eşikleme/morfoloji/connected components.

    `method` "dinov2" veya "ssim+dinov2" istenip model yüklenemezse SSIM'e düşülür ve bu
    `params["fallback_reason"]` ile `DetectionResult.method` içinde görünür. Thread-safe: aynı
    örnek birden çok thread'den çağrılabilir; `release()` çalışan çıkarımı bekler.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()
        self._dino = Dinov2Extractor(self._config)

    @property
    def cfg(self) -> DetectionConfig:
        """Aktif tespit ayarları."""
        return self._config.detection

    def release(self) -> None:
        """DINOv2 modelini bellekten/VRAM'den bırakır. Sonraki `detect()` modeli tekrar yükler."""
        self._dino.release()

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
        scale = estimate_scale(registration.homography, before.shape[:2])
        # s > 1: after before'a büyütülmüş. Çözünürlük eşitleme: ikisini 1/s'e küçült (INTER_AREA)
        work = scale if scale > SCALE_COMPENSATE_FROM else 1.0
        params["registration_scale"] = round(scale, 3)
        params["working_scale"] = round(work, 3)
        params["working_downscale"] = round(1.0 / work, 3)
        # Alan tabanı yalnızca çalışma küçültmesi varken yükselir (küçültmede JPEG/ölçek artıkları
        # büyük lekeler bırakır). Çözünürlük farkı yoksa config değeri tek taban: büyük görüntüde
        # küçük ama belirgin nesne kaçmaz.
        min_area = int(c.min_region_area_px)
        if work > 1.0:
            min_area = max(min_area, int(round(MIN_AREA_FRACTION * float(valid.sum()))))
        params["min_region_area_effective_px"] = min_area
        params["min_region_area_fraction"] = MIN_AREA_FRACTION
        use_ssim = c.method in ("ssim", "ssim+dinov2")
        use_dino = c.method in ("dinov2", "ssim+dinov2")
        if use_dino:
            ok, why = self._dino.available()
            if not ok:
                use_dino, use_ssim = False, True
                params["fallback_reason"] = why

        # Geçerli alan: kenar erozyonu + SSIM penceresinin kenara taşan etkisi
        erode = int(c.edge_erode_px) + int(np.ceil((_odd(c.ssim_window) // 2 + 2) * work))
        if use_dino:
            # ViT kenar patch'leri (padding/dikkat sınırı) yapay mesafe üretir
            erode += int(np.ceil(DINO_EDGE_ERODE_PX * work))
        safe = self._erode(valid, erode)
        params["effective_erode_px"] = erode

        h, w = before.shape[:2]
        if work > 1.0:
            size = (max(int(round(w / work)), 8), max(int(round(h / work)), 8))
            b_s = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
            a_s = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
            v_s = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            safe_s = cv2.resize(safe.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            b_s, a_s, v_s, safe_s = before, after, valid, safe

        ssim_map = self._ssim_change_map(b_s, a_s, v_s, params) if use_ssim else None
        dino_map = None
        if use_dino:
            try:
                dino_map = self._dino_change_map(b_s, a_s, v_s, safe_s, params)
            except Dinov2Unavailable as exc:
                params["fallback_reason"] = str(exc)
                if ssim_map is None:
                    ssim_map = self._ssim_change_map(b_s, a_s, v_s, params)
        if ssim_map is not None and dino_map is not None:
            ws, wd = max(float(c.ssim_weight), 0.0), max(float(c.dinov2_weight), 0.0)
            if ws + wd <= 0:
                ws, wd = 0.5, 0.5
            ws, wd = ws / (ws + wd), wd / (ws + wd)
            params["ssim_weight_used"] = round(ws, 4)
            params["dinov2_weight_used"] = round(wd, 4)
            mixed = ws * ssim_map + wd * dino_map
            # DINOv2 anlamsal "değişmedi" diyorsa (doku/ton/gölge) yapısal SSIM farkı bastırılır
            agree = np.clip(dino_map / FUSION_GATE, 0.0, 1.0)
            small = np.clip(mixed * (FUSION_FLOOR + (1.0 - FUSION_FLOOR) * agree), 0.0, 1.0).astype(np.float32)
            # DINOv2 patch'inden (14 px) küçük ama SSIM+kromada GÜÇLÜ ve kompakt farklar kapıdan geçer
            compact = self._compact_strong(ssim_map)
            small = np.maximum(small, np.where(compact, ssim_map, 0.0)).astype(np.float32)
            params["fusion_compact_px"] = COMPACT_MAX_PX
            params["fusion_gate"] = FUSION_GATE
            params["fusion_floor"] = FUSION_FLOOR
            method_used = "ssim+dinov2"
        elif dino_map is not None:
            small, method_used = dino_map, "dinov2"
        else:
            small, method_used = ssim_map, "ssim"
        if "fallback_reason" in params:
            params["fell_back"] = 1
        params["method_used"] = method_used
        change = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if work > 1.0 else small
        change[~safe] = 0.0

        mask = self._threshold(change, safe, params)
        regions = self._regions(mask, change, safe, float(params["threshold"]), work, min_area)
        params["num_regions"] = len(regions)
        low_otsu = (params.get("threshold_mode") == "otsu"
                    and float(params["otsu_threshold"]) < 0.8 * float(params.get("min_threshold_effective", c.min_threshold)))
        # Değişiklik yokken Otsu doğal olarak düşüktür; yalnızca çok sayıda bölgeyle birlikteyse şüpheli
        if len(regions) > MAX_REGIONS_RELIABLE or (low_otsu and len(regions) > MAX_REGIONS_RELIABLE // 4):
            params["unreliable"] = 1
            params["unreliable_reason"] = "Otsu çok düşük veya aşırı bölge sayısı (çözünürlük/ışık uyumsuzluğu olabilir)"
        else:
            params["unreliable"] = 0
        return DetectionResult(change_map=change, regions=regions, method=method_used, params=params)

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

    @staticmethod
    def _compact_strong(ssim_map: np.ndarray) -> np.ndarray:
        """SSIM skoru `COMPACT_STRONG` üstünde olan ve alanı `COMPACT_MAX_PX` altında kalan bileşenler (genişletilmiş)."""
        strong = (ssim_map >= COMPACT_STRONG).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] <= COMPACT_MAX_PX) & (stats[1:, cv2.CC_STAT_AREA] >= COMPACT_MIN_PX)
        if int(keep.sum()) > COMPACT_MAX_COUNT:
            # Çok sayıda küçük güçlü leke = yaygın SSIM gürültüsü (doku/tane): muafiyet yok
            return np.zeros(ssim_map.shape, bool)
        m = keep[labels].astype(np.uint8)
        return cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)

    def _dino_change_map(self, before: np.ndarray, after: np.ndarray, valid: np.ndarray,
                         safe: np.ndarray, params: dict) -> np.ndarray:
        """DINOv2 patch cosine distance -> 0..1 değişiklik haritası (aynı ışık normalizasyonundan sonra)."""
        c = self.cfg
        b, a = normalize_pair(before, after, valid, c.light_normalization)
        a = np.where(valid[..., None], a, b)
        if DINO_PREBLUR > 0:    # tane/JPEG farkı patch özniteliklerini bozmasın
            b = cv2.GaussianBlur(b, (0, 0), DINO_PREBLUR)
            a = cv2.GaussianBlur(a, (0, 0), DINO_PREBLUR)
        raw = self._dino.distance_map(b, a, int(c.dinov2_tile_px), bool(c.dinov2_fp16), params)
        params["dinov2_tile_px"] = int(c.dinov2_tile_px)
        params["dinov2_fp16"] = int(bool(c.dinov2_fp16))
        out = calibrate_dino(raw, safe, params)
        out[~valid] = 0.0
        return out

    def _threshold(self, change: np.ndarray, safe: np.ndarray, params: dict) -> np.ndarray:
        c = self.cfg
        if not safe.any():
            params["threshold"] = 1.0
            return np.zeros(change.shape, bool)
        if c.threshold == "otsu":
            vals = np.clip(change[safe] * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
            otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            min_thr = c.min_threshold * (1.0 + THRESHOLD_SCALE_GAIN * max(float(params.get("working_scale", 1.0)) - 1.0, 0.0))
            params["min_threshold_effective"] = round(min_thr, 4)
            base = max(float(otsu) / 255.0, min_thr)
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

    def _regions(self, mask: np.ndarray, change: np.ndarray, safe: np.ndarray, thr: float,
                 work: float = 1.0, min_area: int | None = None) -> list[ChangeRegion]:
        c = self.cfg
        min_area = c.min_region_area_px if min_area is None else min_area
        m = mask.astype(np.uint8)
        if c.morph_kernel_px > 1:
            k = _odd(int(round(c.morph_kernel_px * max(work, 1.0))))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
            m &= safe.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        found: list[tuple[float, BoundingBox, np.ndarray, int]] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < min_area:
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
