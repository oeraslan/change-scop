"""Hizalama: kaba → ince eşleştirme, MAGSAC homografi, warp, geçerlilik maskesi ve kalite değerlendirmesi.

Algoritma
    1. KABA aşama: iki görüntü `match_long_side_px` uzun kenara küçültülüp eşleştirilir. Alan
       genişlikleri çok farklıysa (drone vs uydu) `after` için ek ölçek hipotezleri
       (`COARSE_SCALE_HYPOTHESES`) denenir; en çok inlier'lı olan seçilir. Homografiden ölçek
       oranı (`sqrt(|det A|)`) tahmin edilip `notes`'a yazılır.
    2. İNCE aşama: `after` kaba homografi ile `before` tuvaline warp edilir; artık ikisi yaklaşık
       aynı ölçekte ve hizalıdır. `before` tile_size_px'lik tile'lara (%tile_overlap örtüşme) bölünüp
       her tile çifti yerel çözünürlükte eşleştirilir; eşleşmeler birleştirilir, `after`
       koordinatlarına geri taşınır ve MAGSAC ile son homografi kestirilir.
    3. `after` → `before` warp; `valid_mask` = iki görüntünün de içerik içerdiği piksel (siyah
       kenarlar ve warp dışı alanlar False, `valid_mask_erode_px` kadar aşındırılır).

Kalite eşikleri (varsayılanlar `RegistrationConfig`'ten gelir)
    GOOD   : inlier >= good_min_inliers (60) VE inlier oranı >= good_min_inlier_ratio (0.35)
             VE ortalama reprojeksiyon hatası <= good_max_reproj_error_px (2.0 px)
             VE inlier'lar örtüşme alanının en az `MIN_INLIER_HULL_FRACTION`'ını kapsıyor.
    WEAK   : inlier >= weak_min_inliers (15) VE oran >= `WEAK_MIN_INLIER_RATIO` (0.10)
             VE hata <= weak_max_reproj_error_px (5.0 px); GOOD değilse.
    FAILED : yukarıdakilerden biri sağlanmıyor, homografi geometrik olarak geçersiz
             (dışbükey olmayan/dejenere dörtgen, aşırı ölçek) ya da örtüşme
             `MIN_OVERLAP_FRACTION` altında. Bu durumda homografi None, maske tamamen False.
"""

from __future__ import annotations

import gc
import logging
import threading
import time

import cv2
import numpy as np

from core.config import AppConfig, resolve_device
from core.contracts import AlignmentQuality, PreparedImage, RegistrationResult
from core.registration.matching import LightGlueMatcher, Matcher, SiftMatcher

log = logging.getLogger(__name__)

# --- Config'de karşılığı olmayan sabitler -----------------------------------------------------
WEAK_MIN_INLIER_RATIO = 0.10        # Bunun altındaki inlier oranı rastgele eşleşme sayılır → FAILED
MIN_INLIER_HULL_FRACTION = 0.15     # Inlier dışbükey zarfı / örtüşme alanı; altındaysa GOOD → WEAK
MIN_OVERLAP_FRACTION = 0.005        # valid_mask / before alanı bunun altındaysa FAILED
MIN_OVERLAP_PIXELS = 64 * 64
COARSE_MIN_INLIERS = 8              # Kaba aşamada bundan azsa ince aşamaya geçilmez → FAILED
COARSE_SCALE_HYPOTHESES = (1.0, 1.6, 0.6, 2.5, 0.4)   # after uzun kenarı = match_long_side * f
COARSE_MAX_LONG_SIDE = 2048         # VRAM sınırı: kaba aşamada hiçbir görüntü bundan büyük olmaz
MIN_TILE_VALID_FRACTION = 0.25      # Tile'ın bu kadarı iki görüntüde de geçerli değilse atlanır
REFINE_RESIDUAL_GATE_FRAC = 0.05    # Kaba hizadan bu kadar (uzun kenarın oranı) sapan eşleşme atılır
MIN_SCALE, MAX_SCALE = 0.02, 50.0   # Kabul edilen homografi ölçek aralığı (after → before)
DARK_BORDER_MAX_VALUE = 8           # Siyah kenar tespitinde "karanlık" piksel üst sınırı
DARK_BORDER_MIN_AREA_FRAC = 0.002


def content_mask(rgb: np.ndarray) -> np.ndarray:
    """Görüntünün kenara değen saf siyah (warp/padding) bölgeleri dışındaki piksellerini işaretler."""
    h, w = rgb.shape[:2]
    dark = (rgb.max(axis=2) <= DARK_BORDER_MAX_VALUE).astype(np.uint8)
    valid = np.ones((h, w), bool)
    if not dark.any():
        return valid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches and area >= DARK_BORDER_MIN_AREA_FRAC * h * w:
            valid[labels == i] = False
    return valid


def estimate_scale(H: np.ndarray) -> float:
    """Homografinin yerel ölçeğini (görüntü merkezinde değil, afin kısımdan) `sqrt(|det A|)` olarak verir."""
    A = H[:2, :2] / H[2, 2]
    return float(np.sqrt(abs(np.linalg.det(A))))


def _tile_ranges(n: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    if n <= tile:
        return [(0, n)]
    stride = max(1, int(tile * (1.0 - overlap)))
    starts = list(range(0, n - tile + 1, stride))
    if starts[-1] + tile < n:
        starts.append(n - tile)
    return [(s, s + tile) for s in starts]


def _resize_long(img: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp), nw / w


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def _fit_homography(src: np.ndarray, dst: np.ndarray, thr: float) -> tuple[np.ndarray | None, np.ndarray]:
    """USAC_MAGSAC ile src → dst homografisi; (H, inlier bool maskesi) döndürür."""
    empty = np.zeros(len(src), bool)
    if len(src) < 4:
        return None, empty
    H, mask = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), cv2.USAC_MAGSAC,
                                 thr, maxIters=10000, confidence=0.999)
    if H is None or mask is None or not np.all(np.isfinite(H)):
        return None, empty
    return H / H[2, 2], mask.ravel().astype(bool)


def _geometry_ok(H: np.ndarray, size_a: tuple[int, int], size_b: tuple[int, int]) -> str | None:
    """Homografi makul değilse nedenini, makulse None döndürür. size = (w, h)."""
    wa, ha = size_a
    wb, hb = size_b
    corners = np.array([[0, 0], [wa, 0], [wa, ha], [0, ha]], np.float64)
    q = _apply_h(H, corners)
    if not np.all(np.isfinite(q)):
        return "homografi sonlu değil"
    if np.abs(q).max() > 30 * max(wb, hb):
        return "after dörtgeni aşırı büyük"
    quad = q.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(quad):
        return "warp edilen after dörtgeni dışbükey değil"
    if cv2.contourArea(quad) < 16:
        return "warp edilen after dörtgeni dejenere"
    s = estimate_scale(H)
    if not (MIN_SCALE <= s <= MAX_SCALE):
        return f"ölçek makul aralık dışında ({s:.3f})"
    return None


# Bu boyutun altındaki görüntülerde güvenilir özellik eşleştirmesi beklenmez.
SMALL_IMAGE_PX = 64


class DefaultRegistrar:
    """`Registrar` uygulaması. Modeller ilk `register()` çağrısında lazy yüklenir; `release()` bırakır."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._matcher: Matcher | None = None
        self._fallback_reason: str | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ model yaşam döngüsü
    @property
    def device(self) -> str:
        """Çözümlenmiş torch cihazı ("cuda" ya da "cpu")."""
        return resolve_device(self.config.runtime.device)

    @property
    def matcher_name(self) -> str | None:
        """Yüklü eşleştiricinin adı ("lightglue" / "sift"); henüz yüklenmediyse None."""
        return self._matcher.name if self._matcher else None

    def release(self) -> None:
        """Modelleri serbest bırakır (del + gc + CUDA varsa empty_cache). Sonraki register() yeniden yükler."""
        with self._lock:
            matcher, self._matcher = self._matcher, None
            self._fallback_reason = None
            if isinstance(matcher, LightGlueMatcher):
                matcher.close()
            del matcher
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _get_matcher(self) -> Matcher:
        if self._matcher is not None:
            return self._matcher
        cfg = self.config.registration
        if cfg.matcher == "lightglue":
            try:
                self._matcher = LightGlueMatcher(self.config, self.device)
                return self._matcher
            except Exception as exc:  # noqa: BLE001 - her yükleme hatası SIFT'e düşürür
                self._fallback_reason = f"LightGlue yüklenemedi ({type(exc).__name__}: {exc})"
                log.warning("%s; SIFT kullanılacak.", self._fallback_reason)
        else:
            self._fallback_reason = "SIFT yapılandırmada seçildi"
        self._matcher = SiftMatcher(cfg.max_keypoints * 2, cfg.sift_ratio)
        return self._matcher

    # ------------------------------------------------------------------ ana giriş
    def register(self, before: PreparedImage, after: PreparedImage) -> RegistrationResult:
        """`after` görüntüsünü `before`'un koordinat sistemine hizalar."""
        with self._lock:
            t0 = time.perf_counter()
            notes: list[str] = []
            matcher = self._get_matcher()
            if matcher.name == "sift" and self._fallback_reason:
                notes.append(f"SIFT + ratio test geri dönüşü kullanıldı: {self._fallback_reason}")
            try:
                result = self._register(matcher, before, after, notes)
            except Exception as exc:  # noqa: BLE001
                if not self._is_cuda_oom(exc) or not isinstance(matcher, LightGlueMatcher):
                    raise
                import torch
                torch.cuda.empty_cache()
                matcher.to("cpu")
                notes = [n for n in notes if "CUDA" not in n]
                notes.append("CUDA belleği yetmedi; eşleştirme CPU'da yeniden çalıştırıldı.")
                result = self._register(matcher, before, after, notes)
            result.notes.append(f"Hizalama süresi {time.perf_counter() - t0:.2f} s "
                                f"({matcher.name}, {getattr(matcher, 'device', 'cpu')}).")
            return result

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"

    # ------------------------------------------------------------------ algoritma
    def _register(self, matcher: Matcher, before: PreparedImage, after: PreparedImage,
                  notes: list[str]) -> RegistrationResult:
        cfg = self.config.registration
        notes = list(notes)
        pb, pa = before.pixels, after.pixels
        hb, wb = pb.shape[:2]
        ha, wa = pa.shape[:2]
        thr = cfg.ransac_reproj_threshold_px

        def failed(reason: str, n_matches: int = 0, n_inliers: int = 0, err: float = float("inf")):
            notes.append(f"Hizalama başarısız: {reason}")
            return RegistrationResult(
                before=before, after_warped=np.zeros_like(pb), valid_mask=np.zeros((hb, wb), bool),
                homography=None, num_matches=int(n_matches), num_inliers=int(n_inliers),
                reprojection_error_px=float(err), quality=AlignmentQuality.FAILED, notes=notes)

        if min(hb, wb, ha, wa) < SMALL_IMAGE_PX:
            notes.append(f"Görüntü hizalama için çok küçük ({wb}x{hb} / {wa}x{ha} px; "
                         f"en az {SMALL_IMAGE_PX} px önerilir).")

        # ---- 1) kaba aşama
        Hc, c_matches, c_inliers, c_err, c_hull = self._coarse(matcher, pb, pa, notes)
        if Hc is None or c_inliers < COARSE_MIN_INLIERS:
            return failed(f"kaba eşleştirmede yeterli inlier yok ({c_inliers})", c_matches, c_inliers)
        bad = _geometry_ok(Hc, (wa, ha), (wb, hb))
        if bad:
            return failed(f"kaba homografi geçersiz: {bad}", c_matches, c_inliers)
        notes.append(f"Kaba aşama: {c_inliers}/{c_matches} inlier, tahmini ölçek (after→before) "
                     f"{estimate_scale(Hc):.3f}.")

        # ---- 2) ince aşama (tile'lı)
        mask_b = content_mask(pb)
        mask_a = content_mask(pa)
        H, n_matches, n_inliers, err, hull_area = self._refine(
            matcher, pb, pa, mask_b, mask_a, Hc, notes)
        if H is None:
            # İnce aşama başarısız: kaba homografiyi kullan.
            H = Hc
            n_matches, n_inliers = c_matches, c_inliers
            notes.append("İnce aşama yetersiz; kaba homografi kullanıldı.")
            err, hull_area = c_err, c_hull
        bad = _geometry_ok(H, (wa, ha), (wb, hb))
        if bad:
            return failed(f"homografi geçersiz: {bad}", n_matches, n_inliers, err)

        # ---- 3) warp + maske
        flags = cv2.INTER_CUBIC if estimate_scale(H) > 1.0 else cv2.INTER_LINEAR
        warped = cv2.warpPerspective(pa, H, (wb, hb), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        wmask = cv2.warpPerspective(mask_a.astype(np.uint8), H, (wb, hb), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        valid = wmask & mask_b
        overlap_px = int(valid.sum())
        if cfg.valid_mask_erode_px > 0:
            k = 2 * cfg.valid_mask_erode_px + 1
            valid = cv2.erode(valid.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
        if overlap_px < MIN_OVERLAP_PIXELS or overlap_px / (hb * wb) < MIN_OVERLAP_FRACTION:
            return failed(f"örtüşme alanı çok küçük ({overlap_px} px)", n_matches, n_inliers, err)

        # ---- 4) kalite
        ratio = n_inliers / max(n_matches, 1)
        hull_frac = hull_area / max(overlap_px, 1)
        quality = self._grade(n_inliers, ratio, err, hull_frac, notes)
        if quality == AlignmentQuality.FAILED:
            return failed(f"inlier={n_inliers}, oran={ratio:.2f}, hata={err:.2f}px eşiklerin altında",
                          n_matches, n_inliers, err)
        notes.append(f"Son homografi: {n_inliers}/{n_matches} inlier (oran {ratio:.2f}), "
                     f"reprojeksiyon hatası {err:.2f} px, ölçek {estimate_scale(H):.3f}.")
        return RegistrationResult(
            before=before, after_warped=warped, valid_mask=valid, homography=H,
            num_matches=int(n_matches), num_inliers=int(n_inliers),
            reprojection_error_px=float(err), quality=quality, notes=notes)

    def _grade(self, n_inliers: int, ratio: float, err: float, hull_frac: float,
               notes: list[str]) -> AlignmentQuality:
        c = self.config.registration
        if (n_inliers < c.weak_min_inliers or ratio < WEAK_MIN_INLIER_RATIO
                or not np.isfinite(err) or err > c.weak_max_reproj_error_px):
            return AlignmentQuality.FAILED
        if (n_inliers >= c.good_min_inliers and ratio >= c.good_min_inlier_ratio
                and err <= c.good_max_reproj_error_px):
            if hull_frac < MIN_INLIER_HULL_FRACTION:
                notes.append("Inlier'lar örtüşme alanının küçük bir kısmında toplanmış; güvenilirlik düşük.")
                return AlignmentQuality.WEAK
            return AlignmentQuality.GOOD
        return AlignmentQuality.WEAK

    # ---- kaba aşama
    def _coarse(self, matcher: Matcher, pb: np.ndarray, pa: np.ndarray,
                notes: list[str]) -> tuple[np.ndarray | None, int, int, float, float]:
        """(H, eşleşme sayısı, inlier sayısı, ortalama hata, inlier zarf alanı) döndürür."""
        cfg = self.config.registration
        L = cfg.match_long_side_px
        b_img, sb = _resize_long(pb, min(L, COARSE_MAX_LONG_SIDE))
        best: tuple[int, np.ndarray | None, int, float] = (-1, None, 0, 1.0)
        best_err, best_hull = float("inf"), 0.0
        for f in COARSE_SCALE_HYPOTHESES:
            a_img, sa = _resize_long(pa, int(min(L * f, COARSE_MAX_LONG_SIDE)))
            p0, p1 = matcher.match(b_img, a_img)
            pts_b, pts_a = p0 / sb, p1 / sa
            thr = max(cfg.ransac_reproj_threshold_px, 2.0 / sb)
            H, inl = _fit_homography(pts_a, pts_b, thr)
            n_in = int(inl.sum())
            if n_in > best[0]:
                best = (n_in, H, len(pts_a), f)
                best_err = (float(np.linalg.norm(_apply_h(H, pts_a[inl]) - pts_b[inl], axis=1).mean())
                            if H is not None and n_in else float("inf"))
                best_hull = (float(cv2.contourArea(cv2.convexHull(pts_b[inl].astype(np.float32))))
                             if n_in >= 3 else 0.0)
            if (n_in >= cfg.good_min_inliers and len(pts_a)
                    and n_in / len(pts_a) >= cfg.good_min_inlier_ratio):
                break
        n_in, H, n_m, f = best
        if f != 1.0:
            notes.append(f"Ölçek farkı nedeniyle after {f:.1f}x ölçek hipoteziyle eşleştirildi.")
        return H, n_m, max(n_in, 0), best_err, best_hull

    # ---- ince aşama
    def _refine(self, matcher: Matcher, pb: np.ndarray, pa: np.ndarray, mask_b: np.ndarray,
                mask_a: np.ndarray, Hc: np.ndarray, notes: list[str]):
        cfg = self.config.registration
        hb, wb = pb.shape[:2]
        warp_a = cv2.warpPerspective(pa, Hc, (wb, hb), flags=cv2.INTER_LINEAR)
        warp_m = cv2.warpPerspective(mask_a.astype(np.uint8), Hc, (wb, hb),
                                     flags=cv2.INTER_NEAREST).astype(bool)
        both = warp_m & mask_b
        Hc_inv = np.linalg.inv(Hc)
        gate = REFINE_RESIDUAL_GATE_FRAC * max(hb, wb)

        all_b: list[np.ndarray] = []
        all_a: list[np.ndarray] = []
        n_tiles = 0
        for y0, y1 in _tile_ranges(hb, cfg.tile_size_px, cfg.tile_overlap):
            for x0, x1 in _tile_ranges(wb, cfg.tile_size_px, cfg.tile_overlap):
                if both[y0:y1, x0:x1].mean() < MIN_TILE_VALID_FRACTION:
                    continue
                n_tiles += 1
                p0, p1 = matcher.match(pb[y0:y1, x0:x1], warp_a[y0:y1, x0:x1])
                if len(p0) == 0:
                    continue
                off = np.array([x0, y0], np.float32)
                p0, p1 = p0 + off, p1 + off
                keep = np.linalg.norm(p0 - p1, axis=1) <= gate
                all_b.append(p0[keep])
                all_a.append(_apply_h(Hc_inv, p1[keep]).astype(np.float32))
        if not all_b:
            return None, 0, 0, float("inf"), 0.0
        pts_b = np.concatenate(all_b)
        pts_a = np.concatenate(all_a)
        # Tile örtüşmesinde mükerrer eşleşmeleri ele
        _, idx = np.unique(np.round(pts_b / 2.0).astype(np.int64), axis=0, return_index=True)
        pts_b, pts_a = pts_b[idx], pts_a[idx]
        H, inl = _fit_homography(pts_a, pts_b, cfg.ransac_reproj_threshold_px)
        n_in = int(inl.sum())
        notes.append(f"İnce aşama: {n_tiles} tile, {len(pts_a)} eşleşme, {n_in} inlier.")
        if H is None or n_in < cfg.weak_min_inliers:
            return None, len(pts_a), n_in, float("inf"), 0.0
        err = float(np.linalg.norm(_apply_h(H, pts_a[inl]) - pts_b[inl], axis=1).mean())
        hull = float(cv2.contourArea(cv2.convexHull(pts_b[inl].astype(np.float32)))) if n_in >= 3 else 0.0
        return H, len(pts_a), n_in, err, hull
