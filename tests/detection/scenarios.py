"""Yöntem karşılaştırması için sentetik senaryolar ve metrikler (gölge, doku/ton kayması dahil)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.contracts import DetectionResult, RegistrationResult
from scripts.make_synthetic_pair import PairParams, SyntheticPair, make_pair
from tests.detection.helpers import make_registration, registration_from_pair

SCENARIOS = ("default", "scale0.4", "strong_light_rot20_jpeg50", "shadow", "texture_tone", "no_change")


@dataclass
class Scenario:
    """Bir test senaryosu: hizalanmış çift + gerçek değişiklik maskesi (before koordinatları)."""

    name: str
    registration: RegistrationResult
    truth: np.ndarray                         # bool HxW
    bboxes: list[tuple[int, int, int, int]]   # (x, y, w, h)


def soft_polygon_mask(shape: tuple[int, int], pts: np.ndarray, feather: float = 12.0) -> np.ndarray:
    """Yumuşak kenarlı poligon maskesi (0..1 float32)."""
    m = np.zeros(shape, np.float32)
    cv2.fillPoly(m, [pts.astype(np.int32)], 1.0)
    return np.clip(cv2.GaussianBlur(m, (0, 0), feather), 0.0, 1.0)


def _free_spot(shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]], size: tuple[int, int],
               prefer: list[tuple[float, float]]) -> tuple[int, int]:
    """Nesne bbox'larıyla (boşluklu) çakışmayan sol-üst köşe; `prefer` oranlarını sırayla dener."""
    h, w = shape
    sw, sh = size
    for fx, fy in prefer:
        x, y = int((w - sw) * fx), int((h - sh) * fy)
        if all(x + sw + 40 < bx or bx + bw + 40 < x or y + sh + 40 < by or by + bh + 40 < y
               for bx, by, bw, bh in bboxes):
            return x, y
    return 20, 20


def add_shadow(img: np.ndarray, valid: np.ndarray, bboxes: list[tuple[int, int, int, int]],
               darken: float = 0.40) -> np.ndarray:
    """Görüntünün bir bölgesini `darken` oranında karartan yumuşak kenarlı gölge poligonu ekler."""
    h, w = img.shape[:2]
    sw, sh = int(w * 0.28), int(h * 0.26)
    x, y = _free_spot((h, w), bboxes, (sw, sh), [(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9), (0.5, 0.5)])
    pts = np.array([[x, y + sh * 0.2], [x + sw * 0.7, y], [x + sw, y + sh * 0.6], [x + sw * 0.3, y + sh]])
    m = soft_polygon_mask((h, w), pts)
    out = img.astype(np.float32) * (1.0 - darken * m[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def add_texture_tone_shift(img: np.ndarray, seed: int = 3) -> np.ndarray:
    """Mevsim benzeri kayma: yeşil alanlar sarı-kahveye döner, renk dengesi kayar, ince doku değişir."""
    rng = np.random.default_rng(seed)
    f = img.astype(np.float32)
    green = np.clip((f[..., 1] - np.maximum(f[..., 0], f[..., 2]) + 5.0) / 40.0, 0.0, 1.0)
    green = cv2.GaussianBlur(green, (0, 0), 3.0)[..., None]
    autumn = f * np.array([1.18, 0.95, 0.70], np.float32) + np.array([14.0, 4.0, -6.0], np.float32)
    f = f * (1 - green) + autumn * green
    f *= np.array([1.05, 1.0, 0.92], np.float32)                 # genel renk dengesi kayması
    h, w = img.shape[:2]
    coarse = cv2.resize(rng.normal(0, 1, (h // 24 + 2, w // 24 + 2)).astype(np.float32), (w, h),
                        interpolation=cv2.INTER_CUBIC)
    f *= (1.0 + 0.08 * coarse)[..., None]                         # orta ölçek doku
    f += rng.normal(0, 7.0, f.shape).astype(np.float32)           # ince doku/tane
    return np.clip(f, 0, 255).astype(np.uint8)


def tiny_object_scenario(scale: float, size: int = 20, width: int = 1600, height: int = 1200) -> Scenario:
    """Nesnesiz sahneye `size` px kırmızı kare ekler; after `scale` çözünürlüğünde + JPEG70 (integration `_tiny_pair`)."""
    pr = make_pair(PairParams(width=width, height=height, scale=scale, n_new_objects=0))
    ch = pr.before.copy()
    cx, cy = int(width * 0.45), int(height * 0.45)
    cv2.rectangle(ch, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), (230, 30, 30), -1)
    H = np.linalg.inv(pr.homography_after_to_before)
    after = cv2.warpPerspective(ch, H, (int(width * scale), int(height * scale)), flags=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    after = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    from dataclasses import replace
    reg = registration_from_pair(replace(pr, after=after))
    truth = np.zeros((height, width), bool)
    truth[cy - size // 2:cy + size // 2 + 1, cx - size // 2:cx + size // 2 + 1] = True
    return Scenario(f"tiny{size}_s{scale}", reg, truth, [(cx - size // 2, cy - size // 2, size + 1, size + 1)])


def build_scenario(name: str, seed: int = 42) -> Scenario:
    """Adlandırılmış senaryoyu kurar (bkz. `SCENARIOS`)."""
    if name == "default":
        p = PairParams(seed=seed)
    elif name == "scale0.4":
        p = PairParams(seed=seed, scale=0.4)
    elif name == "strong_light_rot20_jpeg50":
        p = PairParams(seed=seed, brightness=45.0, contrast=1.25, rotate_deg=20.0, jpeg_quality=50)
    elif name in ("shadow", "texture_tone"):
        p = PairParams(seed=seed, scale=1.0, rotate_deg=3.0, shift_px=(10.0, -6.0), jpeg_quality=80)
    elif name == "no_change":
        p = PairParams(seed=seed, n_new_objects=0)
    else:
        raise ValueError(name)
    pair: SyntheticPair = make_pair(p)
    reg = registration_from_pair(pair)
    after = reg.after_warped
    if name == "shadow":
        after = add_shadow(after, reg.valid_mask, pair.added_bboxes)
    elif name == "texture_tone":
        after = add_texture_tone_shift(after)
    if after is not reg.after_warped:
        reg = make_registration(pair.before, after, reg.valid_mask, homography=reg.homography)
    return Scenario(name, reg, pair.truth_mask > 0, list(pair.added_bboxes))


def regions_union(result: DetectionResult, shape: tuple[int, int]) -> np.ndarray:
    """Tüm bölge maskelerinin birleşimi (HxW bool)."""
    out = np.zeros(shape, bool)
    for r in result.regions:
        b = r.bbox
        out[b.y:b.y + b.height, b.x:b.x + b.width] |= r.mask
    return out


def evaluate(result: DetectionResult, sc: Scenario) -> dict[str, float]:
    """Bölge sayısı, isabet (nesne merkezi bir bölgede), sahte bölge (gerçek nesneye değmeyen) ve piksel P/R."""
    shape = sc.truth.shape
    union = regions_union(result, shape)
    near = cv2.dilate(sc.truth.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    hits = 0
    for x, y, w, h in sc.bboxes:
        cx, cy = x + w // 2, y + h // 2
        if any(r.bbox.x <= cx < r.bbox.x + r.bbox.width and r.bbox.y <= cy < r.bbox.y + r.bbox.height
               and r.mask[cy - r.bbox.y, cx - r.bbox.x] for r in result.regions) or \
                _any_overlap(result, sc, x, y, w, h):
            hits += 1
    false_regions = 0
    for r in result.regions:
        b = r.bbox
        if not (r.mask & near[b.y:b.y + b.height, b.x:b.x + b.width]).any():
            false_regions += 1
    tp = int((union & sc.truth).sum())
    prec = tp / union.sum() if union.any() else (1.0 if not sc.truth.any() else 0.0)
    rec = tp / sc.truth.sum() if sc.truth.any() else 1.0
    return {"regions": len(result.regions), "hits": hits, "objects": len(sc.bboxes),
            "false_regions": false_regions, "precision": float(prec), "recall": float(rec)}


def _any_overlap(result: DetectionResult, sc: Scenario, x: int, y: int, w: int, h: int) -> bool:
    """Nesne bbox'ının alanının en az yarısı bölge maskeleriyle örtüşüyor mu."""
    union = regions_union(result, sc.truth.shape)
    box = sc.truth[y:y + h, x:x + w]
    return bool(box.any() and (union[y:y + h, x:x + w] & box).sum() >= 0.5 * box.sum())
