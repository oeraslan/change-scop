"""UI testleri için sahte sonuç ve sahte pipeline bileşenleri (hızlı, model gerektirmez)."""

from __future__ import annotations

import threading
import time

import numpy as np
from PIL import Image

from core.config import AppConfig
from core.contracts import (
    AlignmentQuality,
    BoundingBox,
    ChangeRegion,
    ChangeVerdict,
    ComparisonResult,
    DetectionResult,
    ImageInput,
    PreparedImage,
    RegistrationResult,
)
from core.pipeline import Pipeline

SUMMARY = (
    "Kuzeydoğuda yeni bir yapı inşa edilmiş, yol kenarındaki iki araç artık yok. "
    "Batıdaki farklar gölge kaynaklıdır."
)


def scene(width: int, height: int, seed: int = 1, shift: int = 0) -> np.ndarray:
    """Basit, tanınabilir bir RGB sahne (gradyan + bloklar)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    img = np.stack([
        60 + 90 * (x / width), 90 + 80 * (y / height), 70 + 40 * np.sin(x / 40.0 + y / 55.0)
    ], axis=-1)
    img += rng.normal(0, 3, img.shape)
    for k in range(6):
        bx = (k * 131 + 40 + shift) % max(1, width - 60)
        by = (k * 97 + 30) % max(1, height - 50)
        img[by:by + 36, bx:bx + 52] = (200 - 20 * k, 180, 150 + 10 * k)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_regions(width: int, height: int, with_llm: bool) -> list[ChangeRegion]:
    """3 bölge: yeni yapı (gerçek), kayıp araç (gerçek), gölge (yanlış alarm)."""
    specs = [
        (0.62, 0.12, 0.18, 0.2, 0.91, ChangeVerdict.REAL_CHANGE, "Yeni bir yapı inşa edilmiş.", 0.93),
        (0.30, 0.62, 0.10, 0.1, 0.74, ChangeVerdict.REAL_CHANGE, "Yol kenarındaki araç kaybolmuş.", 0.81),
        (0.08, 0.35, 0.16, 0.18, 0.42, ChangeVerdict.FALSE_ALARM, "Gölge kaynaklı fark (yanlış alarm).", 0.88),
    ]
    regions = []
    for i, (fx, fy, fw, fh, score, verdict, desc, conf) in enumerate(specs, start=1):
        w, h = max(8, int(width * fw)), max(8, int(height * fh))
        x, y = int(width * fx), int(height * fy)
        region = ChangeRegion(
            region_id=i, bbox=BoundingBox(x, y, w, h), mask=np.ones((h, w), dtype=bool),
            score=score, area_px=w * h,
        )
        if with_llm:
            region.verdict, region.description, region.llm_confidence = verdict, desc, conf
        regions.append(region)
    return regions


def make_result(
    width: int = 320, height: int = 240, with_llm: bool = False,
    quality: AlignmentQuality = AlignmentQuality.GOOD, original=None,
) -> ComparisonResult:
    """Sahte `ComparisonResult` (FAILED ise detection=None)."""
    before_px = scene(width, height, seed=1)
    after_px = scene(width, height, seed=1, shift=12)
    original = original or ImageInput(path=__import__("pathlib").Path("before.png"))
    registration = RegistrationResult(
        before=PreparedImage(original=original, pixels=before_px, scale_from_original=1.0,
                             was_compressed=False),
        after_warped=after_px, valid_mask=np.ones((height, width), dtype=bool),
        homography=None if quality is AlignmentQuality.FAILED else np.eye(3),
        num_matches=120, num_inliers=90, reprojection_error_px=1.2, quality=quality,
        notes=["sahte"],
    )
    detection = None
    if quality is not AlignmentQuality.FAILED:
        cmap = np.zeros((height, width), dtype=np.float32)
        regions = make_regions(width, height, with_llm)
        for r in regions:
            b = r.bbox
            cmap[b.y:b.y + b.height, b.x:b.x + b.width] = r.score
        detection = DetectionResult(change_map=cmap, regions=regions, method="ssim")
    return ComparisonResult(
        registration=registration, detection=detection, llm_used=with_llm, elapsed_s=0.1,
        summary=SUMMARY if with_llm else None,
    )


# ---------------------------------------------------------------------------
# Sahte pipeline bileşenleri
# ---------------------------------------------------------------------------

class InstantPreprocessor:
    def prepare(self, image: ImageInput) -> PreparedImage:
        with Image.open(image.path) as img:
            px = np.array(img.convert("RGB"))
        return PreparedImage(original=image, pixels=px, scale_from_original=1.0, was_compressed=False)


class InstantRegistrar:
    def __init__(self, quality: AlignmentQuality = AlignmentQuality.GOOD, delay: float = 0.0) -> None:
        self.quality, self.delay = quality, delay

    def register(self, before: PreparedImage, after: PreparedImage) -> RegistrationResult:
        h, w = before.pixels.shape[:2]
        warped = np.array(Image.fromarray(after.pixels).resize((w, h)))
        if self.delay:
            time.sleep(self.delay)
        return RegistrationResult(
            before=before, after_warped=warped, valid_mask=np.ones((h, w), dtype=bool),
            homography=None if self.quality is AlignmentQuality.FAILED else np.eye(3),
            num_matches=100, num_inliers=80, reprojection_error_px=1.0, quality=self.quality,
        )


class InstantDetector:
    def detect(self, registration: RegistrationResult) -> DetectionResult:
        h, w = registration.before.pixels.shape[:2]
        regions = make_regions(w, h, with_llm=False)
        cmap = np.zeros((h, w), dtype=np.float32)
        for r in regions:
            cmap[r.bbox.y:r.bbox.y + r.bbox.height, r.bbox.x:r.bbox.x + r.bbox.width] = r.score
        return DetectionResult(change_map=cmap, regions=regions, method="ssim")


class InstantAnalyst:
    def is_available(self) -> bool:
        return True

    def analyze(self, registration, detection):
        h, w = registration.before.pixels.shape[:2]
        for mine, theirs in zip(make_regions(w, h, True), detection.regions):
            theirs.verdict, theirs.description = mine.verdict, mine.description
            theirs.llm_confidence = mine.llm_confidence
        return detection

    def summarize(self, registration, detection) -> str | None:
        return SUMMARY


def make_pipeline(
    quality: AlignmentQuality = AlignmentQuality.GOOD, delay: float = 0.0, with_llm: bool = True
) -> Pipeline:
    """Sahte bileşenlerle `Pipeline` (config: varsayılan, diskten okumaz)."""
    config = AppConfig()
    config.runtime.release_cv_models_before_llm = False   # torch içe aktarma maliyetinden kaçın
    return Pipeline(
        InstantPreprocessor(), InstantRegistrar(quality, delay), InstantDetector(),
        analyst_factory=(lambda: InstantAnalyst()) if with_llm else None, config=config,
    )


class BlockingRegistrar(InstantRegistrar):
    """İptal testi için: `release` event'i set edilene kadar bekler."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()

    def register(self, before, after):
        self.entered.set()
        self.gate.wait(5)
        return super().register(before, after)
