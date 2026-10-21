"""Geliştirme demosu: gerçek modüller bağlanana kadar OpenCV tabanlı SAHTE pipeline bileşenleri.

`python -m app.main --demo` bu bileşenleri `core.pipeline.set_default_pipeline` ile kurar ve geçici
bir kütüphaneyi sentetik görüntü çiftleriyle doldurur. Üretim kodu bunları kullanmaz.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.storage import Library
from core.config import AppConfig
from core.contracts import (
    AlignmentQuality,
    BoundingBox,
    ChangeRegion,
    ChangeVerdict,
    DetectionResult,
    ImageInput,
    ImageSource,
    PreparedImage,
    RegistrationResult,
)
from core.pipeline import Pipeline, set_default_pipeline

DEMO_SUMMARY = (
    "Kuzeydoğuda yeni bir yapı inşa edilmiş, yol kenarındaki araç artık yok. "
    "Batıdaki farklar gölge ve parlaklık kaynaklı olup gerçek değişiklik değildir."
)


class DemoPreprocessor:
    """Dosyayı RGB uint8 olarak okur (denoise yok)."""

    def prepare(self, image: ImageInput) -> PreparedImage:
        with Image.open(image.path) as img:
            px = np.array(img.convert("RGB"))
        return PreparedImage(image, px, 1.0, image.path.suffix.lower() in {".jpg", ".jpeg"})


class DemoRegistrar:
    """SIFT + RANSAC homografisi ile hizalar (yalnızca demo)."""

    def register(self, before: PreparedImage, after: PreparedImage) -> RegistrationResult:
        time.sleep(0.4)
        h, w = before.pixels.shape[:2]
        g1 = cv2.cvtColor(before.pixels, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(after.pixels, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(3000)
        k1, d1 = sift.detectAndCompute(g1, None)
        k2, d2 = sift.detectAndCompute(g2, None)
        H, inliers, matches = None, 0, 0
        if d1 is not None and d2 is not None and len(k1) > 8 and len(k2) > 8:
            pairs = cv2.BFMatcher().knnMatch(d2, d1, k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
            matches = len(good)
            if matches >= 8:
                src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                inliers = int(mask.sum()) if mask is not None else 0
        if H is None or inliers < 15:
            return RegistrationResult(before, after.pixels.copy() if after.pixels.shape == before.pixels.shape
                                      else cv2.resize(after.pixels, (w, h)),
                                      np.zeros((h, w), bool), None, matches, inliers, 99.0,
                                      AlignmentQuality.FAILED, ["Yeterli eşleşme bulunamadı"])
        warped = cv2.warpPerspective(after.pixels, H, (w, h))
        valid = cv2.warpPerspective(np.full(after.pixels.shape[:2], 255, np.uint8), H, (w, h)) > 200
        valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        quality = AlignmentQuality.GOOD if inliers >= 60 else AlignmentQuality.WEAK
        return RegistrationResult(before, warped, valid, H, matches, inliers, 1.0, quality,
                                  ["Demo hizalama (SIFT)"] if quality is AlignmentQuality.WEAK else [])


class DemoDetector:
    """Işık normalizasyonu + bulanık mutlak fark + Otsu + bağlı bileşenler (yalnızca demo)."""

    def detect(self, registration: RegistrationResult) -> DetectionResult:
        time.sleep(0.4)
        gb = cv2.cvtColor(registration.before.pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ga = cv2.cvtColor(registration.after_warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
        valid = registration.valid_mask
        if valid.sum() > 100:
            ga = (ga - ga[valid].mean()) / (ga[valid].std() + 1e-6) * gb[valid].std() + gb[valid].mean()
        diff = np.abs(cv2.GaussianBlur(gb, (0, 0), 3) - cv2.GaussianBlur(ga, (0, 0), 3))
        diff[~valid] = 0
        scale = np.percentile(diff[valid], 99.0) + 1e-6 if valid.any() else 1.0
        cmap = np.clip(diff / scale, 0, 1).astype(np.float32)
        u8 = (cmap * 255).astype(np.uint8)
        otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (u8 > max(otsu * 0.8, 0.28 * 255)).astype(np.uint8)
        k = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        found = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 400:
                continue
            m = labels[y:y + h, x:x + w] == i
            found.append((float(cmap[y:y + h, x:x + w][m].mean()), int(x), int(y), int(w), int(h), int(area), m))
        found.sort(key=lambda t: -t[0])
        regions = [
            ChangeRegion(i, BoundingBox(x, y, w, h), m, min(1.0, s), a)
            for i, (s, x, y, w, h, a, m) in enumerate(found[:12], start=1)
        ]
        return DetectionResult(cmap, regions, "demo-diff", {"otsu": float(otsu), "method_requested": AppConfig().detection.method})


class DemoAnalyst:
    """Skora göre sahte verdict/açıklama atar (yalnızca demo)."""

    def is_available(self) -> bool:
        return True

    def analyze(self, registration: RegistrationResult, detection: DetectionResult) -> DetectionResult:
        time.sleep(0.8)
        texts = [
            (ChangeVerdict.REAL_CHANGE, "Yeni bir yapı ya da nesne eklenmiş görünüyor.", 0.92),
            (ChangeVerdict.REAL_CHANGE, "Bir nesne kaybolmuş ya da yer değiştirmiş.", 0.80),
            (ChangeVerdict.UNCERTAIN, "Fark belirgin ama nedeni net değil; elle kontrol önerilir.", 0.55),
            (ChangeVerdict.FALSE_ALARM, "Gölge veya parlaklık kaynaklı fark; gerçek değişiklik değil.", 0.86),
        ]
        for i, region in enumerate(detection.regions):
            verdict, text, conf = texts[min(i, len(texts) - 1)] if i < 3 else texts[3]
            region.verdict, region.description, region.llm_confidence = verdict, text, conf
        return detection

    def summarize(self, registration: RegistrationResult, detection: DetectionResult) -> str | None:
        time.sleep(0.4)
        return DEMO_SUMMARY


def install_demo_pipeline(with_llm: bool = True) -> Pipeline:
    """Sahte bileşenlerle bir `Pipeline` kurar ve varsayılan yapar."""
    config = AppConfig()
    config.runtime.release_cv_models_before_llm = False   # sahte bileşenlerde VRAM boşaltmaya (torch içe aktarmaya) gerek yok
    pipeline = Pipeline(
        DemoPreprocessor(), DemoRegistrar(), DemoDetector(),
        analyst_factory=(lambda: DemoAnalyst()) if with_llm else None, config=config,
    )
    set_default_pipeline(pipeline)
    return pipeline


def seed_demo_library(library: Library, work_dir: Path | None = None, size: tuple[int, int] = (960, 720)) -> None:
    """Boş kütüphaneyi örnek klasörler ve sentetik görüntü çiftleriyle doldurur."""
    from scripts.make_synthetic_pair import PairParams, make_pair

    work = Path(work_dir or tempfile.mkdtemp(prefix="changescope_seed_"))
    work.mkdir(parents=True, exist_ok=True)
    repo = library.repo
    x = repo.create_folder("X Bölgesi")
    y = repo.create_folder("Y Dağı", x.id)
    north = repo.create_folder("Kuzey Yamaç", y.id)
    south = repo.create_folder("Güney Vadi", y.id)
    port = repo.create_folder("Z Limanı")
    plan = [
        (north.id, 42, ImageSource.DRONE, (2026, 10, 12), (2026, 10, 19), ImageSource.UAV),
        (south.id, 7, ImageSource.SATELLITE, (2026, 9, 28), (2026, 10, 5), ImageSource.SATELLITE),
        (port.id, 99, ImageSource.UAV, (2026, 8, 3), (2026, 8, 24), ImageSource.DRONE),
    ]
    for folder_id, seed, src_a, date_a, date_b, src_b in plan:
        pair = make_pair(PairParams(seed=seed, width=size[0], height=size[1]))
        b_path, a_path = work / f"sahne{seed}_once.png", work / f"sahne{seed}_sonra.jpg"
        Image.fromarray(pair.before).save(b_path)
        Image.fromarray(pair.after).save(a_path, quality=82)
        library.import_image(b_path, folder_id, datetime(*date_a, 10, 30), src_a)
        library.import_image(a_path, folder_id, datetime(*date_b, 11, 15), src_b)
    # Galeriyi dolduran ek görüntüler (aynı sahnenin farklı ışık/açı varyantları)
    for k, (folder_id, seed) in enumerate([(north.id, 42), (north.id, 5), (south.id, 7)]):
        pair = make_pair(PairParams(seed=seed + 100, width=size[0], height=size[1], rotate_deg=4 + k))
        p = work / f"ek{k}.jpg"
        Image.fromarray(pair.after).save(p, quality=85)
        library.import_image(p, folder_id, datetime(2026, 10, 1 + 3 * k, 9, 0), ImageSource.DRONE)
