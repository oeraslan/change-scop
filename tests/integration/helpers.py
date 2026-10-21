"""Entegrasyon testleri için ortak yardımcılar: sentetik çift yazma, isabet değerlendirme, sonuç tablosu."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.contracts import AlignmentQuality, ComparisonResult, ImageInput
from core.pipeline import Pipeline
from scripts.make_synthetic_pair import PairParams, SyntheticPair, make_pair

# "Büyük" sahte bölge eşiği: görüntü alanının %0.1'i (1600x1200 için ~1900 px).
LARGE_AREA_FRACTION = 0.001
# Gerçek nesne bbox'ına bu kadar piksel yakın bölge "komşu" sayılır (hizalama kayması payı).
NEAR_PX = 40

RESULTS: list[dict] = []


def write_pair(pair: SyntheticPair, directory: Path, tag: str = "p") -> tuple[Path, Path]:
    """Çifti `before` PNG + `after` JPEG(100) olarak yazar (after zaten sıkıştırılmış)."""
    directory.mkdir(parents=True, exist_ok=True)
    before, after = directory / f"{tag}_before.png", directory / f"{tag}_after.jpg"
    cv2.imwrite(str(before), cv2.cvtColor(pair.before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(after), cv2.cvtColor(pair.after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    return before, after


def save_image(path: Path, array: np.ndarray) -> Path:
    """RGB/gri/RGBA/16-bit diziyi Pillow ile diske yazar."""
    Image.fromarray(array).save(path)
    return path


@dataclass
class Evaluation:
    """Bir sonucun gerçek (truth) ile karşılaştırması."""

    quality: str
    n_regions: int | None
    hits: list[bool]
    n_large_extra: int
    n_extra: int
    elapsed_s: float


def evaluate(pair: SyntheticPair, result: ComparisonResult) -> Evaluation:
    """Eklenen her nesne için bir bölge var mı; gerçek dışı (büyük) bölge kaç tane."""
    det = result.detection
    regions = det.regions if det else []
    hits = []
    for x, y, w, h in pair.added_bboxes:
        cx, cy = x + w / 2, y + h / 2
        hits.append(any(r.bbox.x <= cx <= r.bbox.x + r.bbox.width and
                        r.bbox.y <= cy <= r.bbox.y + r.bbox.height for r in regions))
    ih, iw = pair.before.shape[:2]
    large = LARGE_AREA_FRACTION * ih * iw
    extra = large_extra = 0
    for r in regions:
        b = r.bbox
        near = any(b.x < x + w + NEAR_PX and b.x + b.width > x - NEAR_PX and
                   b.y < y + h + NEAR_PX and b.y + b.height > y - NEAR_PX
                   for x, y, w, h in pair.added_bboxes)
        if not near:
            extra += 1
            large_extra += int(r.area_px >= large)
    return Evaluation(result.registration.quality.value, None if det is None else len(regions),
                      hits, large_extra, extra, result.elapsed_s)


def run_pair(pipeline: Pipeline, params: PairParams, directory: Path, tag: str) -> tuple[SyntheticPair, ComparisonResult]:
    """Sentetik çift üretir, diske yazar, pipeline'dan geçirir."""
    pair = make_pair(params)
    before, after = write_pair(pair, directory, tag)
    return pair, pipeline.run(ImageInput(before), ImageInput(after))


def record(scenario: str, ev: Evaluation) -> None:
    """Terminal özet tablosu için sonuç kaydeder."""
    RESULTS.append({"senaryo": scenario, "kalite": ev.quality, "bölge": ev.n_regions,
                    "isabet": f"{sum(ev.hits)}/{len(ev.hits)}", "büyük_fazla": ev.n_large_extra,
                    "fazla": ev.n_extra, "süre": f"{ev.elapsed_s:.1f}s"})


def assert_not_failed(result: ComparisonResult) -> None:
    """Hizalama FAILED değil ve tespit var."""
    assert result.registration.quality is not AlignmentQuality.FAILED, result.registration.notes
    assert result.detection is not None
