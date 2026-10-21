"""Gerçek modüllerle uçtan uca duman testi (sentetik çift)."""

from __future__ import annotations

import cv2
import pytest

from core.config import AppConfig
from core.contracts import AlignmentQuality, ImageInput
from core.pipeline import build_default_pipeline
from scripts.make_synthetic_pair import make_pair


def test_default_pipeline_finds_added_objects(tmp_path):
    pair = make_pair()
    before, after = tmp_path / "before.png", tmp_path / "after.jpg"
    cv2.imwrite(str(before), cv2.cvtColor(pair.before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(after), cv2.cvtColor(pair.after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

    pipe = build_default_pipeline(AppConfig())   # kullanıcının config.json'u testi etkilemesin
    stages: list[str] = []
    result = pipe.run(ImageInput(before), ImageInput(after), lambda s, f, m="": stages.append(s),  # LLM: tests/llm
                      use_llm=False)

    assert result.registration.quality is AlignmentQuality.GOOD
    assert result.detection is not None and result.llm_used is False
    regions = result.detection.regions
    assert 1 <= len(regions) <= 4
    for x, y, w, h in pair.added_bboxes:
        cx, cy = x + w / 2, y + h / 2
        assert any(r.bbox.x <= cx <= r.bbox.x + r.bbox.width and
                   r.bbox.y <= cy <= r.bbox.y + r.bbox.height for r in regions), (x, y, w, h)
    assert stages[-1] == "done"
    pipe.shutdown()
