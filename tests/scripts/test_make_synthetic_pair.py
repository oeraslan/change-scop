"""scripts/make_synthetic_pair.py testleri."""

from __future__ import annotations

import json

import cv2
import numpy as np

from scripts.make_synthetic_pair import PairParams, make_pair, save_pair


def test_pair_is_deterministic():
    a = make_pair(PairParams(seed=3, width=400, height=300))
    b = make_pair(PairParams(seed=3, width=400, height=300))
    assert np.array_equal(a.before, b.before) and np.array_equal(a.after, b.after)


def test_homography_aligns_unchanged_area_and_truth_is_visible():
    pair = make_pair(PairParams(width=800, height=600))
    h, w = pair.before.shape[:2]
    assert pair.after.shape[:2] == (int(h * 0.75), int(w * 0.75))

    warped = cv2.warpPerspective(pair.after, pair.homography_after_to_before, (w, h))
    valid = cv2.warpPerspective(np.ones(pair.after.shape[:2], np.uint8),
                                pair.homography_after_to_before, (w, h))
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0
    truth = pair.truth_mask > 0
    assert truth.any() and (truth & valid).sum() == truth.sum()   # değişiklik örtüşme alanında

    diff = np.abs(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(float)
                  - cv2.cvtColor(pair.before, cv2.COLOR_RGB2GRAY).astype(float))
    # Değişen bölge, değişmeyen bölgeden (ışık farkına rağmen) belirgin şekilde farklı olmalı
    assert diff[truth].mean() > 2 * diff[valid & ~truth].mean()


def test_save_pair_writes_files(tmp_path):
    pair = make_pair(PairParams(width=400, height=300))
    save_pair(pair, tmp_path)
    for name in ("before.png", "after.jpg", "truth_mask.png", "truth.json"):
        assert (tmp_path / name).is_file()
    truth = json.loads((tmp_path / "truth.json").read_text(encoding="utf-8"))
    assert len(truth["added_bboxes_xywh"]) == 2
    assert np.array(truth["homography_after_to_before"]).shape == (3, 3)
