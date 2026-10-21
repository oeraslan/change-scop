"""Tespit testleri için yardımcılar: sentetik çiftten RegistrationResult kurar."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import AppConfig
from core.contracts import AlignmentQuality, ImageInput, PreparedImage, RegistrationResult
from scripts.make_synthetic_pair import PairParams, SyntheticPair, make_pair


def ssim_config() -> AppConfig:
    """Yöntemi açıkça "ssim" olan config (testler config varsayılanından bağımsız olsun)."""
    cfg = AppConfig()
    cfg.detection.method = "ssim"
    return cfg


def prepared(pixels: np.ndarray) -> PreparedImage:
    """Piksel dizisinden PreparedImage üretir."""
    return PreparedImage(ImageInput(path=Path("x.png")), pixels, 1.0, False)


def make_registration(
    before: np.ndarray,
    after_warped: np.ndarray,
    valid_mask: np.ndarray | None = None,
    quality: AlignmentQuality = AlignmentQuality.GOOD,
    homography: np.ndarray | None = None,
) -> RegistrationResult:
    """Hizalanmış iki görüntüden RegistrationResult kurar."""
    h, w = before.shape[:2]
    if valid_mask is None:
        valid_mask = np.ones((h, w), bool)
    return RegistrationResult(
        before=prepared(before), after_warped=after_warped, valid_mask=valid_mask,
        homography=(np.eye(3) if homography is None else homography) if quality != AlignmentQuality.FAILED else None,
        num_matches=100, num_inliers=90, reprojection_error_px=0.5, quality=quality,
    )


def registration_from_pair(pair: SyntheticPair) -> RegistrationResult:
    """Sentetik çiftte after'ı gerçek homografiyle before'a warp eder (registration modülü olmadan)."""
    h, w = pair.before.shape[:2]
    H = pair.homography_after_to_before
    warped = cv2.warpPerspective(pair.after, H, (w, h), flags=cv2.INTER_LINEAR)
    ones = np.full(pair.after.shape[:2], 255, np.uint8)
    vm = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST) > 0
    vm = cv2.erode(vm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return make_registration(pair.before, warped, vm, homography=H)


def pair_registration(params: PairParams | None = None) -> tuple[SyntheticPair, RegistrationResult]:
    """Sentetik çift + hizalanmış RegistrationResult."""
    pair = make_pair(params)
    return pair, registration_from_pair(pair)
