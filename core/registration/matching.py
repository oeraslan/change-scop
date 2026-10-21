"""Özellik eşleştiriciler: SuperPoint + LightGlue (offline) ve SIFT geri dönüşü.

Her eşleştirici `match(img0, img1) -> (pts0, pts1)` sağlar; girdiler RGB uint8, çıktılar
Nx2 float32 piksel koordinatlarıdır (görüntülerin kendi koordinat sisteminde).

Offline kuralı: `lightglue` paketi ağırlıkları `torch.hub.load_state_dict_from_url` ile indirir.
Model kurulurken bu fonksiyon `unittest.mock.patch` ile yerel bir yükleyiciye yönlendirilir;
dosya yoksa `FileNotFoundError` fırlatılır (ağa asla çıkılmaz).
"""

from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import Protocol
from unittest import mock

import cv2
import numpy as np

from core.config import AppConfig

log = logging.getLogger(__name__)

_HUB_PATCH_LOCK = threading.Lock()


class Matcher(Protocol):
    name: str

    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def _local_state_dict_loader(config: AppConfig):
    """`torch.hub.load_state_dict_from_url` yerine geçen, yalnızca yerel dosya okuyan fonksiyon."""
    import torch

    def loader(url: str, *args, **kwargs):  # noqa: ANN001
        name = Path(str(url).split("?")[0]).name
        if name.startswith("superpoint_lightglue"):
            path = config.paths.lightglue_weights
        elif name.startswith("superpoint_v1"):
            path = config.paths.superpoint_weights
        else:
            raise FileNotFoundError(f"Bilinmeyen ağırlık isteği (ağa çıkılmaz): {name}")
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model ağırlığı bulunamadı: {path}")
        return torch.load(str(path), map_location="cpu", weights_only=True)

    return loader


class LightGlueMatcher:
    """SuperPoint + LightGlue; ağırlıklar yalnızca `models/` altından yüklenir."""

    name = "lightglue"

    def __init__(self, config: AppConfig, device: str) -> None:
        import torch
        from lightglue import LightGlue, SuperPoint

        self.torch = torch
        self.device = device
        n_kp = config.registration.max_keypoints
        with _HUB_PATCH_LOCK, mock.patch.object(
            torch.hub, "load_state_dict_from_url", _local_state_dict_loader(config)
        ):
            extractor = SuperPoint(max_num_keypoints=n_kp)
            matcher = LightGlue(features="superpoint")
        self.extractor = extractor.eval().to(device)
        self.matcher = matcher.eval().to(device)

    def to(self, device: str) -> None:
        """Modelleri başka bir cihaza taşır (CUDA OOM → CPU geri dönüşü)."""
        self.extractor.to(device)
        self.matcher.to(device)
        self.device = device

    def _tensor(self, rgb: np.ndarray):
        t = self.torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.0
        return t.to(self.device)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            f0 = self.extractor.extract(self._tensor(img0))
            f1 = self.extractor.extract(self._tensor(img1))
            out = self.matcher({"image0": f0, "image1": f1})
            matches = out["matches"][0]
            if matches.numel() == 0:
                return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
            p0 = f0["keypoints"][0][matches[:, 0]].float().cpu().numpy()
            p1 = f1["keypoints"][0][matches[:, 1]].float().cpu().numpy()
        return p0.astype(np.float32), p1.astype(np.float32)

    def close(self) -> None:
        del self.extractor, self.matcher
        gc.collect()
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class SiftMatcher:
    """SIFT + Lowe ratio testi (LightGlue yüklenemezse geri dönüş)."""

    name = "sift"

    def __init__(self, max_keypoints: int = 4096, ratio: float = 0.75) -> None:
        self.ratio = ratio
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _features(self, rgb: np.ndarray):
        gray = self.clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        return self.sift.detectAndCompute(gray, None)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k0, d0 = self._features(img0)
        k1, d1 = self._features(img1)
        if d0 is None or d1 is None or len(k0) < 4 or len(k1) < 4:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
        if not good:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        return p0, p1
