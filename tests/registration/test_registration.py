"""Hizalama testleri (SIFT her zaman; SuperPoint/LightGlue model dosyası varsa)."""

from __future__ import annotations

import socket

import cv2
import numpy as np
import pytest

from core.config import AppConfig
from core.contracts import AlignmentQuality
from core.registration import create_registrar
from core.registration.registrar import (
    DARK_BORDER_MAX_VALUE, content_mask, _tile_ranges)

from .conftest import corner_error_px, prepare_pair, requires_models, sift_config

MAX_CORNER_ERR = 2.0


def _lg_config(device: str = "auto") -> AppConfig:
    cfg = AppConfig()
    cfg.runtime.device = device
    return cfg


def _check_valid_mask(res, pair) -> None:
    """valid_mask: warp sonrası siyah kenarlar False, geçerli piksellerin neredeyse tamamı gerçek içerik."""
    valid = res.valid_mask
    assert valid.dtype == bool and valid.shape == res.after_warped.shape[:2] == res.before.pixels.shape[:2]
    assert 0.3 < valid.mean() < 1.0          # dönme+kaydırma yüzünden siyah kenarlar var
    black = res.after_warped.max(axis=2) <= DARK_BORDER_MAX_VALUE
    assert (black & valid).mean() < 0.002        # geçerli alanda siyah piksel (neredeyse) yok
    # Doğruluk verisine göre after kapsamı dışında kalan before pikselleri geçersiz olmalı
    H_true = pair.homography_after_to_before
    ah, aw = pair.after.shape[:2]
    ys, xs = np.mgrid[0:valid.shape[0]:7, 0:valid.shape[1]:7]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
    back = cv2.perspectiveTransform(pts, np.linalg.inv(H_true)).reshape(-1, 2)
    outside = (back[:, 0] < -3) | (back[:, 0] > aw + 3) | (back[:, 1] < -3) | (back[:, 1] > ah + 3)
    v = valid[ys.ravel(), xs.ravel()]
    assert not v[outside].any()
    assert v[~outside].mean() > 0.85         # içerideki piksellerin çoğu geçerli (erozyon payı)


# ---------------------------------------------------------------------------
# SIFT (her zaman çalışır)
# ---------------------------------------------------------------------------

def test_sift_aligns_synthetic_pair(default_pair, tmp_path):
    cfg = sift_config()
    b, a = prepare_pair(default_pair, tmp_path, cfg)
    reg = create_registrar(cfg)
    res = reg.register(b, a)
    assert res.quality == AlignmentQuality.GOOD, res.notes
    assert res.homography is not None
    assert corner_error_px(res.homography, default_pair.homography_after_to_before,
                           (a.pixels.shape[1], a.pixels.shape[0])) < MAX_CORNER_ERR
    assert any("SIFT" in n for n in res.notes)
    _check_valid_mask(res, default_pair)
    reg.release()


def test_sift_unrelated_images_fail(default_pair, other_scene, tmp_path):
    cfg = sift_config()
    b, _ = prepare_pair(default_pair, tmp_path / "x", cfg)
    _, a = prepare_pair(other_scene, tmp_path / "y", cfg)
    res = create_registrar(cfg).register(b, a)
    assert res.quality == AlignmentQuality.FAILED
    assert res.homography is None and not res.valid_mask.any()
    assert res.after_warped.shape == b.pixels.shape


def test_missing_weights_fall_back_to_sift(default_pair, tmp_path, monkeypatch):
    monkeypatch.setenv("CHANGESCOPE_MODELS_DIR", str(tmp_path / "bos"))
    cfg = AppConfig()              # matcher="lightglue", ama ağırlık yok
    b, a = prepare_pair(default_pair, tmp_path, cfg)
    reg = create_registrar(cfg)
    res = reg.register(b, a)
    assert reg.matcher_name == "sift"
    assert any("LightGlue yüklenemedi" in n and "SIFT" in n for n in res.notes)
    assert res.quality == AlignmentQuality.GOOD


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def test_tile_ranges_overlap_and_cover():
    r = _tile_ranges(2500, 1024, 0.10)
    assert r[0][0] == 0 and r[-1][1] == 2500
    assert all(e - s == 1024 for s, e in r)
    assert all(r[i + 1][0] < r[i][1] for i in range(len(r) - 1))      # örtüşme var
    assert _tile_ranges(800, 1024, 0.1) == [(0, 800)]


def test_content_mask_marks_black_border():
    img = np.full((200, 200, 3), 120, np.uint8)
    img[:, :50] = 0
    m = content_mask(img)
    assert not m[:, :50].any() and m[:, 60:].all()


# ---------------------------------------------------------------------------
# SuperPoint + LightGlue
# ---------------------------------------------------------------------------

@requires_models
def test_lightglue_aligns_synthetic_pair(default_pair, tmp_path):
    cfg = _lg_config()
    b, a = prepare_pair(default_pair, tmp_path, cfg)
    reg = create_registrar(cfg)
    res = reg.register(b, a)
    assert reg.matcher_name == "lightglue", res.notes
    assert res.quality == AlignmentQuality.GOOD, res.notes
    assert res.num_inliers >= 60 and res.num_inliers / res.num_matches >= 0.35
    assert res.reprojection_error_px <= 2.0
    err = corner_error_px(res.homography, default_pair.homography_after_to_before,
                          (a.pixels.shape[1], a.pixels.shape[0]))
    assert err < MAX_CORNER_ERR
    _check_valid_mask(res, default_pair)
    reg.release()
    assert reg.matcher_name is None


@requires_models
def test_lightglue_large_scale_difference(scale_pair, tmp_path):
    cfg = _lg_config()
    b, a = prepare_pair(scale_pair, tmp_path, cfg)
    assert b.pixels.shape[1] / a.pixels.shape[1] > 2.4      # before ~2.5x daha büyük
    res = create_registrar(cfg).register(b, a)
    assert res.quality == AlignmentQuality.GOOD, res.notes
    assert corner_error_px(res.homography, scale_pair.homography_after_to_before,
                           (a.pixels.shape[1], a.pixels.shape[0])) < 3.0
    assert any("ölçek" in n.lower() for n in res.notes)


@requires_models
def test_lightglue_different_field_of_view(tmp_path):
    """Drone-uydu benzeri: after, before'un yalnızca bir kısmını (yakın plan) kapsar."""
    from dataclasses import replace
    from scripts.make_synthetic_pair import PairParams, make_pair
    pair = make_pair(PairParams(seed=11))
    crop = pair.before[300:700, 400:1000]            # 600x400 yakın çekim
    big = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)     # 2x çözünürlük
    H_true = np.array([[0.5, 0, 400], [0, 0.5, 300], [0, 0, 1]], np.float64)       # after -> before
    pair2 = replace(pair, after=big, homography_after_to_before=H_true)
    cfg = _lg_config()
    b, a = prepare_pair(pair2, tmp_path, cfg)
    res = create_registrar(cfg).register(b, a)
    assert res.quality != AlignmentQuality.FAILED, res.notes
    assert corner_error_px(res.homography, H_true, (a.pixels.shape[1], a.pixels.shape[0])) < 4.0


@requires_models
def test_lightglue_unrelated_images_fail(default_pair, other_scene, tmp_path):
    cfg = _lg_config()
    b, _ = prepare_pair(default_pair, tmp_path / "x", cfg)
    _, a = prepare_pair(other_scene, tmp_path / "y", cfg)
    res = create_registrar(cfg).register(b, a)
    assert res.quality == AlignmentQuality.FAILED, res.notes
    assert res.homography is None and not res.valid_mask.any()


@requires_models
def test_lightglue_noise_images_fail(tmp_path):
    rng = np.random.default_rng(1)
    from core.contracts import ImageInput
    from core.registration import create_preprocessor
    import cv2 as _cv
    paths = []
    for i in range(2):
        p = tmp_path / f"n{i}.png"
        _cv.imwrite(str(p), rng.integers(0, 255, (600, 800, 3), dtype=np.uint8))
        paths.append(p)
    pre = create_preprocessor()
    res = create_registrar(_lg_config()).register(
        pre.prepare(ImageInput(path=paths[0])), pre.prepare(ImageInput(path=paths[1])))
    assert res.quality == AlignmentQuality.FAILED


@requires_models
def test_lightglue_loads_with_network_disabled(default_pair, tmp_path, monkeypatch):
    def no_net(*args, **kwargs):
        raise OSError("Ağ erişimi test sırasında yasak")

    cfg = _lg_config("cpu")
    b, a = prepare_pair(default_pair, tmp_path, cfg)
    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.setattr(socket, "create_connection", no_net)
    reg = create_registrar(cfg)
    res = reg.register(b, a)
    assert reg.matcher_name == "lightglue", res.notes      # indirme denenseydi SIFT'e düşerdi
    assert res.quality == AlignmentQuality.GOOD
    reg.release()


@requires_models
def test_lightglue_cpu_small_image(tmp_path):
    from scripts.make_synthetic_pair import PairParams, make_pair
    pair = make_pair(PairParams(width=640, height=480, seed=5, rotate_deg=6.0, scale=0.9,
                                shift_px=(10, -8), jpeg_quality=80))
    cfg = _lg_config("cpu")
    b, a = prepare_pair(pair, tmp_path, cfg)
    reg = create_registrar(cfg)
    res = reg.register(b, a)
    assert reg.matcher_name == "lightglue" and reg._matcher.device == "cpu"
    assert res.quality != AlignmentQuality.FAILED, res.notes
    assert corner_error_px(res.homography, pair.homography_after_to_before,
                           (a.pixels.shape[1], a.pixels.shape[0])) < 3.0


@requires_models
def test_release_then_reuse(default_pair, tmp_path):
    cfg = _lg_config()
    b, a = prepare_pair(default_pair, tmp_path, cfg)
    reg = create_registrar(cfg)
    r1 = reg.register(b, a)
    reg.release()
    reg.release()                          # ikinci çağrı zararsız
    r2 = reg.register(b, a)                # lazy yeniden yükleme
    assert r1.quality == r2.quality == AlignmentQuality.GOOD
