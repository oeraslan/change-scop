"""Ön işleme testleri."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from core.config import AppConfig
from core.contracts import ImageInput
from core.registration import create_preprocessor


def _scene(w: int = 320, h: int = 240) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2.5)
    cv2.rectangle(img, (40, 40), (160, 120), (200, 30, 30), -1)
    return img


def test_png_rgb_not_compressed(tmp_path):
    img = _scene()
    p = tmp_path / "a.png"
    Image.fromarray(img).save(p)
    out = create_preprocessor().prepare(ImageInput(path=p))
    assert out.pixels.shape == img.shape and out.pixels.dtype == np.uint8
    assert np.array_equal(out.pixels, img)          # RGB sırası korunur
    assert out.scale_from_original == 1.0
    assert out.was_compressed is False


def test_low_quality_jpeg_detected_and_denoised(tmp_path):
    img = _scene()
    p = tmp_path / "a.jpg"
    Image.fromarray(img).save(p, quality=40)
    out = create_preprocessor().prepare(ImageInput(path=p))
    assert out.was_compressed is True
    decoded = np.asarray(Image.open(p).convert("RGB"))
    assert not np.array_equal(out.pixels, decoded)  # denoise uygulandı
    assert np.abs(out.pixels.astype(int) - decoded.astype(int)).mean() < 8   # ama hafif


def test_high_quality_jpeg_not_flagged(tmp_path):
    p = tmp_path / "a.jpg"
    Image.fromarray(_scene()).save(p, quality=97)
    assert create_preprocessor().prepare(ImageInput(path=p)).was_compressed is False


def test_denoise_can_be_disabled(tmp_path):
    p = tmp_path / "a.jpg"
    Image.fromarray(_scene()).save(p, quality=40)
    cfg = AppConfig()
    cfg.preprocess.denoise_if_compressed = False
    out = create_preprocessor(cfg).prepare(ImageInput(path=p))
    assert out.was_compressed is True
    assert np.array_equal(out.pixels, np.asarray(Image.open(p).convert("RGB")))


def test_exif_orientation_applied(tmp_path):
    img = _scene(320, 240)
    im = Image.fromarray(img)
    exif = Image.Exif()
    exif[0x0112] = 6                     # 90 derece saat yönü döndür
    p = tmp_path / "rot.jpg"
    im.save(p, quality=98, exif=exif)
    out = create_preprocessor().prepare(ImageInput(path=p))
    assert out.pixels.shape[:2] == (320, 240)


def test_rgba_and_grayscale_to_rgb(tmp_path):
    Image.fromarray(np.full((64, 64), 128, np.uint8)).save(tmp_path / "g.png")
    Image.fromarray(np.dstack([_scene(64, 64), np.full((64, 64), 255, np.uint8)])).save(tmp_path / "rgba.png")
    pre = create_preprocessor()
    assert pre.prepare(ImageInput(path=tmp_path / "g.png")).pixels.shape == (64, 64, 3)
    assert pre.prepare(ImageInput(path=tmp_path / "rgba.png")).pixels.shape == (64, 64, 3)


def test_large_image_downscaled(tmp_path):
    cfg = AppConfig()
    cfg.preprocess.max_side_px = 500
    p = tmp_path / "big.png"
    Image.fromarray(_scene(1000, 600)).save(p)
    out = create_preprocessor(cfg).prepare(ImageInput(path=p))
    assert out.pixels.shape[:2] == (300, 500)
    assert out.scale_from_original == pytest.approx(0.5)


def test_corrupt_file_raises_value_error(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"bu bir goruntu degil" * 10)
    with pytest.raises(ValueError, match="okunamadı"):
        create_preprocessor().prepare(ImageInput(path=p))


def test_truncated_png_raises_value_error(tmp_path):
    p = tmp_path / "t.png"
    Image.fromarray(_scene()).save(p)
    p.write_bytes(p.read_bytes()[:200])
    with pytest.raises(ValueError):
        create_preprocessor().prepare(ImageInput(path=p))


def test_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="bulunamadı"):
        create_preprocessor().prepare(ImageInput(path=tmp_path / "yok.png"))
