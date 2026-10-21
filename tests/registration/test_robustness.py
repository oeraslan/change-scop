"""Düzeltme turu testleri: 16-bit PNG, decompression bomb, küçük görüntü, torch.hub yaması."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from core.config import AppConfig
from core.contracts import ImageInput
from core.registration import create_preprocessor, create_registrar
from core.registration.matching import _local_state_dict_loader

from .conftest import sift_config


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (200, 240, 3), dtype=np.uint8), (0, 0), 3)
    return cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)


def test_16bit_gray_png_normalised(tmp_path):
    gray8 = cv2.cvtColor(_scene(), cv2.COLOR_RGB2GRAY)
    path = tmp_path / "g16.png"
    cv2.imwrite(str(path), gray8.astype(np.uint16) * 257)
    px = create_preprocessor(AppConfig()).prepare(ImageInput(path)).pixels
    assert px.shape[2] == 3
    assert abs(float(px[..., 0].mean()) - float(gray8.mean())) < 10
    assert np.array_equal(px[..., 0], px[..., 1])


def test_16bit_gray_low_range_is_stretched(tmp_path):
    """Düşük dinamik aralıklı 16-bit görüntü beyaza doymamalı, kontrast açılmalı."""
    gray = cv2.cvtColor(_scene(), cv2.COLOR_RGB2GRAY).astype(np.uint16) * 16
    path = tmp_path / "low.png"
    cv2.imwrite(str(path), gray)
    px = create_preprocessor(AppConfig()).prepare(ImageInput(path)).pixels
    assert px.max() >= 250 and px.min() <= 5 and 60 < px.mean() < 200


def test_16bit_rgb_png_normalised(tmp_path):
    rgb8 = _scene()
    path = tmp_path / "c16.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR).astype(np.uint16) * 257)
    px = create_preprocessor(AppConfig()).prepare(ImageInput(path)).pixels
    assert px.shape == rgb8.shape
    for c in range(3):   # kanal sırası korunmalı
        assert abs(float(px[..., c].mean()) - float(rgb8[..., c].mean())) < 12


def test_pillow_i16_mode_normalised(tmp_path):
    arr = (np.linspace(0, 1, 100 * 120).reshape(100, 120) * 60000).astype(np.uint16)
    path = tmp_path / "i16.tiff"
    Image.fromarray(arr).save(path)
    px = create_preprocessor(AppConfig()).prepare(ImageInput(path)).pixels
    assert 100 < px.mean() < 155


def test_decompression_bomb_message(monkeypatch, tmp_path):
    path = tmp_path / "bomb.png"
    cv2.imwrite(str(path), np.zeros((300, 300, 3), np.uint8))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(ValueError, match="çok büyük"):
        create_preprocessor(AppConfig()).prepare(ImageInput(path))


def test_pixel_limit_raised_but_bounded():
    from core.registration import preprocess
    assert 89_000_000 < Image.MAX_IMAGE_PIXELS <= preprocess.MAX_IMAGE_PIXELS < 1_000_000_000


@pytest.mark.parametrize("size", [16, 40])
def test_tiny_images_get_explanatory_note(tmp_path, size):
    img = cv2.resize(_scene(), (size, size))
    path = tmp_path / f"t{size}.png"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    p = create_preprocessor(sift_config()).prepare(ImageInput(path))
    res = create_registrar(sift_config()).register(p, p)
    assert any("çok küçük" in n for n in res.notes), res.notes


def test_patched_loader_rejects_foreign_url_without_network(monkeypatch):
    """Başka bir thread/kütüphane yamalı fonksiyonu çağırsa bile ağa çıkılmaz."""
    import socket

    def no_net(*a, **k):
        raise AssertionError("ağ erişimi denendi")

    monkeypatch.setattr(socket.socket, "connect", no_net)
    loader = _local_state_dict_loader(AppConfig())
    with pytest.raises(FileNotFoundError):
        loader("https://example.com/resnet50-0676ba61.pth")
    with pytest.raises(FileNotFoundError):
        loader("https://evil.example/other.pth?x=1")


def test_hub_patch_is_restored_after_loading():
    """Model kurulumu bittikten sonra torch.hub orijinal haline dönmüş olmalı."""
    torch = pytest.importorskip("torch")
    cfg = AppConfig()
    if not (cfg.paths.superpoint_weights.is_file() and cfg.paths.lightglue_weights.is_file()):
        pytest.skip("model dosyaları yok")
    from core.registration.matching import LightGlueMatcher
    orig = torch.hub.load_state_dict_from_url
    m = LightGlueMatcher(cfg, "cpu")
    try:
        assert torch.hub.load_state_dict_from_url is orig
    finally:
        m.close()
