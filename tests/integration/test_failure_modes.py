"""Hata durumları: alakasız görüntü, bozuk/küçük dosya, farklı piksel biçimleri, eksik model."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from core.config import AppConfig
from core.contracts import AlignmentQuality, ImageInput
from core.pipeline import build_default_pipeline
from core.registration import create_preprocessor
from scripts.make_synthetic_pair import PairParams, make_pair
from tests.integration.helpers import evaluate, record, save_image, write_pair


@pytest.fixture(scope="module")
def pair():
    return make_pair(PairParams())


@pytest.fixture(scope="module")
def files(pair, tmp_path_factory):
    """Ortak giriş dosyaları."""
    d = tmp_path_factory.mktemp("fail")
    before, after = write_pair(pair, d, "ok")
    return {"dir": d, "before": before, "after": after}


# --------------------------------------------------------------------------- alakasız görüntüler

def test_unrelated_scenes_fail_alignment(real_pipeline, files, tmp_path) -> None:
    """Aynı bölge olmayan iki sahne → FAILED, detection None, homografi None, maske tamamen False."""
    other = make_pair(PairParams(seed=999))
    other_path = save_image(tmp_path / "other.png", other.before)
    result = real_pipeline.run(ImageInput(files["before"]), ImageInput(other_path))
    assert result.registration.quality is AlignmentQuality.FAILED, result.registration.notes
    assert result.detection is None
    assert result.registration.homography is None
    assert not result.registration.valid_mask.any()
    assert result.registration.notes


def test_random_noise_fails_alignment(real_pipeline, files, tmp_path) -> None:
    """Rastgele gürültü → FAILED (çökmeden)."""
    noise = np.random.default_rng(5).integers(0, 255, (900, 1200, 3), dtype=np.uint8)
    result = real_pipeline.run(ImageInput(files["before"]), ImageInput(save_image(tmp_path / "n.png", noise)))
    assert result.registration.quality is AlignmentQuality.FAILED
    assert result.detection is None


def test_flat_black_image_fails_cleanly(real_pipeline, files, tmp_path) -> None:
    """Tamamen siyah görüntü: çökmeden FAILED."""
    black = np.zeros((600, 800, 3), np.uint8)
    result = real_pipeline.run(ImageInput(files["before"]), ImageInput(save_image(tmp_path / "k.png", black)))
    assert result.registration.quality is AlignmentQuality.FAILED
    assert result.detection is None


# --------------------------------------------------------------------------- bozuk dosyalar

@pytest.mark.parametrize("content", [b"bu bir goruntu degil", b"", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_corrupt_file_raises_value_error(content, real_pipeline, files, tmp_path) -> None:
    """Bozuk dosya anlaşılır `ValueError` ile reddedilir (UI bunu 'failed' sinyaline çevirir)."""
    bad = tmp_path / "bozuk.png"
    bad.write_bytes(content)
    with pytest.raises(ValueError):
        real_pipeline.run(ImageInput(bad), ImageInput(files["after"]))
    with pytest.raises(ValueError):
        real_pipeline.run(ImageInput(files["before"]), ImageInput(bad))


def test_truncated_png_raises_value_error(real_pipeline, files, tmp_path) -> None:
    """Yarım inmiş PNG (sohbet uygulamalarında sık) → ValueError."""
    data = files["before"].read_bytes()
    cut = tmp_path / "yarim.png"
    cut.write_bytes(data[: len(data) // 3])
    with pytest.raises(ValueError):
        real_pipeline.run(ImageInput(cut), ImageInput(files["after"]))


def test_missing_file_raises_value_error(real_pipeline, files, tmp_path) -> None:
    """Olmayan dosya → ValueError."""
    with pytest.raises(ValueError):
        real_pipeline.run(ImageInput(tmp_path / "yok.png"), ImageInput(files["after"]))


def test_fake_extension_still_loads(real_pipeline, files, tmp_path, pair) -> None:
    """Uzantısı yanlış (JPEG içerikli .png) dosya içerikten tanınmalı."""
    fake = tmp_path / "jpeg_ama.png"
    fake.write_bytes(files["after"].read_bytes())
    result = real_pipeline.run(ImageInput(files["before"]), ImageInput(fake))
    assert result.registration.quality is not AlignmentQuality.FAILED


def test_models_released_after_exception(real_pipeline, files, tmp_path) -> None:
    """Hata sonrası pipeline kullanılabilir kalır (sonraki geçerli çalışma başarılı)."""
    bad = tmp_path / "b.png"
    bad.write_bytes(b"xx")
    with pytest.raises(ValueError):
        real_pipeline.run(ImageInput(bad), ImageInput(files["after"]))
    ok = real_pipeline.run(ImageInput(files["before"]), ImageInput(files["after"]))
    assert ok.registration.quality is AlignmentQuality.GOOD


# --------------------------------------------------------------------------- boyutlar

def test_below_minimum_size_raises(real_pipeline, files, tmp_path, pair) -> None:
    """8x8 görüntü ön işlemede reddedilir."""
    tiny = save_image(tmp_path / "t8.png", pair.before[:8, :8])
    with pytest.raises(ValueError):
        real_pipeline.run(ImageInput(tiny), ImageInput(tiny))


@pytest.mark.parametrize("size", [16, 40])
def test_tiny_images_do_not_crash(size, real_pipeline, tmp_path, pair) -> None:
    """16x16 / 40x40: çökmemeli; sonuç ya FAILED+detection None ya da geçerli sonuç."""
    tiny = save_image(tmp_path / f"t{size}.png", pair.before[300:300 + size, 300:300 + size])
    result = real_pipeline.run(ImageInput(tiny), ImageInput(tiny))
    if result.registration.quality is AlignmentQuality.FAILED:
        assert result.detection is None
    else:
        assert result.detection is not None


def test_tiny_vs_full_fails_cleanly(real_pipeline, files, tmp_path, pair) -> None:
    """200x200 kırpık vs tam görüntü → çökmeden sonuç döner (FAILED ya da kırpığın bulunması)."""
    crop = save_image(tmp_path / "c200.png", pair.before[400:600, 600:800])
    result = real_pipeline.run(ImageInput(crop), ImageInput(files["before"]))
    assert result.registration.quality in set(AlignmentQuality)
    if result.registration.quality is AlignmentQuality.FAILED:
        assert result.detection is None


def test_large_image_is_downscaled_and_works(real_pipeline, files, tmp_path, pair) -> None:
    """6000x4500 before (4096 sınırına küçülür) vs 1600x1200 after → çalışır, koordinatlar before.pixels'te."""
    big = cv2.resize(pair.before, (6000, 4500), interpolation=cv2.INTER_CUBIC)
    big_path = save_image(tmp_path / "big.png", big)
    result = real_pipeline.run(ImageInput(big_path), ImageInput(files["before"]))
    reg = result.registration
    assert reg.quality is not AlignmentQuality.FAILED, reg.notes
    assert max(reg.before.pixels.shape[:2]) == 4096
    assert reg.before.scale_from_original == pytest.approx(4096 / 6000, rel=1e-3)
    assert reg.after_warped.shape == reg.before.pixels.shape
    assert result.detection is not None and result.detection.regions == []


# --------------------------------------------------------------------------- piksel biçimleri

def _expect_objects(real_pipeline, before_path, after_path, pair, name):
    result = real_pipeline.run(ImageInput(before_path), ImageInput(after_path))
    ev = evaluate(pair, result)
    record(name, ev)
    assert result.registration.quality is not AlignmentQuality.FAILED, result.registration.notes
    assert all(ev.hits), (name, ev.hits)
    assert ev.n_large_extra == 0
    return result


def test_rgba_png(real_pipeline, files, tmp_path, pair) -> None:
    """RGBA (opak) PNG before → normal sonuç."""
    rgba = np.dstack([pair.before, np.full(pair.before.shape[:2], 255, np.uint8)])
    _expect_objects(real_pipeline, save_image(tmp_path / "rgba.png", rgba), files["after"], pair, "rgba_png")


def test_rgba_png_with_transparent_corner(real_pipeline, files, tmp_path, pair) -> None:
    """Köşesi şeffaf RGBA PNG: şeffaf bölge beyaza bileşilir; eklenen nesneler yine bulunmalı."""
    rgba = np.dstack([pair.before, np.full(pair.before.shape[:2], 255, np.uint8)])
    rgba[:150, :150, 3] = 0
    _expect_objects(real_pipeline, save_image(tmp_path / "rgba_t.png", rgba), files["after"], pair,
                    "rgba_şeffaf_köşe")


def test_grayscale_png_pair(real_pipeline, tmp_path, pair) -> None:
    """Gri (L modu) PNG/JPEG çifti."""
    gb = cv2.cvtColor(pair.before, cv2.COLOR_RGB2GRAY)
    ga = cv2.cvtColor(pair.after, cv2.COLOR_RGB2GRAY)
    _expect_objects(real_pipeline, save_image(tmp_path / "gb.png", gb), save_image(tmp_path / "ga.png", ga),
                    pair, "gri_çift")


def test_palette_png(real_pipeline, files, tmp_path, pair) -> None:
    """Paletli (P modu) PNG before."""
    p = Image.fromarray(pair.before).quantize(256)
    path = tmp_path / "pal.png"
    p.save(path)
    result = real_pipeline.run(ImageInput(path), ImageInput(files["after"]))
    assert result.registration.quality is not AlignmentQuality.FAILED, result.registration.notes


def test_16bit_grayscale_png_is_normalised(tmp_path, pair) -> None:
    """16-bit gri PNG (uydu/termal kaynaklarda yaygın) 8-bit karşılığına yakın yüklenmeli."""
    gray8 = cv2.cvtColor(pair.before, cv2.COLOR_RGB2GRAY)
    path = tmp_path / "g16.png"
    cv2.imwrite(str(path), gray8.astype(np.uint16) * 257)
    prepared = create_preprocessor(AppConfig()).prepare(ImageInput(path))
    got = prepared.pixels[..., 0].astype(np.float32)
    assert abs(float(got.mean()) - float(gray8.mean())) < 8.0, (got.mean(), gray8.mean())


def test_exif_rotated_jpeg_is_upright(tmp_path, pair) -> None:
    """EXIF yönü 6 (90° CW) olan JPEG, ön işlemede döndürülmüş boyutta gelmeli."""
    img = Image.fromarray(pair.before[:300, :500])
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "exif.jpg"
    img.save(path, exif=exif)
    prepared = create_preprocessor(AppConfig()).prepare(ImageInput(path))
    assert prepared.pixels.shape[:2] == (500, 300)


# --------------------------------------------------------------------------- eksik model

def test_missing_model_files_fall_back_to_sift(files, tmp_path, pair) -> None:
    """models/ boş → LightGlue yüklenemez, SIFT'e düşer, ağa çıkmaz, sonuç yine doğru."""
    import socket

    cfg = AppConfig()
    cfg.paths.models_dir = tmp_path / "bos_models"
    pipe = build_default_pipeline(cfg)
    calls = []
    real_connect = socket.socket.connect

    def guard(self, *a, **k):  # noqa: ANN001
        calls.append(a)
        raise OSError("ağ kapalı (test)")

    socket.socket.connect = guard  # type: ignore[method-assign]
    try:
        result = pipe.run(ImageInput(files["before"]), ImageInput(files["after"]))
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        pipe.shutdown()
    assert calls == [], f"ağ bağlantısı denendi: {calls}"
    assert any("SIFT" in n for n in result.registration.notes), result.registration.notes
    ev = evaluate(pair, result)
    record("model_yok→SIFT", ev)
    assert result.registration.quality is not AlignmentQuality.FAILED
    assert all(ev.hits)


def test_sift_matcher_configured_works(files, pair) -> None:
    """`matcher="sift"` yapılandırması da çalışır."""
    cfg = AppConfig()
    cfg.registration.matcher = "sift"
    pipe = build_default_pipeline(cfg)
    result = pipe.run(ImageInput(files["before"]), ImageInput(files["after"]))
    ev = evaluate(pair, result)
    record("sift_ayarı", ev)
    assert result.registration.quality is not AlignmentQuality.FAILED
    assert all(ev.hits)


def test_decompression_bomb_raises_value_error(monkeypatch, tmp_path, files) -> None:
    """Pillow'un 'decompression bomb' koruması da anlaşılır `ValueError` olmalı."""
    path = save_image(tmp_path / "bomb.png", np.zeros((300, 300, 3), np.uint8))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)   # 300*300 > 2*1000 → DecompressionBombError
    with pytest.raises(ValueError):
        create_preprocessor(AppConfig()).prepare(ImageInput(path))
