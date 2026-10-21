"""DINOv2 yöntemi ve ssim+dinov2 birleştirme testleri (model yoksa ilgili testler atlanır)."""

from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from core.config import AppConfig
from core.detection import create_detector
from core.detection.dinov2_features import Dinov2Extractor
from scripts.make_synthetic_pair import PairParams
from tests.detection.helpers import make_registration, pair_registration
from tests.detection.scenarios import build_scenario, evaluate

pytestmark = pytest.mark.filterwarnings("ignore")

SMALL = dict(width=420, height=330, scale=1.0, rotate_deg=0.0, shift_px=(0.0, 0.0), perspective=0.0)


def _cfg(method: str, device: str = "auto") -> AppConfig:
    cfg = AppConfig()
    cfg.detection.method = method
    cfg.runtime.device = device
    return cfg


needs_model = pytest.mark.skipif(not Dinov2Extractor(AppConfig()).available()[0],
                                 reason="DINOv2 ağırlığı/kaynağı yok (CHANGESCOPE_MODELS_DIR)")


@pytest.fixture
def no_network(monkeypatch):
    """Her ağ bağlantısı denemesi test hatası olur."""
    def boom(*a, **k):
        raise AssertionError("ağ çağrısı yapıldı")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


@needs_model
def test_load_and_detect_without_network(no_network) -> None:
    """Model yerel dosyadan yüklenir, ağa çıkılmaz."""
    sc = build_scenario("default")
    det = create_detector(_cfg("dinov2"))
    res = det.detect(sc.registration)
    det.release()
    assert res.method == "dinov2" and res.params["method_used"] == "dinov2"
    assert evaluate(res, sc)["hits"] == 2


@needs_model
def test_cpu_path_small_image() -> None:
    """Cihaz cpu iken küçük (14'ün katı olmayan boyutlu) görüntüde çalışır."""
    pair, reg = pair_registration(PairParams(**SMALL))
    det = create_detector(_cfg("ssim+dinov2", "cpu"))
    res = det.detect(reg)
    det.release()
    assert res.change_map.shape == pair.before.shape[:2]
    assert res.params["dinov2_device"] == "cpu"
    assert res.method == "ssim+dinov2"
    assert 0.0 <= res.change_map.min() and res.change_map.max() <= 1.0


@needs_model
def test_release_then_reuse() -> None:
    """release() modeli bırakır; sonraki detect() yeniden yükler."""
    pair, reg = pair_registration(PairParams(**SMALL))
    det = create_detector(_cfg("dinov2", "cpu"))
    r1 = det.detect(reg)
    assert det._dino.loaded
    det.release()
    assert not det._dino.loaded
    r2 = det.detect(reg)
    det.release()
    assert r2.params.get("dinov2_loaded_now") == 1
    np.testing.assert_allclose(r1.change_map, r2.change_map, atol=1e-4)


@needs_model
def test_concurrent_detect_and_release() -> None:
    """Çok thread'li detect + araya giren release() hata üretmez, sonuçlar tutarlı kalır."""
    pair, reg = pair_registration(PairParams(**SMALL))
    det = create_detector(_cfg("dinov2", "cpu"))
    ref = det.detect(reg).change_map
    errors: list[BaseException] = []
    outs: list[np.ndarray] = []

    def work() -> None:
        try:
            outs.append(det.detect(reg).change_map)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(5):
        det.release()
    for t in threads:
        t.join()
    det.release()
    assert not errors, errors
    assert len(outs) == 4
    for o in outs:
        np.testing.assert_allclose(o, ref, atol=1e-4)


@pytest.mark.parametrize("method", ["dinov2", "ssim+dinov2"])
def test_fallback_when_model_missing(tmp_path, method, no_network) -> None:
    """Model yoksa SSIM'e düşer; method/params bunu yansıtır; ağa çıkılmaz."""
    cfg = _cfg(method)
    cfg.paths.models_dir = tmp_path / "yok"
    pair, reg = pair_registration(PairParams(n_new_objects=0, width=400, height=300))
    res = create_detector(cfg).detect(reg)
    assert res.method == "ssim"
    assert res.params["method_requested"] == method
    assert res.params["method_used"] == "ssim"
    assert "fallback_reason" in res.params


def test_fallback_when_weights_corrupt(tmp_path) -> None:
    """Bozuk ağırlık/kaynak dosyası da SSIM'e düşürür."""
    cfg = _cfg("dinov2", "cpu")
    cfg.paths.models_dir = tmp_path
    hub = tmp_path / "dinov2" / "dinov2_src" / "dinov2" / "hub"
    hub.mkdir(parents=True)
    (hub / "backbones.py").write_text("raise RuntimeError('x')")
    (tmp_path / "dinov2" / "dinov2_vits14_pretrain.pth").write_bytes(b"bozuk")
    pair, reg = pair_registration(PairParams(n_new_objects=0, width=400, height=300))
    res = create_detector(cfg).detect(reg)
    assert res.method == "ssim" and "fallback_reason" in res.params


@needs_model
def test_fusion_weights_recorded_and_normalized() -> None:
    """Birleşimde ağırlıklar normalize edilip params'a yazılır."""
    cfg = _cfg("ssim+dinov2")
    cfg.detection.ssim_weight, cfg.detection.dinov2_weight = 1.0, 3.0
    det = create_detector(cfg)
    res = det.detect(build_scenario("default").registration)
    det.release()
    assert res.params["ssim_weight_used"] == pytest.approx(0.25)
    assert res.params["dinov2_weight_used"] == pytest.approx(0.75)


@needs_model
@pytest.mark.parametrize("name", ["default", "no_change"])
def test_fused_finds_objects_without_false_regions(name) -> None:
    """Birleşik yöntem nesneleri bulur, nesnesiz çiftte bölge üretmez."""
    sc = build_scenario(name)
    det = create_detector(_cfg("ssim+dinov2"))
    res = det.detect(sc.registration)
    det.release()
    e = evaluate(res, sc)
    assert e["hits"] == e["objects"]
    assert e["false_regions"] == 0


@needs_model
def test_shadow_scenario_fused_not_worse_than_ssim() -> None:
    """Gölge: nesneler bulunur ve birleşik yöntem SSIM'den fazla sahte bölge üretmez."""
    sc = build_scenario("shadow")
    out = {}
    for m in ("ssim", "ssim+dinov2"):
        det = create_detector(_cfg(m))
        out[m] = evaluate(det.detect(sc.registration), sc)
        det.release()
    assert out["ssim+dinov2"]["hits"] == out["ssim+dinov2"]["objects"]
    assert out["ssim+dinov2"]["false_regions"] <= out["ssim"]["false_regions"]


@needs_model
def test_texture_tone_fused_much_better_than_ssim() -> None:
    """Doku/ton kayması: birleşik yöntem SSIM'in sahte bölgelerinin en fazla üçte birini üretir."""
    sc = build_scenario("texture_tone")
    out = {}
    for m in ("ssim", "ssim+dinov2"):
        det = create_detector(_cfg(m))
        out[m] = evaluate(det.detect(sc.registration), sc)
        det.release()
    assert out["ssim+dinov2"]["hits"] == out["ssim+dinov2"]["objects"]
    assert out["ssim+dinov2"]["false_regions"] * 3 <= out["ssim"]["false_regions"]


@needs_model
def test_invalid_mask_respected_with_dinov2() -> None:
    """valid_mask dışındaki farklar dinov2/ssim+dinov2'de de sayılmaz."""
    rng = np.random.default_rng(0)
    base = np.clip(np.kron(rng.random((20, 28, 3)), np.ones((14, 14, 1))) * 120 + 60, 0, 255).astype(np.uint8)
    after = base.copy()
    after[100:200, 200:300] = 255 - after[100:200, 200:300]
    valid = np.ones(base.shape[:2], bool)
    valid[80:220, 180:320] = False
    for m in ("dinov2", "ssim+dinov2"):
        det = create_detector(_cfg(m, "cpu"))
        res = det.detect(make_registration(base, after, valid))
        det.release()
        assert res.regions == []
        assert res.change_map[~valid].max() == 0.0


@needs_model
def test_cuda_oom_falls_back_to_cpu(monkeypatch) -> None:
    """CUDA OOM yakalanırsa CPU'ya düşülür ve params'a yazılır."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA yok")
    pair, reg = pair_registration(PairParams(**SMALL))
    det = create_detector(_cfg("dinov2", "cuda"))
    real = det._dino._run
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("sahte OOM")
        return real(*a, **k)

    monkeypatch.setattr(det._dino, "_run", flaky)
    res = det.detect(reg)
    det.release()
    assert res.params["dinov2_oom_fallback"] == 1
    assert res.params["dinov2_device"] == "cpu"
    assert res.method == "dinov2"
