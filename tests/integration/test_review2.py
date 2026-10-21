"""İkinci inceleme turu: çözünürlük farkı (iki yön), küçük nesneler, ısı haritası, pipeline yarışları, SQLite eşzamanlılığı."""

from __future__ import annotations

import copy
import os
import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import pipeline as core_pipeline
from core.config import AppConfig
from core.contracts import AlignmentQuality, ImageInput
from scripts.make_synthetic_pair import PairParams, make_pair
from tests.integration.helpers import LARGE_AREA_FRACTION, record, run_pair, write_pair, evaluate

SHOT_DIR = Path(__file__).resolve().parent / "screenshots"
SAVE_SHOTS = os.environ.get("CHANGESCOPE_SAVE_SHOTS") == "1"


def _write(img: np.ndarray, path: Path, jpeg: bool = False) -> Path:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 100] if jpeg else [])
    return path


def _hits(regions, centers) -> list[bool]:
    return [any(r.bbox.x <= cx <= r.bbox.x + r.bbox.width and r.bbox.y <= cy <= r.bbox.y + r.bbox.height
                for r in regions) for cx, cy in centers]


def _large_extra(result, centers, near: int = 80) -> int:
    h, w = result.registration.before.pixels.shape[:2]
    large = LARGE_AREA_FRACTION * h * w
    n = 0
    for r in result.detection.regions:
        mx, my = r.bbox.x + r.bbox.width / 2, r.bbox.y + r.bbox.height / 2
        if not any(abs(mx - cx) < near and abs(my - cy) < near for cx, cy in centers) and r.area_px >= large:
            n += 1
    return n


# --------------------------------------------------------------------------- çözünürlük farkı: ileri yön

@pytest.mark.parametrize("scale", [0.3, 0.25])
def test_no_change_pair_at_extreme_downscale_has_no_large_regions(scale, real_pipeline, tmp_path) -> None:
    """Değişiklik yok, after çok düşük çözünürlük: büyük sahte bölge çıkmamalı (0.25'te küçük artık kabul)."""
    pair, result = run_pair(real_pipeline, PairParams(scale=scale, n_new_objects=0), tmp_path, "nc")
    assert result.detection is not None
    assert _large_extra(result, []) == 0
    assert len(result.detection.regions) <= 2
    assert not result.detection.params["unreliable"]


@pytest.mark.parametrize("scale", [0.3, 0.25])
def test_extreme_downscale_still_finds_added_objects(scale, real_pipeline, tmp_path) -> None:
    """Ölçek 0.3 ve 0.25'te de eklenen bina+araç grubu bulunmalı."""
    pair, result = run_pair(real_pipeline, PairParams(scale=scale), tmp_path, "ex")
    ev = evaluate(pair, result)
    record(f"ölçek{scale}", ev)
    assert all(ev.hits) and ev.n_large_extra == 0


# --------------------------------------------------------------------------- ters yön: after daha yüksek çözünürlüklü

@pytest.mark.parametrize("scale", [1.5, 2.0, 3.0])
def test_after_higher_resolution_upsampled(scale, real_pipeline, tmp_path) -> None:
    """PairParams(scale>1): after büyütülmüş (s<1). İsabet tam, sahte bölge yok."""
    pair, result = run_pair(real_pipeline, PairParams(scale=scale), tmp_path, "up")
    ev = evaluate(pair, result)
    record(f"ters_ölçek{scale}", ev)
    assert result.registration.quality is not AlignmentQuality.FAILED
    assert all(ev.hits), ev.hits
    assert ev.n_large_extra == 0 and ev.n_extra == 0


@pytest.mark.parametrize("scale", [0.6, 0.5, 0.4])
def test_before_low_res_after_high_res(scale, real_pipeline, tmp_path) -> None:
    """before = küçültülmüş görüntü (nesneli), after = tam çözünürlüklü orijinal (nesnesiz): nesneler 'kaldırılmış'
    bölge olarak bulunmalı; s<1 yolu sahte bölge seli üretmemeli."""
    pair = make_pair(PairParams(scale=scale))
    Hbs = np.linalg.inv(pair.homography_after_to_before)
    centers = []
    for x, y, w, h in pair.added_bboxes:
        p = cv2.perspectiveTransform(np.array([[[x + w / 2, y + h / 2]]], np.float64), Hbs)[0, 0]
        centers.append((float(p[0]), float(p[1])))
    b = _write(pair.after, tmp_path / "small_before.png")
    a = _write(pair.before, tmp_path / "big_after.jpg", jpeg=True)
    result = real_pipeline.run(ImageInput(b), ImageInput(a))
    assert result.detection is not None, result.registration.notes
    assert result.detection.params["registration_scale"] < 1.0
    assert all(_hits(result.detection.regions, centers)), centers
    assert _large_extra(result, centers) == 0
    assert not result.detection.params["unreliable"]


@pytest.mark.parametrize("scale", [0.6, 0.4])
def test_before_low_res_after_high_res_no_change(scale, real_pipeline, tmp_path) -> None:
    """Aynı sahne, before düşük çözünürlük, after yüksek, nesne yok → bölge yok."""
    pair = make_pair(PairParams(scale=scale, n_new_objects=0))
    b = _write(pair.after, tmp_path / "sb.png")
    a = _write(pair.before, tmp_path / "ba.jpg", jpeg=True)
    result = real_pipeline.run(ImageInput(b), ImageInput(a))
    assert result.detection is not None
    assert result.detection.regions == [], [(r.bbox, r.area_px) for r in result.detection.regions][:5]


# --------------------------------------------------------------------------- küçük nesneler

def _tiny_pair(scale: float, size: int, width: int = 1600, height: int = 1200, same_res: bool = False):
    pr = make_pair(PairParams(width=width, height=height, scale=scale, n_new_objects=0,
                              **({"rotate_deg": 0.0, "shift_px": (0.0, 0.0), "perspective": 0.0} if same_res else {})))
    ch = pr.before.copy()
    cx, cy = int(width * 0.45), int(height * 0.45)
    cv2.rectangle(ch, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), (230, 30, 30), -1)
    H = np.linalg.inv(pr.homography_after_to_before)
    after = cv2.warpPerspective(ch, H, (int(width * scale), int(height * scale)), flags=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    after = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return pr.before, after, (cx, cy)


@pytest.mark.parametrize("scale", [0.6, 0.4, 0.3])
def test_20px_object_found_under_scale_compensation(scale, real_pipeline, tmp_path) -> None:
    """20x20 px kırmızı kare (after'da 6-12 px): ölçek telafisine rağmen bulunmalı. (15 px ve altı fiziksel/alan
    tabanı sınırı: ölçek 1.0'da bile bulunmaz; hata sayılmaz.)"""
    before, after, c = _tiny_pair(scale, 20)
    result = real_pipeline.run(ImageInput(_write(before, tmp_path / "b.png")),
                               ImageInput(_write(after, tmp_path / "a.jpg", jpeg=True)))
    assert result.detection is not None
    assert _hits(result.detection.regions, [c]) == [True]
    assert _large_extra(result, [c]) == 0


def test_40px_object_in_3200x2400_same_resolution_pair_is_found(real_pipeline, tmp_path) -> None:
    """Aynı çözünürlük, büyük görüntü: 40x40 px belirgin (kırmızı) nesne bölge olmalı."""
    before, after, c = _tiny_pair(1.0, 40, width=3200, height=2400, same_res=True)
    result = real_pipeline.run(ImageInput(_write(before, tmp_path / "b.png")),
                               ImageInput(_write(after, tmp_path / "a.jpg", jpeg=True)))
    assert result.detection is not None
    assert _hits(result.detection.regions, [c]) == [True], result.detection.params.get("min_region_area_effective_px")


# --------------------------------------------------------------------------- ısı haritası (gerçek change_map)

def test_heatmap_from_real_change_map_is_meaningful(real_pipeline, tmp_path) -> None:
    """Gerçek pipeline `change_map`'i → `heatmap_to_qimage`: opak piksellerin tamamı bölgelerin içinde,
    geçersiz alan saydam, genel 'sis' hafif. PNG: CHANGESCOPE_SAVE_SHOTS=1 ise screenshots/, değilse tmp_path."""
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QImage

    from app.ui.qt_utils import heatmap_to_qimage

    pair, result = run_pair(real_pipeline, PairParams(), tmp_path, "hm")
    det = result.detection
    assert det is not None and len(det.regions) >= 2
    qi = heatmap_to_qimage(det.change_map).convertToFormat(QImage.Format.Format_RGBA8888)
    arr = np.frombuffer(qi.constBits(), np.uint8).reshape(qi.height(), qi.bytesPerLine() // 4, 4)[:, :qi.width()]
    alpha = arr[..., 3]

    def small(mask: np.ndarray) -> np.ndarray:
        return cv2.resize(mask.astype(np.uint8), (qi.width(), qi.height()), interpolation=cv2.INTER_NEAREST).astype(bool)

    regions = np.zeros(det.change_map.shape, bool)
    for r in det.regions:
        b = r.bbox
        regions[b.y:b.y + b.height, b.x:b.x + b.width] |= r.mask
    rs, valid = small(regions), small(result.registration.valid_mask)
    assert not ((alpha > 0) & ~valid).any(), "geçersiz alanda ısı haritası görünüyor"
    strong = alpha > 128
    assert strong.mean() < 0.05, "ısı haritası ekranı boğuyor"
    opaque = alpha > 200      # ağaç kenarı artıkları (min alan altı) bir miktar opak nokta bırakır: ~%83 içeride
    assert (opaque & rs).sum() / max(opaque.sum(), 1) > 0.7, "opak ısı büyük oranda bölgelerin dışında"
    assert (strong & rs).sum() / rs.sum() > 0.9, "bölgeler ısı haritasında görünmüyor"
    assert alpha[~rs].mean() < 25, f"bölge dışı sis yoğun: {alpha[~rs].mean():.1f}"

    base = cv2.resize(result.registration.before.pixels, (qi.width(), qi.height()))
    a = (alpha / 255.0 * 0.6)[..., None]
    comp = (base * (1 - a) + arr[..., :3] * a).astype(np.uint8)
    out_dir = SHOT_DIR if SAVE_SHOTS else tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "real_pipeline_heatmap.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- core.pipeline yarışları

@pytest.fixture
def default_pipeline_state():
    """Modül düzeyi varsayılan pipeline durumunu korur/geri yükler."""
    saved = (core_pipeline._default_pipeline, core_pipeline._default_config)
    core_pipeline._default_pipeline = None
    yield
    core_pipeline.shutdown_default_pipeline()
    core_pipeline._default_pipeline, core_pipeline._default_config = saved


@pytest.fixture(scope="module")
def race_files(tmp_path_factory):
    pair = make_pair(PairParams(width=800, height=600))
    return write_pair(pair, tmp_path_factory.mktemp("race"), "r")


@pytest.fixture
def slow_register(monkeypatch):
    """`DefaultRegistrar._register`'ı yavaşlatır; içine girildiğinde `entered` set edilir."""
    from core.registration.registrar import DefaultRegistrar

    entered = threading.Event()
    orig = DefaultRegistrar._register

    def slow(self, *a, **k):
        entered.set()
        time.sleep(2.0)
        return orig(self, *a, **k)

    monkeypatch.setattr(DefaultRegistrar, "_register", slow)
    return entered


def _start_run(race_files, out: dict):
    before, after = race_files

    def target():
        try:
            out["result"] = core_pipeline.run_comparison(ImageInput(before), ImageInput(after),
                                                         lambda *a, **k: None, False)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def test_configure_during_running_comparison_is_safe(default_pipeline_state, race_files, slow_register) -> None:
    """Karşılaştırma sürerken `configure_default_pipeline`: kilitlenme yok, çalışan karşılaştırma geçerli sonuç
    verir, sonraki `get_default_pipeline` YENİ ayarla kurulur."""
    cfg1, cfg2 = AppConfig(), AppConfig()
    cfg2.detection.sensitivity = 0.9
    core_pipeline.configure_default_pipeline(cfg1)
    out: dict = {}
    t = _start_run(race_files, out)
    assert slow_register.wait(60)
    old = core_pipeline._default_pipeline
    core_pipeline.configure_default_pipeline(cfg2)
    t.join(120)
    assert not t.is_alive(), "karşılaştırma kilitlendi"
    assert "error" not in out, out.get("error")
    assert out["result"].detection is not None
    new = core_pipeline.get_default_pipeline()
    assert new is not old and new.config is cfg2


def test_shutdown_during_running_comparison_is_safe(default_pipeline_state, race_files, slow_register) -> None:
    """Kapanışta worker hâlâ çalışıyorsa: `shutdown_default_pipeline` çakışmaz/çökertmez; worker sonuç ya da
    anlaşılır hata ile biter; sonrasında yeni karşılaştırma yeniden çalışır."""
    core_pipeline.configure_default_pipeline(AppConfig())
    out: dict = {}
    t = _start_run(race_files, out)
    assert slow_register.wait(60)
    core_pipeline.shutdown_default_pipeline()
    t.join(120)
    assert not t.is_alive()
    assert "error" not in out, out.get("error")
    assert out["result"].registration.quality is not AlignmentQuality.FAILED
    # kapanıştan sonra pipeline yeniden kurulabilir
    out2: dict = {}
    t2 = _start_run(race_files, out2)
    t2.join(120)
    assert "error" not in out2 and out2["result"].detection is not None


def test_configure_does_not_block_caller_for_the_whole_registration(default_pipeline_state, race_files,
                                                                    slow_register) -> None:
    """Çağıran (UI thread'i olabilir) 1 sn'den fazla bloklanmamalı."""
    core_pipeline.configure_default_pipeline(AppConfig())
    out: dict = {}
    t = _start_run(race_files, out)
    assert slow_register.wait(60)
    t0 = time.perf_counter()
    core_pipeline.configure_default_pipeline(AppConfig())
    blocked = time.perf_counter() - t0
    t.join(120)
    assert blocked < 1.0, f"configure {blocked:.1f} s bloklandı"


def test_configure_and_runs_stress_no_exceptions(default_pipeline_state, tmp_path) -> None:
    """4 eşzamanlı run_comparison + 12 configure + 1 shutdown (küçük görüntü, SIFT): hiçbir thread istisna almaz."""
    pair = make_pair(PairParams(width=480, height=360))
    before, after = write_pair(pair, tmp_path, "s")
    cfg = AppConfig()
    cfg.registration.matcher = "sift"
    core_pipeline.configure_default_pipeline(cfg)
    errors: list[BaseException] = []
    results: list[object] = []

    def worker():
        try:
            for _ in range(2):
                results.append(core_pipeline.run_comparison(ImageInput(before), ImageInput(after),
                                                            lambda *a, **k: None, False))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(12):
        core_pipeline.configure_default_pipeline(copy.deepcopy(cfg))
        time.sleep(0.05)
    core_pipeline.shutdown_default_pipeline()
    for t in threads:
        t.join(180)
    assert not any(t.is_alive() for t in threads), "kilitlenme"
    assert not errors, errors
    assert len(results) == 8


# --------------------------------------------------------------------------- SQLite eşzamanlılığı

def _repo(tmp_path):
    from app.storage.repository import LibraryRepository
    return LibraryRepository(tmp_path / "db.sqlite3")


def test_sqlite_concurrent_writers_and_readers(tmp_path) -> None:
    """8 yazıcı (klasör + görüntü kaydı) ve 4 okuyucu eşzamanlı: istisna yok, sayılar tam."""
    repo = _repo(tmp_path)
    root = repo.create_folder("kök")
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(i: int) -> None:
        try:
            for j in range(40):
                f = repo.create_folder(f"w{i}-{j}", root.id)
                repo.add_image(f.id, f"images/{i}-{j}.png", f"{i}-{j}.png", None, "unknown", 10, 10, None)
                repo.move_folder(f.id, root.id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                repo.list_folders()
                repo.list_images(root.id, include_subfolders=True)
                repo.total_image_count()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ws = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    rs = [threading.Thread(target=reader) for _ in range(4)]
    for t in ws + rs:
        t.start()
    for t in ws:
        t.join(120)
    stop.set()
    for t in rs:
        t.join(30)
    assert not errors, errors[:3]
    assert repo.total_image_count() == 320
    assert len(repo.list_folders()) == 321
    assert repo._tx_depth == 0 if hasattr(repo, "_tx_depth") else True


def test_sqlite_failed_transaction_rolls_back_and_does_not_poison_others(tmp_path) -> None:
    """Transaction içinde istisna → ROLLBACK, derinlik sıfırlanır; başka thread'in yazması etkilenmez."""
    repo = _repo(tmp_path)
    root = repo.create_folder("a")
    with pytest.raises(KeyError):
        repo.move_image(999, 12345)        # hedef klasör yok -> tx içinde KeyError
    with pytest.raises(RuntimeError):
        with repo._tx():
            repo.create_folder("yarım", root.id)
            raise RuntimeError("iç hata")
    assert [f.name for f in repo.list_folders()] == ["a"], "ROLLBACK uygulanmadı"
    done: list[str] = []
    t = threading.Thread(target=lambda: done.append(repo.create_folder("b", root.id).name))
    t.start()
    t.join(10)
    assert done == ["b"]
    assert not repo._con.in_transaction


def test_sqlite_begin_failure_does_not_leak_lock(tmp_path) -> None:
    """BEGIN başarısız olunca kilit sızmamalı: başka thread `get_folder` ile bloklanmamalı."""
    repo = _repo(tmp_path)
    repo.create_folder("x")
    real_con = repo._con

    class Boom:
        def execute(self, sql, *a):
            if sql == "BEGIN":
                raise sqlite3.OperationalError("disk I/O error")
            return real_con.execute(sql, *a)

    repo._con = Boom()      # type: ignore[assignment]
    t1 = threading.Thread(target=lambda: _swallow(lambda: repo.create_folder("y")))
    t1.start()
    t1.join(10)
    repo._con = real_con
    res: list[object] = []
    t2 = threading.Thread(target=lambda: res.append(repo.get_folder(1)), daemon=True)
    t2.start()
    t2.join(5)
    assert not t2.is_alive(), "kilit sızdı: ikinci thread bloklandı"


def _swallow(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


def test_library_concurrent_import_and_save_comparison(tmp_path) -> None:
    """`Library` düzeyinde: eşzamanlı import_image (kopya + thumbnail + kayıt) tüm kayıtları üretir."""
    from app.storage.library import Library

    lib = Library(tmp_path / "lib")
    folder = lib.repo.create_folder("f")
    srcs = []
    for i in range(6):
        arr = np.full((60, 80, 3), i * 30, np.uint8)
        p = tmp_path / f"i{i}.png"
        cv2.imwrite(str(p), arr)
        srcs.append(p)
    errors: list[BaseException] = []
    recs: list[object] = []

    def imp(p: Path) -> None:
        try:
            recs.append(lib.import_image(p, folder.id))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=imp, args=(p,)) for p in srcs for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert not errors, errors[:3]
    assert len(recs) == 18 and lib.repo.total_image_count() == 18
    assert len({r.file_name for r in recs}) == 18
    lib.close()
