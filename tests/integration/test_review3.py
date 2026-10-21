"""Faz 2 sonu bağımsız inceleme: ssim+dinov2 varsayılanı, ignore_mask uçtan uca, şema v2, yetim dosyalar,
önbellek tutarlılığı, eşzamanlılık, gerçek UI ile maske akışı."""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.storage import Library
from app.storage import library as library_mod
from core.config import AppConfig
from core.contracts import AlignmentQuality, ImageInput, ImageSource
from core.pipeline import build_default_pipeline
from scripts.make_synthetic_pair import PairParams, SyntheticPair, make_pair
from tests.integration.helpers import evaluate, write_pair

MARGIN = 40


# --------------------------------------------------------------------------- yardımcılar

def _separated_pair(seed: int = 42) -> SyntheticPair:
    """İki eklenen nesnesi birbirinden en az 200 px uzak çift (biri maskelenecek, diğeri kalacak)."""
    for s in range(seed, seed + 40):
        pair = make_pair(PairParams(seed=s))
        if len(pair.added_bboxes) < 2:
            continue
        (x0, y0, w0, h0), (x1, y1, w1, h1) = pair.added_bboxes[:2]
        gap_x = max(x0 - (x1 + w1), x1 - (x0 + w0))
        gap_y = max(y0 - (y1 + h1), y1 - (y0 + h0))
        if max(gap_x, gap_y) > 200:
            return pair
    pytest.skip("yeterince ayrık nesneli çift üretilemedi")


def _mask_rect(shape: tuple[int, int], bbox: tuple[int, int, int, int], k: float = 1.0,
               margin: int = MARGIN) -> np.ndarray:
    """before koordinatlarındaki bbox'ı (+marj) `k` ölçeğinde bir bool maske olarak çizer."""
    h, w = shape
    mh, mw = max(1, round(h * k)), max(1, round(w * k))
    m = np.zeros((mh, mw), bool)
    x, y, bw, bh = bbox
    x0, y0 = max(0, round((x - margin) * k)), max(0, round((y - margin) * k))
    x1, y1 = min(mw, round((x + bw + margin) * k)), min(mh, round((y + bh + margin) * k))
    m[y0:y1, x0:x1] = True
    return m


def _touches(result, bbox: tuple[int, int, int, int]) -> bool:
    """Herhangi bir bölge nesne bbox'ıyla kesişiyor mu."""
    x, y, w, h = bbox
    regions = result.detection.regions if result.detection else []
    return any(r.bbox.x < x + w and r.bbox.x + r.bbox.width > x and
               r.bbox.y < y + h and r.bbox.y + r.bbox.height > y for r in regions)


def _write_rgb_jpeg(path: Path, arr: np.ndarray, exif_orientation: int | None = None) -> Path:
    exif = Image.Exif()
    if exif_orientation:
        exif[0x0112] = exif_orientation
    Image.fromarray(arr).save(path, quality=100, exif=exif)
    return path


@pytest.fixture(scope="module")
def sep_pair() -> SyntheticPair:
    return _separated_pair()


@pytest.fixture(scope="module")
def sep_files(sep_pair, tmp_path_factory) -> tuple[Path, Path]:
    return write_pair(sep_pair, tmp_path_factory.mktemp("sep"), "m")


# --------------------------------------------------------------------------- 1. varsayılan yöntem / VRAM / offline

def test_default_run_really_uses_dinov2(real_pipeline, sep_files, sep_pair) -> None:
    """Varsayılan koşu gerçekten ssim+dinov2 kullanmalı (sessiz SSIM'e düşüş olmamalı)."""
    res = real_pipeline.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]))
    det = res.detection
    assert det is not None
    assert det.method == "ssim+dinov2", (det.method, det.params.get("fallback_reason"))
    assert "fallback_reason" not in det.params
    assert all(evaluate(sep_pair, res).hits)


def test_dinov2_released_after_run_and_vram_zero(sep_files) -> None:
    """Pipeline.run sonrası DINOv2 serbest: `loaded` False, CUDA varsa memory_allocated ~0."""
    torch = pytest.importorskip("torch")
    pipe = build_default_pipeline(AppConfig())
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        base = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        res = pipe.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]))
        assert res.detection is not None and res.detection.method == "ssim+dinov2"
        assert res.detection.params.get("dinov2_loaded_now") == 1
        assert not pipe.detector._dino.loaded, "DINOv2 koşu sonrası hâlâ bellekte"  # type: ignore[attr-defined]
        if torch.cuda.is_available():
            assert res.detection.params.get("dinov2_device") == "cuda"
            assert torch.cuda.memory_allocated() - base <= 8 * 2**20
    finally:
        pipe.shutdown()


def test_dinov2_load_offline_no_hub_no_socket(monkeypatch, sep_files) -> None:
    """Ağ + torch.hub kapalı; yeni pipeline DINOv2'yi yerelden yükleyip ssim+dinov2 ile bitirmeli."""
    import socket
    torch = pytest.importorskip("torch")
    attempts: list = []

    def refuse(*a, **k):  # noqa: ANN002, ANN003
        attempts.append(a)
        raise OSError("ağ kapalı")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    for name in ("load", "load_state_dict_from_url", "download_url_to_file"):
        monkeypatch.setattr(torch.hub, name, refuse)
    pipe = build_default_pipeline(AppConfig())
    try:
        res = pipe.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]))
    finally:
        pipe.shutdown()
    assert attempts == []
    assert res.detection is not None and res.detection.method == "ssim+dinov2"
    assert "fallback_reason" not in res.detection.params


def test_missing_dinov2_model_falls_back_to_ssim_end_to_end(tmp_path, sep_files, sep_pair) -> None:
    """models/ boş: hizalama SIFT'e, tespit SSIM'e düşer; sonuç yine geçerli ve nedeni params'ta yazılı."""
    cfg = AppConfig()
    cfg.paths.models_dir = tmp_path / "bos_models"
    pipe = build_default_pipeline(cfg)
    try:
        res = pipe.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]))
    finally:
        pipe.shutdown()
    det = res.detection
    assert det is not None and det.method == "ssim"
    assert det.params.get("fallback_reason") and det.params.get("fell_back") == 1
    assert all(evaluate(sep_pair, res).hits)


# --------------------------------------------------------------------------- 2. ignore_mask uçtan uca

@pytest.mark.parametrize("k", [1.0, 0.5, 2.0], ids=["aynı-çözünürlük", "yarı", "iki-kat"])
def test_ignore_mask_removes_only_masked_change(k, real_pipeline, sep_files, sep_pair) -> None:
    """Maskelenen nesne sonuçta yok; maskelenmeyen nesne duruyor. Maske her çözünürlükte çalışmalı."""
    masked, kept = sep_pair.added_bboxes[0], sep_pair.added_bboxes[1]
    mask = _mask_rect(sep_pair.before.shape[:2], masked, k)
    res = real_pipeline.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]), ignore_mask=mask)
    assert res.registration.quality is not AlignmentQuality.FAILED
    assert not _touches(res, masked), "maskelenen alanda hâlâ bölge var"
    cx, cy = kept[0] + kept[2] / 2, kept[1] + kept[3] / 2
    assert any(r.bbox.x <= cx <= r.bbox.x + r.bbox.width and r.bbox.y <= cy <= r.bbox.y + r.bbox.height
               for r in res.detection.regions), "maskelenmeyen gerçek değişiklik kayboldu"
    x, y, w, h = masked
    assert not res.registration.valid_mask[y + h // 2, x + w // 2], "maske valid_mask'a yansımadı"


def test_ignore_mask_uint8_and_degenerate_masks(real_pipeline, sep_files, sep_pair) -> None:
    """uint8 (0/255) maske çalışır; tamamen dolu maske 0 bölge; boş maske maskesiz ile aynı bölge sayısı."""
    shape = sep_pair.before.shape[:2]
    base = real_pipeline.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]))
    empty = real_pipeline.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]),
                              ignore_mask=np.zeros(shape, bool))
    assert len(empty.detection.regions) == len(base.detection.regions)
    full = real_pipeline.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]),
                             ignore_mask=np.full(shape, 255, np.uint8))
    assert full.detection is not None and full.detection.regions == []


def test_ignore_mask_wrong_ndim_raises_clear_error(real_pipeline, sep_files, sep_pair) -> None:
    """3 boyutlu maske net bir ValueError üretmeli."""
    bad = np.zeros(sep_pair.before.shape, bool)
    with pytest.raises(ValueError, match="2 boyutlu"):
        real_pipeline.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]), ignore_mask=bad)


def test_ignore_mask_with_exif_rotated_before(real_pipeline, sep_pair, tmp_path) -> None:
    """EXIF yönlü (piksel olarak döndürülmüş saklanan) 'önce' görüntüsünde maske EKRANDA görünen yönde uygulanır."""
    # orientation 6: gösterim = saklanan 90° CW döndürülmüş. saklanan = gösterimin 90° CCW döndürülmüşü.
    stored_before = np.ascontiguousarray(np.rot90(sep_pair.before, 1))
    stored_after = np.ascontiguousarray(np.rot90(sep_pair.after, 1))
    b = _write_rgb_jpeg(tmp_path / "b.jpg", stored_before, 6)
    a = _write_rgb_jpeg(tmp_path / "a.jpg", stored_after, 6)
    masked, kept = sep_pair.added_bboxes[0], sep_pair.added_bboxes[1]
    mask = _mask_rect(sep_pair.before.shape[:2], masked, 1.0)       # gösterim (EXIF düzeltilmiş) koordinatı
    res = real_pipeline.run(ImageInput(b), ImageInput(a), ignore_mask=mask)
    assert res.registration.before.pixels.shape[:2] == sep_pair.before.shape[:2], "EXIF düzeltilmedi"
    assert res.registration.quality is not AlignmentQuality.FAILED, res.registration.notes
    assert not _touches(res, masked)
    assert _touches(res, kept)


# --------------------------------------------------------------------------- 3. şema migration

def _downgrade_to_v1(db: Path) -> None:
    con = sqlite3.connect(str(db))
    for stmt in ("ALTER TABLE images DROP COLUMN mask_path", "ALTER TABLE images DROP COLUMN mask_hash",
                 "ALTER TABLE comparisons DROP COLUMN mask_hash", "ALTER TABLE comparisons DROP COLUMN method",
                 "UPDATE schema_version SET version = 1"):
        con.execute(stmt)
    con.commit()
    con.close()


@pytest.fixture(scope="module")
def real_result(real_pipeline, sep_files):
    return real_pipeline.run(ImageInput(sep_files[0]), ImageInput(sep_files[1]))


def _src_images(tmp_path: Path, n: int = 3) -> list[Path]:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        arr = (rng.random((90, 120, 3)) * 255).astype(np.uint8)
        out.append(_write_rgb_jpeg(src / f"im{i}.jpg", arr))
    return out


def test_v1_library_opens_as_v2_and_keeps_data(tmp_path, real_result) -> None:
    """v1 şemalı gerçek kütüphane: açılınca v2, klasör/görüntü/karşılaştırma sağlam, sonuç okunuyor, maske kaydedilebiliyor."""
    root = tmp_path / "lib"
    lib = Library(root)
    f1 = lib.repo.create_folder("X Bölgesi")
    f2 = lib.repo.create_folder("Y Dağı", f1.id)
    paths = _src_images(tmp_path)
    ra = lib.import_image(paths[0], f2.id, datetime(2026, 1, 1), ImageSource.DRONE)
    rb = lib.import_image(paths[1], f1.id, datetime(2026, 1, 8), ImageSource.UAV)
    crec = lib.save_comparison(ra, rb, real_result)
    lib.close()
    _downgrade_to_v1(root / "library.sqlite3")
    con = sqlite3.connect(str(root / "library.sqlite3"))
    assert "mask_path" not in [r[1] for r in con.execute("PRAGMA table_info(images)")]
    con.close()

    lib2 = Library(root)
    try:
        assert lib2.repo.schema_version() == 2
        assert [f.name for f in lib2.repo.list_folders()] == ["X Bölgesi", "Y Dağı"]
        imgs = lib2.repo.list_images()
        assert {i.original_name for i in imgs} == {"im0.jpg", "im1.jpg"}
        assert all(i.mask_path is None and i.mask_hash == "" for i in imgs)
        old = lib2.repo.get_comparison(crec.id)
        assert old is not None and old.mask_hash == "" and old.method == ""
        ra2 = lib2.repo.get_image(ra.id)
        rb2 = lib2.repo.get_image(rb.id)
        assert lib2.find_comparison(ra2, rb2) is not None, "v1 sonucu maskesiz anahtar ile bulunmalı"
        loaded = lib2.load_comparison(old)
        assert len(loaded.detection.regions) == len(real_result.detection.regions)
        m = np.zeros((90, 120), bool)
        m[10:30, 10:40] = True
        ra3 = lib2.save_mask(ra2, m)
        assert ra3.mask_hash and lib2.find_comparison(ra3, rb2) is None, "maske eklenince eski sonuç kullanılmamalı"
    finally:
        lib2.close()
    lib3 = Library(root)       # ikinci açılış idempotent
    try:
        assert lib3.repo.schema_version() == 2
    finally:
        lib3.close()


# --------------------------------------------------------------------------- 4. önbellek / maske tutarlılığı

def test_cache_key_follows_mask_and_only_latest_is_kept(tmp_path, real_result) -> None:
    """Maske değişince önbellek kaçırılır; maske GERİ alınınca da (yalnızca son sonuç saklanır) yeniden hesap gerekir."""
    lib = Library(tmp_path / "lib")
    try:
        f = lib.repo.create_folder("F")
        p = _src_images(tmp_path, 2)
        a = lib.import_image(p[0], f.id, datetime(2026, 1, 1))
        b = lib.import_image(p[1], f.id, datetime(2026, 1, 8))
        lib.save_comparison(a, b, real_result)                                   # maskesiz
        assert lib.find_comparison(a, b) is not None
        m1 = np.zeros((90, 120), bool)
        m1[5:20, 5:20] = True
        a = lib.save_mask(a, m1)
        assert lib.find_comparison(a, b) is None
        lib.save_comparison(a, b, real_result)                                   # m1 ile
        assert lib.find_comparison(a, b) is not None
        a = lib.clear_mask(a)                                                    # geri al
        assert lib.find_comparison(a, b) is None, "davranış değişti: eski maskesiz sonuç yeniden bulunuyor"
        assert len(lib.repo.list_comparisons()) == 1, "çift başına tek kayıt beklenir"
        # Aynı maske yeniden kaydedilirse özet aynı: önbellek tutarlı
        a = lib.save_mask(a, m1)
        lib.save_comparison(a, b, real_result)
        a2 = lib.save_mask(a, m1.copy())
        assert a2.mask_hash == a.mask_hash and lib.find_comparison(a2, b) is not None
    finally:
        lib.close()


@pytest.mark.xfail(strict=True, reason="upsert_comparison satırı yeni mask_hash'e çevirir, dosyalar SONRA yazılır "
                   "(app/storage/library.py:save_comparison ~satır 255-259): arada eski sonuç yeni maskeli önbellek "
                   "gibi sunulur; satır dosyalar yazıldıktan sonra güncellenmeli")
def test_cache_not_served_while_result_files_are_being_rewritten(tmp_path, real_result, monkeypatch) -> None:
    """upsert satırı yeni maske özetine çevirir, dosyalar sonra yazılır: arada okuyan ESKİ sonucu yeni maske sanmamalı."""
    lib = Library(tmp_path / "lib")
    try:
        f = lib.repo.create_folder("F")
        p = _src_images(tmp_path, 2)
        a = lib.import_image(p[0], f.id, datetime(2026, 1, 1))
        b = lib.import_image(p[1], f.id, datetime(2026, 1, 8))
        lib.save_comparison(a, b, real_result)            # eski (maskesiz) sonuç diskte
        m = np.zeros((90, 120), bool)
        m[5:20, 5:20] = True
        a = lib.save_mask(a, m)
        entered, gate = threading.Event(), threading.Event()
        orig = library_mod._write_result

        def slow(directory, result):  # noqa: ANN001
            entered.set()
            gate.wait(20)
            orig(directory, result)

        monkeypatch.setattr(library_mod, "_write_result", slow)
        t = threading.Thread(target=lib.save_comparison, args=(a, b, real_result), kwargs={"mask_hash": a.mask_hash})
        t.start()
        try:
            assert entered.wait(20)
            hit = lib.find_comparison(a, b)
        finally:
            gate.set()
            t.join(30)
        assert hit is None, "yeni maskeli sonuç yazılmadan eski sonuç yeni maskenin önbelleği gibi sunuldu"
    finally:
        lib.close()


# --------------------------------------------------------------------------- 5. cascade / yetim dosyalar

def _db_expected_files(lib: Library) -> set[str]:
    exp: set[str] = set()
    con = lib.repo._con
    for r in con.execute("SELECT file_name, thumb_path, mask_path FROM images"):
        exp.update(x for x in (r["file_name"], r["thumb_path"], r["mask_path"]) if x)
    for r in con.execute("SELECT result_dir FROM comparisons"):
        d = lib.abs_path(r["result_dir"])
        exp.update(p.relative_to(lib.root).as_posix() for p in d.rglob("*") if p.is_file())
    return exp


def _disk_files(lib: Library) -> set[str]:
    return {p.relative_to(lib.root).as_posix() for p in lib.root.rglob("*")
            if p.is_file() and not p.name.startswith("library.sqlite3")}


def _assert_no_orphans(lib: Library, where: str) -> None:
    disk, exp = _disk_files(lib), _db_expected_files(lib)
    assert disk == exp, f"{where}: yetim={sorted(disk - exp)} eksik={sorted(exp - disk)}"


def test_cascade_deletes_leave_no_orphan_files(tmp_path, real_result) -> None:
    """Görüntü / karşılaştırma / klasör silmeleri sonrası results/, masks/, thumbs/, images/ altında yetim dosya kalmaz."""
    lib = Library(tmp_path / "lib")
    try:
        root = lib.repo.create_folder("X")
        sub = lib.repo.create_folder("Y", root.id)
        other = lib.repo.create_folder("Z")
        p = _src_images(tmp_path, 4)
        a = lib.import_image(p[0], sub.id, datetime(2026, 1, 1))
        b = lib.import_image(p[1], sub.id, datetime(2026, 1, 8))
        c = lib.import_image(p[2], other.id, datetime(2026, 1, 9))
        d = lib.import_image(p[3], root.id, datetime(2026, 1, 10))
        m = np.zeros((90, 120), bool)
        m[5:30, 5:30] = True
        a = lib.save_mask(a, m)
        m2 = m.copy()
        m2[40:60, 40:60] = True
        a = lib.save_mask(a, m2)                  # eski maske dosyası silinmeli
        b = lib.save_mask(b, m)
        lib.save_comparison(a, b, real_result)
        lib.save_comparison(a, c, real_result)    # başka klasördeki görüntüyle
        lib.save_comparison(c, d, real_result)
        lib.save_comparison(a, b, real_result)    # yeniden kayıt: aynı dizin
        _assert_no_orphans(lib, "kurulum")
        assert len(list((lib.root / "masks").glob("*.png"))) == 2

        lib.delete_comparison(lib.repo.list_comparisons()[0].id)
        _assert_no_orphans(lib, "delete_comparison")

        lib.delete_image(a.id)
        assert lib.repo.get_image(a.id) is None
        _assert_no_orphans(lib, "delete_image(a)")
        assert lib.repo.count_comparisons_for_image(a.id) == 0

        lib.save_comparison(b, c, real_result)
        lib.delete_folder(root.id)               # root > sub(b) ; d de root'ta ; c/other'da karşılaştırmaları var
        _assert_no_orphans(lib, "delete_folder(root)")
        assert {i.id for i in lib.repo.list_images()} == {c.id}
        assert lib.repo.list_comparisons() == [], "silinen görüntüye bağlı karşılaştırma kaydı kaldı"
        lib.delete_image(c.id)
        _assert_no_orphans(lib, "tümü")
        assert _disk_files(lib) == set()
    finally:
        lib.close()


@pytest.mark.xfail(strict=True, reason="comparisons.folder_id 'önce' görüntüsünün klasörünün KOPYASI; move_image güncellemiyor "
                   "(app/storage/repository.py:move_image): görüntü taşınınca geçmiş eski klasörde görünür ve eski klasör "
                   "silinince (ON DELETE CASCADE) görüntüleri yaşayan karşılaştırma + sonuç dosyaları da silinir")
def test_moving_before_image_keeps_history_attached_to_its_new_folder(tmp_path, real_result) -> None:
    """'Önce' görüntüsü başka klasöre taşınıp eski klasör silinince karşılaştırma kaydı kaybolmamalı."""
    lib = Library(tmp_path / "lib")
    try:
        old, new = lib.repo.create_folder("Eski"), lib.repo.create_folder("Yeni")
        p = _src_images(tmp_path, 2)
        a = lib.import_image(p[0], old.id, datetime(2026, 1, 1))
        b = lib.import_image(p[1], new.id, datetime(2026, 1, 8))
        lib.save_comparison(a, b, real_result)
        lib.repo.move_image(a.id, new.id)
        assert [c.before_image_id for c in lib.repo.list_comparisons(new.id)] == [a.id], "geçmiş yeni klasörde görünmüyor"
        lib.delete_folder(old.id)
        assert lib.repo.get_image(a.id) is not None and lib.repo.get_image(b.id) is not None
        assert len(lib.repo.list_comparisons()) == 1, "görüntüleri yaşayan karşılaştırma eski klasörle birlikte silindi"
    finally:
        lib.close()


# --------------------------------------------------------------------------- 6. EXIF'li görüntünün küçük resmi

@pytest.mark.xfail(strict=True, reason="make_thumbnail EXIF yönünü uygulamıyor: galeri küçük resmi ve ImageRecord "
                   "width/height, ön işlemenin (exif_transpose) gördüğünden farklı yönde (app/storage/library.py:make_thumbnail)")
def test_thumbnail_respects_exif_orientation(tmp_path) -> None:
    """Gösterim 160x120 (yatay), saklanan 120x160 + EXIF=6: küçük resim yatay olmalı."""
    rng = np.random.default_rng(1)
    display = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    path = _write_rgb_jpeg(tmp_path / "rot.jpg", np.ascontiguousarray(np.rot90(display, 1)), 6)
    lib = Library(tmp_path / "lib")
    try:
        f = lib.repo.create_folder("F")
        rec = lib.import_image(path, f.id, datetime(2026, 1, 1))
        with Image.open(lib.thumb_path(rec)) as th:
            assert th.width > th.height, f"küçük resim dikey: {th.size}"
        assert rec.width > rec.height, f"kayıtlı boyut dikey: {rec.width}x{rec.height}"
    finally:
        lib.close()


# --------------------------------------------------------------------------- 7. eşzamanlılık (gerçek modeller)

def test_concurrent_runs_with_release_spam(sep_files, sep_pair) -> None:
    """İki eşzamanlı run + sürekli release_cv_models: istisna yok, sonuçlar geçerli (DINOv2 RLock'u)."""
    pipe = build_default_pipeline(AppConfig())
    errors: list[BaseException] = []
    results: list = []
    stop = threading.Event()

    def runner() -> None:
        try:
            results.append(pipe.run(ImageInput(sep_files[0]), ImageInput(sep_files[1])))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def spammer() -> None:
        while not stop.is_set():
            try:
                pipe.detector.release()                    # type: ignore[attr-defined]
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            time.sleep(0.01)

    threads = [threading.Thread(target=runner) for _ in range(2)]
    sp = threading.Thread(target=spammer)
    sp.start()
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(300)
    finally:
        stop.set()
        sp.join(10)
        pipe.shutdown()
    assert errors == [], errors
    assert len(results) == 2
    for r in results:
        assert r.detection is not None and all(evaluate(sep_pair, r).hits)


# --------------------------------------------------------------------------- 8. gerçek UI + gerçek pipeline: maske akışı

pytest.importorskip("pytestqt")


def _pump(app, ms: int) -> None:
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def _wait(app, cond, timeout_s: float = 240.0) -> bool:
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture()
def ui_env(qapp, tmp_path, sep_pair):
    from app.prefs import UiPrefs
    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme
    from core.pipeline import set_default_pipeline, shutdown_default_pipeline

    apply_theme("dark")
    before, after = write_pair(sep_pair, tmp_path / "src", "ui")
    lib = Library(tmp_path / "library")
    cfg = AppConfig()
    cfg.paths.library_dir = lib.root
    win = MainWindow(lib, cfg, UiPrefs(), tmp_path / "config.json", tmp_path / "prefs.json", probe_device=False)
    win.resize(1440, 900)
    win.show()
    set_default_pipeline(build_default_pipeline(AppConfig()))
    f = lib.repo.create_folder("X Bölgesi")
    recs = win.library_view.import_files([before, after], f.id, ImageSource.DRONE, sync=True)
    by = {r.original_name: r for r in recs}
    yield win, lib, by[before.name], by[after.name]
    win.compare_view.shutdown() if hasattr(win.compare_view, "shutdown") else None
    shutdown_default_pipeline()
    win.close()
    lib.close()


def test_ui_mask_draw_save_recompare_removes_region(qapp, ui_env, sep_pair) -> None:
    """Gerçek UI: çiz -> kaydet ve yeniden karşılaştır -> bölge kayboluyor; geçmişten açma maskeli sonucu getiriyor;
    maske temizlenince bölge geri geliyor."""
    win, lib, rb, ra = ui_env
    cv = win.compare_view
    masked, kept = sep_pair.added_bboxes[0], sep_pair.added_bboxes[1]

    win.open_compare(rb, ra)
    assert _wait(qapp, lambda: not cv.is_busy() and cv._result is not None), "ilk karşılaştırma bitmedi"
    first = cv._result
    assert _touches(first, masked) and _touches(first, kept)
    assert first.detection.method == "ssim+dinov2"

    # Maske modu: çalışan maske yükleme worker'ı bitsin, sonra fırça vuruşları
    cv.enter_mask_mode()
    assert cv.mask_mode_active()
    assert _wait(qapp, lambda: not any(w.isRunning() for w in cv._mask_workers), 20)
    ed = cv.mask_editor
    ed.tool = "brush"
    ed.brush_px = 60
    x, y, w, h = masked
    pts = [(xx, yy) for yy in range(y - 30, y + h + 31, 40) for xx in range(x - 30, x + w + 31, 40)]
    ed.stroke(pts)
    assert ed.has_content() and ed.is_modified()
    cv.save_mask_and_recompare()
    assert _wait(qapp, lambda: not cv._mask_saving and not cv.is_busy() and cv._result is not first, 240), \
        "maskeli yeniden karşılaştırma tamamlanmadı"
    second = cv._result
    assert not _touches(second, masked), "maskelenen bölge sonuçta hâlâ var"
    assert _touches(second, kept), "maskelenmeyen bölge kayboldu"
    assert cv.mask_badge.isVisibleTo(cv) or lib.repo.get_image(rb.id).mask_hash
    fresh_before = lib.repo.get_image(rb.id)
    assert fresh_before.mask_hash
    crec = lib.find_comparison(fresh_before, ra)
    assert crec is not None and crec.mask_hash == fresh_before.mask_hash
    # kutular iki görünümde de aynı sayıda
    n = len(second.detection.regions)
    assert cv.visible_region_count() == (n, n) or not cv.diff_switch.isChecked()

    # Geçmişten aç: önbellekten (maskeli) gelmeli
    cv.open_comparison(crec)
    assert _wait(qapp, lambda: not cv.is_busy() and cv._result is not None and cv._result is not second, 60)
    assert not _touches(cv._result, masked)

    # Maskeyi sil -> yeniden hesapla -> bölge geri gelir
    cleared = lib.clear_mask(fresh_before)
    prev = cv._result
    cv.open_pair(cleared, ra)
    assert _wait(qapp, lambda: not cv.is_busy() and cv._result is not None and cv._result is not prev, 240)
    assert _touches(cv._result, masked)


# --------------------------------------------------------------------------- 9. dışa aktarma varsayılan yolu

@pytest.mark.xfail(strict=True, reason="export_dialog varsayılanı göreli 'karsilastirma.png': çalışma dizinine "
                   "(proje köküne) yazar (app/ui/compare_view.py:734); Resimler klasörü kullanılmalı")
def test_export_dialog_default_path_is_absolute_user_folder(qapp, tmp_path, monkeypatch) -> None:
    """Kaydet penceresi varsayılan yolu mutlak (kullanıcının Resimler klasörü) olmalı."""
    from app.ui import compare_view as cvmod
    from tests.ui.helpers import make_result

    seen: dict = {}

    def fake(parent, caption, default, flt):  # noqa: ANN001
        seen["default"] = default
        return "", ""

    monkeypatch.setattr(cvmod.QFileDialog, "getSaveFileName", staticmethod(fake))
    lib = Library(tmp_path / "lib")
    try:
        view = cvmod.CompareView(lib)
        view.show_result(make_result())
        view.export_dialog()
        assert "default" in seen
        assert Path(seen["default"]).is_absolute(), seen["default"]
    finally:
        lib.close()
