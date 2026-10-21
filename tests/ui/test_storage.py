"""app/storage testleri: klasör ağacı, içe aktarma, karşılaştırma kaydet/yükle."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.storage import Library
from core.contracts import (
    AlignmentQuality,
    BoundingBox,
    ChangeRegion,
    ChangeVerdict,
    ComparisonResult,
    DetectionResult,
    ImageSource,
)
from tests.ui.helpers import make_result


@pytest.fixture()
def lib(tmp_path: Path) -> Library:
    library = Library(tmp_path / "library")
    yield library
    library.close()


def _png(path: Path, color=(120, 160, 90), size=(640, 480)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_schema_version_and_reopen(tmp_path: Path) -> None:
    first = Library(tmp_path / "lib")
    assert first.repo.schema_version() == 1
    folder = first.repo.create_folder("A")
    first.close()
    second = Library(tmp_path / "lib")  # migration tekrar çalışmaz, veri kalır
    assert second.repo.schema_version() == 1
    assert [f.name for f in second.repo.list_folders()] == ["A"]
    assert second.repo.get_folder(folder.id) is not None
    second.close()


def test_nested_folders_unlimited_depth(lib: Library) -> None:
    parent = None
    for i in range(12):
        parent = lib.repo.create_folder(f"Seviye {i}", parent.id if parent else None)
    assert [f.name for f in lib.repo.folder_path(parent.id)][:2] == ["Seviye 0", "Seviye 1"]
    assert len(lib.repo.folder_path(parent.id)) == 12
    root = lib.repo.list_folders()[0]
    assert len(lib.repo.descendant_ids(root.id)) == 12


def test_move_folder_and_cycle_protection(lib: Library) -> None:
    a = lib.repo.create_folder("A")
    b = lib.repo.create_folder("B", a.id)
    c = lib.repo.create_folder("C", b.id)
    d = lib.repo.create_folder("D")
    lib.repo.move_folder(c.id, d.id)
    assert lib.repo.get_folder(c.id).parent_id == d.id
    with pytest.raises(ValueError):
        lib.repo.move_folder(a.id, b.id)      # kendi alt klasörüne
    with pytest.raises(ValueError):
        lib.repo.move_folder(a.id, a.id)
    lib.repo.move_folder(c.id, None)
    assert lib.repo.get_folder(c.id).parent_id is None


def test_rename_and_empty_name(lib: Library) -> None:
    f = lib.repo.create_folder("Eski")
    lib.repo.rename_folder(f.id, "Yeni")
    assert lib.repo.get_folder(f.id).name == "Yeni"
    with pytest.raises(ValueError):
        lib.repo.create_folder("   ")


def test_delete_folder_cascades_including_files(lib: Library, tmp_path: Path) -> None:
    a = lib.repo.create_folder("A")
    b = lib.repo.create_folder("B", a.id)
    img1 = lib.import_image(_png(tmp_path / "1.png"), a.id)
    img2 = lib.import_image(_png(tmp_path / "2.png"), b.id)
    other = lib.repo.create_folder("Diğer")
    img3 = lib.import_image(_png(tmp_path / "3.png"), other.id)
    cmp_rec = lib.save_comparison(img2, img3, make_result(img2.width, img2.height))
    files = [lib.image_path(img1), lib.thumb_path(img2), lib.abs_path(cmp_rec.result_dir)]
    assert all(p.exists() for p in files)

    lib.delete_folder(a.id)

    assert [f.name for f in lib.repo.list_folders()] == ["Diğer"]
    assert lib.repo.get_image(img1.id) is None and lib.repo.get_image(img2.id) is None
    assert lib.repo.get_image(img3.id) is not None
    assert lib.repo.list_comparisons() == []        # b'deki görüntüye bağlı karşılaştırma da gitti
    assert not any(p.exists() for p in files)
    assert lib.image_path(img3).exists()


def test_import_copies_and_makes_thumbnail(lib: Library, tmp_path: Path) -> None:
    src = _png(tmp_path / "kaynak.png", size=(1000, 500))
    folder = lib.repo.create_folder("X")
    rec = lib.import_image(src, folder.id, captured_at=datetime(2026, 10, 12, 9, 30),
                           source=ImageSource.DRONE)
    assert src.exists()                              # orijinale dokunulmadı
    assert lib.image_path(rec).is_file() and lib.image_path(rec) != src
    assert (rec.width, rec.height) == (1000, 500)
    assert rec.captured_at == "2026-10-12T09:30:00" and rec.source == "drone"
    with Image.open(lib.thumb_path(rec)) as thumb:
        assert max(thumb.size) == 320
    assert lib.repo.image_counts() == {folder.id: 1}


def test_import_reads_exif_date_then_falls_back_to_file_date(lib: Library, tmp_path: Path) -> None:
    folder = lib.repo.create_folder("X")
    with_exif = tmp_path / "exif.jpg"
    img = Image.new("RGB", (200, 100), "white")
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = "2025:03:04 05:06:07"
    img.save(with_exif, exif=exif)
    rec = lib.import_image(with_exif, folder.id)
    assert rec.captured_at == "2025-03-04T05:06:07"

    plain = _png(tmp_path / "plain.png")
    rec2 = lib.import_image(plain, folder.id)
    expected = datetime.fromtimestamp(plain.stat().st_mtime).replace(microsecond=0).isoformat()
    assert rec2.captured_at == expected


def test_list_images_sorted_and_filtered(lib: Library, tmp_path: Path) -> None:
    a = lib.repo.create_folder("A")
    b = lib.repo.create_folder("B", a.id)
    late = lib.import_image(_png(tmp_path / "gec.png"), a.id, datetime(2026, 10, 19), ImageSource.SATELLITE)
    early = lib.import_image(_png(tmp_path / "erken.png"), b.id, datetime(2026, 10, 12), ImageSource.DRONE)
    assert [i.id for i in lib.repo.list_images(a.id)] == [late.id]
    assert [i.id for i in lib.repo.list_images(a.id, include_subfolders=True)] == [early.id, late.id]
    assert [i.id for i in lib.repo.list_images(search="ERK")] == [early.id]
    assert [i.id for i in lib.repo.list_images(source="satellite")] == [late.id]
    assert [i.id for i in lib.repo.list_images(date_from="2026-10-15")] == [late.id]
    assert lib.repo.list_images(search="%") == []     # joker karakter kaçışı


def test_move_and_delete_image(lib: Library, tmp_path: Path) -> None:
    a, b = lib.repo.create_folder("A"), lib.repo.create_folder("B")
    rec = lib.import_image(_png(tmp_path / "i.png"), a.id)
    lib.repo.move_image(rec.id, b.id)
    assert lib.repo.get_image(rec.id).folder_id == b.id
    path = lib.image_path(rec)
    lib.delete_image(rec.id)
    assert not path.exists() and lib.repo.get_image(rec.id) is None


def test_save_and_load_comparison_roundtrip(lib: Library, tmp_path: Path) -> None:
    f = lib.repo.create_folder("X")
    before = lib.import_image(_png(tmp_path / "b.png"), f.id, datetime(2026, 10, 12))
    after = lib.import_image(_png(tmp_path / "a.png"), f.id, datetime(2026, 10, 19))
    result = make_result(before.width, before.height, with_llm=True)
    assert lib.find_comparison(before, after) is None

    rec = lib.save_comparison(before, after, result)
    assert rec.region_count == 3 and rec.quality == "good" and rec.summary == result.summary

    found = lib.find_comparison(before, after)
    assert found is not None and found.id == rec.id
    loaded = lib.load_comparison(found)
    assert loaded.summary == result.summary and loaded.llm_used
    assert loaded.registration.quality is AlignmentQuality.GOOD
    np.testing.assert_array_equal(loaded.registration.after_warped, result.registration.after_warped)
    np.testing.assert_array_equal(loaded.registration.before.pixels, result.registration.before.pixels)
    assert loaded.registration.before.original.path == lib.image_path(before)
    np.testing.assert_allclose(loaded.detection.change_map, result.detection.change_map, atol=1e-3)
    assert [r.region_id for r in loaded.detection.regions] == [1, 2, 3]
    r0, o0 = loaded.detection.regions[0], result.detection.regions[0]
    assert r0.bbox == o0.bbox and r0.verdict is ChangeVerdict.REAL_CHANGE
    assert r0.description == o0.description and r0.llm_confidence == o0.llm_confidence
    np.testing.assert_array_equal(r0.mask, o0.mask)

    # Aynı çift tekrar kaydedilince yeni kayıt açılmaz, güncellenir.
    rec2 = lib.save_comparison(before, after, result)
    assert rec2.id == rec.id and len(lib.repo.list_comparisons()) == 1


def test_failed_result_roundtrip(lib: Library, tmp_path: Path) -> None:
    f = lib.repo.create_folder("X")
    before = lib.import_image(_png(tmp_path / "b.png"), f.id)
    after = lib.import_image(_png(tmp_path / "a.png"), f.id)
    result = make_result(64, 48, quality=AlignmentQuality.FAILED)
    assert result.detection is None
    rec = lib.save_comparison(before, after, result)
    loaded = lib.load_comparison(rec)
    assert loaded.detection is None and loaded.registration.quality is AlignmentQuality.FAILED
    assert loaded.registration.homography is None


def test_empty_regions_result(lib: Library, tmp_path: Path) -> None:
    f = lib.repo.create_folder("X")
    before = lib.import_image(_png(tmp_path / "b.png"), f.id)
    after = lib.import_image(_png(tmp_path / "a.png"), f.id)
    base = make_result(64, 48)
    base.detection = DetectionResult(change_map=base.detection.change_map, regions=[], method="ssim")
    rec = lib.save_comparison(before, after, base)
    assert lib.load_comparison(rec).detection.regions == []


def test_region_dataclass_defaults() -> None:
    r = ChangeRegion(1, BoundingBox(0, 0, 2, 2), np.ones((2, 2), bool), 0.5, 4)
    assert r.verdict is ChangeVerdict.UNREVIEWED
