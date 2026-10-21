"""Kütüphane: görüntü kopyalama, thumbnail, EXIF tarihi ve karşılaştırma sonuçlarının diske kaydı."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from app.storage.models import ComparisonRecord, Folder, ImageRecord
from app.storage.repository import LibraryRepository
from core.config import user_data_dir
from core.contracts import (
    AlignmentQuality,
    BoundingBox,
    ChangeRegion,
    ChangeVerdict,
    ComparisonResult,
    DetectionResult,
    ImageInput,
    ImageSource,
    PreparedImage,
    RegistrationResult,
)

THUMB_SIZE = 320
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
RESULT_FORMAT_VERSION = 1

# Uydu görüntüleri büyük olabilir: Pillow sınırı 400 MP'ye yükseltilir ama korumayı tamamen kapatmaz
# (core/registration/preprocess.py ile aynı sınır; süreç geneli ayar olduğu için None yapılmamalı).
MAX_IMAGE_PIXELS = 400_000_000
if Image.MAX_IMAGE_PIXELS is not None and Image.MAX_IMAGE_PIXELS < MAX_IMAGE_PIXELS:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def default_library_dir() -> Path:
    """Varsayılan kütüphane konumu: `%LOCALAPPDATA%/ChangeScope/library`."""
    return user_data_dir() / "library"


def is_image_file(path: Path) -> bool:
    """Uzantıya göre desteklenen bir görüntü dosyası mı?"""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def read_exif_date(path: Path) -> datetime | None:
    """EXIF çekim tarihini (DateTimeOriginal/DateTime) okur; yoksa None."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            candidates = []
            try:
                sub = exif.get_ifd(0x8769)
                candidates += [sub.get(36867), sub.get(36868)]
            except Exception:
                pass
            candidates.append(exif.get(306))
    except Exception:
        return None
    for value in candidates:
        if not value:
            continue
        text = value.decode(errors="ignore") if isinstance(value, bytes) else str(value)
        text = text.strip().strip("\x00")
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
            try:
                return datetime.strptime(text[:19] if "%H" in fmt else text[:10], fmt)
            except ValueError:
                continue
    return None


def file_date(path: Path) -> datetime:
    """Dosyanın değiştirilme tarihi (EXIF yoksa varsayılan çekim tarihi)."""
    return datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0)


def make_thumbnail(src: Path, dst: Path, size: int = THUMB_SIZE) -> tuple[int, int]:
    """`src` görüntüsünden `size` px'e sığan JPEG thumbnail üretir; (genişlik, yükseklik) döndürür."""
    with Image.open(src) as img:
        width, height = img.size
        img.draft("RGB", (size * 2, size * 2))
        img = img.convert("RGB")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=88)
    return width, height


class Library:
    """Uygulamanın görüntü kütüphanesi: klasör kökü + `library.sqlite3`.

    Dosya yerleşimi (köke göre):
        images/<uuid>.<ext>     kopyalanmış görüntüler (orijinale dokunulmaz)
        thumbs/<uuid>.jpg       320 px thumbnail'ler
        results/<id>/           karşılaştırma sonuçları
        library.sqlite3
    """

    def __init__(self, root: Path) -> None:
        """Kütüphaneyi açar; klasörler ve veritabanı yoksa oluşturulur."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo = LibraryRepository(self.root / "library.sqlite3")

    def close(self) -> None:
        """Veritabanını kapatır."""
        self.repo.close()

    # -- yollar --------------------------------------------------------------

    def abs_path(self, relative: str) -> Path:
        """Kütüphane köküne göre göreli yolu mutlak yola çevirir."""
        return self.root / relative

    def image_path(self, rec: ImageRecord) -> Path:
        """Görüntünün kütüphanedeki tam boyutlu dosyası."""
        return self.abs_path(rec.file_name)

    def thumb_path(self, rec: ImageRecord) -> Path | None:
        """Thumbnail dosyası (yoksa None)."""
        return self.abs_path(rec.thumb_path) if rec.thumb_path else None

    def to_image_input(self, rec: ImageRecord) -> ImageInput:
        """Pipeline girdisine çevirir."""
        try:
            source = ImageSource(rec.source)
        except ValueError:
            source = ImageSource.UNKNOWN
        return ImageInput(path=self.image_path(rec), captured_at=rec.captured_at, source=source)

    # -- klasörler -----------------------------------------------------------

    def delete_folder(self, folder_id: int) -> None:
        """Klasörü (alt klasörler, görüntüler, sonuçlar dahil) veritabanından ve diskten siler."""
        self._remove_paths(self.repo.delete_folder(folder_id))

    def delete_image(self, image_id: int) -> None:
        """Görüntüyü, thumbnail'ini ve bağlı karşılaştırma sonuçlarını siler."""
        self._remove_paths(self.repo.delete_image(image_id))

    def delete_comparison(self, comparison_id: int) -> None:
        """Karşılaştırma kaydını ve sonuç dosyalarını siler."""
        self._remove_paths(self.repo.delete_comparison(comparison_id))

    def _remove_paths(self, relative_paths: list[str]) -> None:
        for rel in relative_paths:
            target = self.abs_path(rel)
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()
            except OSError:
                pass

    # -- içe aktarma ---------------------------------------------------------

    def import_image(
        self,
        src: Path,
        folder_id: int,
        captured_at: datetime | str | None = None,
        source: ImageSource | str = ImageSource.UNKNOWN,
    ) -> ImageRecord:
        """Görüntüyü kütüphaneye KOPYALAR, thumbnail üretir ve kaydeder.

        Args:
            src: Orijinal dosya (değiştirilmez).
            folder_id: Hedef klasör.
            captured_at: Çekim tarihi. None ise EXIF, o da yoksa dosya tarihi kullanılır.
            source: Kaynak etiketi.
        """
        src = Path(src)
        if captured_at is None:
            captured_at = read_exif_date(src) or file_date(src)
        if isinstance(captured_at, datetime):
            captured_at = captured_at.replace(microsecond=0).isoformat()
        source_value = source.value if isinstance(source, ImageSource) else str(source)

        stem = uuid.uuid4().hex
        rel_image = f"images/{stem}{src.suffix.lower()}"
        rel_thumb = f"thumbs/{stem}.jpg"
        dst = self.abs_path(rel_image)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        try:
            width, height = make_thumbnail(dst, self.abs_path(rel_thumb))
        except Exception:
            dst.unlink(missing_ok=True)
            raise
        return self.repo.add_image(
            folder_id, rel_image, src.name, captured_at, source_value, width, height, rel_thumb
        )

    # -- karşılaştırma sonuçları --------------------------------------------

    def find_comparison(self, before: ImageRecord, after: ImageRecord) -> ComparisonRecord | None:
        """Aynı çift için kaydedilmiş sonuç var mı?"""
        rec = self.repo.find_comparison(before.id, after.id)
        if rec is not None and not (self.abs_path(rec.result_dir) / "meta.json").is_file():
            return None
        return rec

    def save_comparison(
        self, before: ImageRecord, after: ImageRecord, result: ComparisonResult
    ) -> ComparisonRecord:
        """Karşılaştırma sonucunu `results/<id>/` altına yazar ve kaydı ekler/günceller.

        change_map, bölgeler (maskeleriyle), AI yorumları, hizalanmış görüntü ve özet saklanır.
        """
        existing = self.repo.find_comparison(before.id, after.id)
        folder_id: int | None = before.folder_id
        quality = result.registration.quality.value
        n_regions = len(result.detection.regions) if result.detection else 0
        if existing is None:
            rel_dir = f"results/{uuid.uuid4().hex[:12]}"
            rec = self.repo.add_comparison(
                folder_id, before.id, after.id, rel_dir, quality, n_regions, result.summary
            )
        else:
            rec, rel_dir = existing, existing.result_dir
            self.repo.update_comparison(rec.id, quality, n_regions, result.summary)
        _write_result(self.abs_path(rel_dir), result)
        return self.repo.get_comparison(rec.id)  # type: ignore[return-value]

    def load_comparison(self, rec: ComparisonRecord) -> ComparisonResult:
        """Kaydedilmiş sonucu yeniden hesaplamadan okur."""
        before = self.repo.get_image(rec.before_image_id)
        if before is None:
            raise FileNotFoundError("Karşılaştırmanın 'önce' görüntüsü kütüphanede yok")
        after = self.repo.get_image(rec.after_image_id)
        return _read_result(
            self.abs_path(rec.result_dir),
            self.to_image_input(before),
            self.to_image_input(after) if after else None,
        )


# ---------------------------------------------------------------------------
# Sonuç (de)serileştirme
# ---------------------------------------------------------------------------

def _save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(arr)).save(path, "PNG", compress_level=3)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _write_result(directory: Path, result: ComparisonResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    reg = result.registration
    _save_rgb(directory / "before.png", reg.before.pixels)
    _save_rgb(directory / "after_warped.png", reg.after_warped)
    arrays: dict[str, np.ndarray] = {"valid_mask": reg.valid_mask.astype(bool)}
    if reg.homography is not None:
        arrays["homography"] = np.asarray(reg.homography, dtype=np.float64)
    regions_meta: list[dict] = []
    det = result.detection
    if det is not None:
        arrays["change_map"] = det.change_map.astype(np.float16)
        for r in det.regions:
            arrays[f"mask_{r.region_id}"] = r.mask.astype(bool)
            regions_meta.append({
                "region_id": r.region_id,
                "bbox": [r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height],
                "score": float(r.score),
                "area_px": int(r.area_px),
                "verdict": r.verdict.value,
                "description": r.description,
                "llm_confidence": r.llm_confidence,
            })
    np.savez_compressed(directory / "arrays.npz", **arrays)
    meta = {
        "format": RESULT_FORMAT_VERSION,
        "quality": reg.quality.value,
        "num_matches": reg.num_matches,
        "num_inliers": reg.num_inliers,
        "reprojection_error_px": reg.reprojection_error_px,
        "notes": list(reg.notes),
        "scale_from_original": reg.before.scale_from_original,
        "was_compressed": reg.before.was_compressed,
        "has_detection": det is not None,
        "method": det.method if det else None,
        "params": det.params if det else {},
        "regions": regions_meta,
        "llm_used": result.llm_used,
        "elapsed_s": result.elapsed_s,
        "summary": result.summary,
    }
    tmp = directory / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, directory / "meta.json")  # meta.json en son: yarım yazılmış sonuç "yok" sayılır


def _read_result(directory: Path, before_in: ImageInput, after_in: ImageInput | None) -> ComparisonResult:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    before_px = _load_rgb(directory / "before.png")
    after_warped = _load_rgb(directory / "after_warped.png")
    with np.load(directory / "arrays.npz") as npz:
        valid_mask = npz["valid_mask"].astype(bool)
        homography = npz["homography"] if "homography" in npz.files else None
        change_map = npz["change_map"].astype(np.float32) if "change_map" in npz.files else None
        masks = {name: npz[name].astype(bool) for name in npz.files if name.startswith("mask_")}
    prepared = PreparedImage(
        original=before_in, pixels=before_px,
        scale_from_original=float(meta.get("scale_from_original", 1.0)),
        was_compressed=bool(meta.get("was_compressed", False)),
    )
    registration = RegistrationResult(
        before=prepared, after_warped=after_warped, valid_mask=valid_mask, homography=homography,
        num_matches=int(meta["num_matches"]), num_inliers=int(meta["num_inliers"]),
        reprojection_error_px=float(meta["reprojection_error_px"]),
        quality=AlignmentQuality(meta["quality"]), notes=list(meta.get("notes", [])),
    )
    detection: DetectionResult | None = None
    if meta.get("has_detection") and change_map is not None:
        regions = []
        for r in meta["regions"]:
            x, y, w, h = r["bbox"]
            regions.append(ChangeRegion(
                region_id=r["region_id"], bbox=BoundingBox(x, y, w, h),
                mask=masks.get(f"mask_{r['region_id']}", np.ones((h, w), dtype=bool)),
                score=r["score"], area_px=r["area_px"], verdict=ChangeVerdict(r["verdict"]),
                description=r.get("description"), llm_confidence=r.get("llm_confidence"),
            ))
        detection = DetectionResult(
            change_map=change_map, regions=regions, method=meta.get("method") or "",
            params=meta.get("params", {}),
        )
    return ComparisonResult(
        registration=registration, detection=detection, llm_used=bool(meta["llm_used"]),
        elapsed_s=float(meta["elapsed_s"]), summary=meta.get("summary"),
    )


__all__ = [
    "IMAGE_EXTENSIONS", "Library", "default_library_dir", "file_date", "is_image_file",
    "make_thumbnail", "read_exif_date", "Folder",
]
