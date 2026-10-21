"""SQLite erişim katmanı. UI ve diğer katmanlar SQL'i yalnızca buradan çağırır."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from app.storage.models import ComparisonRecord, Folder, ImageRecord

# ---------------------------------------------------------------------------
# Migration'lar: SCHEMA_MIGRATIONS[i], şemayı sürüm i -> i+1 yapar.
# ---------------------------------------------------------------------------


def _migrate_v1(con: sqlite3.Connection) -> None:
    for stmt in (
        """CREATE TABLE folders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id   INTEGER NULL REFERENCES folders(id) ON DELETE CASCADE,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            sort_order  INTEGER NOT NULL DEFAULT 0
        )""",
        "CREATE INDEX idx_folders_parent ON folders(parent_id)",
        """CREATE TABLE images (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id     INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
            file_name     TEXT NOT NULL,
            original_name TEXT NOT NULL,
            captured_at   TEXT NULL,
            source        TEXT NOT NULL DEFAULT 'unknown',
            width         INTEGER NOT NULL DEFAULT 0,
            height        INTEGER NOT NULL DEFAULT 0,
            thumb_path    TEXT NULL,
            created_at    TEXT NOT NULL
        )""",
        "CREATE INDEX idx_images_folder ON images(folder_id)",
        """CREATE TABLE comparisons (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id       INTEGER NULL REFERENCES folders(id) ON DELETE CASCADE,
            before_image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            after_image_id  INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            result_dir      TEXT NOT NULL,
            quality         TEXT NOT NULL,
            region_count    INTEGER NOT NULL DEFAULT 0,
            summary         TEXT NULL,
            created_at      TEXT NOT NULL
        )""",
        "CREATE INDEX idx_comparisons_pair ON comparisons(before_image_id, after_image_id)",
    ):
        con.execute(stmt)


def _migrate_v2(con: sqlite3.Connection) -> None:
    """Faz 2: görüntüye bağlı yok sayma maskesi + karşılaştırmanın hangi maskeyle/yöntemle çalıştığı."""
    for stmt in (
        "ALTER TABLE images ADD COLUMN mask_path TEXT NULL",
        "ALTER TABLE images ADD COLUMN mask_hash TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE comparisons ADD COLUMN mask_hash TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE comparisons ADD COLUMN method TEXT NOT NULL DEFAULT ''",
    ):
        con.execute(stmt)


SCHEMA_MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_migrate_v1, _migrate_v2]
SCHEMA_VERSION = len(SCHEMA_MIGRATIONS)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _folder(row: sqlite3.Row) -> Folder:
    return Folder(row["id"], row["parent_id"], row["name"], row["created_at"], row["sort_order"])


def _image(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        row["id"], row["folder_id"], row["file_name"], row["original_name"], row["captured_at"],
        row["source"], row["width"], row["height"], row["thumb_path"], row["created_at"],
        row["mask_path"], row["mask_hash"],
    )


def _comparison(row: sqlite3.Row) -> ComparisonRecord:
    return ComparisonRecord(
        row["id"], row["folder_id"], row["before_image_id"], row["after_image_id"],
        row["result_dir"], row["quality"], row["region_count"], row["summary"], row["created_at"],
        row["mask_hash"], row["method"],
    )


class LibraryRepository:
    """Klasör / görüntü / karşılaştırma tablolarına erişim.

    Thread-safe (tek bağlantı + kilit). Dosya sistemi işlemleri yapmaz; silme metotları
    diskten de silinmesi gereken göreli yolları döndürür (bkz. `Library`).
    """

    def __init__(self, db_path: Path) -> None:
        """Veritabanını açar (yoksa oluşturur) ve migration'ları uygular."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # -- altyapı -------------------------------------------------------------

    def _tx(self) -> "_Tx":
        return _Tx(self)

    def _migrate(self) -> None:
        with self._lock:
            self._con.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self._con.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._con.execute("INSERT INTO schema_version(version) VALUES (0)")
                current = 0
            else:
                current = row["version"]
            if current > SCHEMA_VERSION:
                raise RuntimeError(f"Veritabanı şeması daha yeni ({current} > {SCHEMA_VERSION})")
            for version in range(current, SCHEMA_VERSION):
                with self._tx():
                    SCHEMA_MIGRATIONS[version](self._con)
                    self._con.execute("UPDATE schema_version SET version = ?", (version + 1,))

    def schema_version(self) -> int:
        """Veritabanındaki şema sürümü."""
        with self._lock:
            return int(self._con.execute("SELECT version FROM schema_version").fetchone()["version"])

    def close(self) -> None:
        """Bağlantıyı kapatır."""
        with self._lock:
            self._con.close()

    # -- klasörler -----------------------------------------------------------

    def create_folder(self, name: str, parent_id: int | None = None) -> Folder:
        """Yeni klasör oluşturur (sınırsız derinlik)."""
        name = name.strip()
        if not name:
            raise ValueError("Klasör adı boş olamaz")
        with self._tx():
            if parent_id is not None and self.get_folder(parent_id) is None:
                raise KeyError(f"Üst klasör yok: {parent_id}")
            order = self._con.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM folders WHERE parent_id IS ?",
                (parent_id,),
            ).fetchone()[0]
            cur = self._con.execute(
                "INSERT INTO folders(parent_id, name, created_at, sort_order) VALUES (?,?,?,?)",
                (parent_id, name, _now(), order),
            )
            new_id = int(cur.lastrowid)
        return self.get_folder(new_id)  # type: ignore[return-value]

    def get_folder(self, folder_id: int) -> Folder | None:
        """Klasörü döndürür; yoksa None."""
        with self._lock:
            row = self._con.execute("SELECT * FROM folders WHERE id = ?", (folder_id,)).fetchone()
        return _folder(row) if row else None

    def list_folders(self) -> list[Folder]:
        """Tüm klasörler (sıra, ad sırasıyla)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT * FROM folders ORDER BY sort_order, name COLLATE NOCASE, id"
            ).fetchall()
        return [_folder(r) for r in rows]

    def rename_folder(self, folder_id: int, name: str) -> None:
        """Klasörü yeniden adlandırır."""
        name = name.strip()
        if not name:
            raise ValueError("Klasör adı boş olamaz")
        with self._tx():
            self._con.execute("UPDATE folders SET name = ? WHERE id = ?", (name, folder_id))

    def descendant_ids(self, folder_id: int) -> list[int]:
        """Klasör ve tüm alt klasörlerinin kimlikleri (klasörün kendisi dahil)."""
        with self._lock:
            rows = self._con.execute(
                """
                WITH RECURSIVE sub(id) AS (
                    SELECT id FROM folders WHERE id = ?
                    UNION ALL
                    SELECT f.id FROM folders f JOIN sub ON f.parent_id = sub.id
                )
                SELECT id FROM sub
                """,
                (folder_id,),
            ).fetchall()
        return [r["id"] for r in rows]

    def move_folder(self, folder_id: int, new_parent_id: int | None) -> None:
        """Klasörü başka bir klasörün altına (veya köke) taşır. Döngüye izin vermez."""
        with self._tx():
            if new_parent_id is not None:
                if new_parent_id in self.descendant_ids(folder_id):
                    raise ValueError("Klasör kendisinin ya da alt klasörünün içine taşınamaz")
                if self.get_folder(new_parent_id) is None:
                    raise KeyError(f"Hedef klasör yok: {new_parent_id}")
            order = self._con.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM folders WHERE parent_id IS ?",
                (new_parent_id,),
            ).fetchone()[0]
            self._con.execute(
                "UPDATE folders SET parent_id = ?, sort_order = ? WHERE id = ?",
                (new_parent_id, order, folder_id),
            )

    def folder_path(self, folder_id: int) -> list[Folder]:
        """Kökten klasöre giden zincir (ör. X Bölgesi, Y Dağı, Kuzey Yamaç)."""
        chain: list[Folder] = []
        current = self.get_folder(folder_id)
        while current is not None:
            chain.append(current)
            current = self.get_folder(current.parent_id) if current.parent_id is not None else None
        return list(reversed(chain))

    def delete_folder(self, folder_id: int) -> list[str]:
        """Klasörü alt klasörleri, görüntüleri ve karşılaştırmalarıyla birlikte siler (cascade).

        Returns:
            Diskten de silinmesi gereken göreli yollar (görüntü, thumbnail, sonuç klasörleri).
        """
        with self._tx():
            ids = self.descendant_ids(folder_id)
            marks = ",".join("?" * len(ids))
            paths: list[str] = []
            for row in self._con.execute(
                f"SELECT file_name, thumb_path, mask_path FROM images WHERE folder_id IN ({marks})", ids
            ).fetchall():
                paths += [p for p in (row["file_name"], row["thumb_path"], row["mask_path"]) if p]
            # Bu klasörlerdeki görüntülere bağlı tüm karşılaştırmalar (başka klasördekiler dahil).
            sub = f"SELECT id FROM images WHERE folder_id IN ({marks})"
            for row in self._con.execute(
                f"""SELECT DISTINCT result_dir FROM comparisons WHERE folder_id IN ({marks})
                    OR before_image_id IN ({sub}) OR after_image_id IN ({sub})""",
                ids * 3,
            ).fetchall():
                paths.append(row["result_dir"])
            self._con.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        return paths

    def image_counts(self) -> dict[int, int]:
        """Klasör başına DOĞRUDAN görüntü sayısı."""
        with self._lock:
            rows = self._con.execute(
                "SELECT folder_id, COUNT(*) AS n FROM images GROUP BY folder_id"
            ).fetchall()
        return {r["folder_id"]: r["n"] for r in rows}

    def total_image_count(self) -> int:
        """Kütüphanedeki toplam görüntü sayısı."""
        with self._lock:
            return int(self._con.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    # -- görüntüler ----------------------------------------------------------

    def add_image(
        self, folder_id: int, file_name: str, original_name: str, captured_at: str | None,
        source: str, width: int, height: int, thumb_path: str | None,
    ) -> ImageRecord:
        """Görüntü kaydı ekler (dosyaların kopyalanması `Library`'nin işi)."""
        with self._tx():
            cur = self._con.execute(
                """INSERT INTO images(folder_id, file_name, original_name, captured_at, source,
                   width, height, thumb_path, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (folder_id, file_name, original_name, captured_at, source, width, height,
                 thumb_path, _now()),
            )
            new_id = int(cur.lastrowid)
        return self.get_image(new_id)  # type: ignore[return-value]

    def get_image(self, image_id: int) -> ImageRecord | None:
        """Görüntü kaydı; yoksa None."""
        with self._lock:
            row = self._con.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return _image(row) if row else None

    def list_images(
        self,
        folder_id: int | None = None,
        include_subfolders: bool = False,
        search: str = "",
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ImageRecord]:
        """Görüntüleri çekim tarihine göre (eskiden yeniye) listeler.

        Args:
            folder_id: None ise tüm kütüphane.
            include_subfolders: Alt klasörlerdeki görüntüleri de kat.
            search: Dosya adında geçen metin.
            source: Kaynak filtresi (drone/uav/satellite/unknown).
            date_from, date_to: ISO tarih (dahil) sınırları; `captured_at` üzerinde.
        """
        sql = "SELECT * FROM images WHERE 1=1"
        args: list[object] = []
        if folder_id is not None:
            ids = self.descendant_ids(folder_id) if include_subfolders else [folder_id]
            sql += f" AND folder_id IN ({','.join('?' * len(ids))})"
            args += ids
        if search.strip():
            sql += " AND original_name LIKE ? ESCAPE '\\'"
            esc = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            args.append(f"%{esc}%")
        if source:
            sql += " AND source = ?"
            args.append(source)
        if date_from:
            sql += " AND captured_at >= ?"
            args.append(date_from)
        if date_to:
            sql += " AND captured_at <= ?"
            args.append(date_to)
        sql += " ORDER BY COALESCE(captured_at, created_at), id"
        with self._lock:
            rows = self._con.execute(sql, args).fetchall()
        return [_image(r) for r in rows]

    def move_image(self, image_id: int, folder_id: int) -> None:
        """Görüntüyü başka klasöre taşır."""
        with self._tx():
            if self.get_folder(folder_id) is None:
                raise KeyError(f"Hedef klasör yok: {folder_id}")
            self._con.execute("UPDATE images SET folder_id = ? WHERE id = ?", (folder_id, image_id))

    def update_image(
        self, image_id: int, captured_at: str | None = None, source: str | None = None
    ) -> None:
        """Görüntünün çekim tarihini ve/veya kaynağını günceller."""
        with self._tx():
            if captured_at is not None:
                self._con.execute("UPDATE images SET captured_at = ? WHERE id = ?", (captured_at, image_id))
            if source is not None:
                self._con.execute("UPDATE images SET source = ? WHERE id = ?", (source, image_id))

    def delete_image(self, image_id: int) -> list[str]:
        """Görüntüyü, maskesini ve ona bağlı karşılaştırmaları siler (cascade).

        Kural: bir görüntü silinince onu "önce" ya da "sonra" olarak kullanan TÜM karşılaştırma kayıtları
        ve sonuç klasörleri de silinir (yarım / "kaynak silindi" kayıtları bırakılmaz). Diskten silinecek
        göreli yolları döndürür.
        """
        with self._tx():
            img = self.get_image(image_id)
            if img is None:
                return []
            paths = [p for p in (img.file_name, img.thumb_path, img.mask_path) if p]
            for row in self._con.execute(
                "SELECT result_dir FROM comparisons WHERE before_image_id = ? OR after_image_id = ?",
                (image_id, image_id),
            ).fetchall():
                paths.append(row["result_dir"])
            self._con.execute("DELETE FROM images WHERE id = ?", (image_id,))
        return paths

    def set_image_mask(self, image_id: int, mask_path: str | None, mask_hash: str) -> str | None:
        """Görüntünün maske dosyası/özetini ayarlar (None → maske kaldırılır). Önceki maske yolunu döndürür."""
        with self._tx():
            row = self._con.execute("SELECT mask_path FROM images WHERE id = ?", (image_id,)).fetchone()
            if row is None:
                raise KeyError(f"Görüntü yok: {image_id}")
            self._con.execute(
                "UPDATE images SET mask_path = ?, mask_hash = ? WHERE id = ?",
                (mask_path, mask_hash if mask_path else "", image_id),
            )
            return row["mask_path"]

    def count_comparisons_for_image(self, image_id: int) -> int:
        """Görüntüyü "önce" ya da "sonra" olarak kullanan karşılaştırma sayısı."""
        with self._lock:
            return int(self._con.execute(
                "SELECT COUNT(*) FROM comparisons WHERE before_image_id = ? OR after_image_id = ?",
                (image_id, image_id),
            ).fetchone()[0])

    # -- karşılaştırmalar ----------------------------------------------------

    def add_comparison(
        self, folder_id: int | None, before_image_id: int, after_image_id: int, result_dir: str,
        quality: str, region_count: int, summary: str | None, mask_hash: str = "", method: str = "",
    ) -> ComparisonRecord:
        """Karşılaştırma kaydı ekler."""
        with self._tx():
            return self._insert_comparison(
                folder_id, before_image_id, after_image_id, result_dir, quality, region_count, summary,
                mask_hash, method,
            )

    def _insert_comparison(
        self, folder_id, before_image_id, after_image_id, result_dir, quality, region_count, summary,
        mask_hash, method,
    ) -> ComparisonRecord:
        cur = self._con.execute(
            """INSERT INTO comparisons(folder_id, before_image_id, after_image_id, result_dir,
               quality, region_count, summary, created_at, mask_hash, method) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (folder_id, before_image_id, after_image_id, result_dir, quality, region_count,
             summary, _now(), mask_hash, method),
        )
        return self.get_comparison(int(cur.lastrowid))  # type: ignore[return-value]

    def upsert_comparison(
        self, folder_id: int | None, before_image_id: int, after_image_id: int, new_result_dir: str,
        quality: str, region_count: int, summary: str | None, mask_hash: str = "", method: str = "",
    ) -> tuple[ComparisonRecord, bool]:
        """(önce, sonra) çifti için kaydı TEK transaction içinde ekler ya da günceller.

        Çift için kayıt varsa güncellenir (mevcut `result_dir` korunur; maske/yöntem yeni değerlere
        geçer). Yoksa `new_result_dir` ile eklenir. Eşzamanlı çağrılar tek satır üretir.

        Returns:
            (kayıt, yeni_mi)
        """
        with self._tx():
            row = self._con.execute(
                """SELECT id FROM comparisons WHERE before_image_id = ? AND after_image_id = ?
                   ORDER BY id DESC LIMIT 1""",
                (before_image_id, after_image_id),
            ).fetchone()
            if row is None:
                rec = self._insert_comparison(
                    folder_id, before_image_id, after_image_id, new_result_dir, quality, region_count,
                    summary, mask_hash, method)
                return rec, True
            self._con.execute(
                """UPDATE comparisons SET folder_id=?, quality=?, region_count=?, summary=?, created_at=?,
                   mask_hash=?, method=? WHERE id=?""",
                (folder_id, quality, region_count, summary, _now(), mask_hash, method, row["id"]),
            )
            return self.get_comparison(int(row["id"])), False  # type: ignore[return-value]

    def update_comparison(
        self, comparison_id: int, quality: str, region_count: int, summary: str | None,
        mask_hash: str | None = None, method: str | None = None,
    ) -> None:
        """Mevcut karşılaştırma özetini günceller (ör. AI yorumu sonradan geldiğinde)."""
        with self._tx():
            self._con.execute(
                "UPDATE comparisons SET quality=?, region_count=?, summary=?, created_at=? WHERE id=?",
                (quality, region_count, summary, _now(), comparison_id),
            )
            if mask_hash is not None:
                self._con.execute("UPDATE comparisons SET mask_hash=? WHERE id=?", (mask_hash, comparison_id))
            if method is not None:
                self._con.execute("UPDATE comparisons SET method=? WHERE id=?", (method, comparison_id))

    def get_comparison(self, comparison_id: int) -> ComparisonRecord | None:
        """Karşılaştırma kaydı; yoksa None."""
        with self._lock:
            row = self._con.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
        return _comparison(row) if row else None

    def find_comparison(
        self, before_image_id: int, after_image_id: int, mask_hash: str | None = None
    ) -> ComparisonRecord | None:
        """Aynı (önce, sonra) çifti için kaydedilmiş en yeni karşılaştırma.

        `mask_hash` verilirse yalnızca O maskeyle hesaplanmış kayıt döner (önbellek anahtarı);
        "" = maskesiz. None = maskeye bakma.
        """
        sql = "SELECT * FROM comparisons WHERE before_image_id = ? AND after_image_id = ?"
        args: list[object] = [before_image_id, after_image_id]
        if mask_hash is not None:
            sql += " AND mask_hash = ?"
            args.append(mask_hash)
        with self._lock:
            row = self._con.execute(sql + " ORDER BY id DESC LIMIT 1", args).fetchone()
        return _comparison(row) if row else None

    def list_comparisons(
        self, folder_id: int | None = None, search: str = "", sort: str = "newest"
    ) -> list[ComparisonRecord]:
        """Karşılaştırma geçmişi. `folder_id` verilirse alt klasörler dahil.

        Args:
            folder_id: Karşılaştırmanın kayıtlı olduğu (önce görüntüsünün) klasörü; None = hepsi.
            search: Görüntü adları, klasör adı ya da özet metninde geçen metin.
            sort: "newest" (yeni→eski, varsayılan), "oldest", "regions" (bölge sayısı, çoktan aza).
        """
        sql = (
            "SELECT c.* FROM comparisons c"
            " JOIN images b ON b.id = c.before_image_id"
            " JOIN images a ON a.id = c.after_image_id"
            " LEFT JOIN folders f ON f.id = c.folder_id WHERE 1=1"
        )
        args: list[object] = []
        if folder_id is not None:
            ids = self.descendant_ids(folder_id)
            sql += f" AND c.folder_id IN ({','.join('?' * len(ids))})"
            args += ids
        if search.strip():
            esc = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += (" AND (b.original_name LIKE ? ESCAPE '\\' OR a.original_name LIKE ? ESCAPE '\\'"
                    " OR c.summary LIKE ? ESCAPE '\\' OR f.name LIKE ? ESCAPE '\\')")
            args += [f"%{esc}%"] * 4
        order = {
            "newest": "c.created_at DESC, c.id DESC",
            "oldest": "c.created_at ASC, c.id ASC",
            "regions": "c.region_count DESC, c.created_at DESC, c.id DESC",
        }.get(sort, "c.created_at DESC, c.id DESC")
        with self._lock:
            rows = self._con.execute(f"{sql} ORDER BY {order}", args).fetchall()
        return [_comparison(r) for r in rows]

    def delete_comparison(self, comparison_id: int) -> list[str]:
        """Karşılaştırma kaydını siler; diskten silinecek `result_dir` yolunu döndürür."""
        with self._tx():
            rec = self.get_comparison(comparison_id)
            if rec is None:
                return []
            self._con.execute("DELETE FROM comparisons WHERE id = ?", (comparison_id,))
        return [rec.result_dir]


class _Tx:
    """Kilit + açık transaction bağlam yöneticisi (isolation_level=None ile elle yönetilir)."""

    def __init__(self, repo: LibraryRepository) -> None:
        self._repo = repo

    def __enter__(self) -> None:
        self._repo._lock.acquire()
        try:
            self._depth = getattr(self._repo, "_tx_depth", 0)
            if self._depth == 0:
                self._repo._con.execute("BEGIN")
            self._repo._tx_depth = self._depth + 1  # type: ignore[attr-defined]
        except BaseException:
            # `__exit__` çağrılmayacağı için kilit burada bırakılır (aksi halde diğer thread'ler kilitlenir).
            self._repo._lock.release()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._repo._tx_depth = self._depth  # type: ignore[attr-defined]
            if self._depth == 0:
                self._repo._con.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self._repo._lock.release()
