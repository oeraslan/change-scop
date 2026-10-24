"""Geliştirme sırasında model ağırlıklarını indirir.

UYARI: Bu script SADECE geliştirme/kurulum içindir. Uygulama çalışma zamanında asla ağa çıkmaz.

Hedefler:
    superpoint   models/superpoint/superpoint_v1.pth                (cvg/LightGlue release)
    lightglue    models/lightglue/superpoint_lightglue.pth          (cvg/LightGlue release)
    dinov2       models/dinov2/dinov2_vits14_pretrain.pth + dinov2_src/ (Meta, sabit commit)

Örnekler:
    .venv\\Scripts\\python.exe scripts/download_models.py                 # hepsi
    .venv\\Scripts\\python.exe scripts/download_models.py --only superpoint,lightglue --force
    .venv\\Scripts\\python.exe scripts/download_models.py --list

İndirmeler `<dosya>.part` olarak yazılır; yarıda kesilirse (Ctrl+C, ağ hatası) script tekrar
çalıştırıldığında HTTP Range ile kaldığı yerden devam eder. Dosyalar boyutla doğrulanır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm requirements ile gelir
    tqdm = None

MODELS_DIR = PROJECT_ROOT / "models"

USER_AGENT = "ChangeScope-model-downloader/1.0"
CHUNK = 1024 * 1024
MAX_RETRIES = 6

LIGHTGLUE_RELEASE = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv"
DINOV2_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
# facebookresearch/dinov2 kaynak kodu sabit bir commit'ten (tekrarlanabilirlik)
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_SRC_URL = f"https://github.com/facebookresearch/dinov2/archive/{DINOV2_COMMIT}.zip"

ALL_TARGETS = ["superpoint", "lightglue", "dinov2"]
DEFAULT_TARGETS = list(ALL_TARGETS)


class DownloadError(RuntimeError):
    """İndirme kalıcı olarak başarısız oldu."""


# ---------------------------------------------------------------------------
# HTTP yardımcıları
# ---------------------------------------------------------------------------

def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    h = {"User-Agent": USER_AGENT}
    if "api.github.com" in url and os.environ.get("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    if "huggingface.co" in url and os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    h.update(headers or {})
    return urllib.request.Request(url, headers=h)


def fetch_json(url: str) -> object:
    """Bir JSON API yanıtını indirir."""
    with urllib.request.urlopen(_request(url), timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def sha256_file(path: Path) -> str:
    """Dosyanın SHA-256 özetini hesaplar."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """`url`'yi `dest`'e indirir; yarım kalan `.part` dosyasından devam eder.

    Args:
        url: Kaynak adres (yönlendirmeler izlenir).
        dest: Hedef dosya.
        expected_size: Biliniyorsa bayt cinsinden boyut (doğrulama + atlama kararı).
        expected_sha256: Biliniyorsa SHA-256 (indirme sonrası doğrulanır).
        force: Mevcut dosyayı yok sayıp baştan indir.

    Returns:
        `dest`.

    Raises:
        DownloadError: Tekrar denemelere rağmen indirilemezse veya doğrulama başarısızsa.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if force:
        dest.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    elif dest.is_file() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  ✓ zaten var: {dest.relative_to(PROJECT_ROOT)}")
        return dest

    for attempt in range(1, MAX_RETRIES + 1):
        offset = part.stat().st_size if part.exists() else 0
        if expected_size is not None and offset > expected_size:
            part.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urllib.request.urlopen(_request(url, headers), timeout=60) as resp:
                if offset and resp.status != 206:
                    # Sunucu Range desteklemiyor → baştan
                    offset = 0
                total = _total_size(resp, offset) or expected_size
                mode = "ab" if offset else "wb"
                bar = _progress(dest.name, total, offset)
                try:
                    with part.open(mode) as f:
                        while chunk := resp.read(CHUNK):
                            f.write(chunk)
                            if bar is not None:
                                bar.update(len(chunk))
                finally:
                    if bar is not None:
                        bar.close()
            if total is not None and part.stat().st_size < total:
                # Sunucu bağlantıyı gövde ortasında kapattı; read() hata vermeden boş dönebilir.
                raise ConnectionError(f"eksik veri ({part.stat().st_size}/{total} bayt)")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and expected_size is not None and offset == expected_size:
                break  # .part zaten tam
            if exc.code in (401, 403, 404):
                raise DownloadError(f"{url} → HTTP {exc.code} {exc.reason}") from exc
            _retry_wait(attempt, exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            _retry_wait(attempt, exc)
    else:
        raise DownloadError(f"{url} indirilemedi ({MAX_RETRIES} deneme). Tekrar çalıştırınca devam eder.")

    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise DownloadError(
            f"{dest.name}: boyut uyuşmuyor ({size} != {expected_size}). Tekrar çalıştırın."
        )
    if expected_sha256:
        print(f"  … SHA-256 doğrulanıyor: {dest.name}")
        digest = sha256_file(part)
        if digest.lower() != expected_sha256.lower():
            part.unlink()
            raise DownloadError(f"{dest.name}: SHA-256 uyuşmuyor; bozuk dosya silindi.")
    part.replace(dest)
    print(f"  ✓ indirildi: {dest.relative_to(PROJECT_ROOT)} ({_human(size)})")
    return dest


def _total_size(resp, offset: int) -> int | None:
    content_range = resp.headers.get("Content-Range")  # "bytes 100-999/1000"
    if content_range and "/" in content_range:
        tail = content_range.rsplit("/", 1)[1]
        if tail.isdigit():
            return int(tail)
    length = resp.headers.get("Content-Length")
    if length and length.isdigit():
        return int(length) + (offset if resp.status == 206 else 0)
    return None


def _progress(name: str, total: int | None, initial: int):
    if tqdm is None:
        print(f"  ↓ {name} ({_human(total) if total else '?'})")
        return None
    return tqdm(
        total=total, initial=initial, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"  ↓ {name[:40]}", dynamic_ncols=True, mininterval=0.5,
    )


def _retry_wait(attempt: int, exc: BaseException) -> None:
    if attempt >= MAX_RETRIES:
        return
    wait = min(2 ** attempt, 30)
    print(f"  ! ağ hatası ({exc}); {wait} sn sonra kaldığı yerden tekrar deneniyor "
          f"({attempt}/{MAX_RETRIES - 1})")
    time.sleep(wait)


# ---------------------------------------------------------------------------
# Arşiv açma
# ---------------------------------------------------------------------------

def safe_extract(
    archive: Path,
    target: Path,
    strip_prefix: str | None = None,
    include_prefix: str | None = None,
) -> int:
    """ZIP'i güvenli şekilde açar (zip-slip korumalı) ve açılan dosya sayısını döndürür.

    Args:
        archive: ZIP dosyası.
        target: Hedef klasör.
        strip_prefix: Arşiv içindeki yolların başından atılacak klasör (ör. "repo-abc/").
        include_prefix: Verilirse sadece bu önekle başlayan (strip sonrası) girdiler açılır.
    """
    target = target.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if strip_prefix:
                if not name.startswith(strip_prefix):
                    continue
                name = name[len(strip_prefix):]
            if not name or (include_prefix and not name.startswith(include_prefix)):
                continue
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or ":" in name:
                raise DownloadError(f"Güvensiz arşiv girdisi: {info.filename}")
            out = (target / Path(*rel.parts)).resolve()
            if target not in out.parents and out != target:
                raise DownloadError(f"Güvensiz arşiv girdisi: {info.filename}")
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            count += 1
    return count


def common_root(archive: Path) -> str | None:
    """Arşivdeki tüm girdiler tek bir üst klasördeyse onun adını ("ad/") döndürür."""
    with zipfile.ZipFile(archive) as zf:
        tops = {n.split("/", 1)[0] for n in zf.namelist() if n}
        dirs = all("/" in n for n in zf.namelist() if n)
    return f"{next(iter(tops))}/" if len(tops) == 1 and dirs else None


# ---------------------------------------------------------------------------
# Hedefler
# ---------------------------------------------------------------------------

def get_superpoint(force: bool) -> None:
    download_file(f"{LIGHTGLUE_RELEASE}/superpoint_v1.pth",
                  MODELS_DIR / "superpoint" / "superpoint_v1.pth", force=force)


def get_lightglue(force: bool) -> None:
    download_file(f"{LIGHTGLUE_RELEASE}/superpoint_lightglue.pth",
                  MODELS_DIR / "lightglue" / "superpoint_lightglue.pth", force=force)


def get_dinov2(force: bool) -> None:
    root = MODELS_DIR / "dinov2"
    download_file(DINOV2_WEIGHTS_URL, root / "dinov2_vits14_pretrain.pth", force=force)

    src_dir = root / "dinov2_src"
    marker = src_dir / "COMMIT.txt"
    if not force and marker.is_file() and marker.read_text().strip() == DINOV2_COMMIT:
        print("  ✓ zaten var: models/dinov2/dinov2_src")
        return
    zip_path = download_file(DINOV2_SRC_URL, root / f"_dinov2-{DINOV2_COMMIT[:8]}.zip", force=True)
    if src_dir.exists():
        shutil.rmtree(src_dir)
    prefix = f"dinov2-{DINOV2_COMMIT}/"
    n = safe_extract(zip_path, src_dir, strip_prefix=prefix, include_prefix="dinov2/")
    safe_extract(zip_path, src_dir, strip_prefix=prefix, include_prefix="LICENSE")
    marker.write_text(DINOV2_COMMIT + "\n")
    zip_path.unlink()
    print(f"  ✓ dinov2 kaynak kodu açıldı ({n} dosya) → models/dinov2/dinov2_src/")


def _check_disk(path: Path, needed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if needed and free < needed * 1.05:
        raise DownloadError(f"Yetersiz disk alanı: gerekli ~{_human(needed)}, boş {_human(free)}.")


# ---------------------------------------------------------------------------
# Durum listesi
# ---------------------------------------------------------------------------

def target_status(target: str) -> tuple[bool, str]:
    """Bir hedefin kurulu olup olmadığını ve kısa açıklamasını döndürür."""
    if target == "superpoint":
        p = MODELS_DIR / "superpoint" / "superpoint_v1.pth"
        return p.is_file(), str(p.relative_to(PROJECT_ROOT))
    if target == "lightglue":
        p = MODELS_DIR / "lightglue" / "superpoint_lightglue.pth"
        return p.is_file(), str(p.relative_to(PROJECT_ROOT))
    if target == "dinov2":
        ok = (MODELS_DIR / "dinov2" / "dinov2_vits14_pretrain.pth").is_file() and \
             (MODELS_DIR / "dinov2" / "dinov2_src" / "dinov2").is_dir()
        return ok, "models/dinov2/"
    raise ValueError(target)


def parse_targets(only: list[str] | None, all_: bool) -> list[str]:
    """`--only` değerlerini (virgülle ayrılmış veya tekrarlı) doğrulayıp sıralı hedef listesi üretir."""
    if all_:
        return list(ALL_TARGETS)
    if not only:
        return list(DEFAULT_TARGETS)
    chosen: set[str] = set()
    for item in only:
        for t in item.split(","):
            t = t.strip().lower()
            if not t:
                continue
            if t not in ALL_TARGETS:
                raise SystemExit(f"Bilinmeyen hedef: {t!r}. Geçerli: {', '.join(ALL_TARGETS)}")
            chosen.add(t)
    return [t for t in ALL_TARGETS if t in chosen]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", metavar="HEDEF",
                        help=f"Sadece bu hedef(ler): {', '.join(ALL_TARGETS)} (virgülle veya tekrarla)")
    parser.add_argument("--all", action="store_true", help="Tüm hedefler (varsayılanla aynı)")
    parser.add_argument("--force", action="store_true", help="Var olan dosyaları yeniden indir")
    parser.add_argument("--list", action="store_true", help="Sadece kurulum durumunu göster")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Çıktı boruya yönlenince cp125x'te ✓/↓ çökmesin
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if args.list:
        for t in ALL_TARGETS:
            ok, desc = target_status(t)
            print(f"  {'✓' if ok else '✗'} {t:<11} {desc}")
        return 0

    targets = parse_targets(args.only, args.all)
    print(f"İndirilecek: {', '.join(targets)}")
    failures: list[str] = []
    for t in targets:
        print(f"\n[{t}]")
        try:
            if t == "superpoint":
                get_superpoint(args.force)
            elif t == "lightglue":
                get_lightglue(args.force)
            elif t == "dinov2":
                get_dinov2(args.force)
        except KeyboardInterrupt:
            print("\nİptal edildi. Tekrar çalıştırınca yarım kalan dosyalar kaldığı yerden devam eder.")
            return 130
        except (DownloadError, urllib.error.URLError, OSError) as exc:
            print(f"  ✗ {t}: {exc}")
            failures.append(t)

    print()
    if failures:
        print(f"Başarısız: {', '.join(failures)}")
        return 1
    print("Tamamlandı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
