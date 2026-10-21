"""Uygulamanın tüm ayarları tek yerde.

Model yolları, eşikler, tile boyutu, cihaz seçimi ve LLM ayarları burada tanımlanır. Modüller bu
dosyayı SADECE OKUR; değişiklik ana oturuma aittir.

Kullanım:
    from core.config import load_config
    cfg = load_config()                 # varsayılanlar + kullanıcı config.json üzerine yazılır
    cfg.llm.model_spec.gguf_path(cfg.paths.models_dir)

Kullanıcı ayarları `%LOCALAPPDATA%/ChangeScope/config.json` içinde, sadece varsayılandan farklı
alanlar yazılarak saklanır (bkz. `save_config`). Bilinmeyen anahtarlar yok sayılır.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

APP_NAME = "ChangeScope"


# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------

def app_root() -> Path:
    """Uygulama kök klasörünü döndürür.

    PyInstaller onedir paketinde exe'nin bulunduğu klasör, geliştirmede proje kökü.
    `models/` ve `bin/` bu klasörün altındadır.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Kullanıcıya ait veri klasörü: `%LOCALAPPDATA%/ChangeScope` (yoksa `~/.changescope`)."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def _env_path(var: str, default: Path) -> Path:
    """Ortam değişkeni tanımlıysa onun yolunu, değilse `default`'u döndürür.

    Geliştirmede git worktree'leri `models/` ve `bin/` içermez; ajanlar ana klasörü
    `CHANGESCOPE_MODELS_DIR` / `CHANGESCOPE_BIN_DIR` ile gösterir.
    """
    value = os.environ.get(var)
    return Path(value) if value else default


def default_config_path() -> Path:
    """Kullanıcı ayar dosyasının varsayılan konumu."""
    return user_data_dir() / "config.json"


@dataclass
class PathsConfig:
    """Dosya sistemi konumları. Varsayılanlar `app_root()` ve `user_data_dir()` altındadır."""

    models_dir: Path = field(default_factory=lambda: _env_path("CHANGESCOPE_MODELS_DIR", app_root() / "models"))
    bin_dir: Path = field(default_factory=lambda: _env_path("CHANGESCOPE_BIN_DIR", app_root() / "bin"))
    library_dir: Path = field(default_factory=lambda: user_data_dir() / "library")

    @property
    def llama_server_exe(self) -> Path:
        """Gömülü llama-server.exe yolu."""
        return self.bin_dir / "llama" / "llama-server.exe"

    @property
    def superpoint_weights(self) -> Path:
        return self.models_dir / "superpoint" / "superpoint_v1.pth"

    @property
    def lightglue_weights(self) -> Path:
        return self.models_dir / "lightglue" / "superpoint_lightglue.pth"

    @property
    def dinov2_weights(self) -> Path:
        return self.models_dir / "dinov2" / "dinov2_vits14_pretrain.pth"

    @property
    def dinov2_source_dir(self) -> Path:
        """facebookresearch/dinov2 kaynak kodu (model mimarisi için; `import dinov2` buradan)."""
        return self.models_dir / "dinov2" / "dinov2_src"


# ---------------------------------------------------------------------------
# Cihaz
# ---------------------------------------------------------------------------

DeviceName = Literal["auto", "cuda", "cpu"]


@dataclass
class RuntimeConfig:
    """Donanım seçimi. RTX 2060 (6 GB) hedeflenir; CUDA yoksa her şey CPU'da çalışır."""

    device: DeviceName = "auto"
    # CV adımlarından sonra, LLM'den önce CV modellerini VRAM'den boşalt (6 GB'ta zorunlu).
    release_cv_models_before_llm: bool = True


def resolve_device(preference: DeviceName = "auto") -> str:
    """`"auto"`/`"cuda"`/`"cpu"` tercihinden gerçek torch cihaz adını üretir.

    CUDA istenip bulunamazsa uyarı loglanır ve `"cpu"` döner. torch kurulu değilse `"cpu"`.
    """
    if preference == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        log.warning("CUDA istendi ama bulunamadı; CPU kullanılacak.")
    return "cpu"


# ---------------------------------------------------------------------------
# Ön işleme + hizalama
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    max_side_px: int = 4096                 # Daha büyük görüntüler işlem için küçültülür
    denoise_if_compressed: bool = True
    denoise_h: float = 3.0                  # fastNlMeansDenoisingColored h (düşük = hafif)
    jpeg_quality_suspect: int = 90          # Tahmini kalite bunun altındaysa "sıkıştırılmış"


@dataclass
class RegistrationConfig:
    matcher: Literal["lightglue", "sift"] = "lightglue"   # lightglue yüklenemezse sift'e düşer
    max_keypoints: int = 2048
    match_long_side_px: int = 1024          # Kaba eşleştirme çözünürlüğü
    tile_size_px: int = 1024
    tile_overlap: float = 0.10
    sift_ratio: float = 0.75
    ransac_reproj_threshold_px: float = 3.0
    # Kalite eşikleri
    good_min_inliers: int = 60
    good_min_inlier_ratio: float = 0.35
    good_max_reproj_error_px: float = 2.0
    weak_min_inliers: int = 15
    weak_max_reproj_error_px: float = 5.0
    valid_mask_erode_px: int = 4


# ---------------------------------------------------------------------------
# Değişiklik tespiti
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    method: Literal["ssim", "dinov2", "ssim+dinov2"] = "ssim"
    light_normalization: Literal["none", "histogram", "clahe", "histogram+clahe"] = "histogram+clahe"
    ssim_window: int = 7
    blur_sigma: float = 1.5                 # Sıkıştırma gürültüsüne karşı ön bulanıklaştırma
    threshold: float | Literal["otsu"] = "otsu"
    min_threshold: float = 0.25             # Otsu çok düşük çıkarsa alt sınır
    sensitivity: float = 0.5                # 0..1, UI kaydırıcısı; eşik ölçeklemesi detector'da
    morph_kernel_px: int = 5
    min_region_area_px: int = 400
    edge_erode_px: int = 6
    ssim_weight: float = 0.5                # "ssim+dinov2" birleştirme ağırlıkları
    dinov2_weight: float = 0.5
    dinov2_tile_px: int = 896               # 14'ün katı (ViT-S/14 patch)
    dinov2_fp16: bool = True                # Sadece CUDA'da etkili


# ---------------------------------------------------------------------------
# LLM (Gemma 4 + llama-server)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMModelSpec:
    """İndirilebilir bir GGUF modelinin tanımı. `scripts/download_models.py` de bunu kullanır."""

    key: str
    display_name: str
    hf_repo: str
    gguf_file: str
    mmproj_file: str
    n_layers: int                           # -ngl "auto" hesabı için
    approx_vram_mb: int                     # Tam offload için kaba tahmin (gguf + mmproj)

    def model_dir(self, models_dir: Path) -> Path:
        return models_dir / "gemma" / self.key

    def gguf_path(self, models_dir: Path) -> Path:
        return self.model_dir(models_dir) / self.gguf_file

    def mmproj_path(self, models_dir: Path) -> Path:
        return self.model_dir(models_dir) / self.mmproj_file


LLM_MODELS: dict[str, LLMModelSpec] = {
    "gemma-4-12b": LLMModelSpec(
        key="gemma-4-12b",
        display_name="Gemma 4 12B",
        hf_repo="google/gemma-4-12B-it-qat-q4_0-gguf",
        gguf_file="gemma-4-12b-it-qat-q4_0.gguf",
        mmproj_file="mmproj-gemma-4-12b-it-qat-q4_0.gguf",
        n_layers=48,
        approx_vram_mb=7150,
    ),
    "gemma-4-e4b": LLMModelSpec(
        key="gemma-4-e4b",
        display_name="Gemma 4 E4B",
        hf_repo="google/gemma-4-E4B-it-qat-q4_0-gguf",
        gguf_file="gemma-4-E4B_q4_0-it.gguf",
        mmproj_file="gemma-4-E4B-it-mmproj.gguf",
        n_layers=42,
        approx_vram_mb=6150,
    ),
}

DEFAULT_LLM_MODEL = "gemma-4-12b"


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = DEFAULT_LLM_MODEL          # LLM_MODELS anahtarı
    # "auto": başlatma anındaki boş VRAM'e göre hesaplanır; tam sayı: elle (-ngl), 0 = tamamen CPU.
    n_gpu_layers: int | Literal["auto"] = "auto"
    vram_safety_margin_mb: int = 600        # "auto" hesabında boş VRAM'den düşülen pay
    mmproj_offload: bool = True             # False → --no-mmproj-offload (görü kodlayıcı CPU'da)
    ctx_size: int = 8192
    threads: int | None = None              # None → llama-server varsayılanı
    temperature: float = 0.1
    max_tokens: int = 400
    max_regions: int = 20                   # Sadece en yüksek skorlu N bölge yorumlanır
    crop_margin: float = 0.30
    crop_max_side_px: int = 512
    startup_timeout_s: float = 180.0        # Kısmi offload ile yükleme uzun sürebilir
    request_timeout_s: float = 120.0
    extra_server_args: list[str] = field(default_factory=list)

    @property
    def model_spec(self) -> LLMModelSpec:
        """Seçili modelin tanımı; bilinmeyen anahtar → varsayılan model."""
        spec = LLM_MODELS.get(self.model)
        if spec is None:
            log.warning("Bilinmeyen LLM modeli %r; %s kullanılacak.", self.model, DEFAULT_LLM_MODEL)
            return LLM_MODELS[DEFAULT_LLM_MODEL]
        return spec


# ---------------------------------------------------------------------------
# Kök
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON'a yazılabilir sözlük (Path → str)."""
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _apply_overrides(target: Any, overrides: dict[str, Any], prefix: str = "") -> None:
    """`overrides` sözlüğünü dataclass `target` üzerine yerinde uygular."""
    known = {f.name: f for f in fields(target)}
    for key, value in overrides.items():
        if key not in known:
            log.warning("Bilinmeyen ayar yok sayıldı: %s%s", prefix, key)
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value, f"{prefix}{key}.")
        elif isinstance(current, Path):
            setattr(target, key, Path(value))
        else:
            setattr(target, key, value)


def _diff(current: Any, default: Any) -> Any:
    """Varsayılandan farklı alanları döndürür (iç içe sözlüklerde yinelemeli)."""
    if isinstance(current, dict) and isinstance(default, dict):
        out = {}
        for key, value in current.items():
            d = _diff(value, default.get(key))
            if d is not _SAME:
                out[key] = d
        return out if out else _SAME
    return _SAME if current == default else current


_SAME = object()


def load_config(path: Path | None = None) -> AppConfig:
    """Varsayılan ayarları oluşturur ve varsa kullanıcı JSON'unu üzerine uygular.

    Dosya yoksa veya bozuksa varsayılanlarla devam edilir (bozuk bir ayar dosyası uygulamanın açılmasını engellemez).
    """
    cfg = AppConfig()
    path = path or default_config_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _apply_overrides(cfg, data)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Ayar dosyası okunamadı (%s): %s; varsayılanlar kullanılıyor.", path, exc)
    return cfg


def save_config(cfg: AppConfig, path: Path | None = None) -> Path:
    """Sadece varsayılandan farklı ayarları JSON olarak yazar ve dosya yolunu döndürür."""
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    diff = _diff(cfg.to_dict(), AppConfig().to_dict())
    payload = {} if diff is _SAME else diff
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path
