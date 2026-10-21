"""Bölge analizi (`analyze`) ve genel özet (`summarize`): Gemma 4 + llama-server.

LLM değişiklik BULMAZ; CV pipeline'ın bulduğu aday bölgelerin önce/sonra crop'larına bakarak
(a) değişikliği tanımlar, (b) gölge/ışık/sıkıştırma kaynaklı yanlış alarmları işaretler.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import cv2
import httpx
import numpy as np

from core.config import AppConfig
from core.contracts import (
    BoundingBox,
    ChangeRegion,
    ChangeVerdict,
    DetectionResult,
    RegistrationResult,
)
from core.llm.server import LlamaServer, StartInfo

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt'lar
# ---------------------------------------------------------------------------

REGION_PROMPT = (
    "Sen hava/uydu görüntülerini karşılaştıran bir analistsin. Sana aynı yerin iki görüntüsü verilir: "
    "İLK görüntü ÖNCE, İKİNCİ görüntü SONRA zamanına aittir. İkisi aynı bölgeyi gösterecek şekilde hizalanmıştır. "
    "{marker}"
    "Görev: Bu iki görüntü arasında GERÇEK bir değişiklik var mı karar ver "
    "(nesne eklenmiş/kaldırılmış/taşınmış, yapı, araç, bitki örtüsü veya zemin gerçekten değişmiş).\n"
    "Şunları GERÇEK DEĞİŞİKLİK SAYMA (bunlar yanlış alarmdır): gölge ve ışık/parlaklık/renk farkları, mevsim veya "
    "bulut etkisi, JPEG sıkıştırma ve gürültü artefaktları, hafif hizalama kayması (kenarların birkaç piksel "
    "kayması), bulanıklık veya keskinlik farkı, aynı nesnenin biraz farklı çizilmesi/yeniden çizilmiş ince ayrıntılar "
    "(ör. gökyüzü, ince ip/kablo/çizgi, yaprak dokusu). Sadece nesnenin eklenmesi, kaldırılması, taşınması ya da "
    "yapının/zeminin belirgin biçimde değişmesi gerçek değişikliktir; ince çizgi veya doku farkı için false_alarm de; "
    "emin olamazsan uncertain de.\n"
    "Açıklamaya 'Çerçevenin içinde', 'İşaretli alanda' gibi ifadelerle BAŞLAMA ve çerçeveden/görüntü sıralamasından söz etme; yalnızca sahnedeki nesneyi/farkı anlat.\n"
    "Cevabını SADECE tek bir JSON nesnesi olarak ver, başka hiçbir metin yazma:\n"
    '{{"verdict": "real_change" | "false_alarm" | "uncertain", '
    '"description": "<Türkçe, 1-2 kısa cümle: ne değişti ya da neden yanlış alarm>", '
    '"confidence": <0.0-1.0 arası sayı>}}'
)

MARKER_TEXT = "Her iki görüntüde de aday bölge ince sarı bir çerçeveyle gösterilmiştir; asıl odak bu çerçevenin içidir. "

RETRY_SUFFIX = (
    "\n\nÖNEMLİ: Yalnızca geçerli bir JSON nesnesi döndür. Açıklama, markdown veya kod bloğu ekleme. "
    'Alanlar: "verdict" (real_change, false_alarm veya uncertain), "description" (Türkçe), "confidence" (0-1).'
)

SUMMARY_PROMPT = (
    "Aşağıda iki zamanlı görüntü karşılaştırmasında bulunan bölgelerin analiz sonuçları var. "
    "Konumlar görüntünün üstü kuzey kabul edilerek verilmiştir.\n\n"
    "{items}\n\n"
    "Bu bilgilere dayanarak Türkçe, 2 ila 4 cümlelik kısa bir genel özet yaz. Kurallar:\n"
    "- Yalnızca gerçek değişiklikleri (real_change) ne olduğu ve kabaca nerede olduğu ile anlat; "
    "konumu 'kuzeydoğu', 'merkez', 'batı' gibi yönlerle belirt.\n"
    "- Yanlış alarmları (false_alarm) özetin SONUNDA tek kısa cümleyle an (ör. 'Batıdaki farklar gölge kaynaklıdır').\n"
    "- Hiç gerçek değişiklik yoksa bunu açıkça söyle.\n"
    "- Çerçeve, görüntü numarası gibi teknik ayrıntılardan söz etme.\n"
    "- Listede olmayan bir şey uydurma. Madde işareti, başlık veya markdown kullanma; düz paragraf yaz."
)

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["real_change", "false_alarm", "uncertain"]},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "description", "confidence"],
}


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def expanded_box(bbox: BoundingBox, img_w: int, img_h: int, margin: float, min_pad_px: int = 12) -> tuple[int, int, int, int]:
    """bbox'ı her yönde `margin` oranında (en az `min_pad_px`) genişletip görüntü sınırlarına kırpar.

    Returns:
        (x0, y0, x1, y1), x1/y1 dışlayıcı. Her zaman en az 1x1 piksel.
    """
    px = max(int(round(bbox.width * margin)), min_pad_px)
    py = max(int(round(bbox.height * margin)), min_pad_px)
    x0 = max(0, bbox.x - px)
    y0 = max(0, bbox.y - py)
    x1 = min(img_w, bbox.x + bbox.width + px)
    y1 = min(img_h, bbox.y + bbox.height + py)
    x0 = min(x0, img_w - 1)
    y0 = min(y0, img_h - 1)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return x0, y0, x1, y1


def make_crop(image: np.ndarray, bbox: BoundingBox, margin: float, max_side: int,
              mark_bbox: bool = True, min_side: int = 192) -> np.ndarray:
    """Bölge çevresinden crop alır, makul boyuta getirir, istenirse bbox'ı ince sarı çerçeveyle işaretler.

    Crop `max_side`'tan büyükse küçültülür; `min_side`'tan küçükse büyütülür (vision encoder için).
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = expanded_box(bbox, w, h, margin)
    crop = image[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]
    scale = 1.0
    longest = max(ch, cw)
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side:
        scale = min_side / longest
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))), interpolation=interp)
    if mark_bbox:
        rx0 = round((bbox.x - x0) * scale)
        ry0 = round((bbox.y - y0) * scale)
        rx1 = round((bbox.x + bbox.width - x0) * scale) - 1
        ry1 = round((bbox.y + bbox.height - y0) * scale) - 1
        cv2.rectangle(crop, (rx0, ry0), (max(rx0, rx1), max(ry0, ry1)), (255, 235, 0), 1)
    return crop


def encode_png_data_uri(rgb: np.ndarray) -> str:
    """RGB uint8 görüntüyü `data:image/png;base64,...` URI'sine çevirir."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG kodlanamadı")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Yanıt ayrıştırma
# ---------------------------------------------------------------------------

@dataclass
class ParsedVerdict:
    verdict: ChangeVerdict
    description: str
    confidence: float | None


def parse_verdict(text: str) -> ParsedVerdict | None:
    """LLM çıktısından {verdict, description, confidence} JSON'unu çıkarır; olmazsa None."""
    if not text:
        return None
    obj = _extract_json_object(text)
    if not isinstance(obj, dict):
        return None
    raw = str(obj.get("verdict", "")).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "real_change": ChangeVerdict.REAL_CHANGE, "real": ChangeVerdict.REAL_CHANGE,
        "false_alarm": ChangeVerdict.FALSE_ALARM, "false": ChangeVerdict.FALSE_ALARM,
        "uncertain": ChangeVerdict.UNCERTAIN, "unsure": ChangeVerdict.UNCERTAIN,
    }
    verdict = mapping.get(raw)
    if verdict is None:
        return None
    desc = obj.get("description")
    desc = strip_frame_phrases(str(desc).strip()) if desc is not None else ""
    conf: float | None
    try:
        conf = float(obj.get("confidence"))
        conf = min(1.0, max(0.0, conf)) if conf == conf else None
    except (TypeError, ValueError):
        conf = None
    return ParsedVerdict(verdict, desc, conf)


_FRAME_PREFIX = re.compile(
    r"^\s*(?:(?:sarı\s+)?çerçeve(?:nin)?\s+(?:içinde|içerisinde|içindeki|içerisindeki)|"
    r"işaretli\s+(?:alan|bölge)(?:da|de|ın|in|nın|nin)?(?:\s+(?:içinde|içerisinde))?|"
    r"işaretlenen\s+(?:alan|bölge)(?:da|de)?)\s*[,:;\-–]?\s*",
    re.IGNORECASE,
)


def strip_frame_phrases(text: str) -> str:
    """Baştaki 'Çerçevenin içinde', 'İşaretli alanda' gibi ifadeleri kırpar, ilk harfi büyütür."""
    out = text
    for _ in range(2):
        new = _FRAME_PREFIX.sub("", out, count=1)
        if new == out:
            break
        out = new
    if out and out != text:
        out = out[0].upper() + out[1:]
    return out.strip()


def _extract_json_object(text: str) -> object | None:
    """Metindeki ilk geçerli JSON nesnesini bulur (kod bloğu çitleri ve çevre metni tolere edilir)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Konum -> yön
# ---------------------------------------------------------------------------

def direction_label(bbox: BoundingBox, img_w: int, img_h: int) -> str:
    """bbox merkezini 3x3 ızgarada kaba yöne çevirir (üst = kuzey)."""
    cx = (bbox.x + bbox.width / 2) / max(img_w, 1)
    cy = (bbox.y + bbox.height / 2) / max(img_h, 1)
    col = 0 if cx < 1 / 3 else (2 if cx > 2 / 3 else 1)
    row = 0 if cy < 1 / 3 else (2 if cy > 2 / 3 else 1)
    names = [
        ["kuzeybatı", "kuzey", "kuzeydoğu"],
        ["batı", "merkez", "doğu"],
        ["güneybatı", "güney", "güneydoğu"],
    ]
    return names[row][col]


# ---------------------------------------------------------------------------
# HTTP istemcisi (enjekte edilebilir sunucu)
# ---------------------------------------------------------------------------

class ServerLike(Protocol):
    """Analistin ihtiyaç duyduğu sunucu arayüzü (testlerde sahtesi kullanılır)."""

    info: StartInfo

    def files_present(self) -> bool: ...
    def start(self) -> str | None: ...
    def stop(self) -> None: ...
    @property
    def is_running(self) -> bool: ...


class LLMTransportError(Exception):
    """Sunucuya ulaşılamadı / zaman aşımı / HTTP hatası."""


@dataclass
class AnalysisStats:
    """Son `analyze` çağrısının ölçümleri."""

    regions_analyzed: int = 0
    seconds_per_region: list[float] = field(default_factory=list)
    completion_tokens: int = 0
    completion_seconds: float = 0.0

    @property
    def tokens_per_s(self) -> float | None:
        return self.completion_tokens / self.completion_seconds if self.completion_seconds > 0 else None


class GemmaRegionAnalyst:
    """`RegionAnalyst` uygulaması: gömülü llama-server üzerinden Gemma 4.

    Sunucu LAZY başlatılır (ilk `analyze`/`summarize`). `is_available()` sunucuyu başlatmaz.
    `release()` süreci öldürür (VRAM boşalır); sonraki çağrı yeniden başlatır.
    """

    def __init__(self, config: AppConfig, server: ServerLike | None = None, mark_bbox: bool = True) -> None:
        self.config = config
        self.server: ServerLike = server if server is not None else LlamaServer(config)
        self.mark_bbox = mark_bbox
        self.stats = AnalysisStats()
        self.last_error: str | None = None
        # (biten, toplam, mesaj); pipeline UI ilerlemesi için ayarlar. İstisnası analizi bozmaz.
        self.progress_callback: Callable[[int, int, str], None] | None = None
        self._max_consecutive_transport_errors = 3

    # -- RegionAnalyst -----------------------------------------------------------

    def is_available(self) -> bool:
        """LLM açık mı ve exe + gguf + mmproj var mı? Sunucuyu başlatmaz."""
        return bool(self.config.llm.enabled and self.server.files_present())

    def analyze(self, registration: RegistrationResult, detection: DetectionResult) -> DetectionResult:
        """En yüksek skorlu `max_regions` bölgeyi yorumlar; kalanlar UNREVIEWED kalır. Asla istisna fırlatmaz."""
        self.stats = AnalysisStats()
        try:
            return self._analyze(registration, detection)
        except Exception:  # noqa: BLE001
            log.exception("Bölge analizi beklenmedik şekilde başarısız")
            return detection

    def summarize(self, registration: RegistrationResult, detection: DetectionResult) -> str | None:
        """Bölge sonuçlarından (metin olarak) 2-4 cümlelik Türkçe özet. Başarısızsa None."""
        try:
            return self._summarize(registration, detection)
        except Exception:  # noqa: BLE001
            log.exception("Özet üretimi başarısız")
            return None

    def release(self) -> None:
        """llama-server sürecini sonlandırır (VRAM'i boşaltır)."""
        try:
            self.server.stop()
        except Exception:  # noqa: BLE001
            log.exception("llama-server durdurulamadı")

    # -- Bilgi -------------------------------------------------------------------

    @property
    def is_server_running(self) -> bool:
        return bool(self.server.is_running)

    @property
    def last_start_info(self) -> StartInfo:
        """Son başlatmanın bilgisi: seçilen ngl, başlatma süresi, denemeler, hata."""
        return self.server.info

    # -- İç ----------------------------------------------------------------------

    def _analyze(self, registration: RegistrationResult, detection: DetectionResult) -> DetectionResult:
        cfg = self.config.llm
        ranked = sorted(detection.regions, key=lambda r: r.score, reverse=True)[: max(0, cfg.max_regions)]
        if not ranked:
            return detection
        total = len(ranked)
        self._notify(0, total, "Yapay zekâ modeli yükleniyor…")
        base_url = self.server.start()
        if base_url is None:
            self.last_error = self.server.info.error
            log.warning("LLM sunucusu yok; bölgeler yorumlanmadan bırakıldı (%s)", self.last_error)
            return detection

        before = registration.before.pixels
        after = registration.after_warped
        consecutive_errors = 0
        with httpx.Client(base_url=base_url, timeout=cfg.request_timeout_s) as client:
            for idx, region in enumerate(ranked, start=1):
                t0 = time.monotonic()
                try:
                    parsed = self._analyze_region(client, before, after, region)
                    consecutive_errors = 0
                except LLMTransportError as exc:
                    consecutive_errors += 1
                    self.last_error = str(exc)
                    log.warning("Bölge %d için istek başarısız: %s", region.region_id, exc)
                    region.verdict = ChangeVerdict.UNCERTAIN
                    region.description = None
                    region.llm_confidence = None
                    self._notify(idx, total, f"Bölge {idx}/{total} yorumlandı")
                    if consecutive_errors >= self._max_consecutive_transport_errors:
                        log.error("Art arda %d istek hatası; kalan bölgeler atlanıyor", consecutive_errors)
                        break
                    continue
                self.stats.seconds_per_region.append(time.monotonic() - t0)
                self.stats.regions_analyzed += 1
                if parsed is None:
                    region.verdict = ChangeVerdict.UNCERTAIN
                    region.description = None
                    region.llm_confidence = None
                else:
                    region.verdict = parsed.verdict
                    region.description = parsed.description or None
                    region.llm_confidence = parsed.confidence
                self._notify(idx, total, f"Bölge {idx}/{total} yorumlandı")
        return detection

    def _notify(self, done: int, total: int, message: str) -> None:
        """İlerleme geri çağrısını çağırır; hatası analizi bozmaz."""
        cb = self.progress_callback
        if cb is None:
            return
        try:
            cb(done, total, message)
        except Exception:  # noqa: BLE001
            log.exception("progress_callback hatası yok sayıldı")

    def _analyze_region(self, client: httpx.Client, before: np.ndarray, after: np.ndarray,
                        region: ChangeRegion) -> ParsedVerdict | None:
        """Tek bölge: bir istek + (parse olmazsa) bir tekrar. Parse edilemezse None."""
        cfg = self.config.llm
        crops = [
            encode_png_data_uri(make_crop(img, region.bbox, cfg.crop_margin, cfg.crop_max_side_px, self.mark_bbox))
            for img in (before, after)
        ]
        prompt = REGION_PROMPT.format(marker=MARKER_TEXT if self.mark_bbox else "")
        for attempt in range(2):
            text = self._chat(
                client,
                [{"role": "user", "content": [
                    {"type": "text", "text": prompt + (RETRY_SUFFIX if attempt else "")},
                    {"type": "image_url", "image_url": {"url": crops[0]}},
                    {"type": "image_url", "image_url": {"url": crops[1]}},
                ]}],
                schema=_VERDICT_SCHEMA if attempt == 0 else None,
            )
            parsed = parse_verdict(text)
            if parsed is not None:
                return parsed
            log.info("Bölge %d: JSON ayrıştırılamadı (deneme %d): %r", region.region_id, attempt + 1, text[:200])
        return None

    def _chat(self, client: httpx.Client, messages: list[dict], schema: dict | None,
              max_tokens: int | None = None) -> str:
        """`/v1/chat/completions` çağrısı; yanıt metnini döndürür. Ağ/HTTP hatasında LLMTransportError."""
        cfg = self.config.llm
        payload: dict = {
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": max_tokens or cfg.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": schema},
            }
        t0 = time.monotonic()
        try:
            resp = client.post("/v1/chat/completions", json=payload)
            if resp.status_code >= 400 and schema is not None:
                # Bazı sürümler response_format'ı reddedebilir: şemasız bir kez dene
                payload.pop("response_format", None)
                resp = client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMTransportError(f"{type(exc).__name__}: {exc}") from exc
        dt = time.monotonic() - t0
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMTransportError(f"beklenmeyen yanıt biçimi: {exc}") from exc
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        n = int(usage.get("completion_tokens") or 0)
        if n:
            self.stats.completion_tokens += n
            # Üretim süresi varsa onu kullan (prompt işleme hariç); yoksa toplam süre
            gen_ms = timings.get("predicted_ms")
            self.stats.completion_seconds += (gen_ms / 1000.0) if gen_ms else dt
        return content if isinstance(content, str) else ""

    # -- Özet --------------------------------------------------------------------

    def _summarize(self, registration: RegistrationResult, detection: DetectionResult) -> str | None:
        regions = detection.regions
        if not regions:
            return "İki görüntü arasında anlamlı bir değişiklik tespit edilmedi."
        reviewed = [r for r in regions if r.verdict != ChangeVerdict.UNREVIEWED]
        if not reviewed:
            return None
        h, w = registration.before.pixels.shape[:2]
        label = {
            ChangeVerdict.REAL_CHANGE: "real_change",
            ChangeVerdict.FALSE_ALARM: "false_alarm",
            ChangeVerdict.UNCERTAIN: "uncertain",
        }
        lines = []
        for r in sorted(reviewed, key=lambda r: r.score, reverse=True):
            lines.append(
                f"- konum: {direction_label(r.bbox, w, h)}; karar: {label[r.verdict]}; "
                f"açıklama: {r.description or '(yok)'}"
            )
        prompt = SUMMARY_PROMPT.format(items="\n".join(lines))

        base_url = self.server.start()
        if base_url is None:
            self.last_error = self.server.info.error
            return None
        with httpx.Client(base_url=base_url, timeout=self.config.llm.request_timeout_s) as client:
            try:
                text = self._chat(client, [{"role": "user", "content": prompt}], schema=None,
                                  max_tokens=max(self.config.llm.max_tokens, 300))
            except LLMTransportError as exc:
                self.last_error = str(exc)
                log.warning("Özet isteği başarısız: %s", exc)
                return None
        return clean_summary(text)


def clean_summary(text: str, max_sentences: int = 4) -> str | None:
    """Markdown işaretlerini temizler, en fazla `max_sentences` cümleye kırpar. Boşsa None."""
    t = re.sub(r"[*_`#>]+", "", text or "")
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(sentences[:max_sentences]).strip()
