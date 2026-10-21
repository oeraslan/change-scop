"""Analist testleri: sahte HTTP sunucu ile JSON parse, retry, fallback, max_regions, özet."""

from __future__ import annotations

import json

import numpy as np

from core.config import AppConfig
from core.contracts import BoundingBox, ChangeVerdict
from core.llm.analyst import (
    GemmaRegionAnalyst,
    clean_summary,
    direction_label,
    expanded_box,
    make_crop,
    parse_verdict,
    strip_frame_phrases,
)
from tests.llm.conftest import FakeServer, make_analyst, make_scene

GOOD = json.dumps({"verdict": "real_change", "description": "Yeni bir yapı eklenmiş.", "confidence": 0.9})
FALSE = json.dumps({"verdict": "false_alarm", "description": "Gölge farkı.", "confidence": 0.8})


def test_parse_verdict_variants():
    assert parse_verdict(GOOD).verdict == ChangeVerdict.REAL_CHANGE
    fenced = "```json\n" + FALSE + "\n```"
    assert parse_verdict(fenced).verdict == ChangeVerdict.FALSE_ALARM
    assert parse_verdict("Tabii! " + GOOD + " umarım yardımcı olur").description.startswith("Yeni")
    p = parse_verdict('{"verdict":"uncertain","description":"x","confidence":7}')
    assert p.confidence == 1.0
    assert parse_verdict('{"verdict":"bilmem","description":"x"}') is None
    assert parse_verdict("hiç json yok") is None
    assert parse_verdict("") is None


def test_analyze_fills_regions_and_sends_two_images(mock_llm, scene):
    srv = mock_llm(default=GOOD)
    analyst, fake = make_analyst(srv.url)
    reg, det = scene
    out = analyst.analyze(reg, det)
    assert out is det
    assert all(r.verdict == ChangeVerdict.REAL_CHANGE for r in det.regions)
    assert det.regions[0].description == "Yeni bir yapı eklenmiş."
    assert det.regions[0].llm_confidence == 0.9
    assert fake.start_calls == 1
    req = srv.requests[0]
    parts = req["messages"][0]["content"]
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert len(urls) == 2 and all(u.startswith("data:image/png;base64,") for u in urls)
    assert req["temperature"] == 0.1
    assert "ÖNCE" in parts[0]["text"] and "gölge" in parts[0]["text"]


def test_retry_after_bad_json_then_ok(mock_llm, scene):
    srv = mock_llm(responses=["bozuk çıktı", GOOD], default=GOOD)
    analyst, _ = make_analyst(srv.url, max_regions=1)
    reg, det = scene
    analyst.analyze(reg, det)
    top = max(det.regions, key=lambda r: r.score)
    assert top.verdict == ChangeVerdict.REAL_CHANGE
    assert len(srv.requests) == 2
    assert "response_format" in srv.requests[0] and "response_format" not in srv.requests[1]


def test_bad_json_twice_gives_uncertain(mock_llm, scene):
    srv = mock_llm(default="asla json değil")
    analyst, _ = make_analyst(srv.url, max_regions=1)
    reg, det = scene
    analyst.analyze(reg, det)
    top = max(det.regions, key=lambda r: r.score)
    assert top.verdict == ChangeVerdict.UNCERTAIN
    assert len(srv.requests) == 2


def test_max_regions_picks_highest_scores(mock_llm, scene):
    srv = mock_llm(default=FALSE)
    analyst, _ = make_analyst(srv.url, max_regions=2)
    reg, det = scene
    analyst.analyze(reg, det)
    verdicts = {r.score: r.verdict for r in det.regions}
    assert verdicts[0.9] == ChangeVerdict.FALSE_ALARM
    assert verdicts[0.7] == ChangeVerdict.FALSE_ALARM
    assert verdicts[0.5] == ChangeVerdict.UNREVIEWED
    assert len(srv.requests) == 2


def test_http_error_response_format_fallback(mock_llm, scene):
    srv = mock_llm(responses=[("status", 400), GOOD], default=GOOD)
    analyst, _ = make_analyst(srv.url, max_regions=1)
    reg, det = scene
    analyst.analyze(reg, det)
    assert max(det.regions, key=lambda r: r.score).verdict == ChangeVerdict.REAL_CHANGE


def test_timeout_does_not_raise(mock_llm, scene):
    srv = mock_llm(default=("sleep", 1.5, GOOD))
    analyst, _ = make_analyst(srv.url, request_timeout_s=0.3)
    reg, det = scene
    out = analyst.analyze(reg, det)       # istisna fırlatmamalı
    assert out is det
    assert all(r.verdict != ChangeVerdict.REAL_CHANGE for r in det.regions)
    assert analyst.last_error


def test_server_unavailable_leaves_unreviewed(scene):
    analyst, fake = make_analyst(None)
    reg, det = scene
    analyst.analyze(reg, det)
    assert all(r.verdict == ChangeVerdict.UNREVIEWED for r in det.regions)
    assert analyst.summarize(reg, det) is None


def test_no_regions_does_not_start_server():
    analyst, fake = make_analyst("http://127.0.0.1:1")
    reg, det = make_scene()
    det.regions = []
    analyst.analyze(reg, det)
    assert fake.start_calls == 0
    assert "değişiklik" in analyst.summarize(reg, det)
    assert fake.start_calls == 0


def test_is_available_does_not_start(tmp_path):
    cfg = AppConfig()
    cfg.paths.bin_dir = tmp_path / "bin"
    cfg.paths.models_dir = tmp_path / "models"
    from core.llm import create_analyst
    analyst = create_analyst(cfg)
    assert analyst.is_available() is False
    assert analyst.is_server_running is False
    assert analyst.analyze(*make_scene()) is not None   # istisna yok
    assert analyst.last_start_info.error
    analyst.release()


def test_is_available_respects_enabled_flag():
    analyst = GemmaRegionAnalyst(AppConfig(), server=FakeServer("http://x"))
    assert analyst.is_available() is True
    analyst.config.llm.enabled = False
    assert analyst.is_available() is False


# -- Crop -----------------------------------------------------------------------

def test_expanded_box_margin_and_clamp():
    # %30 pay: 100x100 kutu -> 30 px her yönde
    assert expanded_box(BoundingBox(200, 200, 100, 100), 1000, 1000, 0.3) == (170, 170, 330, 330)
    # sol-üst kenarda kırpma
    assert expanded_box(BoundingBox(0, 0, 100, 100), 1000, 1000, 0.3) == (0, 0, 130, 130)
    # sağ-alt kenar
    assert expanded_box(BoundingBox(950, 950, 50, 50), 1000, 1000, 0.3) == (935, 935, 1000, 1000)
    # küçük kutuda asgari pay
    x0, y0, x1, y1 = expanded_box(BoundingBox(100, 100, 4, 4), 1000, 1000, 0.3)
    assert (x1 - x0, y1 - y0) == (28, 28)


def test_make_crop_sizes_and_marker():
    img = np.zeros((2000, 3000, 3), np.uint8)
    big = make_crop(img, BoundingBox(500, 500, 1000, 800), 0.3, 512)
    assert max(big.shape[:2]) == 512
    tiny = make_crop(img, BoundingBox(10, 10, 5, 5), 0.3, 512)
    assert max(tiny.shape[:2]) == 192
    # işaretçi: işaretsizde sıfır görüntüde hiç renk yok, işaretlide var
    assert (make_crop(img, BoundingBox(500, 500, 100, 100), 0.3, 512, mark_bbox=False) == 0).all()
    assert (make_crop(img, BoundingBox(500, 500, 100, 100), 0.3, 512, mark_bbox=True) != 0).any()
    # köşe bbox'ı hata vermez
    assert make_crop(img, BoundingBox(2990, 1990, 10, 10), 0.3, 512).size > 0


# -- Özet -----------------------------------------------------------------------

def test_direction_labels():
    assert direction_label(BoundingBox(300, 10, 40, 40), 400, 300) == "kuzeydoğu"
    assert direction_label(BoundingBox(180, 140, 30, 30), 400, 300) == "merkez"
    assert direction_label(BoundingBox(10, 250, 50, 30), 400, 300) == "güneybatı"
    assert direction_label(BoundingBox(0, 130, 20, 40), 400, 300) == "batı"


def test_summarize_prompt_has_text_only_and_directions(mock_llm, scene):
    srv = mock_llm(default=GOOD)
    analyst, _ = make_analyst(srv.url)
    reg, det = scene
    analyst.analyze(reg, det)
    det.regions[1].verdict = ChangeVerdict.FALSE_ALARM
    det.regions[1].description = "Gölge."
    srv.responses = ["**Kuzeydoğuda** yeni yapı var. Merkezdeki fark gölge kaynaklı. Üçüncü. Dördüncü. Beşinci."]
    out = analyst.summarize(reg, det)
    req = srv.requests[-1]
    content = req["messages"][0]["content"]
    assert isinstance(content, str)                    # görüntü yok
    assert "kuzeydoğu" in content and "merkez" in content and "false_alarm" in content
    assert out.startswith("Kuzeydoğuda")
    assert out.count(".") <= 4 and "*" not in out


def test_summarize_failure_returns_none(mock_llm, scene):
    srv = mock_llm(default=("status", 500))
    analyst, _ = make_analyst(srv.url)
    reg, det = scene
    for r in det.regions:
        r.verdict = ChangeVerdict.REAL_CHANGE
    assert analyst.summarize(reg, det) is None


def test_summarize_unreviewed_returns_none(scene):
    analyst, _ = make_analyst("http://127.0.0.1:1")
    assert analyst.summarize(*scene) is None


def test_clean_summary():
    assert clean_summary("") is None
    assert clean_summary("# Başlık\n- Bir. İki. Üç. Dört. Beş.") == "Başlık Bir. İki. Üç. Dört."


def test_release_calls_server_stop():
    analyst, fake = make_analyst("http://x")
    analyst.release()
    assert fake.stop_calls == 1


def test_strip_frame_phrases():
    assert strip_frame_phrases("Çerçevenin içinde kırmızı bir nesne eklenmiş.") == "Kırmızı bir nesne eklenmiş."
    assert strip_frame_phrases("Çerçeve içinde yeni bina var.") == "Yeni bina var."
    assert strip_frame_phrases("Sarı çerçevenin içerisinde, araç görünüyor.") == "Araç görünüyor."
    assert strip_frame_phrases("İşaretli alanda yeni bir yol var.") == "Yeni bir yol var."
    assert strip_frame_phrases("Yeni bir bina eklenmiş.") == "Yeni bir bina eklenmiş."
    p = parse_verdict('{"verdict":"real_change","description":"Çerçevenin içinde top var.","confidence":1}')
    assert p.description == "Top var."


def test_progress_callback(mock_llm, scene):
    srv = mock_llm(default=GOOD)
    analyst, _ = make_analyst(srv.url, max_regions=2)
    calls = []
    analyst.progress_callback = lambda d, t, m: calls.append((d, t, m))
    reg, det = scene
    analyst.analyze(reg, det)
    assert calls[0] == (0, 2, "Yapay zekâ modeli yükleniyor…")
    assert calls[1:] == [(1, 2, "Bölge 1/2 yorumlandı"), (2, 2, "Bölge 2/2 yorumlandı")]


def test_progress_callback_exception_is_ignored(mock_llm, scene):
    srv = mock_llm(default=GOOD)
    analyst, _ = make_analyst(srv.url)

    def boom(*a):
        raise RuntimeError("ui")

    analyst.progress_callback = boom
    reg, det = scene
    analyst.analyze(reg, det)
    assert all(r.verdict == ChangeVerdict.REAL_CHANGE for r in det.regions)
