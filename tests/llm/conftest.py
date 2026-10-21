"""Ortak test yardımcıları: sahte HTTP (OpenAI uyumlu) sunucu, sentetik kayıt/tespit nesneleri."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from core.config import AppConfig
from core.contracts import (
    AlignmentQuality,
    BoundingBox,
    ChangeRegion,
    DetectionResult,
    ImageInput,
    PreparedImage,
    RegistrationResult,
)
from core.llm.analyst import GemmaRegionAnalyst
from core.llm.server import StartInfo


class MockLLMServer:
    """`/v1/chat/completions` için betiklenebilir sahte sunucu.

    `responses`: sırayla tüketilen yanıtlar. Her öğe str (içerik), ("status", kod) ya da ("sleep", sn, str).
    Tükenince `default` kullanılır. Alınan istekler `requests` içinde saklanır.
    """

    def __init__(self, responses: list | None = None, default=None) -> None:
        self.responses = list(responses or [])
        self.default = default
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                item = outer.responses.pop(0) if outer.responses else outer.default
                if isinstance(item, tuple) and item[0] == "status":
                    self.send_response(item[1])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if isinstance(item, tuple) and item[0] == "sleep":
                    time.sleep(item[1])
                    item = item[2]
                payload = {"choices": [{"message": {"content": item}}],
                           "usage": {"completion_tokens": 20}, "timings": {"predicted_ms": 1000.0}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeServer:
    """`ServerLike` uygulaması: `start()` sabit bir URL döndürür (süreç yok)."""

    def __init__(self, url: str | None, present: bool = True) -> None:
        self.url = url
        self.present = present
        self.info = StartInfo()
        self.start_calls = 0
        self.stop_calls = 0

    def files_present(self) -> bool:
        return self.present

    def start(self) -> str | None:
        self.start_calls += 1
        return self.url

    def stop(self) -> None:
        self.stop_calls += 1

    @property
    def is_running(self) -> bool:
        return self.url is not None


@pytest.fixture
def mock_llm():
    servers: list[MockLLMServer] = []

    def make(responses=None, default=None) -> MockLLMServer:
        s = MockLLMServer(responses, default)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def make_scene(w: int = 400, h: int = 300) -> tuple[RegistrationResult, DetectionResult]:
    """Basit bir kayıt sonucu + 3 bölge (skorlar 0.9/0.5/0.7)."""
    rng = np.random.default_rng(0)
    before = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    after = before.copy()
    prepared = PreparedImage(ImageInput(Path("b.png")), before, 1.0, False)
    reg = RegistrationResult(
        before=prepared, after_warped=after, valid_mask=np.ones((h, w), bool), homography=np.eye(3),
        num_matches=100, num_inliers=90, reprojection_error_px=0.5, quality=AlignmentQuality.GOOD,
    )
    boxes = [(300, 20, 40, 40), (180, 140, 30, 30), (10, 250, 50, 30)]
    scores = [0.9, 0.5, 0.7]
    regions = [
        ChangeRegion(region_id=i, bbox=BoundingBox(*b), mask=np.ones((b[3], b[2]), bool), score=s, area_px=b[2] * b[3])
        for i, (b, s) in enumerate(zip(boxes, scores))
    ]
    det = DetectionResult(change_map=np.zeros((h, w), np.float32), regions=regions, method="ssim")
    return reg, det


@pytest.fixture
def scene():
    return make_scene()


def make_analyst(url: str | None, **llm_overrides) -> tuple[GemmaRegionAnalyst, FakeServer]:
    cfg = AppConfig()
    for k, v in llm_overrides.items():
        setattr(cfg.llm, k, v)
    fake = FakeServer(url)
    return GemmaRegionAnalyst(cfg, server=fake), fake
