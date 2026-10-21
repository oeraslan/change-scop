"""scripts/download_models.py testleri — sadece yerel (127.0.0.1) sahte sunucu, internet yok."""

from __future__ import annotations

import hashlib
import http.server
import threading
import zipfile
from pathlib import Path

import pytest

from scripts import download_models as dm

PAYLOAD = bytes(range(256)) * 4096  # 1 MB


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Range destekli, ilk isteği yarıda kesebilen minimal sunucu."""

    drop_first = False
    requests: list[str | None] = []

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        type(self).requests.append(rng)
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).drop_first and len(type(self).requests) == 1:
            self.wfile.write(body[: len(body) // 3])   # bağlantı yarıda koptu
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _RangeHandler.requests = []
    _RangeHandler.drop_first = False
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/f.bin"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dm, "_retry_wait", lambda attempt, exc: None)
    monkeypatch.setattr(dm, "tqdm", None)


def test_download_verifies_sha(server, tmp_path):
    dest = tmp_path / "a" / "f.bin"
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    dm.download_file(server, dest, expected_size=len(PAYLOAD), expected_sha256=sha)
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name("f.bin.part").exists()


def test_resumes_existing_part_file(server, tmp_path):
    dest = tmp_path / "f.bin"
    dest.with_name("f.bin.part").write_bytes(PAYLOAD[:1000])
    dm.download_file(server, dest, expected_size=len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD
    assert _RangeHandler.requests == ["bytes=1000-"]


def test_resumes_after_dropped_connection(server, tmp_path):
    _RangeHandler.drop_first = True
    dest = tmp_path / "f.bin"
    dm.download_file(server, dest, expected_size=len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD
    assert _RangeHandler.requests[0] is None and _RangeHandler.requests[1].startswith("bytes=")


def test_bad_sha_deletes_part(server, tmp_path):
    dest = tmp_path / "f.bin"
    with pytest.raises(dm.DownloadError):
        dm.download_file(server, dest, expected_size=len(PAYLOAD), expected_sha256="0" * 64)
    assert not dest.exists() and not dest.with_name("f.bin.part").exists()


def test_skips_existing_complete_file(server, tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(PAYLOAD)
    dm.download_file(server, dest, expected_size=len(PAYLOAD))
    assert _RangeHandler.requests == []


def test_safe_extract_blocks_zip_slip(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ok/a.txt", "a")
        zf.writestr("../kacak.txt", "x")
    with pytest.raises(dm.DownloadError):
        dm.safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "kacak.txt").exists()


def test_safe_extract_strip_and_include(tmp_path):
    z = tmp_path / "repo.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("repo-abc/dinov2/__init__.py", "")
        zf.writestr("repo-abc/dinov2/hub/backbones.py", "")
        zf.writestr("repo-abc/notebooks/x.ipynb", "")
    assert dm.common_root(z) == "repo-abc/"
    n = dm.safe_extract(z, tmp_path / "out", strip_prefix="repo-abc/", include_prefix="dinov2/")
    assert n == 2
    assert (tmp_path / "out" / "dinov2" / "hub" / "backbones.py").is_file()
    assert not (tmp_path / "out" / "notebooks").exists()


def _asset(name: str) -> dict:
    return {"name": name, "size": 1, "browser_download_url": f"https://x/{name}"}


def test_pick_llama_release_skips_non_build_tags():
    releases = [
        {"tag_name": "v0.7.0", "assets": [_asset("llama-v0.7.0-bin-win-cuda-12.4-x64.zip")]},
        {"tag_name": "b200", "assets": [_asset("llama-b200-bin-win-cpu-x64.zip")]},
        {"tag_name": "b199", "assets": [_asset("llama-b199-bin-win-cuda-12.4-x64.zip"),
                                       _asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
                                       _asset("llama-b199-bin-win-cuda-13.4-x64.zip")]},
    ]
    tag, main, rt = dm.pick_llama_release(releases, "12.4")
    assert tag == "b199"
    assert main["name"] == "llama-b199-bin-win-cuda-12.4-x64.zip"
    assert rt["name"] == "cudart-llama-bin-win-cuda-12.4-x64.zip"
    with pytest.raises(dm.DownloadError):
        dm.pick_llama_release(releases, "12.4", tag="b200")


def test_parse_targets():
    assert dm.parse_targets(None, False) == dm.DEFAULT_TARGETS
    assert "gemma-e4b" not in dm.DEFAULT_TARGETS
    assert dm.parse_targets(None, True) == dm.ALL_TARGETS
    assert dm.parse_targets(["llama,superpoint", "gemma"], False) == ["superpoint", "gemma-12b", "llama"]
    with pytest.raises(SystemExit):
        dm.parse_targets(["yok"], False)
