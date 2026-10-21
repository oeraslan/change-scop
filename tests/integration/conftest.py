"""Entegrasyon testi ortamı: gerçek pipeline (oturum boyunca tek örnek), hermetik yapılandırma."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from core.config import AppConfig  # noqa: E402
from core.pipeline import Pipeline, build_default_pipeline  # noqa: E402
from tests.integration import helpers  # noqa: E402


@pytest.fixture(scope="session")
def real_pipeline() -> Pipeline:
    """Gerçek modüllerle pipeline. Kullanıcının config.json'una bağlı değil: `AppConfig()` varsayılanları."""
    pipe = build_default_pipeline(AppConfig())
    yield pipe
    pipe.shutdown()


def pytest_terminal_summary(terminalreporter) -> None:  # noqa: ANN001
    """Senaryo → kalite/bölge/isabet/süre tablosu."""
    if not helpers.RESULTS:
        return
    cols = ["senaryo", "kalite", "bölge", "isabet", "büyük_fazla", "fazla", "süre"]
    rows = [[str(r[c]) for c in cols] for r in helpers.RESULTS]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    terminalreporter.section("Entegrasyon sonuç tablosu")
    terminalreporter.write_line("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        terminalreporter.write_line("  ".join(v.ljust(w) for v, w in zip(r, widths)))
