"""LLM modülü: gömülü llama-server + Gemma 4 ile bölge yorumu ve genel özet.

Dışa açık API: `create_analyst(config)` → `RegionAnalyst` (+ `release()`).
"""

from __future__ import annotations

from core.config import AppConfig, load_config
from core.llm.analyst import GemmaRegionAnalyst

__all__ = ["GemmaRegionAnalyst", "create_analyst"]


def create_analyst(config: AppConfig | None = None) -> GemmaRegionAnalyst:
    """Gemma tabanlı `RegionAnalyst` üretir. Sunucu burada BAŞLATILMAZ (lazy).

    Args:
        config: Uygulama ayarları; None → `load_config()`.

    Returns:
        `is_available()`, `analyze()`, `summarize()`, `release()`, `is_server_running`, `last_start_info`
        sağlayan analist.
    """
    return GemmaRegionAnalyst(config or load_config())
