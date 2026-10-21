"""Görüntü hizalama modülü (ön işleme + SuperPoint/LightGlue + homografi).

Dışa aktarılan fabrikalar:
    create_preprocessor(config) -> Preprocessor
    create_registrar(config)    -> Registrar (ek olarak `release()` metodu vardır)
"""

from __future__ import annotations

from core.config import AppConfig
from core.contracts import Preprocessor, Registrar

__all__ = ["create_preprocessor", "create_registrar"]


def create_preprocessor(config: AppConfig | None = None) -> Preprocessor:
    """Config'deki `preprocess` ayarlarıyla bir `Preprocessor` üretir."""
    from core.registration.preprocess import DefaultPreprocessor

    return DefaultPreprocessor((config or AppConfig()).preprocess)


def create_registrar(config: AppConfig | None = None) -> Registrar:
    """Bir `Registrar` üretir. Modeller ilk `register()` çağrısında yüklenir; `release()` bırakır."""
    from core.registration.registrar import DefaultRegistrar

    return DefaultRegistrar(config or AppConfig())
