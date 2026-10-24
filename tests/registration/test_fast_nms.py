"""CPU için hızlı SuperPoint NMS'i (cv2.dilate) özgün `max_pool2d` sürümüyle birebir aynı sonuç vermeli."""

from __future__ import annotations

import pytest
import torch

from core.registration.matching import _install_fast_cpu_nms, fast_simple_nms


def _original():
    from lightglue import superpoint as sp

    return getattr(sp, "_changescope_original_nms", sp.simple_nms)


@pytest.mark.parametrize("shape", [(1, 64, 48), (2, 37, 53), (1, 128, 96)])
@pytest.mark.parametrize("radius", [1, 4])
def test_fast_nms_equals_original(shape, radius) -> None:
    g = torch.Generator().manual_seed(sum(shape) + radius)
    scores = torch.rand(shape, generator=g)
    scores[..., :5, :5] = 0.5                            # eşit değerli plato (eşitlik davranışı)
    assert torch.equal(fast_simple_nms(scores, radius), _original()(scores, radius))


def test_install_is_idempotent_and_keeps_original() -> None:
    from lightglue import superpoint as sp

    _install_fast_cpu_nms()
    _install_fast_cpu_nms()
    assert sp.simple_nms is fast_simple_nms
    assert sp._changescope_original_nms is not fast_simple_nms
