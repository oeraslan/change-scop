"""Faz 2 sonu düzeltmeleri: yatay kart kaydırma, karşılaştırılamayan alan işareti, örtüşme bilgisi."""

from __future__ import annotations

import numpy as np

from core.contracts import BoundingBox, ChangeRegion
from tests.ui.helpers import make_result


def _partial_result():
    res = make_result(640, 480)
    vm = np.ones((480, 640), bool)
    vm[:, :134] = False                      # ~%21 örtüşmüyor
    res.registration.valid_mask = vm
    return res


def test_cards_strip_scrolls_to_last_card(window, qtbot) -> None:
    view = window.compare_view
    window.switchTo(view)
    res = make_result(640, 480)
    base = res.detection.regions
    regions = []
    for i in range(1, 21):
        b = base[(i - 1) % 3].bbox
        regions.append(ChangeRegion(region_id=i, bbox=BoundingBox(b.x, b.y, b.width, b.height),
                                    mask=np.ones((b.height, b.width), bool), score=0.5, area_px=b.width * b.height))
    res.detection.regions = regions
    view.show_result(res)
    window.resize(1200, 800)
    window.show()
    scroll = view.result_panel.scroll
    qtbot.waitUntil(lambda: scroll.horizontalScrollBar().maximum() > 0)
    view.result_panel.highlight(20)
    last = view.result_panel.cards()[20]
    qtbot.waitUntil(lambda: scroll.viewport().rect().contains(last.mapTo(scroll.viewport(), last.rect().center())))


def test_invalid_area_overlay_follows_diff_switch(window, qtbot) -> None:
    view = window.compare_view
    window.switchTo(view)
    view.show_result(_partial_result())
    for v in (view.before_view, view.after_view):
        assert v.invalid_overlay_visible()
    view.diff_switch.setChecked(False)
    qtbot.waitUntil(lambda: not view.before_view.invalid_overlay_visible()
                    and not view.after_view.invalid_overlay_visible())


def test_overlap_infobar_and_full_overlap_has_no_overlay(window) -> None:
    view = window.compare_view
    window.switchTo(view)
    view.show_result(_partial_result())
    assert any("örtüşüyor" in t for _k, t in view.active_infobars())
    view.show_result(make_result(640, 480))
    assert not any("örtüşüyor" in t for _k, t in view.active_infobars())
    assert not view.before_view.invalid_overlay_visible()
