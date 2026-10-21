"""Pencere açılıyor mu?"""

from __future__ import annotations


def test_window_opens(window) -> None:
    assert window.isVisible()
    assert window.windowTitle() == "ChangeScope"
    assert window.library_view.tree.item_count(None) == 4
    assert window.status_bar.llm_label.text().startswith("Gemma:")
