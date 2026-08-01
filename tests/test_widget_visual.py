"""Smoke tests for offscreen widget snapshot helpers."""

import pytest
from PySide6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def test_label_widget_snapshot(app):
    """A simple QLabel renders deterministically offscreen."""
    from tests.widget_visual import assert_widget_snapshot

    label = QLabel("Widget snapshot smoke test")
    label.setStyleSheet("background-color: #336699; color: #ffffff; padding: 8px;")
    label.adjustSize()

    assert_widget_snapshot(label, "label_smoke", width=240, height=48)
