"""Store listing art: sizes, and the Store logos are the app icon."""

import importlib.util
import os

import pytest
from PySide6.QtGui import QGuiApplication, QImage

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "generate_store_listing_assets.py",
)
_spec = importlib.util.spec_from_file_location("store_art", _SCRIPT)
art = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(art)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture(scope="module")
def images():
    return art.generated_images()


@pytest.mark.parametrize("name, size", [
    ("poster_720x1080.png", (720, 1080)),
    ("box_1080x1080.png", (1080, 1080)),
    ("tile_300x300.png", (300, 300)),
    ("logo_150x150.png", (150, 150)),
    ("logo_71x71.png", (71, 71)),
])
def test_sizes(images, name, size):
    assert (images[name].width(), images[name].height()) == size


def test_store_logos_are_the_app_icon(images):
    icons = art._app_icons()
    for name, size in art.STORE_LOGO_SIZES.items():
        assert images[name] == icons.render(size), name


def test_committed_art_matches_the_generator(images):
    """Partner Center gets these files as committed; regenerate after a
    logo change. Compared by pixels with a small antialiasing tolerance."""
    for name, image in images.items():
        committed = QImage(os.path.join(art.OUT_DIR, name)).convertToFormat(
            image.format(),
        )
        assert (committed.width(), committed.height()) == (
            image.width(), image.height(),
        ), name
        worst = 0
        for y in range(0, image.height(), 3):
            for x in range(0, image.width(), 3):
                a, b = committed.pixelColor(x, y), image.pixelColor(x, y)
                worst = max(worst, abs(a.red() - b.red()),
                            abs(a.green() - b.green()),
                            abs(a.blue() - b.blue()))
        assert worst <= 8, name


@pytest.mark.parametrize("name", ["poster_720x1080.png", "box_1080x1080.png"])
def test_brand_art_is_centred_on_its_ink(images, name):
    """The logo's viewBox has uneven padding; centring on it put the art
    visibly left of centre."""
    image = images[name]
    left, _top, right, _bottom = art.ink_bounds(image)
    assert abs(left - (image.width() - 1 - right)) <= 2


@pytest.mark.parametrize("name", ["poster_720x1080.png", "box_1080x1080.png"])
def test_brand_art_stays_out_of_the_bottom_third(images, name):
    """The Store may lay text over the bottom third of these images."""
    image = images[name]
    _left, _top, _right, bottom = art.ink_bounds(image)
    assert bottom < image.height() * 2 / 3
