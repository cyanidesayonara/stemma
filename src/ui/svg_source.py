"""Read the brand SVGs as element trees for the animated logo widgets.

The animated logos draw some parts of an SVG themselves (noteheads,
waves, letters) and let QSvgRenderer draw the rest. Selecting those parts
by element rather than by matching lines of text keeps the widgets
working when an SVG is regenerated or reformatted: the file stays the
single source of the artwork.

The SVGs are bundled application assets, not untrusted input, so the
standard library parser is appropriate here.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable

SVG_NS = "http://www.w3.org/2000/svg"

# Serialize SVG elements with a default xmlns instead of "ns0:" prefixes.
# (tostring's default_namespace option cannot be used: it rejects the
# unprefixed attribute names every SVG element has.) The registry is
# process-wide; nothing else in stemma uses ElementTree namespaces.
ET.register_namespace("", SVG_NS)


def read_svg(path: str) -> ET.Element:
    """Parse the SVG file at *path* and return its root element."""
    return ET.parse(path).getroot()


def local_name(element: ET.Element) -> str:
    """The element's tag without its namespace (``"path"``, ``"text"``)."""
    return element.tag.rsplit("}", 1)[-1]


def remove_elements(
    root: ET.Element, predicate: Callable[[ET.Element], bool],
) -> int:
    """Remove every element under *root* matching *predicate*.

    Matches are removed at any depth, together with their children.

    Returns:
        The number of elements removed.
    """
    doomed = [
        (parent, child)
        for parent in root.iter()
        for child in parent
        if predicate(child)
    ]
    for parent, child in doomed:
        parent.remove(child)
    return len(doomed)


def with_children(root: ET.Element, children: list[ET.Element]) -> ET.Element:
    """A copy of the ``<svg>`` root holding only *children*.

    The copy keeps the root's attributes (size and viewBox), so it renders
    in the same coordinate space as the full document.
    """
    subset = ET.Element(root.tag, dict(root.attrib))
    subset.extend(children)
    return subset


def to_svg_text(root: ET.Element) -> str:
    """Serialize *root* back to SVG markup QSvgRenderer can load."""
    return ET.tostring(root, encoding="unicode")
