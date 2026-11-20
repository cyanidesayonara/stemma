"""Assemble the stemma website from site/ and a released commit.

The pages in ``site/`` are templates: every ``<!-- gen:NAME -->`` ...
``<!-- /gen:NAME -->`` block is filled here, from the same sources the Store
listing uses, so the site always says what the released app does:

    version      the released version ("3.0.0")
    released     the release date ("November 19, 2026")
    pitch        the Store short description
    features     cards for the Store product features (site/features.json)
    whatsnew     the Store "What's new" for that version, one item per paragraph
    shots       every Store screenshot, with its caption
    meta        link-preview tags, with the first screenshot as the image
    themecolor  the browser's theme colour, the app's mantle in each theme
    jsonld      the schema.org SoftwareApplication record for search engines
    credits     every mixer demo song, credited, from site/demo/songs.json

The copy and the screenshots come from ``--from`` (a checkout of the released
tag), while the pages, styles and script come from this checkout, so a site
change can ship without a release and a release can never ship main's
unreleased copy.

    python scripts/build_site.py --out build/site
        local preview from this checkout, version from store/listing.yaml
    python scripts/build_site.py --from rel --version 3.0.0 \\
        --released 2026-11-19 --out _site
        what .github/workflows/pages.yml runs after a release

A template missing one of its markers, or with an unclosed or unknown one,
fails the build.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import load_listing, png_size  # noqa: E402
from src.ui.styles import DARK_COLORS, LIGHT_COLORS  # noqa: E402

SITE = ROOT / "site"
# the feature cards and their icons: read at build, not published as files
FEATURES = SITE / "features.json"
FEATURE_ICONS = SITE / "feature-icons"
# where the site is published, for the absolute URLs link previews need;
# the Pages workflow passes the real one (a custom domain included)
DEFAULT_SITE_URL = "https://cyanidesayonara.github.io/stemma/"
SHOTS = Path("assets") / "store_listing" / "screenshots"

# each page and the blocks it must carry
PAGES = {
    "index.html": [
        "version", "released", "pitch", "features", "whatsnew", "shots",
        "meta", "themecolor", "jsonld", "credits",
    ],
}

# icons copied into the site, next to the pages
STATIC_ASSETS = {
    "assets/icons/icon_256.png": "img/icon.png",
    "assets/icons/icon_32.png": "img/favicon.png",
    "assets/icons/logo_arpeggio_dark.svg": "img/logo-dark.svg",
    "assets/icons/logo_arpeggio_light.svg": "img/logo-light.svg",
}

BLOCK = re.compile(r"(<!-- gen:(\w+) -->)(.*?)(<!-- /gen:\2 -->)", re.S)
OPEN = re.compile(r"<!-- gen:(\w+) -->")
CLOSE = re.compile(r"<!-- /gen:(\w+) -->")
WHATS_NEW_HEADING = re.compile(r"^What's new in version \S+\s*$", re.I)


def _e(text: str) -> str:
    # the listing's plain-text " -- " reads as a dash on a web page
    return html.escape(text.replace(" -- ", " – "), quote=True)


def screenshots(source: Path) -> list[tuple[str, str]]:
    """Return (file name, caption) for each Store screenshot in order.

    ``captions.txt`` (``01_practice.png: caption``) sets the order and the
    captions when it exists; older releases without it list every PNG.
    """
    folder = source / SHOTS
    captions = folder / "captions.txt"
    shots: list[tuple[str, str]] = []
    if captions.exists():
        for line in captions.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            name, caption = line.split(":", 1)
            name = name.strip()
            if not (folder / name).is_file():
                raise SystemExit(f"captions.txt names a missing file: {name}")
            shots.append((name, caption.strip()))
    else:
        shots = [(p.name, "") for p in sorted(folder.glob("*.png"))]
    if not shots:
        raise SystemExit(f"no screenshots in {folder}")
    return shots


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def feature_cards(released: list[str], version: str) -> str:
    """The Features section: a card per group of released Store features.

    A card in features.json shows only when every feature it covers is in
    the released listing, so the site never describes unreleased work; a
    released feature no card covers still gets a plain card of its own. A
    card marked ``"coming": "X.Y"`` is a labelled preview of the next
    version, shown only while the released version is older than that.
    """
    cards = json.loads(FEATURES.read_text(encoding="utf-8"))["cards"]
    shown = set(released)
    covered: set[str] = set()
    items = []
    for card in cards:
        coming = card.get("coming")
        if coming:
            if _version_key(version) < _version_key(coming):
                items.append((card["id"], card["title"], card["body"],
                              coming))
            continue
        if not set(card["covers"]) <= shown:
            continue
        covered.update(card["covers"])
        items.append((card["id"], card["title"], card["body"], None))
    for feature in released:
        if feature not in covered:
            items.append(("note", feature, "", None))
    # the preview goes last, after everything that ships today
    items.sort(key=lambda item: item[3] is not None)
    out = []
    for icon, title, body, coming in items:
        svg = (FEATURE_ICONS / f"{icon}.svg").read_text(encoding="utf-8")
        text = ""
        if coming:
            text += f'<span class="soon">Coming in {_e(coming)}</span>'
        text += f"<h3>{_e(title)}</h3>"
        if body:
            text += f"<p>{_e(body)}</p>"
        attrs = ' class="coming"' if coming else ""
        out.append(f'  <li{attrs}><div class="art">{svg.strip()}</div>'
                   f'<div class="copy">{text}</div></li>')
    return "\n".join(out)


def screenshot_alt(name: str) -> str:
    """Alt text from an uncaptioned file name: 02_player_light.png ->
    "stemma screenshot: player, light theme"."""
    words = re.sub(r"^\d+_", "", Path(name).stem).split("_")
    theme = [w for w in words if w in ("dark", "light")]
    rest = " ".join(w for w in words if w not in ("dark", "light"))
    text = f"stemma screenshot: {rest}" if rest else "stemma screenshot"
    if theme:
        text += f", {theme[0]} theme"
    return text


def whats_new_items(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [p for p in paragraphs if not WHATS_NEW_HEADING.match(p)]


def long_date(iso: str) -> str:
    day = dt.date.fromisoformat(iso[:10])
    return f"{day:%B} {day.day}, {day.year}"


def share_meta(site_url: str, shot: str, alt: str, size: tuple[int, int]
               ) -> str:
    """The link-preview tags: canonical URL and a large image card."""
    image = f"{site_url}img/shots/{shot}"
    return "\n".join([
        f'<link rel="canonical" href="{_e(site_url)}">',
        f'<meta property="og:url" content="{_e(site_url)}">',
        f'<meta property="og:image" content="{_e(image)}">',
        f'<meta property="og:image:width" content="{size[0]}">',
        f'<meta property="og:image:height" content="{size[1]}">',
        f'<meta property="og:image:alt" content="{_e(alt)}">',
        f'<meta name="twitter:image" content="{_e(image)}">',
        f'<meta name="twitter:image:alt" content="{_e(alt)}">',
    ])


def theme_color() -> str:
    """The browser chrome colour per theme: the page's background."""
    return "\n".join([
        '<meta name="theme-color" media="(prefers-color-scheme: dark)" '
        f'content="{DARK_COLORS["mantle"]}">',
        '<meta name="theme-color" media="(prefers-color-scheme: light)" '
        f'content="{LIGHT_COLORS["mantle"]}">',
    ])


def store_url(source: Path) -> str:
    """The public Store listing URL, from the release's listing.yaml; from
    this checkout's when that release predates the store block."""
    for checkout in (source, ROOT):
        store = load_listing(checkout / "store" / "listing.yaml").store
        if store is not None:
            return store.url
    raise SystemExit("store/listing.yaml has no store url")


def json_ld(version: str, description: str, download_url: str) -> str:
    """A schema.org SoftwareApplication record for the released app."""
    record = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "stemma",
        "operatingSystem": "Windows 10, Windows 11",
        "applicationCategory": "MultimediaApplication",
        "softwareVersion": version,
        "description": description,
        "downloadUrl": download_url,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
    }
    text = json.dumps(record, ensure_ascii=False, indent=2)
    # nothing in the record may close the script element early
    text = text.replace("</", "<\\/")
    return f'<script type="application/ld+json">\n{text}\n</script>'


def demo_credits(songs: list[dict]) -> str:
    """One line per demo song: title, the artist as the song asks to be
    credited, licence and source. Static, so every song is credited, with
    or without JavaScript."""
    items = []
    for song in songs:
        who = song.get("credit_as") or song["artist"]
        items.append(
            f"      <li>\u201c{_e(song['title'])}\u201d by {_e(who)}, "
            f'<a href="{_e(song["licence_url"])}">{_e(song["licence"])}</a>'
            f' (<a href="{_e(song["source"])}">source</a>)</li>'
        )
    return "\n".join(items)


def blocks(source: Path, version: str, released: str,
           site_url: str = DEFAULT_SITE_URL) -> dict[str, str]:
    listing = load_listing(source / "store" / "listing.yaml")
    if version not in listing.whats_new:
        raise SystemExit(f"store/listing.yaml has no whats_new for {version}")
    if not site_url.endswith("/"):
        site_url += "/"
    shots = screenshots(source)
    first, first_caption = shots[0]
    figures = []
    for name, caption in shots:
        alt = caption or screenshot_alt(name)
        width, height = png_size(source / SHOTS / name)
        figures.append(
            f'  <figure><a href="img/shots/{_e(name)}"><img '
            f'src="img/shots/{_e(name)}" alt="{_e(alt)}" width="{width}" '
            f'height="{height}" loading="lazy"></a>'
            + (f"<figcaption>{_e(caption)}</figcaption>" if caption else "")
            + "</figure>"
        )
    return {
        "version": _e(version),
        "released": _e(long_date(released)),
        "pitch": _e(listing.short_description),
        "features": feature_cards(list(listing.features), version),
        "whatsnew": "\n".join(
            f"  <li>{_e(item)}</li>"
            for item in whats_new_items(listing.whats_new[version])
        ),
        "shots": "\n".join(figures),
        "meta": share_meta(
            site_url, first, first_caption or screenshot_alt(first),
            png_size(source / SHOTS / first),
        ),
        "themecolor": theme_color(),
        "jsonld": json_ld(version,
                          " ".join(listing.short_description.split()),
                          store_url(source)),
        # the demo ships with the site, from this checkout
        "credits": demo_credits(json.loads(
            (SITE / "demo" / "songs.json").read_text(encoding="utf-8")
        )["songs"]),
    }


def check_markers(name: str, page: str, want: list[str]) -> None:
    opens = OPEN.findall(page)
    closes = CLOSE.findall(page)
    whole = [m.group(2) for m in BLOCK.finditer(page)]
    problems = [f"no <!-- gen:{n} --> marker" for n in want if n not in opens]
    for n in set(opens) | set(closes):
        if not opens.count(n) == closes.count(n) == whole.count(n):
            problems.append(f"gen:{n} has an opening or closing marker "
                            "without its pair")
    if problems:
        raise SystemExit(f"site/{name}: " + "; ".join(sorted(set(problems))))


def fill(page: str, values: dict[str, str]) -> str:
    def sub(match: re.Match) -> str:
        name = match.group(2)
        if name not in values:
            raise SystemExit(f"unknown block gen:{name}")
        value = values[name]
        if "\n" in value:
            return f"{match.group(1)}\n{value}\n{match.group(4)}"
        return f"{match.group(1)}{value}{match.group(4)}"

    return BLOCK.sub(sub, page)


def build(source: Path, version: str, released: str, out: Path,
          site_url: str = DEFAULT_SITE_URL) -> None:
    values = blocks(source, version, released, site_url)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SITE, out, ignore=shutil.ignore_patterns(
        FEATURES.name, FEATURE_ICONS.name))
    for name, want in PAGES.items():
        page = (out / name).read_text(encoding="utf-8")
        check_markers(name, page, want)
        (out / name).write_text(fill(page, values), encoding="utf-8",
                                newline="\n")
    for src, dest in STATIC_ASSETS.items():
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        # the released icon, so the site shows the one people download;
        # this checkout's when that release predates the file
        origin = source / src if (source / src).is_file() else ROOT / src
        shutil.copy2(origin, target)
    shots_out = out / "img" / "shots"
    shots_out.mkdir(parents=True, exist_ok=True)
    for name, _caption in screenshots(source):
        shutil.copy2(source / SHOTS / name, shots_out / name)
    # GitHub Pages: serve the files as they are, no Jekyll pass
    (out / ".nojekyll").write_text("", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the stemma website."
    )
    parser.add_argument("--from", dest="source", type=Path, default=ROOT,
                        help="checkout the copy and screenshots come from")
    parser.add_argument("--version",
                        help="released version; default: listing_version")
    parser.add_argument("--released",
                        help="release date, ISO 8601; default: today")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL,
                        help="where the site is published")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    version = args.version or load_listing(
        source / "store" / "listing.yaml"
    ).listing_version
    released = args.released or dt.date.today().isoformat()
    build(source, version.lstrip("v"), released, args.out.resolve(),
          args.site_url)
    print(f"Built the site for {version} in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
