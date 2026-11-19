"""Report how often each stemma release has been downloaded from GitHub.

A privacy-friendly view of uptake: GitHub counts downloads of each
release asset, and this script only reads those public totals. Nothing
about the people downloading is collected anywhere, by the site or the app.
(Microsoft Store installs are reported in Partner Center, not here.)

    python scripts/release_stats.py            prints a Markdown table
    python scripts/release_stats.py --json     prints the raw numbers

The weekly "Download stats" workflow runs it and posts the table to the
run's summary. Set GITHUB_TOKEN for a higher API rate limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

REPOSITORY = "cyanidesayonara/stemma"
# the downloads people install; checksums and other files are left out
COUNTED = ("stemma.zip", "stemma.msix")


def fetch_releases(repository: str = REPOSITORY) -> list[dict]:
    """Every published release, newest first, from the GitHub API."""
    releases: list[dict] = []
    page = 1
    while True:
        url = (f"https://api.github.com/repos/{repository}/releases"
               f"?per_page=100&page={page}")
        request = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "stemma-release-stats",
        })
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)
        releases.extend(r for r in batch if not r.get("draft"))
        if len(batch) < 100:
            return releases
        page += 1


def summarise(releases: list[dict]) -> list[dict]:
    """Per release: tag, date, and download count per counted asset."""
    rows = []
    for release in releases:
        counts = {name: 0 for name in COUNTED}
        for asset in release.get("assets", []):
            if asset.get("name") in counts:
                counts[asset["name"]] = int(asset.get("download_count", 0))
        rows.append({
            "tag": release.get("tag_name", ""),
            "published": (release.get("published_at") or "")[:10],
            "prerelease": bool(release.get("prerelease")),
            "downloads": counts,
            "total": sum(counts.values()),
        })
    return rows


def markdown(rows: list[dict]) -> str:
    """The rows as a Markdown table, with a total line."""
    head = ["Release", "Published", *COUNTED, "Total"]
    lines = [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for row in rows:
        tag = row["tag"] + (" (prerelease)" if row["prerelease"] else "")
        cells = [tag, row["published"],
                 *(str(row["downloads"][n]) for n in COUNTED),
                 str(row["total"])]
        lines.append("| " + " | ".join(cells) + " |")
    totals = [str(sum(r["downloads"][n] for r in rows)) for n in COUNTED]
    grand = str(sum(r["total"] for r in rows))
    lines.append("| **All releases** | | " + " | ".join(totals)
                 + f" | **{grand}** |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = summarise(fetch_releases())
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        print("## Downloads from GitHub Releases\n")
        print(markdown(rows))
        print("\nMicrosoft Store installs are in Partner Center > Analytics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
