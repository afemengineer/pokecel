#!/usr/bin/env python3
"""Fetch a Pokémon X/Y model ZIP from The Models Resource and optionally run it."""
from __future__ import annotations

import argparse
import html
import http.cookiejar
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

BASES = (
    "https://models.spriters-resource.com",
    "https://www.models-resource.com",
)
GAME_PATH = "/3ds/pokemonxy/"
USER_AGENT = "PokeCel/0.1 (+https://github.com/afemengineer/pokecel)"
ASSET_RE = re.compile(r"/(?:3ds/pokemonxy/)?(?:asset|model)/(\d+)/?", re.I)
ICON_RE = re.compile(r"/media/asset_icons/(\d+)/(\d+)\.", re.I)
DEX_RE = re.compile(r"^#?(\d{1,4})\s*(.*)$")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []
        self.images: list[str] = []
        self._current: dict[str, str] | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self._current = {
                "href": attrs["href"],
                "title": attrs.get("title", ""),
                "alt": "",
            }
            self._parts = []
        elif tag == "img":
            src = attrs.get("src")
            if src:
                self.images.append(src)
            if self._current is not None:
                alt = attrs.get("alt", "")
                title = attrs.get("title", "")
                if alt:
                    self._current["alt"] += " " + alt
                if title:
                    self._current["title"] += " " + title

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current is not None:
            self._current["text"] = " ".join("".join(self._parts).split())
            self.links.append(self._current)
            self._current = None
            self._parts = []


def normalize(text: str) -> str:
    text = html.unescape(text).replace("♀", " female ").replace("♂", " male ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def pokemon_key(label: str) -> tuple[int | None, str]:
    label = html.unescape(label).strip()
    m = DEX_RE.match(label)
    if m:
        return int(m.group(1)), normalize(m.group(2))
    return None, normalize(label)


def make_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def request(opener, url: str, *, referer: str | None = None, timeout: int = 20):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/zip,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    return opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)


def read_html(opener, url: str) -> str:
    with request(opener, url) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8", "replace")


def parse(html_text: str) -> LinkParser:
    parser = LinkParser()
    parser.feed(html_text)
    return parser


def label_for(link: dict[str, str]) -> str:
    return " ".join(
        x for x in (link.get("text", ""), link.get("alt", ""), link.get("title", "")) if x
    ).strip()


def discover_candidates(opener, base: str, max_pages: int = 12) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    stale_pages = 0
    for page in range(1, max_pages + 1):
        suffix = "" if page == 1 else f"page-{page}/"
        url = urllib.parse.urljoin(base, GAME_PATH + suffix)
        try:
            doc = read_html(opener, url)
        except urllib.error.HTTPError as exc:
            if page == 1:
                raise
            if exc.code in (403, 404):
                break
            raise

        before = len(found)
        for link in parse(doc).links:
            href = link["href"]
            if not ASSET_RE.search(href):
                continue
            label = label_for(link)
            if not label:
                continue
            found.setdefault(urllib.parse.urljoin(url, href), label)

        if len(found) == before:
            stale_pages += 1
            if page > 1 and stale_pages >= 2:
                break
        else:
            stale_pages = 0

    return [(url, label) for url, label in found.items()]


def choose(candidates: list[tuple[str, str]], query: str) -> tuple[str, str]:
    query = query.strip()
    if not query:
        raise ValueError("Pokémon name or National Dex number is empty")

    qdex = int(query.lstrip("#")) if query.lstrip("#").isdigit() else None
    qname = normalize(query)
    scored: list[tuple[int, str, str]] = []

    for url, label in candidates:
        dex, name = pokemon_key(label)
        if qdex is not None:
            if dex == qdex:
                scored.append((0, url, label))
            continue
        if name == qname:
            scored.append((0, url, label))
        elif qname and name.startswith(qname + " "):
            scored.append((1, url, label))
        elif qname and qname in name:
            scored.append((2, url, label))

    if not scored:
        for url, label in candidates:
            if qname and qname in normalize(label):
                scored.append((3, url, label))

    if not scored:
        raise LookupError(
            f"No Pokémon matching {query!r} was found in the Pokémon X/Y model roster"
        )

    scored.sort(key=lambda item: (item[0], len(normalize(item[2])), item[2]))
    _, url, label = scored[0]
    return url, label


def download_url_from_asset(asset_url: str, doc: str) -> str:
    parsed = parse(doc)
    for link in parsed.links:
        words = normalize(label_for(link))
        href = link["href"]
        if "download" in words and (
            "zip" in words or "archive" in words or "model" in words
        ):
            return urllib.parse.urljoin(asset_url, href)
        if href.lower().endswith(".zip"):
            return urllib.parse.urljoin(asset_url, href)

    # Fallback for current pages whose download control is generated separately.
    # Their asset icon and archive live under the same numeric media folder.
    for src in parsed.images:
        match = ICON_RE.search(src)
        if match:
            folder_id, asset_id = match.groups()
            return f"https://www.models-resource.com/media/assets/{folder_id}/{asset_id}.zip"

    match = ICON_RE.search(doc)
    if match:
        folder_id, asset_id = match.groups()
        return f"https://www.models-resource.com/media/assets/{folder_id}/{asset_id}.zip"

    raise ValueError("Could not locate the ZIP download URL on the asset page")


def safe_filename(label: str, asset_url: str) -> str:
    dex, name = pokemon_key(label)
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", name or normalize(label)).strip("_") or "pokemon"
    match = ASSET_RE.search(asset_url)
    asset = match.group(1) if match else "asset"
    prefix = f"{dex:04d}_" if dex is not None else ""
    return f"{prefix}{stem}_{asset}.zip"


def fetch_zip(opener, download_url: str, destination: Path, asset_url: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with request(opener, download_url, referer=asset_url, timeout=60) as src, partial.open("wb") as dst:
            content_type = src.headers.get("Content-Type", "")
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise ValueError(
                f"Server response was not a ZIP archive (Content-Type: {content_type or 'unknown'})"
            )
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def resolve_asset(opener, query: str) -> tuple[str, str]:
    errors: list[str] = []
    for base in BASES:
        try:
            candidates = discover_candidates(opener, base)
            if not candidates:
                errors.append(f"{base}: no model links found")
                continue
            return choose(candidates, query)
        except (urllib.error.URLError, urllib.error.HTTPError, LookupError, ValueError) as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError(
        "Could not resolve that Pokémon from The Models Resource:\n  " + "\n  ".join(errors)
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch a Pokémon X/Y model from The Models Resource"
    )
    ap.add_argument("pokemon", help="Pokémon name or National Dex number, e.g. pikachu or 25")
    ap.add_argument("--run", action="store_true", help="Launch the downloaded ZIP in PokeCel")
    ap.add_argument("--model", help="Model substring for archives containing variants, e.g. PikachuM")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="ZIP cache directory (default: models)",
    )
    ap.add_argument("--refresh", action="store_true", help="Download again even if cached")
    args = ap.parse_args()

    opener = make_opener()
    try:
        asset_url, label = resolve_asset(opener, args.pokemon)
        print(f"Found: {label}")
        print(f"Asset: {asset_url}")
        doc = read_html(opener, asset_url)
        download_url = download_url_from_asset(asset_url, doc)
        destination = args.output_dir / safe_filename(label, asset_url)

        if destination.exists() and zipfile.is_zipfile(destination) and not args.refresh:
            print(f"Using cached ZIP: {destination}")
        else:
            print(f"Downloading to: {destination}")
            fetch_zip(opener, download_url, destination, asset_url)
            print(f"Saved: {destination}")

        if args.run:
            import tempfile
            from pokecel import run
            from pokecel_zip import candidates, safe_extract, select_model

            with tempfile.TemporaryDirectory(prefix="pokecel_") as temp:
                root = Path(temp)
                with zipfile.ZipFile(destination) as zf:
                    safe_extract(zf, root)
                model = select_model(candidates(root), args.model)
                print(f"Using model: {model.relative_to(root)}")
                return run(model, 1100, 800) or 0
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"PokeCel fetch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
