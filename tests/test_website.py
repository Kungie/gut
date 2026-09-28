"""Tests for the website: its documentation pages are built from docs/, and every link on it works.

`scripts/build_site.py` renders docs/*.md into the site. The markdown stays the only source, so
these tests make sure the built pages are never behind it.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DOCS = ROOT / "docs"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_site"] = module  # its dataclasses look themselves up there
    spec.loader.exec_module(module)
    return module


builder = load_builder()


class Links(HTMLParser):
    """Every href and src on a page, and every id it defines."""

    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.targets.append(value)
            if name == "id" and value:
                self.ids.add(value)


def parse(page: Path) -> Links:
    links = Links()
    links.feed(page.read_text(encoding="utf-8"))
    return links


SITE_PAGES = sorted(SITE.glob("*.html")) + sorted((SITE / "docs").glob("*.html"))


# --------------------------------------------------------------------------- built, and current


def test_the_site_is_built_from_the_current_docs() -> None:
    stale = [
        name
        for name, text in builder.pages().items()
        if (SITE / name).read_text(encoding="utf-8") != text
    ]
    assert not stale, f"{', '.join(stale)} behind docs/; run: python scripts/build_site.py"


def test_every_chapter_is_on_the_site_in_the_order_of_the_index() -> None:
    chapters = [slug for slug, _ in builder.CHAPTERS]
    assert set(chapters) == {page.stem for page in DOCS.glob("*.md")} - {"README"}
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    assert re.findall(r"^\| \[[^\]]+\]\(([a-z-]+)\.md\)", index, re.M) == chapters


def test_the_home_page_lists_every_chapter() -> None:
    home = (SITE / "index.html").read_text(encoding="utf-8")
    for slug, title in builder.CHAPTERS:
        assert f'href="docs/{slug}.html"><span class="t">{title}' in home.replace("&#x27;", "'")


def test_the_site_shows_the_current_version() -> None:
    home = (SITE / "index.html").read_text(encoding="utf-8")
    import gut

    assert f"gutfeel {gut.__version__}" in home
    assert not re.findall(rf"gutfeel (?!{re.escape(gut.__version__)})\d+\.\d+\.\d+", home)


# --------------------------------------------------------------------------- the links


@pytest.mark.parametrize("page", SITE_PAGES, ids=lambda page: str(page.relative_to(SITE)))
def test_every_local_link_resolves(page: Path) -> None:
    for target in parse(page).targets:
        parts = urlsplit(target)
        if parts.scheme or target.startswith("mailto:"):
            continue
        destination = (page.parent / parts.path).resolve() if parts.path else page
        assert destination.exists(), f"{page.name} links to {target}, which is not there"
        if parts.fragment and destination.suffix == ".html":
            assert parts.fragment in parse(destination).ids, (
                f"{page.name} links to #{parts.fragment}, which {destination.name} does not have"
            )


@pytest.mark.parametrize("page", SITE_PAGES, ids=lambda page: str(page.relative_to(SITE)))
def test_no_page_sends_a_reader_to_github_for_a_chapter(page: Path) -> None:
    """Chapters are read on the site. A page may only link its own markdown, as its source."""
    to_github = [t for t in parse(page).targets if "github.com/Kungie/gut/blob/main/docs/" in t]
    assert to_github in ([], [f"https://github.com/Kungie/gut/blob/main/docs/{page.stem}.md"])


def test_every_page_starts_dark_and_can_be_switched() -> None:
    for page in SITE_PAGES:
        text = page.read_text(encoding="utf-8")
        assert '<html lang="en" data-theme="dark">' in text, page.name
        assert "data-theme-toggle" in text, page.name
        assert 'id="foot-cmd"><span>pip install</span> gutfeel</code>' in text, page.name


@pytest.mark.parametrize("document", ["README.md", "llms.txt", "skills/gut/SKILL.md"])
def test_links_to_the_site_land_on_built_pages(document: str) -> None:
    text = (ROOT / document).read_text(encoding="utf-8")
    for path, fragment in re.findall(r"https://kungie\.github\.io/gut/([^)\s#>]*)#?([\w-]*)", text):
        page = SITE / path
        if not path or path.endswith("/"):
            page = page / "index.html"
        assert page.exists(), f"{document} links to {path}, which the site does not have"
        if fragment:
            assert fragment in parse(page).ids, f"{document} links to {path}#{fragment}"


# --------------------------------------------------------------------------- the renderer


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("backends.md#cascade", "backends.html#cascade"),
        ("README.md", "index.html"),
        ("../README.md", "../index.html"),
        ("../examples/", "https://github.com/Kungie/gut/tree/main/examples/"),
        ("../llms.txt", "https://github.com/Kungie/gut/blob/main/llms.txt"),
        ("#install", "#install"),
        ("https://pypi.org/project/gutfeel/", "https://pypi.org/project/gutfeel/"),
    ],
)
def test_links_point_at_the_site_or_at_the_repository(url: str, expected: str) -> None:
    assert builder.link_target(url, docs_page=True) == expected


def test_inline_markdown() -> None:
    assert builder.inline("[`Cascade`](backends.md#cascade)") == (
        '<a href="backends.html#cascade"><code class="inline-code">Cascade</code></a>'
    )
    assert builder.inline("**bold** and *italic* and _this_") == (
        "<strong>bold</strong> and <em>italic</em> and <em>this</em>"
    )
    assert builder.inline("before -- after") == "before — after"
    assert builder.inline("`a -- b` & <c>") == (
        '<code class="inline-code">a -- b</code> &amp; &lt;c&gt;'
    )
    assert "↗" in builder.inline("[examples](../examples/)")


def test_headings_get_the_anchors_github_gives_them() -> None:
    assert builder.slugify("Knowing when it doesn't know") == "knowing-when-it-doesnt-know"
    assert builder.slugify("`JevBackend`") == "jevbackend"
    assert builder.slugify("Many subjects, one question") == "many-subjects-one-question"


def test_python_listings_are_highlighted_and_numbered() -> None:
    source = 'import gut\n\nif gut.likely(text, "is spam"):  # hide it\n    pass'
    listing = builder.listing(source, "python", "1.1")
    assert '<span class="k">import</span>' in listing
    assert '<span class="f">likely</span>' in listing
    assert '<span class="s">&quot;is spam&quot;</span>' in listing
    assert '<span class="c"># hide it</span>' in listing
    assert listing.count('<span class="l">') == 4
    assert "listing 1.1" in listing


def test_other_listings_mark_strings_and_comments() -> None:
    listing = builder.listing('pip install "gutfeel[local]"  # or [jev]', "bash", "1.2")
    assert '<span class="s">&quot;gutfeel[local]&quot;</span>' in listing
    assert '<span class="c"># or [jev]</span>' in listing
    assert "<span>terminal</span>" in listing


def test_a_page_renders_headings_tables_and_lists() -> None:
    markdown = (
        "# Title\n\n[← docs index](README.md)\n\nIntro.\n\n## First part\n\n### Detail\n\n"
        "| a | b |\n|---|---|\n| `x` | y |\n\n- one\n  continued\n- two\n"
    )
    page = builder.render(builder.Page(slug="t", number=3), markdown)
    body = "\n".join(page.body)
    assert page.title == "Title"
    assert page.sections == [("first-part", "First part")]
    assert '<h2 id="first-part"><span class="sc">3.1</span>First part</h2>' in body
    assert '<h3 id="detail">Detail</h3>' in body
    assert '<td><code class="inline-code">x</code></td>' in body
    assert "<li>one continued</li><li>two</li>" in body
    assert "docs index" not in body
