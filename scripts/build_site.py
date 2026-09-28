"""Build the website's documentation pages from docs/*.md.

    python scripts/build_site.py      # writes site/docs/*.html, and the contents on site/index.html

The markdown in docs/ stays the only source: it is what the test suite executes, and this script
renders it into the design's documentation template -- listings with line numbers, section marks,
side notes -- so the site can never drift from the tested text. Standard library only, so the
Pages workflow runs it without installing anything.
"""

from __future__ import annotations

import html
import io
import json
import keyword
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SITE = ROOT / "site"
REPO = "https://github.com/Kungie/gut"

CHAPTERS = [
    ("getting-started", "Getting started"),
    ("backends", "Backends"),
    ("knowing-when-it-doesnt-know", "Knowing when it doesn't know"),
    ("batching", "Asking everything at once"),
    ("async", "Async"),
    ("exact-costs", "Exact costs"),
    ("caching-and-observability", "Caching and observability"),
    ("cli", "Command line"),
    ("mcp", "MCP server"),
    ("limitations", "Honest limitations"),
]
SLUGS = {slug for slug, _ in CHAPTERS}

LANG_LABEL = {"python": "python", "bash": "terminal", "json": "json", "": "text"}


# --------------------------------------------------------------------------- inline markdown


def link_target(url: str, *, docs_page: bool) -> str:
    """Where a markdown link should point on the site."""
    if re.match(r"https?://", url) or url.startswith("#"):
        return url
    path, _, anchor = url.partition("#")
    suffix = f"#{anchor}" if anchor else ""
    name = path.rsplit("/", 1)[-1]
    if not path.startswith("../"):
        if name == "README.md":
            return f"index.html{suffix}"
        if name.endswith(".md") and name[:-3] in SLUGS:
            return f"{name[:-3]}.html{suffix}"
    if path == "../README.md":
        return f"../index.html{suffix}"
    rel = path.removeprefix("../")
    kind = "tree" if rel.endswith("/") else "blob"
    return f"{REPO}/{kind}/main/{rel}{suffix}"


def inline(text: str, *, docs_page: bool = True) -> str:
    """Render inline markdown: code, links, bold and italics."""
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(
        r"`([^`]+)`",
        lambda m: keep(f'<code class="inline-code">{html.escape(m.group(1))}</code>'),
        text,
    )
    text = html.escape(text, quote=False)
    # The markdown's ASCII dashes, set as the design sets them. Code spans are already stashed.
    text = re.sub(r"(?<=\s)--(?=\s)|^--$", "—", text)

    def anchor(m: re.Match[str]) -> str:
        href = link_target(html.unescape(m.group(2)), docs_page=docs_page)
        external = href.startswith("http")
        label = m.group(1) + (" ↗" if external and "github.com" in href else "")
        return keep(f'<a href="{html.escape(href)}">{label}</a>')

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", anchor, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w])_(?!\s)(.+?)(?<!\s)_(?![\w])", r"<em>\1</em>", text)
    # A link's label can hold code, so a restored fragment can hold another placeholder.
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


# --------------------------------------------------------------------------- code listings


def python_classes(source: str) -> list[str | None]:
    """A class per character: keywords, strings, comments, and gut's functions."""
    classes: list[str | None] = [None] * len(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def mark(start: tuple[int, int], end: tuple[int, int], cls: str) -> None:
        a = offsets[start[0] - 1] + start[1]
        b = offsets[end[0] - 1] + end[1]
        for i in range(a, min(b, len(classes))):
            classes[i] = cls

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return classes
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.COMMENT:
            mark(tok.start, tok.end, "c")
        elif tok.type == tokenize.STRING or tok.type in _fstring_types():
            mark(tok.start, tok.end, "s")
        elif tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
            mark(tok.start, tok.end, "k")
        elif (
            tok.type == tokenize.NAME
            and i >= 2
            and tokens[i - 1].string == "."
            and tokens[i - 2].string == "gut"
            and i + 1 < len(tokens)
            and tokens[i + 1].string == "("
        ):
            mark(tok.start, tok.end, "f")
    return classes


def _fstring_types() -> set[int]:
    return {
        getattr(tokenize, name)
        for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
        if hasattr(tokenize, name)
    }


def shell_classes(source: str, lang: str) -> list[str | None]:
    """Comments and strings, for terminal and JSON listings."""
    classes: list[str | None] = [None] * len(source)
    for m in re.finditer(r'"(?:[^"\\\n]|\\.)*"|\'[^\'\n]*\'', source):
        for i in range(m.start(), m.end()):
            classes[i] = "s"
    if lang == "bash":
        for m in re.finditer(r"(?:^|(?<=\s))#[^\n]*", source):
            if classes[m.start()] is None:
                for i in range(m.start(), m.end()):
                    classes[i] = "c"
    return classes


def listing(source: str, lang: str, number: str) -> str:
    """A code block as the design prints one: a caption and numbered lines."""
    if lang == "python":
        classes = python_classes(source)
    elif lang in ("bash", "json"):
        classes = shell_classes(source, lang)
    else:
        classes = [None] * len(source)
    lines_html = []
    position = 0
    for line in source.split("\n"):
        parts = []
        current: str | None = None
        chunk = ""
        for offset, char in enumerate(line):
            cls = classes[position + offset] if position + offset < len(classes) else None
            if cls != current and chunk:
                parts.append(
                    f'<span class="{current}">{html.escape(chunk)}</span>'
                    if current
                    else html.escape(chunk)
                )
                chunk = ""
            current = cls
            chunk += char
        if chunk:
            parts.append(
                f'<span class="{current}">{html.escape(chunk)}</span>'
                if current
                else html.escape(chunk)
            )
        position += len(line) + 1
        lines_html.append(f'<span class="l">{"".join(parts) or " "}</span>')
    label = LANG_LABEL.get(lang, lang or "text")
    return (
        '<figure class="listing">\n'
        f'  <figcaption class="listing-head" style="margin:0;font:12px/1.3 var(--mono)"><span>{label}</span><span>listing {number}</span></figcaption>\n'
        f"  <pre><code>{chr(10).join(lines_html)}</code></pre>\n"
        "</figure>"
    )


# --------------------------------------------------------------------------- block markdown


@dataclass
class Page:
    slug: str
    number: int
    title: str = ""
    body: list[str] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)
    has_code: bool = False


def slugify(text: str) -> str:
    """A heading's anchor, spelled as GitHub spells it, so links work in both places."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\- ]", "", text.lower()).strip()
    return text.replace(" ", "-")


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    head, body = (
        cells[0],
        [r for r in cells[1:] if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)],
    )
    th = "".join(f"<th>{inline(c)}</th>" for c in head)
    trs = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="doc-table"><table class="band-table"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def render(page: Page, markdown: str) -> Page:
    lines = markdown.split("\n")
    i = 0
    section = 0
    listing_count = 0
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            page.body.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            listing_count += 1
            page.has_code = page.has_code or lang == "python"
            page.body.append(listing("\n".join(block), lang, f"{page.number}.{listing_count}"))
        elif stripped.startswith("# ") and not page.title:
            flush()
            page.title = stripped[2:].strip()
        elif re.fullmatch(r"\[← docs index\]\(README\.md\)", stripped):
            pass
        elif stripped.startswith("## "):
            flush()
            section += 1
            text = stripped[3:].strip()
            anchor = slugify(text)
            page.sections.append((anchor, re.sub(r"<[^>]+>", "", inline(text))))
            page.body.append(
                f'<h2 id="{anchor}"><span class="sc">{page.number}.{section}</span>{inline(text)}</h2>'
            )
        elif stripped.startswith("### "):
            flush()
            text = stripped[4:].strip()
            page.body.append(f'<h3 id="{slugify(text)}">{inline(text)}</h3>')
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            page.body.append(table(rows))
            continue
        elif re.match(r"^[-*] ", stripped):
            flush()
            items: list[str] = []
            while i < len(lines) and (
                re.match(r"^[-*] ", lines[i].strip()) or (lines[i].startswith("  ") and items)
            ):
                if re.match(r"^[-*] ", lines[i].strip()):
                    items.append(lines[i].strip()[2:])
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            page.body.append(
                "<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>"
            )
            continue
        elif not stripped:
            flush()
        else:
            paragraph.append(stripped)
        i += 1
    flush()
    return page


# --------------------------------------------------------------------------- the page template


def version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.M)
    return match.group(1) if match else "?"


def head(title: str, description: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(description)}">
<meta property="og:type" content="article">
<meta name="theme-color" content="#171614">
<link rel="icon" href="../assets/favicon.svg?v=2" type="image/svg+xml">
<link rel="preload" href="../assets/fonts/9d1b3b9a.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="../assets/style.css">
<script>try{{var t=localStorage.getItem("gut-theme");if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
</head>
<body>
<div style="width: 100%; background-color: var(--paper); color: var(--ink); font-family: var(--sans)">
"""


def header() -> str:
    return f"""<header class="nav">
  <div class="wrap nav-in">
    <a class="mark" href="../index.html" aria-label="gut, home"><b>gut</b><small>docs · {version()}</small></a>
    <ul class="nav-links">
      <li><a href="../index.html#playground">Playground</a></li>
      <li><a href="../index.html#backends">Backends</a></li>
      <li><a href="index.html">Docs</a></li>
      <li><a href="../index.html#agents">Agents</a></li>
      <li><a href="mcp.html">MCP</a></li>
      <li><a class="ext" href="{REPO}">GitHub</a></li>
      <li><a class="ext" href="https://pypi.org/project/gutfeel/">PyPI</a></li>
    </ul>
    <details class="nav-menu">
      <summary>Contents</summary>
      <ul>
        <li><a href="../index.html">Home</a></li>
        <li><a href="../index.html#playground">Playground</a></li>
        <li><a href="index.html">All docs</a></li>
        <li><a href="{REPO}">GitHub ↗</a></li>
      </ul>
    </details>
    <button class="theme-btn" type="button" data-theme-toggle="" aria-label="Switch between light and dark">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><circle cx="8" cy="8" r="6.2"></circle><path d="M8 1.8v12.4A6.2 6.2 0 0 0 8 1.8z" fill="currentColor"></path></svg>
    </button>
  </div>
</header>
"""


def footer() -> str:
    return f"""<footer class="foot">
  <div class="wrap">
    <div class="foot-in">
      <div class="foot-install">
        <code class="big" id="foot-cmd"><span>pip install</span> gutfeel</code>
        <button class="copy" type="button" data-copy="#foot-cmd">Copy</button>
      </div>
      <ul>
        <li>Apache-2.0</li>
        <li><a href="{REPO}">GitHub</a></li>
        <li><a href="https://pypi.org/project/gutfeel/">PyPI (gutfeel)</a></li>
      </ul>
    </div>
  </div>
</footer>
</div>
<script src="../assets/site.js" defer></script>
</body>
</html>
"""


GETTING_STARTED_NOTES = [
    ("Why “gutfeel”", "The name <code>gut</code> was taken on PyPI. The import was not."),
    ("0.1 s", "The local NLI model answering one question on a laptop CPU."),
    (
        "No thresholds",
        "You never write 0.7. You say <code>lean</code>, <code>stakes</code> or <code>ask_human</code>, and gut places the boundaries.",
    ),
]


def notes(page: Page) -> str:
    items = list(GETTING_STARTED_NOTES) if page.slug == "getting-started" else []
    if page.has_code:
        items.append(
            (
                "Tested",
                "Every code sample on this page is run by gut's test suite, so none of it can drift from the library.",
            )
        )
    items.append(
        (
            "This page's source",
            f'<a href="{REPO}/blob/main/docs/{page.slug}.md">docs/{page.slug}.md</a> in the repository. Corrections welcome.',
        )
    )
    letters = "abcdefgh"
    return "\n".join(
        f'    <div class="note"><b><span class="n">{letters[i]}</span>{title}</b>{text}</div>'
        for i, (title, text) in enumerate(items)
    )


def contents_nav(current: str, sections: list[tuple[str, str]]) -> str:
    current_mark = ' aria-current="page"'
    chapters = "\n".join(
        f'      <li><a href="{slug}.html"{current_mark if slug == current else ""}>{html.escape(title)}</a></li>'
        for slug, title in CHAPTERS
    )
    on_page = ""
    if sections:
        entries = "\n".join(
            f'      <li><a href="#{anchor}">{text}</a></li>' for anchor, text in sections
        )
        on_page = f'    <span class="sc">On this page</span>\n    <ol class="onpage">\n{entries}\n    </ol>\n'
    return (
        '  <nav class="doc-nav" aria-label="Documentation">\n'
        '    <span class="sc">Contents</span>\n'
        f"    <ol>\n{chapters}\n    </ol>\n{on_page}  </nav>\n"
    )


def pager(index: int) -> str:
    if index == 0:
        previous = '<a href="index.html"><small>← contents</small><span>All chapters</span></a>'
    else:
        slug, title = CHAPTERS[index - 1]
        previous = f'<a href="{slug}.html"><small>← previous · {index}</small><span>{html.escape(title)}</span></a>'
    if index + 1 < len(CHAPTERS):
        slug, title = CHAPTERS[index + 1]
        following = f'<a href="{slug}.html" style="text-align:right"><small>next · {index + 2}</small><span>{html.escape(title)} →</span></a>'
    else:
        following = '<a href="index.html" style="text-align:right"><small>the end</small><span>All chapters →</span></a>'
    return f'    <nav class="doc-pager" aria-label="Pager">\n      {previous}\n      {following}\n    </nav>\n'


def chapter_page(index: int) -> tuple[str, Page]:
    slug, _ = CHAPTERS[index]
    page = render(
        Page(slug=slug, number=index + 1), (DOCS / f"{slug}.md").read_text(encoding="utf-8")
    )
    first_paragraph = next(
        (re.sub(r"<[^>]+>", "", b) for b in page.body if b.startswith("<p>")), ""
    )
    description = html.unescape(first_paragraph)[:200]
    body = "\n".join(f"    {block}" for block in page.body)
    document = (
        head(f"{page.title} · gut docs", description)
        + header()
        + '\n<div class="wrap doc">\n'
        + contents_nav(slug, page.sections)
        + '\n  <article class="doc-body">\n'
        + f'    <div class="crumbs">docs / {index + 1} / {slug}</div>\n'
        + f"    <h1>{inline(page.title)}</h1>\n"
        + body
        + "\n"
        + pager(index)
        + "  </article>\n\n"
        + '  <aside class="side doc-side">\n'
        + notes(page)
        + "\n  </aside>\n</div>\n\n"
        + footer()
    )
    return document, page


def index_page() -> str:
    """The table of contents, from docs/README.md's own descriptions."""
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    described = {}
    for row in re.findall(r"^\|\s*\[[^\]]+\]\(([a-z-]+)\.md\)\s*\|(.+)\|\s*$", readme, re.M):
        described[row[0]] = row[1].strip()
    items = "\n".join(
        f'      <li><a href="{slug}.html"><span class="t">{html.escape(title)}</span><span class="dots"></span><span class="go">read →</span></a>'
        f'<p class="toc-desc">{inline(described.get(slug, ""))}</p></li>'
        for slug, title in CHAPTERS
    )
    also = (
        readme.split("Also worth knowing about:", 1)[1]
        if "Also worth knowing about:" in readme
        else ""
    )
    extra = render(Page(slug="index", number=0), also).body if also else []
    return (
        head(
            "Documentation · gut", "The gut documentation: every chapter, every code sample tested."
        )
        + header()
        + '\n<div class="wrap doc doc-index">\n'
        + '  <article class="doc-body">\n'
        + '    <div class="crumbs">docs</div>\n'
        + "    <h1>Contents.</h1>\n"
        + f"    <p>{len(CHAPTERS)} chapters. Read them in order if you are new; every code sample on every page is run by the test suite.</p>\n"
        + f'    <ol class="toc">\n{items}\n    </ol>\n'
        + (
            '    <h2 id="also"><span class="sc">+</span>Also worth knowing about</h2>\n'
            + "\n".join(f"    {b}" for b in extra)
            + "\n"
            if extra
            else ""
        )
        + "  </article>\n</div>\n\n"
        + footer()
    )


def home_toc() -> str:
    """The home page's table of contents, which points at these pages."""
    items = "\n".join(
        f'        <li><a href="docs/{slug}.html"><span class="t">{html.escape(title)}</span>'
        '<span class="dots"></span><span class="go">read →</span></a></li>'
        for slug, title in CHAPTERS
    )
    return f'      <!-- docs:toc (written by scripts/build_site.py) -->\n      <ol class="toc">\n{items}\n      </ol>\n      <!-- /docs:toc -->'


def home_page(current: str) -> str:
    """site/index.html with its table of contents and version brought up to date."""
    updated = re.sub(
        r"      <!-- docs:toc .*?<!-- /docs:toc -->", lambda _: home_toc(), current, flags=re.S
    )
    return re.sub(r"gutfeel \d+\.\d+\.\d+", f"gutfeel {version()}", updated)


def pages() -> dict[str, str]:
    """Every file this script writes, by its path under site/."""
    written = {
        f"docs/{slug}.html": chapter_page(index)[0] for index, (slug, _) in enumerate(CHAPTERS)
    }
    written["docs/index.html"] = index_page()
    written["index.html"] = home_page((SITE / "index.html").read_text(encoding="utf-8"))
    return written


HOME = "https://gutpy.dev/"
"""Where the site lives. The old address on GitHub Pages only redirects here."""


def redirect_page(path: str) -> str:
    """A page at the old address that sends its reader to the same page at the new one."""
    target = HOME + ("" if path == "index.html" else path.removesuffix("index.html"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gut has moved to gutpy.dev</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace({json.dumps(target)} + location.hash)</script>
</head>
<body><p>gut has moved to <a href="{target}">{target}</a>.</p></body>
</html>
"""


NOT_FOUND_REDIRECT = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gut has moved to gutpy.dev</title>
<script>location.replace({json.dumps(HOME)} + location.pathname.replace(/^\\/gut\\/?/, "") + location.search + location.hash)</script>
</head>
<body><p>gut has moved to <a href="{HOME}">{HOME}</a>.</p></body>
</html>
"""


def redirects(site: Path = SITE) -> dict[str, str]:
    """The old GitHub Pages site: every page of the new one, as a redirect to it."""
    pages = {
        str(page.relative_to(site)): redirect_page(str(page.relative_to(site)))
        for page in sorted(site.rglob("*.html"))
    }
    pages["404.html"] = NOT_FOUND_REDIRECT
    return pages


def build(site: Path = SITE) -> list[Path]:
    paths = []
    for name, text in pages().items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def build_redirects(out: Path) -> list[Path]:
    paths = []
    for name, text in redirects().items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    if sys.argv[1:2] == ["--redirects"]:
        # For GitHub Pages, which now only points visitors at gutpy.dev.
        written = build_redirects(Path(sys.argv[2]))
    else:
        written = build(Path(sys.argv[1]) if len(sys.argv) > 1 else SITE)
    for path in written:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
