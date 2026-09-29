"""Draw the README's wordmark: "gut" in the site's Instrument Serif italic, with an orange g.

    uv run --with fonttools --with brotli python scripts/make_wordmark.py

The letters are outlines taken from the font the site already ships, so the image needs no font to
show the same everywhere. Two files, because the letters after the g follow GitHub's theme: dark ink
on a light page, the site's cream on a dark one.
"""

from __future__ import annotations

from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "site" / "assets" / "fonts" / "32043434.woff2"  # Instrument Serif italic, Latin
SIGNAL = "#E0452B"
INKS = {"light": "#171614", "dark": "#EDE6D8"}
PAD = 40


def number(n: float) -> str:
    return f"{n:.1f}".rstrip("0").rstrip(".")


def outlines() -> tuple[list[tuple[str, str]], tuple[float, float, float, float]]:
    """Each letter's path, placed side by side, and the bounds of the whole word."""
    font = TTFont(FONT)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    x = 0.0
    paths = []
    bounds = BoundsPen(glyphs)
    for letter in "gut":
        name = cmap[ord(letter)]
        pen = SVGPathPen(glyphs, ntos=number)
        # Font units have y pointing up; SVG's points down.
        glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))
        glyphs[name].draw(TransformPen(bounds, (1, 0, 0, -1, x, 0)))
        paths.append((letter, pen.getCommands()))
        x += glyphs[name].width
    assert bounds.bounds is not None
    return paths, bounds.bounds


def svg(theme: str) -> str:
    paths, (x0, y0, x1, y1) = outlines()
    left, top = x0 - PAD, y0 - PAD
    width, height = x1 - x0 + 2 * PAD, y1 - y0 + 2 * PAD
    box = " ".join(number(n) for n in (left, top, width, height))
    shapes = "\n".join(
        f'  <path fill="{SIGNAL if letter == "g" else INKS[theme]}" d="{d}"/>'
        for letter, d in paths
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}" role="img" aria-label="gut">\n'
        "  <!-- Instrument Serif italic (SIL OFL); drawn by scripts/make_wordmark.py -->\n"
        f"{shapes}\n</svg>\n"
    )


if __name__ == "__main__":
    for theme in INKS:
        out = ROOT / "site" / "assets" / f"wordmark-{theme}.svg"
        out.write_text(svg(theme), encoding="utf-8")
        print(out.relative_to(ROOT))
