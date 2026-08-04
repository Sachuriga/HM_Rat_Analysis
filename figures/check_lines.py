"""Report every stroke width a built PDF actually uses, in points.

Line widths in these scripts are written in points but several are multiplied by a
`scale` factor before they are drawn, so what a script SAYS and what the page GETS
are different numbers. This reads the finished PDF instead: matplotlib writes its
content streams in points, so the `w` operator carries the real width.

Below MIN_PT a stroke is at risk of dropping out in offset print or of rendering
as a single device pixel; below WARN_PT it survives but is a hairline. `0 w` is
worse than either: it asks the device for the thinnest line it can draw, which is
one pixel — invisible on a 2400 dpi imagesetter.

Usage:
    python figures/check_lines.py fig1 fig2 fig3
"""

import argparse
import re
import zlib
from collections import Counter
from pathlib import Path

#: Below this, a stroke can be lost in print. 0.25 pt = 0.09 mm is the usual floor
#: quoted by journals; some ask for 0.5 pt.
MIN_PT = 0.25
#: Above MIN_PT but still a hairline worth knowing about.
WARN_PT = 0.5

OUT_DIR = Path("/Users/sachuriga/Desktop/MSCA_figures")


#: PDF operators that actually paint a stroke. A width is only interesting if one
#: of these follows it: matplotlib emits `0 w` for every filled patch whose edge is
#: "none", and counting those would report a page full of invisible hairlines that
#: are never drawn.
STROKE_OPS = re.compile(rb"(?<![A-Za-z*])(?:S|s|B\*?|b\*?)(?![A-Za-z])")


def stroke_widths(pdf_path):
    """Counter of every line width `pdf_path` actually STROKES with, in points."""
    data = Path(pdf_path).read_bytes()
    widths = Counter()
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        body = m.group(1)
        try:
            body = zlib.decompress(body)
        except zlib.error:
            pass                                   # already plain, or not a stream
        setters = list(re.finditer(rb"(?<![\d.])(\d+\.?\d*)\s+w\b", body))
        for i, w in enumerate(setters):
            end = setters[i + 1].start() if i + 1 < len(setters) else len(body)
            if STROKE_OPS.search(body, w.end(), end):
                widths[round(float(w.group(1)), 3)] += 1
    return widths


def report(name, pdf_path):
    widths = stroke_widths(pdf_path)
    if not widths:
        print(f"{name}: no stroke widths found in {pdf_path}")
        return 0
    total = sum(widths.values())
    bad = {w: n for w, n in widths.items() if w < MIN_PT}
    warn = {w: n for w, n in widths.items() if MIN_PT <= w < WARN_PT}
    print(f"\n=== {name} ===  {total} stroke settings, "
          f"{len(widths)} distinct widths")
    print("   " + "  ".join(f"{w:g}pt×{n}" for w, n in sorted(widths.items())))
    if bad:
        print(f"   BELOW {MIN_PT} pt — at risk of dropping out in print:")
        for w, n in sorted(bad.items()):
            note = " (device-thinnest, one pixel)" if w == 0 else ""
            print(f"     {w:g} pt used {n}×{note}")
    if warn:
        print(f"   hairline ({MIN_PT}-{WARN_PT} pt), legible but thin: "
              + ", ".join(f"{w:g}pt×{n}" for w, n in sorted(warn.items())))
    if not bad and not warn:
        print(f"   OK — every stroke is at or above {WARN_PT} pt")
    return len(bad)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figures", nargs="+", help="stems under the figure directory")
    a = ap.parse_args(argv)
    bad = 0
    for name in a.figures:
        p = Path(name)
        if not p.exists():
            p = OUT_DIR / f"{name}.pdf"
        bad += report(name, p)
    print()
    print("ALL CLEAR" if bad == 0 else f"{bad} width(s) below the print floor")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
