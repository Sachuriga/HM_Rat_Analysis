"""Check a built figure for type below the floor, and for text that overruns.

Point sizes are absolute, so a figure authored at 170 mm keeps its type when it is
re-authored at 110 mm — but the ARTWORK around it shrinks, and a label that fitted
its box at the wide size can run straight out of it at the narrow one. Asserting
"nothing is under 8 pt" is easy; the useful check is the second one, so this walks
the real Text artists of a drawn figure and reports both.

Usage:
    python figures/check_type.py fig2 --width-mm 110
    python figures/check_type.py fig3 --width-mm 110
"""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import MIN_PT                                        # noqa: E402


def undersized(fig):
    """Every Text on `fig` whose size is below the floor."""
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s.strip():
            continue
        size = t.get_fontsize()
        if size < MIN_PT - 1e-6:
            out.append((size, s.replace("\n", " / ")[:48]))
    return sorted(out)


def overflowing(fig, slack_pt=1.0):
    """Text whose drawn box sticks out of the axes it belongs to.

    `slack_pt` forgives a hair of overhang: a descender or an italic side-bearing
    crossing the spine by a fraction of a point is not a layout problem.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        for t in ax.texts:
            s = t.get_text()
            if not s.strip() or not t.get_visible():
                continue
            if t.get_clip_on():
                continue
            tb = t.get_window_extent(r)
            dx = max(ab.x0 - tb.x0, tb.x1 - ab.x1, 0.0)
            dy = max(ab.y0 - tb.y0, tb.y1 - ab.y1, 0.0)
            over_pt = max(dx, dy) * 72.0 / fig.dpi
            if over_pt > slack_pt:
                out.append((over_pt, s.replace("\n", " / ")[:44]))
    return sorted(out, reverse=True)


def colliding(fig, min_overlap_pt=1.5):
    """Pairs of labels in the same panel whose drawn boxes overlap.

    The bounds check above cannot see this: two labels can both sit well inside
    their panel and still print on top of each other. A text box is generous —
    it is the line box, not the ink — so only a real overlap is reported.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        # An annotation WITH AN ARROW is excluded: matplotlib's window extent for
        # one covers the arrow as well as the text, and the arrow is meant to
        # cross the panel to reach what it points at. Including them reported a
        # collision between a caption and a label that are nowhere near each other.
        ts = [t for t in ax.texts
              if t.get_text().strip() and t.get_visible()
              and getattr(t, "arrow_patch", None) is None]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i].get_window_extent(r), ts[j].get_window_extent(r)
                w = min(a.x1, b.x1) - max(a.x0, b.x0)
                h = min(a.y1, b.y1) - max(a.y0, b.y0)
                if w <= 0 or h <= 0:
                    continue
                amount = min(w, h) * 72.0 / fig.dpi
                if amount > min_overlap_pt:
                    out.append((amount, ts[i].get_text().replace("\n", " / ")[:22],
                                ts[j].get_text().replace("\n", " / ")[:22]))
    return sorted(out, reverse=True)


def report(name, fig, width_mm):
    small = undersized(fig)
    over = overflowing(fig)
    hits = colliding(fig)
    w, h = fig.get_size_inches()
    print(f"\n=== {name} at {w * 25.4:.0f} x {h * 25.4:.0f} mm "
          f"(asked {width_mm:.0f} mm) ===")
    if small:
        print(f"  {len(small)} text object(s) BELOW {MIN_PT:g} pt:")
        for size, s in small[:20]:
            print(f"    {size:5.2f} pt  {s!r}")
    else:
        print(f"  type: OK — nothing below {MIN_PT:g} pt")
    if over:
        print(f"  {len(over)} label(s) overrunning their panel:")
        for amount, s in over[:20]:
            print(f"    {amount:5.1f} pt over  {s!r}")
    else:
        print("  fit:  OK — no label overruns its panel")
    if hits:
        print(f"  {len(hits)} label pair(s) printing over each other:")
        for amount, a, b in hits[:20]:
            print(f"    {amount:5.1f} pt  {a!r} × {b!r}")
    else:
        print("  overlap: OK — no two labels collide")
    return len(small), len(over) + len(hits)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figures", nargs="+", help="module names, e.g. msac_fig2")
    ap.add_argument("--width-mm", type=float, default=110.0)
    a = ap.parse_args(argv)

    bad = 0
    for name in a.figures:
        mod = importlib.import_module(name)
        build = getattr(mod, "build_figure", None)
        if build is None:
            print(f"{name}: no build_figure(width_mm) — cannot re-author it")
            bad += 1
            continue
        fig = build(width_mm=a.width_mm)
        s, o = report(name, fig, a.width_mm)
        bad += s + o
    print()
    print("ALL CLEAR" if bad == 0 else f"{bad} problem(s) found")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
