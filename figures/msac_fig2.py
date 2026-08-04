"""MSCA proposal, Figure 2: the i-CRADLE training schedule.

    A  one recording day: 2 implanted + 6 non-implanted animals, maze never idle
    B  session cadence: implanted daily vs non-implanted spaced retrieval
    C  full experiment: 4 build-up goal locations, then the update phase
    D  update manipulation: a barrier on the bridge nearest the goal forces a detour
    E  what a recording day yields

Colour and type come from ``figures/palette.py``, shared with figures 1 and 3.

TYPE NEVER SHRINKS. Every size is in points and floored at ``palette.MIN_PT``, so
a 110 mm column gets the same 8 pt text as a 170 mm one. What gives instead is the
LAYOUT and the WORDING: below ``STACK_BELOW_MM`` panels D and E stack, the legend
gains rows, and labels that live inside a box switch to their short forms
("barrier" becomes "B", explained by the bracket above it). The alternative — 6 pt
inside a 5 mm box — is a label nobody can read in print.

Every string that has to fit somewhere is MEASURED against the real font metrics
(``palette.wrap_mm``), not estimated from a character count: DejaVu Sans runs
about a fifth wider than Times at the same size, and a character-count estimate
errs optimistic exactly where the box is tightest.

Usage:
    python figures/msac_fig2.py [out_stem] [--width-mm 170]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import networkx as nx                                              # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import palette as P                                                # noqa: E402
from hm_rat_analysis import maze                                   # noqa: E402

OUT_DIR = Path(os.environ.get("MSCA_FIG_DIR",
                              "/Users/sachuriga/Desktop/MSCA_figures"))
MM = 1 / 25.4
STACK_BELOW_MM = 100.0

FS = P.scale({"letter": 10.5, "title": 9.0, "body": 8.0, "small": 8.0})

# ---------------------------------------------------------------- palette
# Figure 1 spends the four hues on the ANIMAL; this figure has no animals to
# separate, so the same hues carry the ACTIVITY — one hue per thing that happens
# in a recording day.
C_MAZE = P.ORANGE          # HexMaze navigation + recording — the hero activity
C_TRAIN = P.GREEN_DARK     # non-implanted training
C_GL = P.BLUE              # a goal-location block
C_PRE, C_POST = "#a6c8ee", "#1c5596"   # sleep: ONE hue at two lightness steps
C_BAR = P.RED              # the only red: the one element that BLOCKS something
C_GREY = P.INK2

LINE_MM = 3.4              # one 8 pt line with its leading, in mm


def wrap_to(s, avail_mm, pt_size, weight="normal"):
    """`s` wrapped to `avail_mm`, measured from the font's real metrics.

    `weight` matters: bold runs several percent wider than regular at the same
    size, and the strings in this figure that sit in the tightest boxes — the
    panel titles, the bracket captions in panel C — are bold.
    """
    return P.wrap_mm(s, avail_mm, pt_size, weight=weight)


def fits(s, avail_mm, pt_size, weight="normal"):
    return P.fits_mm(s, avail_mm, pt_size, weight=weight)


def build_figure(width_mm=180.0):
    """The whole figure, authored for a page `width_mm` wide."""
    narrow = width_mm < STACK_BELOW_MM
    plt.rcParams.update({
        "font.family": "serif", "font.serif": P.SERIF_STACK,
        "mathtext.fontset": "stix",
        "font.size": FS["body"], "axes.linewidth": 0.6, **P.VECTOR_TEXT,
    })

    M = 5.0
    inner = width_mm - 2 * M
    GAP, T_H = 4.5, 4.8
    LEFT = 26.0 if not narrow else 24.0    # room for the row names beside A and B

    h_a = 25.0 if not narrow else 30.0
    # four legend entries fit one row at the full column width
    leg_rows = 1 if width_mm >= 175 else (2 if not narrow else 4)
    # the legend hangs under panel A, clear of its tick labels: 4.5 mm for the
    # ticks plus a line per legend row
    h_leg = 4.5 + 3.8 * leg_rows
    h_b = 19.0 if not narrow else 24.0

    # Panel C's height follows its CAPTIONS. They wrap to whatever the page
    # allows, so the wrap is resolved FIRST and the panel is then made tall enough
    # for it — a fixed height is what pushed a three-line caption out of the top of
    # the panel at 110 mm.
    block_mm = (inner - LEFT) / 9
    long_bar = fits("barrier", block_mm - 1.5, FS["small"])
    brackets = [(0, 4, "schema build-up: 4 goal locations", C_GL),
                (4, 9, "update: barrier and new goal alternate" if long_bar else
                 "update: B = barrier, alternating with a new goal", C_BAR)]
    caps = []
    for bx0, bx1, t, _c in brackets:
        # A CENTRED caption may only be as wide as twice its distance to the
        # nearer panel edge. Wrapping it to its bracket's SPAN instead is what let
        # the left one hang off the page: that bracket sits near the edge while
        # its span is wide.
        centre_mm = ((bx0 + bx1) / 2 + 0.1) / 9.2 * (inner - LEFT)
        avail = 2 * min(centre_mm, (inner - LEFT) - centre_mm) - 1.5
        caps.append(wrap_to(t, avail, FS["small"], weight="bold"))
    n_cap = max(c.count("\n") + 1 for c in caps)
    h_c = 12.0 + 3.6 * n_cap


    height_mm = (M + T_H + h_a + h_leg + GAP + T_H + h_b + GAP + T_H + h_c + M)
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor=P.SURFACE)

    def rect(x_mm, y_mm, w_mm_, h_mm_):
        return fig.add_axes([x_mm / width_mm, y_mm / height_mm,
                             w_mm_ / width_mm, h_mm_ / height_mm])

    def ftitle(letter, text, x_mm, y_mm, avail_mm=None):
        fig.text(x_mm / width_mm, y_mm / height_mm, letter.upper(),
                 fontsize=FS["letter"], fontweight="bold", va="bottom",
                 ha="left", color=P.INK)
        if avail_mm is not None:
            text = wrap_to(text, avail_mm, FS["title"], weight="bold")
        fig.text((x_mm + 4.6) / width_mm, y_mm / height_mm, text,
                 fontsize=FS["title"], fontweight="bold", va="bottom",
                 ha="left", color=P.INK, linespacing=1.25)

    def bar(ax, x0, x1, yc, h, colour, text=None, tc=None):
        ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h, facecolor=colour,
                               edgecolor="none", zorder=3))
        if text:
            ax.text((x0 + x1) / 2, yc, text, ha="center", va="center",
                    fontsize=FS["small"],
                    color=P.ink_on(colour) if tc is None else tc, zorder=4)

    y = height_mm - M

    # ================================================================ A
    y -= T_H
    ftitle("a", "One recording day, run in two shifts", M, y + 0.6, inner - 6)
    axa = rect(M + LEFT, y - h_a, inner - LEFT, h_a)
    mm_per_h = (inner - LEFT) / (18.3 - 8.88)
    Y_SHIFT, Y_IMP, Y_NON, Y_OCC = 3.35, 2.35, 1.40, 0.42
    Hb = 0.62

    for x0, x1, t in ((9, 13.5, "morning shift"), (13.5, 18, "afternoon shift")):
        axa.add_patch(Rectangle((x0, Y_SHIFT - 0.30), x1 - x0, 0.60,
                                facecolor=P.FILL, edgecolor=P.RULE,
                                linewidth=0.5, zorder=2))
        label = f"{t}  (2 students)"
        if not fits(label, (x1 - x0) * mm_per_h - 2, FS["small"]):
            label = t
        axa.text((x0 + x1) / 2, Y_SHIFT, label, ha="center", va="center",
                 fontsize=FS["small"], color=P.INK2, zorder=3)
    axa.plot([13.5, 13.5], [-0.05, 3.05], color=P.MUTED, lw=0.7, ls="--",
             zorder=1)

    # the words inside a box switch to their short form when the box is too small
    pre = ("pre-sleep 90 min"
           if fits("pre-sleep 90 min", 1.5 * mm_per_h - 1, FS["small"])
           else "pre-sleep")
    post = "post-sleep 4 h" if fits("post-sleep 4 h", 4 * mm_per_h - 1,
                                    FS["small"]) else "post-sleep"
    bar(axa, 9, 10.5, Y_IMP, Hb, C_PRE, pre)      # 90 min, as section 3.1 says
    bar(axa, 10.5, 11.5, Y_IMP, Hb, C_MAZE, "1")
    bar(axa, 12.0, 13.0, Y_IMP, Hb, C_MAZE, "2")
    bar(axa, 14, 18, Y_IMP, Hb, C_POST, post)
    axa.text(11.75, Y_IMP - 0.58, "maze, 1 h each", ha="center", va="top",
             fontsize=FS["small"], color=C_GREY)

    slots = [9.25, 10.10, 14.25, 15.10, 15.95, 16.80]
    for i, s in enumerate(slots):
        bar(axa, s, s + 1 / 3, Y_NON, Hb, C_TRAIN, f"{i + 1}")
    axa.text(9.72, Y_NON - 0.58, "2 runs", ha="center", va="top",
             fontsize=FS["small"], color=C_GREY)
    axa.text(15.7, Y_NON - 0.58, "4 runs, 20 min each", ha="center", va="top",
             fontsize=FS["small"], color=C_GREY)
    for x0 in (10.5, 12.0):
        bar(axa, x0, x0 + 1.0, Y_OCC, 0.32, C_MAZE)
    for s in slots:
        bar(axa, s, s + 1 / 3, Y_OCC, 0.32, C_TRAIN)

    axa.set_xlim(8.88, 18.3)
    axa.set_ylim(0.05, 3.85)
    # the occupancy row gets a real tick label, so it sits in the left margin the
    # layout already reserves instead of hanging off the axes
    axa.set_yticks([Y_SHIFT, Y_IMP, Y_NON, Y_OCC])
    axa.set_yticklabels(["Staffing", "Implanted 1–2", "Non-implanted 1–6",
                         "HexMaze in use"], fontsize=FS["small"])
    axa.get_yticklabels()[-1].set_style("italic")
    step = 1 if mm_per_h > 9 else 2            # an hourly tick needs ~9 mm
    hours = list(range(9, 19, step))
    axa.set_xticks(hours)
    axa.set_xticklabels([f"{h}:00" for h in hours], fontsize=FS["small"])
    for sp in ("top", "right", "left"):
        axa.spines[sp].set_visible(False)
    axa.spines["bottom"].set_color(P.MUTED)
    axa.tick_params(axis="y", length=0, labelcolor=P.INK2)
    axa.tick_params(axis="x", colors=P.MUTED, labelcolor=P.INK2)

    hl = [Rectangle((0, 0), 1, 1, fc=c) for c in (C_PRE, C_MAZE, C_POST, C_TRAIN)]
    fig.legend(hl, ["pre-sleep (sleep box)", "HexMaze navigation + recording",
                    "post-sleep (sleep box)", "non-implanted training, 20 min"],
               loc="upper center",
               bbox_to_anchor=(0.5, (y - h_a - 4.5) / height_mm),
               ncol=4 // leg_rows, frameon=False, fontsize=FS["small"],
               handlelength=1.3, columnspacing=1.6, handleheight=0.85,
               labelcolor=P.INK2)
    y -= h_a + h_leg

    # ================================================================ B
    y -= GAP + T_H
    ftitle("b", "Session cadence: one session per animal per day", M, y + 0.6,
           inner - 6)
    axb = rect(M + LEFT, y - h_b, inner - LEFT, h_b)
    day_mm = (inner - LEFT) / 14.4
    axb.axvspan(6.55, 13.45, color=P.FILL, zorder=0)
    for wx, wname in ((2.5, "week 1"), (9.5, "week 2")):
        axb.text(wx, 3.10, wname, ha="center", va="top", fontsize=FS["small"],
                 color=C_GREY)

    # Each box is named in full — GL1S1, goal location and session — whenever the
    # box is wide enough to hold it at the type floor. In Times at 8 pt that is
    # true down to about a 7.7 mm box, which a 170 mm page gives. When it is not
    # (a 110 mm column leaves 4.8 mm) the box keeps the SESSION number and the
    # goal is named once under a rule instead, figure 1's convention: the goal is
    # the thing that varies slowly, so it is the part that can be factored out.
    box_mm = 0.9 * day_mm
    full_label = fits("GL1S1", box_mm - 0.6, FS["small"])
    IMP = [(d, 1, s) for d, s in zip(range(5), range(1, 6))] + \
          [(d, 2, s) for d, s in zip(range(7, 12), range(1, 6))]
    NON = [(0, 1, 1), (2, 1, 2), (3, 1, 3), (10, 1, 4), (11, 1, 5)]
    for row_y, colour, rows in ((2.05, C_MAZE, IMP), (0.50, C_TRAIN, NON)):
        for d, gl, s in rows:
            bar(axb, d - 0.45, d + 0.45, row_y, 0.62, colour,
                f"GL{gl}S{s}" if full_label else f"{s}")
    if not full_label:
        for x0, x1, name in ((0, 4, "goal 1"), (7, 11, "goal 2")):
            axb.plot([x0 - 0.45, x1 + 0.45], [1.52, 1.52], color=P.MUTED, lw=0.8,
                     solid_capstyle="butt")
            axb.text((x0 + x1) / 2, 1.40, name, ha="center", va="top",
                     fontsize=FS["small"], color=P.INK2)
    for x0, x1, t in ((0, 2, "48 h"), (2, 3, "24 h"), (3, 10, "7 d"),
                      (10, 11, "24 h")):
        if not fits(t, (x1 - x0) * day_mm, FS["small"]):
            continue
        axb.annotate("", xy=(x1 - 0.45, -0.10), xytext=(x0 + 0.45, -0.10),
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=C_GREY))
        axb.text((x0 + x1) / 2, -0.22, t, ha="center", va="top",
                 fontsize=FS["small"], color=C_GREY)
    axb.set_xlim(-0.85, 13.55)
    axb.set_ylim(-0.85, 3.15)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] * 2
    if not fits("Mon", day_mm, FS["small"]):
        days = [d[0] for d in days]
    axb.set_xticks(range(14))
    axb.set_xticklabels(days, fontsize=FS["small"])
    axb.set_yticks([2.05, 0.50])
    axb.set_yticklabels(["Implanted\n(every day)", "Non-implanted\n(spaced)"],
                        fontsize=FS["small"])
    for s in ("top", "right", "left", "bottom"):
        axb.spines[s].set_visible(False)
    axb.tick_params(axis="y", length=0, labelcolor=P.INK2)
    axb.tick_params(axis="x", length=0, labelcolor=P.INK2)
    y -= h_b

    # ================================================================ C
    y -= GAP + T_H
    ftitle("c", "Full experiment, implanted animal: 1 block = 1 week = 5 sessions",
           M, y + 0.6, inner - 6)
    axc = rect(M + LEFT, y - h_c, inner - LEFT, h_c)
    blocks = [("GL1", C_GL), ("GL2", C_GL), ("GL3", C_GL), ("GL4", C_GL),
              ("barrier" if long_bar else "B", C_BAR), ("GL5", C_GL),
              ("barrier" if long_bar else "B", C_BAR), ("GL6", C_GL),
              ("barrier" if long_bar else "B", C_BAR)]
    for i, (name, c) in enumerate(blocks):
        bar(axc, i + 0.05, i + 0.95, 1.0, 0.72, c, name)
    # the data span is solved so that one caption line is exactly LINE_MM tall,
    # which is what keeps a wrapped caption inside the panel at any width
    span = 1.45 / (1 - LINE_MM * n_cap / h_c)
    for (x0, x1, t, col), cap in zip(brackets, caps):
        axc.plot([x0 + 0.05, x1 - 0.05], [1.72, 1.72], color=col, lw=0.9)
        for xx in (x0 + 0.05, x1 - 0.05):
            axc.plot([xx, xx], [1.64, 1.80], color=col, lw=0.9)
        axc.text((x0 + x1) / 2, 1.90, cap, ha="center", va="bottom",
                 fontsize=FS["small"], color=col, fontweight="bold",
                 linespacing=1.25)
    axc.set_xlim(-0.1, 9.1)
    axc.set_ylim(0.45, 0.45 + span)
    axc.axis("off")
    y -= h_c

    # Panels D and E are gone. The update manipulation now leads FIGURE 1, where
    # it is the thing the whole design turns on; the throughput box was prose
    # repeating section 3.1's own numbers, so it belongs in the caption.
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="fig2",
                    help="output stem (default %(default)s); a bare name lands in "
                         f"{OUT_DIR}")
    ap.add_argument("--width-mm", type=float, default=180.0,
                    help="page width in MILLIMETRES (default %(default).0f). Type "
                         f"stays at or above {P.MIN_PT:g} pt at any width; the "
                         "layout and the wording give instead")
    a = ap.parse_args(argv)
    stem = Path(a.out)
    if not stem.is_absolute() and stem.parent == Path("."):
        stem = OUT_DIR / stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=a.width_mm)
    for ext, extra in (("pdf", {}), ("svg", {}), ("png", dict(dpi=600))):
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor=P.SURFACE, **extra)
    w, h = fig.get_size_inches()
    print(f"wrote {stem}.{{pdf,svg,png}}  ({w * 25.4:.0f} x {h * 25.4:.0f} mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
