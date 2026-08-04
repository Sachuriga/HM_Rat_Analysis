"""MSCA proposal, Figure 1: the pipeline is already running in the host lab.

    A  simultaneously recorded CA1 place cells on the idealised maze, one session
    B  unit yield per session, both animals
    C  behavioural performance per session
    D  one recording day: 2 implanted animals recorded, 6 non-implanted trained

This is the "I can already do this" figure, and it doubles as the reader's only
view of the maze — figure 2 does not repeat it. It is deliberately SMALL: Part B-1
is ten pages and its text already fills them, so anything that could be a sentence
in the caption is a sentence in the caption. Of the six per-session metrics the
summary produces, only two are here: unit yield and behaviour. The other four
(spatial information, field count, field size, map stability) are all confounded
with how much data a session held — ``place_fields.METRIC_NOTES`` says so — and a
feasibility figure should not lead with a number that needs a caveat.

Panel A is drawn by ``msca_fig1a``; panels B-D read the same summary table as the
full figure, so this figure and the report cannot disagree.

Colour and type come from ``figures/palette.py``, shared with figure 2. Type is
Times New Roman floored at ``palette.MIN_PT``; the page is authored at the
document's real column width, so place it at 100% and never rescale it.

Usage:
    python figures/fig_feasibility.py --nwb <session.nwb> --units 149 448 196 307 218 1 \
        --summary <session_summary_*.xlsx> [--out fig1] [--width-mm 180]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import msca_fig1 as F1                                             # noqa: E402
import msca_fig1a as F1A                                           # noqa: E402
import palette as P                                                # noqa: E402

OUT_DIR = Path(os.environ.get("MSCA_FIG_DIR",
                              "/Users/sachuriga/Desktop/MSCA_figures"))
MM = 1 / 25.4
LINE_MM = 3.4

FS = P.scale({"letter": 10.0, "title": 8.5, "body": 8.0, "small": 8.0})

C_MAZE = P.ORANGE          # HexMaze navigation + recording
C_TRAIN = P.GREEN_DARK     # non-implanted training
C_PRE, C_POST = "#a6c8ee", "#1c5596"   # sleep: one hue at two lightness steps


def build_figure(data, summary, width_mm=180.0, animals=None):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": P.SERIF_STACK,
        "mathtext.fontset": "stix", "font.size": FS["body"],
        "axes.linewidth": 0.6, **P.VECTOR_TEXT,
    })
    M, GAP, T_H = 3.0, 11.0, 4.6
    # Panel A's correlogram captions are drawn just ABOVE its axes, so its title
    # needs more clearance than the others or the two print on the same line.
    T_A = 7.6
    inner = width_mm - 2 * M

    # Panel A keeps the maze's own proportions; everything else is sized around
    # what is left. The maze is the one thing here that cannot be reflowed.
    w_a = inner * 0.550
    bbox = data["bbox"]
    h_a = w_a / ((bbox[1] - bbox[0]) / (bbox[3] - bbox[2]))
    w_bc = inner - w_a - GAP
    h_bc = (h_a - T_H) / 2                     # two small panels stacked beside A
    # panel D's axes holds only the bars; its tick row and legend hang BELOW it
    # and need their own room, or the page crops them off
    h_d = LINE_MM * 3.0
    h_d_below = LINE_MM * 2.9
    height_mm = M + T_A + h_a + GAP + T_H + h_d + h_d_below + M
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor=P.SURFACE)

    def rect(x, y, w, h):
        return fig.add_axes([x / width_mm, y / height_mm,
                             w / width_mm, h / height_mm])

    def ftitle(letter, text, x, y):
        fig.text(x / width_mm, y / height_mm, letter.upper(), fontsize=FS["letter"],
                 fontweight="bold", va="bottom", ha="left", color=P.INK)
        fig.text((x + 4.2) / width_mm, y / height_mm, text, fontsize=FS["title"],
                 fontweight="bold", va="bottom", ha="left", color=P.INK)

    def bar(ax, x0, x1, yc, h, colour, text=None):
        ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h, facecolor=colour,
                               edgecolor="none", zorder=3))
        if text:
            ax.text((x0 + x1) / 2, yc, text, ha="center", va="center",
                    fontsize=FS["small"], color=P.ink_on(colour), zorder=4)

    y = height_mm - M

    # ================================================================ A
    y -= T_A
    ftitle("a", "CA1 place cells, one session", M, y + 2.4)
    ax_a = rect(M, y - h_a, w_a, h_a)
    F1A.draw_panel_a(ax_a, data, rasterize=False, scale=w_a / 170.0,
                     fonts=F1A.PRINT_FONT)

    # ---------------------------------------------------------------- B, C
    sess, keys, meta, animals = F1.session_axis(summary, animals)
    pos = F1.slot_positions(meta)
    colours = F1.ANIMAL_COLORS[:len(animals)]
    units = F1._prepared_units(summary, animals)
    trials = summary.get("trials")
    if trials is not None and not trials.empty:
        trials = trials[trials["animal"].isin(animals)].copy()

    x_bc = M + w_a + GAP
    for i, (letter, title) in enumerate((("b", "Unit yield"),
                                         ("c", "Behaviour"))):
        yy = y - (i + 1) * h_bc - i * T_H
        ftitle(letter, title, x_bc, yy + h_bc + 0.5)
        ax = rect(x_bc, yy, w_bc, h_bc)
        if letter == "b":
            F1.units_panel(ax, sess, keys, pos, animals, colours, scale=0.7)
            ax.set_ylabel("units", fontsize=FS["small"], color=P.INK2, labelpad=2)
        else:
            vals = F1._by_animal_key(trials, "performance", keys, animals)
            F1.dist_panel(ax, vals, keys, pos, animals, colours, signed=True,
                          ymax=0.0, scale=0.55)
            ax.set_ylabel("log10 ratio", fontsize=FS["small"], color=P.INK2,
                          labelpad=2)
        # A tick per session would be a dozen numbers in a 60 mm panel; the point
        # here is the TREND across sessions, and the session identities are
        # figure 1's business in the full report, not this one's.
        ax.set_xticks([])
        ax.set_xlim(pos[0] - 0.75, pos[-1] + 0.75)
        ax.tick_params(labelsize=FS["small"], colors=P.MUTED, labelcolor=P.INK2,
                       length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(P.MUTED)
        ax.set_xlabel(f"{len(keys)} sessions →", fontsize=FS["small"],
                      color=P.INK2, labelpad=1)
    y -= h_a

    # ================================================================ D
    y -= GAP + T_H
    ftitle("d", "One recording day, run in two shifts", M, y + 0.5)
    LEFT = 24.0
    axd = rect(M + LEFT, y - h_d, inner - LEFT, h_d)
    mm_per_h = (inner - LEFT) / (18.3 - 8.88)
    Y_IMP, Y_NON = 1.75, 0.62
    Hb = 0.66
    axd.plot([13.5, 13.5], [0.05, 2.25], color=P.MUTED, lw=0.7, ls="--", zorder=1)
    pre = ("pre-sleep 90 min"
           if P.fits_mm("pre-sleep 90 min", 1.5 * mm_per_h - 1,
                        FS["small"]) else "pre-sleep")
    post = "post-sleep 4 h" if P.fits_mm("post-sleep 4 h", 4 * mm_per_h - 1,
                                         FS["small"]) else "post-sleep"
    bar(axd, 9, 10.5, Y_IMP, Hb, C_PRE, pre)      # 90 min, as section 3.1 says
    bar(axd, 10.5, 11.5, Y_IMP, Hb, C_MAZE, "1")
    bar(axd, 12.0, 13.0, Y_IMP, Hb, C_MAZE, "2")
    bar(axd, 14, 18, Y_IMP, Hb, C_POST, post)
    for i, s in enumerate([9.25, 10.10, 14.25, 15.10, 15.95, 16.80]):
        bar(axd, s, s + 1 / 3, Y_NON, Hb, C_TRAIN, f"{i + 1}")
    axd.set_xlim(8.88, 18.3)
    axd.set_ylim(0.22, 2.28)
    axd.set_yticks([Y_IMP, Y_NON])
    axd.set_yticklabels(["Implanted 1–2", "Non-implanted 1–6"],
                        fontsize=FS["small"])
    step = 1 if mm_per_h > 9 else 2
    hours = list(range(9, 19, step))
    axd.set_xticks(hours)
    axd.set_xticklabels([f"{h}:00" for h in hours], fontsize=FS["small"])
    for s in ("top", "right", "left"):
        axd.spines[s].set_visible(False)
    axd.spines["bottom"].set_color(P.MUTED)
    axd.tick_params(axis="y", length=0, labelcolor=P.INK2)
    axd.tick_params(axis="x", colors=P.MUTED, labelcolor=P.INK2)
    hl = [Rectangle((0, 0), 1, 1, fc=c) for c in (C_PRE, C_MAZE, C_POST, C_TRAIN)]
    axd.legend(hl, ["pre-sleep", "HexMaze + recording", "post-sleep",
                    "non-implanted training"],
               loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4,
               frameon=False, fontsize=FS["small"], handlelength=1.2,
               columnspacing=1.3, handleheight=0.85, labelcolor=P.INK2)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nwb", required=True)
    ap.add_argument("--units", type=int, nargs="+", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--animal", default=None, nargs="*")
    ap.add_argument("--out", default="fig1")
    ap.add_argument("--width-mm", type=float, default=180.0,
                    help="page width in MM (default %(default).0f = the Part B-1 "
                         "column: A4 less its 15 mm margins)")
    ap.add_argument("--palette", default="rgbkym",
                    choices=["turbo"] + sorted(F1A.FIXED_PALETTES))
    a = ap.parse_args(argv)

    data = F1A.load_session(a.nwb, a.units, palette=a.palette)
    summary = F1.load_summary(a.summary)
    fig = build_figure(data, summary, width_mm=a.width_mm, animals=a.animal)

    stem = Path(a.out)
    if not stem.is_absolute() and stem.parent == Path("."):
        stem = OUT_DIR / stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext, extra in (("pdf", {}), ("svg", {}), ("png", dict(dpi=600))):
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor=P.SURFACE, **extra)
    w, h = fig.get_size_inches()
    print(f"wrote {stem}.{{pdf,svg,png}}  ({w * 25.4:.0f} x {h * 25.4:.0f} mm)")
    print(f"panel a: Rat {data['animal']} · {data['date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
