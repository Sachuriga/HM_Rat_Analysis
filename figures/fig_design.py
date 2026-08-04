"""MSCA proposal, Figure 2: the schema-update design and what it predicts.

    A  experiment design: a 25-day build-up phase, then a 3-day testing phase
    B  predicted cfos/BrdU recruitment at each of the three killing points

This is the "what I will do" figure. It is deliberately SMALL: Part B-1 is ten
pages and its text already fills them, so every millimetre a figure takes has to
be paid for out of the argument. Anything that could be a sentence in the caption
is a sentence in the caption — the maze itself lives in figure 1 (where the place
cells are drawn on it), and the EBN-before-LBN time course is one clause of the
WP2 prediction rather than a panel of its own.

Colour and type come from ``figures/palette.py``, shared with figure 1. Type is
Times New Roman, floored at ``palette.MIN_PT``; the page is authored at the
document's real column width so it is placed at 100% and never rescaled.

Usage:
    python figures/fig_design.py [out_stem] [--width-mm 180]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyBboxPatch                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette as P                                                # noqa: E402

OUT_DIR = Path(os.environ.get("MSCA_FIG_DIR",
                              "/Users/sachuriga/Desktop/MSCA_figures"))
MM = 1 / 25.4
LINE_MM = 3.4                       # one 8 pt line with its leading

FS = P.scale({"letter": 10.0, "title": 8.5, "body": 8.0, "small": 8.0})

C_EBN, C_LBN = P.GREEN_DARK, P.ORANGE
C_KP1, C_KP2, C_KP3 = P.BLUE, P.AMBER, P.RED
INKC = P.TEXT_ON_SURFACE

PHASES = (("Build-up · Days 1–25", "new information integrated into a stable schema",
           C_KP1),
          ("Testing · Days 25–27", "new memory replayed; old schema updated", C_KP2))

#: How many of five cells are recruited. The prediction is ORDINAL, so it is
#: counted out in cells rather than drawn as a bar, which would imply a number
#: nobody has yet. LBN is the UPPER layer and EBN the lower one, as they sit in
#: the tissue, so a box reads as a slice rather than as a table.
KP_PANELS = ((C_KP1, "KP1 · old goal (D25, s5)", 5, "active", 0, "silent"),
             (C_KP2, "KP2 · new goal (D26, s1)", 3, "moderate", 5, "active ↑↑"),
             (C_KP3, "KP3 · new goal (D27, s2)", 5, "active ↑↑", 1, "fading"))


def build_figure(width_mm=180.0):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": P.SERIF_STACK,
        "mathtext.fontset": "stix", "font.size": FS["body"],
        "axes.linewidth": 0.6, **P.VECTOR_TEXT,
    })
    M, GAP, T_H = 3.0, 4.0, 4.6
    inner = width_mm - 2 * M

    # panel A's height follows the wrap of its phase captions
    n_phase = max(P.wrap_mm(b, inner * 0.47 - 5, FS["small"]).count("\n") + 1
                  for _h, b, _c in PHASES)
    h_a = LINE_MM * (1.3 + 1.3 * n_phase + 0.6 + 1.4 + 0.5 + 1.3 + 0.3)
    h_b = LINE_MM * 5.4
    height_mm = M + T_H + h_a + GAP + T_H + h_b + M
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor=P.SURFACE)

    def rect(x, y, w, h):
        return fig.add_axes([x / width_mm, y / height_mm,
                             w / width_mm, h / height_mm])

    def ftitle(letter, text, x, y):
        fig.text(x / width_mm, y / height_mm, letter.upper(), fontsize=FS["letter"],
                 fontweight="bold", va="bottom", ha="left", color=P.INK)
        fig.text((x + 4.2) / width_mm, y / height_mm, text, fontsize=FS["title"],
                 fontweight="bold", va="bottom", ha="left", color=P.INK)

    def rbox(ax, x, y, w, h, fc, r=0.04):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
            edgecolor="none", linewidth=0, transform=ax.transAxes, zorder=1))

    def dot(ax, x, y, d_in, face, edge, lw=0.6, z=4):
        ax.scatter([x], [y], s=(d_in * 72.0) ** 2, marker="o",
                   transform=ax.transAxes, facecolors=face, edgecolors=edge,
                   linewidths=lw, zorder=z, clip_on=False)

    y = height_mm - M

    # ============================================================ A  design
    y -= T_H
    ftitle("a", "Experiment design", M, y + 0.5)
    axa = rect(M, y - h_a, inner, h_a)
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    axa.axis("off")
    line = LINE_MM / h_a

    #: The 25-day build-up and the 3-day testing phase share one axis but not one
    #: scale: at a common scale the three killing points sit inside 2/27 of the
    #: width and their labels overprint. The break mark says where the scale
    #: changes, so a compressed axis is not read as a linear one.
    BREAK_AT, BREAK_X = 24.0, 0.545

    def day_x(day):
        if day <= BREAK_AT:
            return 0.012 + (day - 1) / (BREAK_AT - 1) * (BREAK_X - 0.012)
        return BREAK_X + (day - BREAK_AT) / (27 - BREAK_AT) * (0.915 - BREAK_X)

    box_h = line * (1.25 + 1.3 * n_phase)
    for x0, bw, (head, body, col) in ((0.005, 0.470, PHASES[0]),
                                      (0.500, 0.495, PHASES[1])):
        rbox(axa, x0, 1.0 - box_h, bw, box_h, P.TINT[col])
        avail = bw * inner - 5.0
        axa.text(x0 + 0.022, 1.0 - line * 0.25,
                 P.wrap_mm(head, avail, FS["body"], weight="bold"),
                 transform=axa.transAxes, fontsize=FS["body"], color=INKC[col],
                 va="top", fontweight="bold")
        axa.text(x0 + 0.022, 1.0 - line * 1.35,
                 P.wrap_mm(body, avail, FS["small"]), transform=axa.transAxes,
                 fontsize=FS["small"], color=INKC[col], va="top", linespacing=1.3)

    # chips and axis are placed FROM the box, so they cannot slide under it
    y_chip = 1.0 - box_h - line * 0.95
    y_ax = y_chip - line * 1.45
    axa.annotate("", xy=(0.995, y_ax), xytext=(0.0, y_ax), xycoords="axes fraction",
                 textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", lw=0.9, color=P.MUTED))
    for off in (-0.006, 0.006):
        axa.plot([BREAK_X + off - 0.005, BREAK_X + off + 0.005],
                 [y_ax - line * 0.22, y_ax + line * 0.22], transform=axa.transAxes,
                 color=P.MUTED, lw=0.8, zorder=5, clip_on=False)
    dot(axa, day_x(1), y_ax, 0.028, P.INK, P.INK, lw=0, z=5)
    axa.text(0.0, y_ax - line * 0.35, "Day 1", transform=axa.transAxes, ha="left",
             va="top", fontsize=FS["small"], color=P.INK2)
    for day, col in ((25, C_KP1), (26, C_KP2), (27, C_KP3)):
        x = day_x(day)
        dot(axa, x, y_ax, 0.032, col, col, lw=0, z=6)
        axa.text(x, y_chip, f"KP{day - 24} · D{day}", transform=axa.transAxes,
                 ha="center", va="center", fontsize=FS["small"], color=INKC[col],
                 zorder=7, bbox=dict(boxstyle="round,pad=0.26",
                                     facecolor=P.TINT[col], edgecolor=col,
                                     linewidth=0.7))
        axa.plot([x, x], [y_ax + line * 0.12, y_chip - line * 0.55],
                 transform=axa.transAxes, color=col, lw=0.6, zorder=4)
        axa.text(x, y_ax - line * 0.35, f"Day {day}", transform=axa.transAxes,
                 ha="center", va="top", fontsize=FS["small"], color=P.INK2)
    y -= h_a

    # ============================================================ B  prediction
    y -= GAP + T_H
    ftitle("b", "Predicted cfos/BrdU recruitment", M, y + 0.5)
    axb = rect(M, y - h_b, inner, h_b)
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.axis("off")

    def pyramidal(ax, x, yc, h, colour, filled, lw=0.65):
        """A pyramidal cell in AXES coordinates, `h` tall in axes-y.

        The soma half-width is converted through the panel's real proportions, or
        it comes out a sliver on a wide panel and a blob on a narrow one.
        """
        w = h * 0.42 * h_b / inner
        ax.add_patch(plt.Polygon([[x, yc + h * 0.42], [x - w, yc - h * 0.18],
                                  [x + w, yc - h * 0.18]], closed=True,
                                 facecolor=colour if filled else "none",
                                 edgecolor=colour, linewidth=lw,
                                 transform=ax.transAxes, zorder=4,
                                 joinstyle="round"))
        ax.plot([x, x], [yc + h * 0.42, yc + h * 0.88], transform=ax.transAxes,
                color=colour, lw=lw, zorder=4, solid_capstyle="round")
        for dx in (-w * 0.85, w * 0.85):
            ax.plot([x, x + dx], [yc + h * 0.88, yc + h * 1.12],
                    transform=ax.transAxes, color=colour, lw=lw * 0.85, zorder=4,
                    solid_capstyle="round")
        for dx in (-w * 0.95, w * 0.95):
            ax.plot([x, x + dx], [yc - h * 0.18, yc - h * 0.48],
                    transform=ax.transAxes, color=colour, lw=lw * 0.85, zorder=4,
                    solid_capstyle="round")
        ax.plot([x, x], [yc - h * 0.18, yc - h * 0.72], transform=ax.transAxes,
                color=colour, lw=lw * 0.7, zorder=4, solid_capstyle="round")

    BW, G = 0.3167, 0.025
    for i, (col, head, ebn_n, ebn_t, lbn_n, lbn_t) in enumerate(KP_PANELS):
        x0 = i * (BW + G)
        rbox(axb, x0, 0.03, BW, 0.94, P.TINT[col], r=0.035)
        axb.text(x0 + BW / 2, 0.875, head, transform=axb.transAxes, ha="center",
                 va="center", fontsize=FS["small"], color=INKC[col], zorder=4)
        for row, (lab, n, note, ccol) in enumerate((("LBN", lbn_n, lbn_t, C_LBN),
                                                    ("EBN", ebn_n, ebn_t, C_EBN))):
            yy = 0.575 - row * 0.345
            axb.text(x0 + 0.020, yy, lab, transform=axb.transAxes, ha="left",
                     va="center", fontsize=FS["body"], color=ccol, zorder=4)
            for k in range(5):
                # 1.84*h is the full height of the glyph; the rows are 0.345
                # apart, so anything above 0.187 makes the layers interleave
                pyramidal(axb, x0 + 0.072 + k * 0.029, yy, 0.165, ccol, k < n)
            # right-ALIGNED at the fill's inner edge, so a wide status like
            # 'active ↑↑' runs inwards instead of over the last cell
            axb.text(x0 + BW - 0.012, yy, note, transform=axb.transAxes,
                     ha="right", va="center", fontsize=FS["small"],
                     color=ccol if n > 0 else P.MUTED, zorder=4)
        axb.plot([x0 + 0.02, x0 + BW - 0.02], [0.4025, 0.4025],
                 transform=axb.transAxes, color=P.MUTED, lw=0.5,
                 ls=(0, (2.2, 1.9)), zorder=2)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="fig2")
    ap.add_argument("--width-mm", type=float, default=180.0,
                    help="page width in MM (default %(default).0f = the Part B-1 "
                         "column: A4 less its 15 mm margins)")
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
