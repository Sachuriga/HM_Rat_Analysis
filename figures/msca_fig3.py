"""MSCA proposal, Figure 3: the HexMaze, the schema-update design, and what the
cfos/BrdU readout is predicted to do.

    A  experiment design: a 25-day build-up phase, then a 3-day testing phase
    B  predicted cfos/BrdU recruitment at each of the three key phases
    C  predicted learning dynamics: EBN as a slow integrator, LBN as fast encoding
    D  the same prediction as a trajectory in EBN-LBN state space

The maze and the room photograph lead FIGURE 1, where the place cells are drawn on
that same lattice; repeating them here would cost a third of this page to say
something already said.

A REDRAW of a figure previously assembled by hand in Illustrator (Fig3.ai).
Nothing in C-F is fitted to data: they are the hypothesis drawn to scale, and the
curves in E are shapes chosen to state a claim, not a model. Panel A is the real
lattice from ``hm_rat_analysis.maze``, so the maze here and the maze in figures 1
and 2 are the same object. Panel B is the one thing no code can regenerate — it
is a photograph, carried over from the Illustrator file (see PHOTO).

Colour and type come from ``figures/palette.py``, shared with figures 1 and 2.

TYPE NEVER SHRINKS. Every size is in points and floored at ``palette.MIN_PT``, so
a 110 mm column gets the same 8 pt text as a 170 mm one; what gives instead is the
LAYOUT — below ``STACK_BELOW_MM`` the side-by-side pairs stack and the three
key-phase boxes become three rows, because the alternative is 6 pt labels
nobody can read in print. The page is authored at the width it will be PLACED at:
insert it at 100% and 8 pt here is 8 pt on the page.

Usage:
    python figures/msca_fig3.py [out_stem] [--width-mm 170]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import palette as P                                                # noqa: E402
from hm_rat_analysis import maze                                   # noqa: E402

OUT_DIR = Path(os.environ.get("MSCA_FIG_DIR",
                              "/Users/sachuriga/Desktop/MSCA_figures"))

#: Panel B is a PHOTOGRAPH and the one part of this figure no code can regenerate;
#: it was lifted out of the Illustrator original. Missing, the panel draws a
#: placeholder and says so rather than failing the whole figure.
PHOTO = OUT_DIR / "fig3_panelB.png"

MM = 1 / 25.4
#: Below this page width the side-by-side pairs stack and panel D becomes three
#: rows. At 110 mm, three KP boxes across the page are 32 mm each and an 8 pt
#: header needs 44 mm — so either the type breaks its floor or the layout gives,
#: and the layout is the thing that is allowed to give.
STACK_BELOW_MM = 100.0

# type sizes in POINTS, floored by the shared palette
FS = P.scale({"letter": 10.5, "title": 9.0, "body": 8.0, "small": 8.0,
              "tiny": 8.0})

C_EBN, C_LBN = P.GREEN_DARK, P.ORANGE
C_KP1, C_KP2, C_KP3 = P.BLUE, P.AMBER, P.RED
INKC = P.TEXT_ON_SURFACE


def wrap_to(s, avail_mm, pt_size, weight="normal"):
    """`s` wrapped to `avail_mm`, measured from the font's real metrics.

    `weight` matters: bold runs several percent wider than regular at the same
    size, and every wrapped string in these figures that sits in a tight box —
    panel titles, the bracket captions in figure 2 panel C — is bold.
    """
    return P.wrap_mm(s, avail_mm, pt_size, weight=weight)


def build_figure(width_mm=180.0):
    """The whole figure, authored for a page `width_mm` wide."""
    stacked = width_mm < STACK_BELOW_MM
    plt.rcParams.update({
        "font.family": "serif", "font.serif": P.SERIF_STACK,
        "mathtext.fontset": "stix", "font.size": FS["body"],
        "axes.linewidth": 0.6, **P.VECTOR_TEXT,
    })

    M = 5.0                                   # page margin, mm
    inner = width_mm - 2 * M
    GAP = 4.5                                 # gap between blocks, mm
    T_H = 4.8                                 # room a panel title needs, mm

    # ---- block heights, in mm. A photo and a maze have fixed aspects, so their
    # heights follow from the width they are given rather than being guessed.
    # Panel C holds five stacked bands of fixed-height type: the phase heading,
    # its body text (which WRAPS, so its line count depends on the page width),
    # the chips, the axis, and the day labels. The height is counted from the
    # wrap rather than guessed, or the body prints over its own heading.
    C_PHASES = (("Build-up phase · Days 1–25",
                 "New information integrated into a stable schema"),
                ("Testing phase · Days 25–27",
                 "New memory replayed; old schema updated"))
    _cw = (0.455 if True else 0) * (width_mm - 2 * 5.0) - 6.0
    n_phase = max(wrap_to(b, _cw, 8.0).count("\n") + 1 for _h, b in C_PHASES)
    # every band, in 8 pt lines: the box (heading + wrapped body), a gap, the KP
    # chips, a gap, the axis, and the day labels under it
    h_c = 3.4 * (1.35 + 1.3 * n_phase + 0.6 + 1.4 + 0.6 + 0.3 + 1.3 + 0.4)
    h_d_row = 14.0
    h_d = h_d_row if not stacked else h_d_row * 3 + 2.0 * 2
    w_ef = inner if stacked else inner * 0.455
    h_ef_one = 38.0
    h_ef = (h_ef_one * 2 + T_H + GAP) if stacked else h_ef_one

    height_mm = (M + T_H + h_c + GAP + T_H + h_d + GAP + T_H + h_ef + M)
    W, H = width_mm * MM, height_mm * MM
    fig = plt.figure(figsize=(W, H), facecolor=P.SURFACE)

    def rect(x_mm, y_mm, w_mm_, h_mm_):
        """A panel rectangle given in MM from the page's bottom-left."""
        return fig.add_axes([x_mm / width_mm, y_mm / height_mm,
                             w_mm_ / width_mm, h_mm_ / height_mm])

    def ftitle(letter, text, x_mm, y_mm):
        fig.text(x_mm / width_mm, y_mm / height_mm, letter.upper(),
                 fontsize=FS["letter"], fontweight="bold", va="bottom",
                 ha="left", color=P.INK)
        fig.text((x_mm + 4.6) / width_mm, y_mm / height_mm, text,
                 fontsize=FS["title"], fontweight="bold", va="bottom",
                 ha="left", color=P.INK)

    def rbox(ax, x, y, w, h, fc, r=0.03, ec="none", lw=0.6, z=1):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, transform=ax.transAxes,
            zorder=z))

    def dot(ax, x, y, size_in, face, edge, lw=0.6, z=4):
        """A round marker of fixed PHYSICAL diameter. scatter, not Circle: a
        Circle in axes coordinates is stretched into an ellipse by the panel's
        aspect."""
        ax.scatter([x], [y], s=(size_in * 72.0) ** 2, marker="o",
                   transform=ax.transAxes, facecolors=face, edgecolors=edge,
                   linewidths=lw, zorder=z, clip_on=False)

    # ---- running y cursor, from the top of the page downwards
    y = height_mm - M
    y += T_H          # the first panel's title is taken off again below

    # Panels A and B are gone: the maze and the room photograph now lead FIGURE 1,
    # where the place cells are drawn on that same lattice. Repeating them here
    # would have cost a third of this page to say something already said.
    # ============================================================ C  design
    y -= GAP + T_H
    ftitle("a", "Experiment design", M, y + 0.6)
    axc = rect(M, y - h_c, inner, h_c)
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")

    #: The 25-day build-up and the 3-day testing phase share one axis but not one
    #: scale: at a common scale the three key phases would sit inside 2/27 of
    #: the width and their labels would overprint. The break mark says where the
    #: scale changes, so a compressed axis is not read as a linear one.
    BREAK_AT, BREAK_X = 24.0, 0.545

    def day_x(day):
        if day <= BREAK_AT:
            return 0.012 + (day - 1) / (BREAK_AT - 1) * (BREAK_X - 0.012)
        # the last chip is pulled in from the right edge by its own half-width, so
        # a nine-character name at 8 pt cannot hang off the page
        return BREAK_X + (day - BREAK_AT) / (27 - BREAK_AT) * (0.915 - BREAK_X)

    line_c = 3.4 / h_c                        # one 8 pt line, as a fraction of C
    box_h = line_c * (1.35 + 1.3 * n_phase)
    for x0, bw, col, (head, body) in ((0.005, 0.455, C_KP1, C_PHASES[0]),
                                      (0.500, 0.495, C_KP2, C_PHASES[1])):
        rbox(axc, x0, 1.0 - box_h, bw, box_h, P.TINT[col], r=0.05)
        avail = bw * inner - 6.0
        axc.text(x0 + 0.025, 1.0 - line_c * 0.30,
                 wrap_to(head, avail, FS["body"], "bold"),
                 transform=axc.transAxes, fontsize=FS["body"],
                 color=INKC[col], va="top", fontweight="bold")
        axc.text(x0 + 0.025, 1.0 - line_c * 1.45, wrap_to(body, avail, FS["small"]),
                 transform=axc.transAxes, fontsize=FS["small"],
                 color=INKC[col], va="top", linespacing=1.3)

    # Chips and axis are placed BELOW the box, measured from it. Fixed fractions
    # are what let the chips slide under the phase text once the body wrapped to
    # two lines and the box grew downwards.
    Y_CHIP = 1.0 - box_h - line_c * 1.0
    Y_AX = Y_CHIP - line_c * 1.5
    axc.annotate("", xy=(0.995, Y_AX), xytext=(0.0, Y_AX),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", lw=0.9, color=P.MUTED))
    for off in (-0.006, 0.006):
        axc.plot([BREAK_X + off - 0.005, BREAK_X + off + 0.005],
                 [Y_AX - line_c * 0.25, Y_AX + line_c * 0.25], transform=axc.transAxes,
                 color=P.MUTED, lw=0.8, zorder=5, clip_on=False)
    dot(axc, day_x(1), Y_AX, 0.030, P.INK, P.INK, lw=0, z=5)
    # left-aligned, not centred: at day 1 the tick is at the very left edge, so a
    # centred label puts half of itself off the page
    axc.text(0.0, Y_AX - line_c * 0.35, "Day 1", transform=axc.transAxes,
             ha="left", va="top", fontsize=FS["small"], color=P.INK2)
    for day, col in ((25, C_KP1), (26, C_KP2), (27, C_KP3)):
        x = day_x(day)
        dot(axc, x, Y_AX, 0.034, col, col, lw=0, z=6)
        axc.text(x, Y_CHIP, f"KP{day - 24} · D{day}", transform=axc.transAxes,
                 ha="center", va="center", fontsize=FS["small"], color=INKC[col],
                 zorder=7, bbox=dict(boxstyle="round,pad=0.28",
                                     facecolor=P.TINT[col], edgecolor=col,
                                     linewidth=0.7))
        axc.plot([x, x], [Y_AX + 0.03, Y_CHIP - line_c * 0.6], transform=axc.transAxes,
                 color=col, lw=0.6, zorder=4)
        axc.text(x, Y_AX - line_c * 0.35, f"Day {day}", transform=axc.transAxes,
                 ha="center", va="top", fontsize=FS["small"], color=P.INK2)
    y -= h_c

    # ============================================================ D  prediction
    y -= GAP + T_H
    ftitle("b", "Predicted cfos/BrdU recruitment", M, y + 0.6)
    axd = rect(M, y - h_d, inner, h_d)
    axd.set_xlim(0, 1)
    axd.set_ylim(0, 1)
    axd.axis("off")

    # How many of five cells are recruited. The prediction is ORDINAL, so it is
    # counted out in cells rather than drawn as a bar, which would imply a number
    # nobody has yet. LBN is the UPPER layer and EBN the lower one, as they sit in
    # the tissue, so the panel reads as a slice rather than a table.
    KP_PANELS = (
        (C_KP1, "KP1: old goal (D25, s5)", 5, "active", 0, "silent"),
        (C_KP2, "KP2: new goal (D26, s1)", 3, "moderate", 5, "active ↑↑"),
        (C_KP3, "KP3: new goal (D27, s2)", 5, "active ↑↑", 1, "fading"),
    )
    n_col = 1 if stacked else 3
    bw = 1.0 if stacked else 0.3167
    gap_x = 0.0 if stacked else 0.025
    box_h = (1.0 / 3) - 0.02 if stacked else 0.94
    cell_h = 0.175 if not stacked else 0.175 / 3

    def pyramidal(ax, x, yc, h, colour, filled, panel_w_mm, panel_h_mm, lw=0.7):
        """A pyramidal cell in AXES coordinates, `h` tall in axes-y.

        The soma half-width is converted through the panel's real proportions, or
        it comes out as a sliver on a wide panel and a blob on a narrow one.
        """
        w = h * 0.42 * panel_h_mm / panel_w_mm
        ax.add_patch(plt.Polygon([[x, yc + h * 0.42], [x - w, yc - h * 0.18],
                                  [x + w, yc - h * 0.18]], closed=True,
                                 facecolor=colour if filled else "none",
                                 edgecolor=colour, linewidth=lw,
                                 transform=ax.transAxes, zorder=4,
                                 joinstyle="round"))
        ax.plot([x, x], [yc + h * 0.42, yc + h * 0.90], transform=ax.transAxes,
                color=colour, lw=lw, zorder=4, solid_capstyle="round")
        for dx_ in (-w * 0.85, w * 0.85):
            ax.plot([x, x + dx_], [yc + h * 0.90, yc + h * 1.16],
                    transform=ax.transAxes, color=colour, lw=lw * 0.85, zorder=4,
                    solid_capstyle="round")
        for dx_ in (-w * 0.95, w * 0.95):
            ax.plot([x, x + dx_], [yc - h * 0.18, yc - h * 0.50],
                    transform=ax.transAxes, color=colour, lw=lw * 0.85, zorder=4,
                    solid_capstyle="round")
        ax.plot([x, x], [yc - h * 0.18, yc - h * 0.78], transform=ax.transAxes,
                color=colour, lw=lw * 0.7, zorder=4, solid_capstyle="round")

    for i, (col, head, ebn_n, ebn_t, lbn_n, lbn_t) in enumerate(KP_PANELS):
        if stacked:
            x0, y0 = 0.0, 1.0 - (i + 1) * (1.0 / 3) + 0.01
        else:
            x0, y0 = i * (bw + gap_x), 0.03
        rbox(axd, x0, y0, bw, box_h, P.TINT[col], r=0.035)
        axd.text(x0 + bw / 2, y0 + box_h * 0.90, head, transform=axd.transAxes,
                 ha="center", va="center", fontsize=FS["small"], color=INKC[col],
                 zorder=4)
        for row, (lab, n, note, ccol) in enumerate((("LBN", lbn_n, lbn_t, C_LBN),
                                                    ("EBN", ebn_n, ebn_t, C_EBN))):
            yy = y0 + box_h * (0.59 - row * 0.38)
            axd.text(x0 + bw * 0.065, yy, lab, transform=axd.transAxes,
                     ha="left", va="center", fontsize=FS["body"], color=ccol,
                     zorder=4)
            for k in range(5):
                pyramidal(axd, x0 + bw * (0.23 + k * 0.095), yy,
                          cell_h * (box_h / 0.94), ccol, k < n, inner, h_d)
            # right-ALIGNED at the fill's inner edge: a wide status like
            # 'active ↑↑' then runs inwards instead of over the last cell
            axd.text(x0 + bw - bw * 0.045, yy, note, transform=axd.transAxes,
                     ha="right", va="center", fontsize=FS["small"],
                     color=ccol if n > 0 else P.MUTED, zorder=4)
        axd.plot([x0 + bw * 0.06, x0 + bw * 0.94],
                 [y0 + box_h * 0.40, y0 + box_h * 0.40], transform=axd.transAxes,
                 color=P.MUTED, lw=0.5, ls=(0, (2.4, 2.0)), zorder=2)
    y -= h_d

    # ============================================================ E / F
    y -= GAP + T_H
    ftitle("c", "Predicted learning dynamics", M, y + 0.6)
    if stacked:
        axe = rect(M + 9.0, y - h_ef_one, w_ef - 12.0, h_ef_one - 9.0)
        ftitle("d", "Memory trajectory, EBN–LBN",
               M, y - h_ef_one - GAP - T_H + 0.6)
        axf = rect(M + 9.0, y - h_ef_one - GAP - T_H - h_ef_one + 9.0,
                   w_ef - 12.0, h_ef_one - 9.0)
    else:
        ftitle("d", "Memory trajectory, EBN–LBN", M + inner - w_ef, y + 0.6)
        axe = rect(M + 9.0, y - h_ef + 9.0, w_ef - 12.0, h_ef - 9.0)
        axf = rect(M + inner - w_ef + 9.0, y - h_ef + 9.0, w_ef - 12.0,
                   h_ef - 9.0)

    tt = np.linspace(0, 10, 800)
    # Drawn shapes, not fits. LBN is a fast encoder: silent until the new goal,
    # a sharp peak, then decay. EBN is a slow integrator that holds the old-schema
    # baseline, is recruited late, and then settles BACK to that same baseline
    # once the new goal is absorbed — the level at the right edge is the level at
    # KP1, not a new plateau.
    T_NEW, EBN_BASE = 2.45, 0.40
    ebn = EBN_BASE + 0.30 * (1 / (1 + np.exp(-(tt - 4.3) / 0.85))) * (
        1 / (1 + np.exp((tt - 6.6) / 1.20)))
    rise = 1 / (1 + np.exp(-(tt - 2.88) / 0.115))
    lbn = 0.05 + 1.02 * rise * np.minimum(1.0, np.exp(-(tt - 3.02) / 2.15))
    lbn = np.where(tt < T_NEW - 0.10, np.nan, lbn)

    axe.axhline(EBN_BASE, color=C_EBN, lw=0.6, ls=(0, (2.2, 2.2)), alpha=0.55,
                zorder=1)
    axe.plot(tt, ebn, color=C_EBN, lw=1.7, zorder=4, solid_capstyle="round")
    axe.plot(tt, lbn, color=C_LBN, lw=1.7, zorder=4, solid_capstyle="round")
    axe.axvline(T_NEW, color=P.MUTED, lw=0.7, ls=(0, (3, 2.6)), zorder=1)
    ipk = int(np.nanargmax(lbn))
    axe.scatter([tt[ipk]], [lbn[ipk]], s=16, color=C_LBN, zorder=6,
                edgecolor=P.SURFACE, linewidth=0.8)
    axe.text(tt[ipk] + 0.25, lbn[ipk], "LBN peak", fontsize=FS["small"],
             color=C_LBN, va="center", ha="left")
    axe.text(5.15, float(ebn.max()) + 0.06, "slow integration",
             fontsize=FS["small"], color=C_EBN, va="bottom", ha="center")
    axe.annotate("back to the\nKP1 baseline", xy=(9.9, float(ebn[-1]) + 0.02),
                 xytext=(10.3, 0.70), fontsize=FS["small"], color=C_EBN,
                 va="bottom", ha="right", linespacing=1.3,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=C_EBN,
                                 connectionstyle="arc3,rad=0.25"))
    # short on purpose: the long form reached across the panel at 110 mm and
    # printed over the 'crossover' label
    axe.text(0.15, 0.335, "EBN baseline", fontsize=FS["small"], color=C_EBN,
             va="top", ha="left")
    # the crossover is FOUND, not placed, so the label cannot drift off it
    after = tt > tt[ipk]
    sign = np.sign(np.nan_to_num(lbn, nan=0.0) - ebn)
    cross = np.flatnonzero(after[:-1] & (sign[:-1] > 0) & (sign[1:] <= 0))
    if cross.size:
        xc = float(tt[cross[0]])
        axe.plot([xc, xc], [0.02, float(ebn[cross[0]])], color=P.MUTED, lw=0.7,
                 ls=(0, (2, 2)), zorder=2)
        axe.text(xc + 0.16, 0.05, "crossover", fontsize=FS["small"],
                 color=P.INK2, ha="left", va="bottom")
    for x, name, col in ((0.95, "KP1", C_KP1), (T_NEW, "KP2", C_KP2),
                         (5.3, "KP3", C_KP3)):
        axe.text(x, 1.32, name, ha="center", va="center", fontsize=FS["small"],
                 color=INKC[col], zorder=7,
                 bbox=dict(boxstyle="round,pad=0.26", facecolor=P.TINT[col],
                           edgecolor=col, linewidth=0.7))
    axe.set_xlim(0, 10.6)
    axe.set_ylim(0, 1.48)
    axe.set_xticks([])
    axe.set_yticks([])
    axe.set_ylabel("Encoding", fontsize=FS["body"], color=P.INK2, labelpad=3)
    axe.set_xlabel("Time →", fontsize=FS["small"], color=P.INK2, labelpad=2)
    for s in ("top", "right"):
        axe.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axe.spines[s].set_color(P.MUTED)
    hl = [plt.Line2D([], [], color=C_EBN, lw=1.7),
          plt.Line2D([], [], color=C_LBN, lw=1.7)]
    axe.legend(hl, ["EBN, slow integrator", "LBN, fast encoding"],
               loc="upper center", bbox_to_anchor=(0.5, -0.06),
               ncol=1 if stacked else 2, frameon=False, fontsize=FS["small"],
               handlelength=1.4, columnspacing=1.1, labelcolor=P.INK2,
               handletextpad=0.5)

    # ---- F
    axf.set_xlim(0, 1)
    axf.set_ylim(0, 1)
    axf.add_patch(Rectangle((0.02, 0.50), 0.66, 0.48, facecolor=C_EBN,
                            alpha=0.10, edgecolor="none", zorder=1))
    axf.plot([0.02, 0.98], [0.02, 0.98], color=P.INK, lw=0.9,
             ls=(0, (3.4, 2.6)), zorder=2)
    KP1_XY, KP2_XY, KP3_XY = (0.165, 0.775), (0.840, 0.300), (0.560, 0.820)
    #: circle diameter in INCHES, sized from the type it has to hold
    r_in = FS["tiny"] * 3.4 / 72.0
    for xy, name, col, face in ((KP1_XY, "KP1", C_KP1, "none"),
                                (KP3_XY, "KP3", C_KP3, "none"),
                                (KP2_XY, "KP2", C_KP2, P.TINT[C_KP2])):
        dot(axf, *xy, r_in, face, col, lw=1.2, z=5)
        axf.text(*xy, name, transform=axf.transAxes, ha="center", va="center",
                 fontsize=FS["tiny"], color=INKC[col], zorder=6)
    ax_f_w = (w_ef - 12.0)
    ax_f_h = (h_ef_one - 9.0)
    rx = r_in * 25.4 / 2 / ax_f_w + 0.012
    ry = r_in * 25.4 / 2 / ax_f_h + 0.012

    def arcto(p0, p1, colour, rad):
        # ends pulled back to each circle's RIM, so no arrowhead is swallowed by
        # a ring or left hanging in space
        d = np.array(p1) - np.array(p0)
        n = np.hypot(d[0] / rx, d[1] / ry) or 1.0
        off = np.array([d[0] / n, d[1] / n])
        axf.annotate("", xy=tuple(np.array(p1) - off),
                     xytext=tuple(np.array(p0) + off), xycoords="axes fraction",
                     textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", lw=1.0, color=colour,
                                     connectionstyle=f"arc3,rad={rad}"), zorder=4)

    arcto(KP1_XY, KP2_XY, P.MUTED, -0.30)
    arcto(KP2_XY, KP3_XY, C_KP3, -0.34)
    arcto(KP3_XY, KP1_XY, C_EBN, 0.30)
    axf.text(0.50, 0.435, "new goal", transform=axf.transAxes,
             fontsize=FS["small"], color=P.MUTED, ha="center", va="top")
    # hung DOWN from the panel's top edge rather than up from the arrow: at
    # 110 mm an 8 pt line placed above the arrow clears the axes entirely
    axf.text(0.365, 0.995, "updating", transform=axf.transAxes,
             fontsize=FS["small"], color=C_EBN, ha="center", va="top")
    # clear of KP1's ring: the ring is a scatter marker, so no text-overlap check
    # will catch a label parked on top of it — its clearance has to be built in
    axf.text(0.035, 0.515, "EBN active,\nLBN silent", transform=axf.transAxes,
             fontsize=FS["small"], color=C_EBN, ha="left", va="bottom",
             linespacing=1.3)
    axf.text(KP2_XY[0], KP2_XY[1] - 0.165, "novel", transform=axf.transAxes,
             fontsize=FS["small"], color=INKC[C_KP2], ha="center", va="top")
    axf.set_xticks([])
    axf.set_yticks([])
    axf.set_xlabel("Encoding in LBN", fontsize=FS["body"], color=C_LBN,
                   labelpad=2)
    axf.set_ylabel("Encoding in EBN", fontsize=FS["body"], color=C_EBN,
                   labelpad=3)
    for s in ("top", "right"):
        axf.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axf.spines[s].set_color(P.MUTED)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="fig3",
                    help="output stem (default %(default)s). A bare name lands in "
                         f"{OUT_DIR}; set MSCA_FIG_DIR to move that")
    ap.add_argument("--width-mm", type=float, default=180.0,
                    help="page width in MILLIMETRES (default %(default).0f = A4 "
                         "less 20 mm margins). Type stays at or above "
                         f"{P.MIN_PT:g} pt at any width; the LAYOUT gives instead, "
                         f"stacking below {STACK_BELOW_MM:.0f} mm")
    a = ap.parse_args(argv)
    stem = Path(a.out)
    if not stem.is_absolute() and stem.parent == Path("."):
        stem = OUT_DIR / stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=a.width_mm)
    # No bbox_inches='tight': the page is authored at exactly --width-mm, and
    # trimming to the ink would make the placed width depend on what was drawn.
    for ext, extra in (("pdf", {}), ("svg", {}), ("png", dict(dpi=600))):
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor=P.SURFACE, **extra)
    w, h = fig.get_size_inches()
    print(f"wrote {stem}.{{pdf,svg,png}}  ({w * 25.4:.0f} x {h * 25.4:.0f} mm)")
    if not PHOTO.exists():
        print(f"NOTE: panel B is a placeholder — no photograph at {PHOTO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
