"""MSCA proposal, Figure 3: the HexMaze, the schema-update design, and what the
cfos/BrdU readout is predicted to do.

    A  the HexMaze layout, with the cue locations and the goal
    B  the maze as built, in the room
    C  experiment design: a 25-day build-up phase, then a 3-day testing phase
    D  predicted cfos/BrdU recruitment at each of the three killing points
    E  predicted learning dynamics: EBN as a slow integrator, LBN as fast encoding
    F  the same prediction as a trajectory in EBN-LBN state space

This is a REDRAW of a figure that was previously assembled by hand in Illustrator
(Fig3.ai). Nothing in panels C-F is fitted to data: they are the hypothesis drawn
to scale, and the curves in E are shapes chosen to state a claim, not a model.
Panel A is the real lattice from ``hm_rat_analysis.maze`` rather than a traced
approximation, so the maze here and the maze in figures 1 and 2 are the same
object. Panel B is the one thing no code can regenerate — it is a photograph,
carried over from the Illustrator file (see PHOTO).

Type is Times New Roman throughout and the colour system is figure 1's, so the
three proposal figures read as one document. The page is 170 mm wide — A4 less
20 mm margins — and every type size is in POINTS, so placing it at 100% puts 8 pt
here on the page as 8 pt. Rescaling it in Word scales the type with it and undoes
that; set the picture width to 17 cm instead.

Usage:
    python figures/msca_fig3.py [out_stem] [--photo PATH]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hm_rat_analysis import maze                                   # noqa: E402

#: Where the finished figures go — the same folder msca_fig1a.OUT_DIR names, kept
#: as its own line so this script stays standalone.
OUT_DIR = Path(os.environ.get("MSCA_FIG_DIR",
                              "/Users/sachuriga/Desktop/MSCA_figures"))

#: Panel B is a PHOTOGRAPH and the one part of this figure no code can regenerate;
#: it was lifted out of the Illustrator original. If the file is missing the panel
#: draws a placeholder and says so, rather than failing the whole figure.
PHOTO = OUT_DIR / "fig3_panelB.png"

MM = 1 / 25.4
W, H = 170 * MM, 186 * MM

# ---------------------------------------------------------------- palette
# Figure 1's colours and neutrals, unchanged. There they separate ANIMALS; here
# they separate the three killing points and the two cell populations — one hue
# per thing the reader has to track, on the same warm neutral ramp.
INK, INK2, MUTED, SURFACE, MAZE_LINE = ("#0b0b0b", "#52514e", "#8a8985",
                                        "#fcfcfb", "#c9c8c2")
BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED = "#d64545"                       # the same red figure 2 gives the barrier
FILL, RULE = "#f2f1ed", MAZE_LINE

C_EBN = "#17976a"      # early-born neurons: GREEN, one lightness step down so a
                       # 7 pt label clears 3:1 on the page
C_LBN = ORANGE         # late-born neurons
C_KP1, C_KP2, C_KP3 = BLUE, AMBER, RED      # the three killing points
C_AMBER_INK = "#8a6100"                     # amber is too pale to set text in
C_CUE, C_GOAL = BLUE, C_EBN                 # panel A markers

#: Tints of the killing-point hues, for the fills their boxes sit on. Lightness
#: says "this box belongs to KP2" while the saturated hue stays for the marker —
#: the same second-dimension-as-lightness rule figure 1 uses inside panel b.
TINT = {C_KP1: "#e5eefb", C_KP2: "#fbf2dc", C_KP3: "#fbe9e9"}
INKC = {C_KP1: C_KP1, C_KP2: C_AMBER_INK, C_KP3: C_KP3}

plt.rcParams.update({
    # Times New Roman throughout, with serif fallbacks so this still renders on a
    # machine without it (the metrics shift; nothing breaks).
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

FS = {"letter": 10.5, "title": 8.4, "body": 7.2, "small": 6.6, "tiny": 6.0}

fig = plt.figure(figsize=(W, H), facecolor=SURFACE)


def panel(x, y, w, h):
    return fig.add_axes([x, y, w, h])


def ftitle(letter, text, x, y):
    """Panel letter and title on one baseline. The letter is UPPERCASE in every
    figure of this proposal."""
    fig.text(x, y, letter.upper(), fontsize=FS["letter"], fontweight="bold",
             va="bottom", ha="left", color=INK)
    fig.text(x + 0.021, y, text, fontsize=FS["title"], fontweight="bold",
             va="bottom", ha="left", color=INK)


def rbox(ax, x, y, w, h, fc, ec="none", r=0.03, lw=0.6, z=1):
    """A rounded panel fill, in axes coordinates."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
        edgecolor=ec, linewidth=lw, transform=ax.transAxes, zorder=z))


def dot(ax, x, y, size_in, face, edge, lw=0.6, z=4):
    """A round marker of a fixed PHYSICAL diameter, in axes coordinates.

    ``scatter`` rather than ``Circle``: a Circle in axes coordinates is stretched
    into an ellipse by whatever aspect the panel happens to have, and these have
    to read as dots in every panel.
    """
    ax.scatter([x], [y], s=(size_in * 72.0) ** 2, marker="o", transform=ax.transAxes,
               facecolors=face, edgecolors=edge, linewidths=lw, zorder=z,
               clip_on=False)


def pyramidal(ax, x, y, h, colour, filled, lw=0.7, z=4):
    """A pyramidal cell at (x, y) in AXES coordinates, `h` tall in axes-y units.

    Triangular soma, one apical dendrite branching into a tuft at the top, two
    basal dendrites and an axon below. `filled` paints the soma in `colour` — a
    recruited cell; an unfilled outline is a cell that stayed silent.

    Widths are converted through the panel's own proportions so the soma is a
    triangle rather than whatever the axes aspect would otherwise stretch it into.
    """
    # one axes-y unit is (H_D * H) inches and one axes-x unit is (0.930 * W), so a
    # width asked for in units of the HEIGHT has to be converted, or the soma comes
    # out as a sliver on a wide panel
    w = h * 0.42 * (H_D * H) / (0.930 * W)
    face = colour if filled else "none"
    ax.add_patch(plt.Polygon([[x, y + h * 0.42], [x - w, y - h * 0.18],
                              [x + w, y - h * 0.18]], closed=True,
                             facecolor=face, edgecolor=colour, linewidth=lw,
                             transform=ax.transAxes, zorder=z, joinstyle="round"))
    # apical dendrite and its tuft
    ax.plot([x, x], [y + h * 0.42, y + h * 0.90], transform=ax.transAxes,
            color=colour, lw=lw, zorder=z, solid_capstyle="round")
    for dx_ in (-w * 0.85, w * 0.85):
        ax.plot([x, x + dx_], [y + h * 0.90, y + h * 1.16], transform=ax.transAxes,
                color=colour, lw=lw * 0.85, zorder=z, solid_capstyle="round")
    # basal dendrites and axon
    for dx_ in (-w * 0.95, w * 0.95):
        ax.plot([x, x + dx_], [y - h * 0.18, y - h * 0.50], transform=ax.transAxes,
                color=colour, lw=lw * 0.85, zorder=z, solid_capstyle="round")
    ax.plot([x, x], [y - h * 0.18, y - h * 0.78], transform=ax.transAxes,
            color=colour, lw=lw * 0.7, zorder=z, solid_capstyle="round")


def chip(ax, x, y, text, colour, fs=None):
    """A killing-point chip: the name in its hue on a tint of it, outlined."""
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fs or FS["small"], color=INKC[colour], zorder=7,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=TINT[colour],
                      edgecolor=colour, linewidth=0.7))


def equal_limits(ax, w_frac, h_frac, span_x):
    """x/y limits that make one data unit square WITHOUT resizing the axes box.

    ``set_aspect('equal')`` would satisfy itself by shrinking the box, which would
    silently undo the page layout. Deriving the y span from the box's real
    proportions instead means equal aspect is already true when it is set, so the
    box stays exactly where the layout put it — and any spare room shows up as
    margin the panel can use for a scale bar.
    """
    return span_x * (h_frac * H) / (w_frac * W)


# --------------------------------------------------------------- geometry
# One vertical rhythm for the whole page: every panel's title sits the same
# distance above its axes, and the gaps between blocks are equal.
Y_AB, H_AB = 0.702, 0.250
Y_C, H_C = 0.548, 0.096
Y_D, H_D = 0.352, 0.158
Y_EF, H_EF = 0.078, 0.246
T_PAD = 0.010

# ================================================================ A  the maze
axa = panel(0.045, Y_AB, 0.495, H_AB)
G = maze.build_graph()
ideal = maze.idealised_positions(G)
# y negated, exactly as figure 2 panel D draws it, so the maze has one orientation
# across the whole proposal
pos = {n: (float(p[0]), -float(p[1])) for n, p in ideal.items()}
xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]

for u, v in G.edges():
    axa.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=MAZE_LINE,
             lw=2.1, zorder=1, solid_capstyle="round")
axa.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
            s=3.0, facecolor=SURFACE, edgecolor=MAZE_LINE, linewidths=0.5, zorder=2)

GOAL = "403"

#: The cues are landmarks standing BESIDE the maze, not positions on it — the palm
#: tree, the lantern and the rest that panel B shows standing around the
#: apparatus. Drawn on a maze node they would read as goal locations, which is the
#: one thing they are not. Placed just outside the maze's own extent, so being
#: off the maze is true by construction; the row BELOW the maze is left clear for
#: the scale bar.
CUE_GAP = 0.78
x0m, x1m, y0m, y1m = min(xs), max(xs), min(ys), max(ys)
wm, ymid = x1m - x0m, (y0m + y1m) / 2
CUES = [(x0m - CUE_GAP, ymid),                     # beside the left arm
        (x0m + 0.30 * wm, y1m + CUE_GAP),          # above, over the left half
        (x0m + 0.76 * wm, y1m + CUE_GAP),          # above, over the right half
        (x1m + CUE_GAP, ymid)]                     # beside the right arm

for cx_, cy_ in CUES:
    axa.scatter(cx_, cy_, s=30, facecolor=SURFACE, edgecolor=C_CUE, linewidths=1.1,
                zorder=5)
    axa.text(cx_, cy_ + 0.26, "C", ha="center", va="bottom", fontsize=FS["small"],
             color=C_CUE, fontweight="bold", zorder=6)
axa.scatter(*pos[GOAL], s=42, color=C_GOAL, edgecolor=SURFACE, linewidths=0.9,
            zorder=6)
axa.text(pos[GOAL][0], pos[GOAL][1] + 0.26, "G", ha="center", va="bottom",
         fontsize=FS["small"], color=C_GOAL, fontweight="bold", zorder=7)

# The scale bar measures the MAZE; the limits have to hold the cues as well.
maze_x0, maze_x1 = x0m, x1m
xs = xs + [c[0] for c in CUES]
ys = ys + [c[1] for c in CUES]
SPAN_X = (max(xs) - min(xs)) + 0.7
span_y = equal_limits(axa, 0.495, H_AB, SPAN_X)
cx = (min(xs) + max(xs)) / 2
# the maze sits in the TOP of the panel; the leftover y is the scale bar's
y_top = max(ys) + 0.28
axa.set_xlim(cx - SPAN_X / 2, cx + SPAN_X / 2)
axa.set_ylim(y_top - span_y, y_top)
axa.set_aspect("equal")
axa.axis("off")

ybar = y_top - span_y + 0.42
axa.annotate("", xy=(maze_x1, ybar), xytext=(maze_x0, ybar),
             arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK2,
                             shrinkA=0, shrinkB=0))
axa.text((maze_x0 + maze_x1) / 2, ybar - 0.10, "9 m", ha="center", va="top",
         fontsize=FS["small"], color=INK2)
axa.text(0.995, 0.985, "C  cue location", transform=axa.transAxes, ha="right",
         va="top", fontsize=FS["small"], color=C_CUE)
axa.text(0.995, 0.895, "G  goal location", transform=axa.transAxes, ha="right",
         va="top", fontsize=FS["small"], color=C_GOAL)
ftitle("a", "HexMaze layout", 0.030, Y_AB + H_AB + T_PAD)

# ================================================================ B  the room
axb = panel(0.570, Y_AB, 0.400, H_AB)
axb.axis("off")


def draw_photo(ax, path):
    p = Path(path)
    if not p.exists():
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=FILL, edgecolor=RULE, linewidth=0.6))
        ax.text(0.5, 0.5, f"panel B photograph not found:\n{p.name}\n"
                          "pass --photo to point at it",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS["small"], color=MUTED, linespacing=1.5)
        return False
    ax.imshow(plt.imread(str(p)), aspect="equal", interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(RULE)
        s.set_linewidth(0.6)
    ax.axis("on")
    return True


draw_photo(axb, PHOTO)
ftitle("b", "The maze as built", 0.560, Y_AB + H_AB + T_PAD)

# ================================================================ C  design
axc = panel(0.045, Y_C, 0.930, H_C)
axc.set_xlim(0, 1)
axc.set_ylim(0, 1)
axc.axis("off")

#: The 25-day build-up and the 3-day testing phase share one axis but not one
#: scale: at a common scale the three killing points would sit inside 2/27 of the
#: width and their labels would overprint. The break mark on the arrow says where
#: the scale changes, so a compressed axis is not read as a linear one.
BREAK_AT, BREAK_X = 24.0, 0.545


def day_x(day):
    if day <= BREAK_AT:
        return 0.012 + (day - 1) / (BREAK_AT - 1) * (BREAK_X - 0.012)
    return BREAK_X + (day - BREAK_AT) / (27 - BREAK_AT) * (0.945 - BREAK_X)


rbox(axc, 0.005, 0.615, 0.455, 0.385, TINT[C_KP1], r=0.05)
axc.text(0.030, 0.895, "Build-up phase · Days 1–25", transform=axc.transAxes,
         fontsize=FS["body"], color=C_KP1, va="center", fontweight="bold")
axc.text(0.030, 0.715, "New information integrated into a stable schema",
         transform=axc.transAxes, fontsize=FS["small"], color=C_KP1, va="center")

rbox(axc, 0.500, 0.615, 0.495, 0.385, TINT[C_KP2], r=0.05)
axc.text(0.525, 0.895, "Testing phase · Days 25–27", transform=axc.transAxes,
         fontsize=FS["body"], color=C_AMBER_INK, va="center", fontweight="bold")
axc.text(0.525, 0.715, "New memory replayed; old schema updated",
         transform=axc.transAxes, fontsize=FS["small"], color=C_AMBER_INK,
         va="center")

# the chips sit in the clear band BETWEEN the phase boxes and the axis; putting
# them level with the boxes is what had KP1 printing over the testing-phase text
Y_AX, Y_CHIP = 0.155, 0.415
axc.annotate("", xy=(0.995, Y_AX), xytext=(0.0, Y_AX), xycoords="axes fraction",
             textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", lw=0.9, color=MUTED))
for off in (-0.006, 0.006):                       # the scale-break mark
    axc.plot([BREAK_X + off - 0.005, BREAK_X + off + 0.005],
             [Y_AX - 0.050, Y_AX + 0.050], transform=axc.transAxes, color=MUTED,
             lw=0.8, zorder=5, clip_on=False)
dot(axc, day_x(1), Y_AX, 0.030, INK, INK, lw=0, z=5)
axc.text(day_x(1), Y_AX - 0.10, "Day 1", transform=axc.transAxes, ha="center",
         va="top", fontsize=FS["small"], color=INK2)
for day, col in [(25, C_KP1), (26, C_KP2), (27, C_KP3)]:
    x = day_x(day)
    dot(axc, x, Y_AX, 0.034, col, col, lw=0, z=6)
    chip(axc, x, Y_CHIP, f"KP{day - 24} · D{day}", col)
    axc.plot([x, x], [Y_AX + 0.03, Y_CHIP - 0.10], transform=axc.transAxes,
             color=col, lw=0.6, zorder=4)
    axc.text(x, Y_AX - 0.10, f"Day {day}", transform=axc.transAxes, ha="center",
             va="top", fontsize=FS["small"], color=INK2)
ftitle("c", "Experiment design", 0.030, Y_C + H_C + T_PAD)

# ================================================================ D  prediction
axd = panel(0.045, Y_D, 0.930, H_D)
axd.set_xlim(0, 1)
axd.set_ylim(0, 1)
axd.axis("off")

# How many of five cells are recruited. The prediction is ORDINAL, not a
# measurement, so it is counted out in cells rather than drawn as a bar, which
# would imply a number nobody has yet. LBN is the UPPER layer and EBN the lower
# one, as they sit in the tissue, so the panel is a slice rather than a table.
KP_PANELS = [
    (C_KP1, "KP1 — old goal (D25, session 5)", 5, "active", 0, "silent"),
    (C_KP2, "KP2 — new goal (D26, session 1)", 3, "moderate", 5, "active ↑↑"),
    (C_KP3, "KP3 — new goal (D27, session 2)", 5, "active ↑↑", 1, "fading"),
]
BW, GAP = 0.3167, 0.025
CELL_H, N_CELLS = 0.175, 5
for i, (col, head, ebn_n, ebn_t, lbn_n, lbn_t) in enumerate(KP_PANELS):
    x0 = i * (BW + GAP)
    rbox(axd, x0, 0.03, BW, 0.94, TINT[col], r=0.035)
    axd.text(x0 + BW / 2, 0.885, head, transform=axd.transAxes, ha="center",
             va="center", fontsize=FS["small"], color=INKC[col], zorder=4)
    for row, (lab, n, note, ccol) in enumerate((("LBN", lbn_n, lbn_t, C_LBN),
                                                ("EBN", ebn_n, ebn_t, C_EBN))):
        yy = 0.585 - row * 0.360               # LBN on top, EBN beneath it
        axd.text(x0 + 0.020, yy, lab, transform=axd.transAxes, ha="left",
                 va="center", fontsize=FS["body"], color=ccol, zorder=4)
        for k in range(N_CELLS):
            pyramidal(axd, x0 + 0.072 + k * 0.030, yy, CELL_H, ccol,
                      filled=k < n)
        # The box is divided into three columns that cannot collide: the layer
        # name, the five cells, and the status right-ALIGNED at the fill's inner
        # edge. Left-aligning the status is what let 'active ↑↑' print over the
        # last cell, since its width depends on the word.
        axd.text(x0 + BW - 0.014, yy, note, transform=axd.transAxes, ha="right",
                 va="center", fontsize=FS["small"],
                 color=ccol if n > 0 else MUTED, zorder=4)
    # the layer boundary, so 'upper' and 'lower' are visible rather than implied
    axd.plot([x0 + 0.020, x0 + BW - 0.020], [0.405, 0.405],
             transform=axd.transAxes, color=MUTED, lw=0.5, ls=(0, (2.4, 2.0)),
             zorder=2)
ftitle("d", "Predicted cfos/BrdU recruitment at each killing point",
       0.030, Y_D + H_D + T_PAD)

# ================================================================ E  dynamics
axe = panel(0.100, Y_EF, 0.360, H_EF)
tt = np.linspace(0, 10, 800)

# Drawn shapes, not fits. LBN is a fast encoder: silent until the new goal
# appears, a sharp peak, then decay. EBN is a slow integrator: it holds the
# old-schema baseline, is recruited late, and then — this is the claim — settles
# BACK to that same baseline once the new goal has been absorbed into the schema,
# so the level at the right edge is the level at KP1, not a new plateau.
T_NEW, EBN_BASE = 2.45, 0.40
ebn = EBN_BASE + 0.30 * (1 / (1 + np.exp(-(tt - 4.3) / 0.85))) * (
    1 / (1 + np.exp((tt - 6.6) / 1.20)))
rise = 1 / (1 + np.exp(-(tt - 2.88) / 0.115))
lbn = 0.05 + 1.02 * rise * np.minimum(1.0, np.exp(-(tt - 3.02) / 2.15))
lbn = np.where(tt < T_NEW - 0.10, np.nan, lbn)

# the baseline drawn as a rule, so "returns to the KP1 level" is something the
# reader can check against a line rather than take on trust
axe.axhline(EBN_BASE, color=C_EBN, lw=0.6, ls=(0, (2.2, 2.2)), alpha=0.55,
            zorder=1)
axe.plot(tt, ebn, color=C_EBN, lw=1.7, zorder=4, solid_capstyle="round")
axe.plot(tt, lbn, color=C_LBN, lw=1.7, zorder=4, solid_capstyle="round")
axe.axvline(T_NEW, color=MUTED, lw=0.7, ls=(0, (3, 2.6)), zorder=1)

ipk = int(np.nanargmax(lbn))
axe.scatter([tt[ipk]], [lbn[ipk]], s=16, color=C_LBN, zorder=6,
            edgecolor=SURFACE, linewidth=0.8)
axe.text(tt[ipk] + 0.22, lbn[ipk], "LBN peak", fontsize=FS["small"], color=C_LBN,
         va="center", ha="left")
axe.text(5.15, float(ebn.max()) + 0.05, "slow integration", fontsize=FS["small"],
         color=C_EBN, va="bottom", ha="center")
axe.annotate("back to the\nKP1 baseline", xy=(9.9, float(ebn[-1]) + 0.02),
             xytext=(10.25, 0.72), fontsize=FS["small"], color=C_EBN, va="bottom",
             ha="right", linespacing=1.35,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=C_EBN,
                             connectionstyle="arc3,rad=0.25"))
axe.text(0.12, 0.335, "EBN baseline\n(old schema)", fontsize=FS["small"],
         color=C_EBN, va="top", ha="left", linespacing=1.35)

# the crossover is FOUND, not placed: the first time the decaying LBN drops
# through EBN, so the label cannot drift away from the thing it names
after = tt > tt[ipk]
sign = np.sign(np.nan_to_num(lbn, nan=0.0) - ebn)
cross = np.flatnonzero(after[:-1] & (sign[:-1] > 0) & (sign[1:] <= 0))
if cross.size:
    xc = float(tt[cross[0]])
    axe.plot([xc, xc], [0.02, float(ebn[cross[0]])], color=MUTED, lw=0.7,
             ls=(0, (2, 2)), zorder=2)
    axe.text(xc + 0.14, 0.06, "crossover", fontsize=FS["small"], color=INK2,
             ha="left", va="bottom")

for x, name, col in [(0.9, "KP1", C_KP1), (T_NEW, "KP2", C_KP2),
                     (5.2, "KP3", C_KP3)]:
    axe.text(x, 1.30, name, ha="center", va="center", fontsize=FS["small"],
             color=INKC[col], zorder=7,
             bbox=dict(boxstyle="round,pad=0.28", facecolor=TINT[col],
                       edgecolor=col, linewidth=0.7))

axe.set_xlim(0, 10.5)
axe.set_ylim(0, 1.44)
axe.set_xticks([])
axe.set_yticks([])
axe.set_ylabel("Encoding", fontsize=FS["body"], color=INK2, labelpad=3)
axe.set_xlabel("Time →", fontsize=FS["small"], color=INK2, labelpad=2)
for s in ("top", "right"):
    axe.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axe.spines[s].set_color(MUTED)
hl = [plt.Line2D([], [], color=C_EBN, lw=1.7),
      plt.Line2D([], [], color=C_LBN, lw=1.7)]
axe.legend(hl, ["EBN, slow integrator", "LBN, fast encoding"],
           loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2, frameon=False,
           fontsize=FS["small"], handlelength=1.5, columnspacing=1.2,
           labelcolor=INK2, handletextpad=0.5)
ftitle("e", "Predicted learning dynamics (WP2)", 0.030, Y_EF + H_EF + T_PAD)

# ================================================================ F  state space
axf = panel(0.605, Y_EF, 0.360, H_EF)
axf.set_xlim(0, 1)
axf.set_ylim(0, 1)

# where the OLD schema is doing the work: EBN high, LBN quiet
axf.add_patch(Rectangle((0.02, 0.50), 0.66, 0.48, facecolor=C_EBN, alpha=0.10,
                        edgecolor="none", zorder=1))
axf.plot([0.02, 0.98], [0.02, 0.98], color=INK, lw=0.9, ls=(0, (3.4, 2.6)),
         zorder=2)

KP1_XY, KP2_XY, KP3_XY = (0.165, 0.775), (0.840, 0.300), (0.560, 0.820)
#: Circle diameter in INCHES, wide enough to hold a three-character name at 6 pt.
#: Sized from the type rather than by eye: at 0.135 in the label overhung the ring.
R_IN = 0.215
for xy, name, col, face in [(KP1_XY, "KP1", C_KP1, "none"),
                            (KP3_XY, "KP3", C_KP3, "none"),
                            (KP2_XY, "KP2", C_KP2, TINT[C_KP2])]:
    dot(axf, *xy, R_IN, face, col, lw=1.2, z=5)
    axf.text(*xy, name, transform=axf.transAxes, ha="center", va="center",
             fontsize=FS["tiny"], color=INKC[col], zorder=6)

# Arrow ends are pulled back to each circle's RIM (r_ax) rather than to its
# centre, so no arrowhead is swallowed by the ring or left hanging in space.
r_ax_x = R_IN / 2 / (0.360 * W) + 0.012
r_ax_y = R_IN / 2 / (H_EF * H) + 0.012


def arc(p0, p1, colour, rad):
    d = np.array(p1) - np.array(p0)
    n = np.hypot(d[0] / r_ax_x, d[1] / r_ax_y) or 1.0
    off = np.array([d[0] / n, d[1] / n])
    axf.annotate("", xy=tuple(np.array(p1) - off), xytext=tuple(np.array(p0) + off),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=colour,
                                 connectionstyle=f"arc3,rad={rad}"), zorder=4)


arc(KP1_XY, KP2_XY, MUTED, -0.30)
axf.text(0.50, 0.435, "new goal", transform=axf.transAxes, fontsize=FS["small"],
         color=MUTED, ha="center", va="top")
arc(KP2_XY, KP3_XY, C_KP3, -0.34)
arc(KP3_XY, KP1_XY, C_EBN, 0.30)
axf.text(0.365, 0.945, "updating", transform=axf.transAxes, fontsize=FS["small"],
         color=C_EBN, ha="center", va="bottom")   # names the GREEN arrow below it
axf.text(0.035, 0.545, "EBN active,\nLBN silent", transform=axf.transAxes,
         fontsize=FS["small"], color=C_EBN, ha="left", va="bottom",
         linespacing=1.35)
axf.text(KP2_XY[0], KP2_XY[1] - 0.085, "novel", transform=axf.transAxes,
         fontsize=FS["small"], color=INKC[C_KP2], ha="center", va="top")

axf.set_xticks([])
axf.set_yticks([])
axf.set_xlabel("Encoding in LBN", fontsize=FS["body"], color=C_LBN, labelpad=2)
axf.set_ylabel("Encoding in EBN", fontsize=FS["body"], color=C_EBN, labelpad=3)
for s in ("top", "right"):
    axf.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axf.spines[s].set_color(MUTED)
ftitle("f", "Memory trajectory in EBN–LBN space", 0.545, Y_EF + H_EF + T_PAD)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="fig3",
                    help="output stem (default %(default)s). A bare name lands in "
                         f"{OUT_DIR}; set MSCA_FIG_DIR to move that, or give a "
                         "path of your own to bypass it")
    a = ap.parse_args(argv)
    stem = Path(a.out)
    if not stem.is_absolute() and stem.parent == Path("."):
        stem = OUT_DIR / stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches='tight': the page is authored at exactly 170 mm and trimming
    # it to the ink would make the placed width depend on what happened to be
    # drawn, which is what the fixed point sizes exist to avoid.
    for ext, extra in (("pdf", {}), ("svg", {}), ("png", dict(dpi=600))):
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor=SURFACE, **extra)
    print(f"wrote {stem}.{{pdf,svg,png}}  ({W / MM:.0f} x {H / MM:.0f} mm)")
    if not PHOTO.exists():
        print(f"NOTE: panel B is a placeholder — no photograph at {PHOTO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
