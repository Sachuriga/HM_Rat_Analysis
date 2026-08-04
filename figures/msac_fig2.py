"""i-CRADLE training schedule figure for MSCA Part B1.

170 mm wide, fully vector, type in the 7-11 pt band (same spec as msca_fig1.py).
  a  one recording day: 2 implanted + 6 non-implanted animals, maze never idle
  b  session cadence: implanted daily vs non-implanted spaced retrieval
  c  full experiment: 4 build-up goal locations, then the update phase
  d  update manipulation: barrier on the bridge nearest the goal forces a detour
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform

#: Where the finished figures go — the same folder msca_fig1a.OUT_DIR names, kept
#: as its own line here so this script stays standalone and does not have to
#: import the NWB-heavy figure 1 module just to learn where to write.
OUT_DIR = Path(os.environ.get("MSCA_FIG_DIR",
                              "/Users/sachuriga/Desktop/MSCA_figures"))

MM = 1 / 25.4
W = 170 * MM
HGT = 7.2
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.transparent": True,
})

# ---------------------------------------------------------------- palette
# One colour system with Figure 1, so the two read as one proposal rather than
# two unrelated pictures. Figure 1 spends its four validated hues on the ANIMAL,
# on a warm neutral ramp; this figure has no animals to separate, so the same
# four hues carry the ACTIVITY instead — one hue per thing that happens in a day
# — and every grey is taken from fig 1's ramp rather than from matplotlib's cool
# blue-greys, which is what made the two pages look like different documents.
INK, INK2, MUTED, SURFACE, MAZE_LINE = ("#0b0b0b", "#52514e", "#8a8985",
                                        "#fcfcfb", "#c9c8c2")
BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
#: a panel/box fill and its rule, on fig 1's warm neutral rather than a blue-grey
FILL, RULE = "#f2f1ed", MAZE_LINE

C_MAZE = ORANGE       # HexMaze navigation + recording — the hero activity
C_TRAIN = "#17976a"   # non-implanted training: GREEN, one lightness step down so
                      # 6 pt white numbers clear 3:1 against it
C_GL = BLUE           # a goal-location block
# Sleep is ONE activity that brackets the session, so it is one hue at two
# lightness steps — the same "lightness carries the second dimension" rule fig 1
# uses for good vs MUA units — pale before the maze, deep after it.
C_PRE, C_POST = "#a6c8ee", "#1c5596"
C_BAR = "#d64545"     # the barrier: the only red on either page, so the one
                      # element that BLOCKS something is never mistaken for the
                      # orange maze activity
C_GREY = INK2


def _lum(c):
    r, g, b = mcolors.to_rgb(c)
    f = lambda u: u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ink_on(color):
    """White or ink, whichever a 6 pt label can actually be read in.

    The palette now spans pale blue to deep blue, so a fixed white label is
    invisible on half of it.
    """
    return "white" if _lum(color) < 0.34 else INK

fig = plt.figure(figsize=(W, HGT))

def panel(x, y, w, h): return fig.add_axes([x, y, w, h])
L_LETTER, L_TITLE = 0.020, 0.055        # one vertical line for every panel
def ftitle(letter, text, y):
    fig.text(L_LETTER, y, letter, fontsize=10.5, fontweight="bold", va="bottom",
             ha="left", color=INK)
    fig.text(L_TITLE, y, text, fontsize=8.4, fontweight="bold", va="bottom",
             ha="left", color=INK)
def title(*a, **k):                      # kept so old calls are inert
    pass
def bar(ax, x0, x1, y, h, color, text=None, fs=7, tc=None):
    ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=color,
                           edgecolor="none", zorder=3))
    if text:
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=fs,
                color=ink_on(color) if tc is None else tc, zorder=4)

# ------------------------------------------------------------------ a
axa = panel(0.150, 0.770, 0.830, 0.175)
Y_SHIFT, Y_IMP, Y_NON, Y_OCC = 3.35, 2.35, 1.40, 0.42
H = 0.62

# shift bands
for x0, x1, t in [(9, 13.5, "morning shift"), (13.5, 18, "afternoon shift")]:
    axa.add_patch(Rectangle((x0, Y_SHIFT - 0.30), x1 - x0, 0.60, facecolor=FILL,
                            edgecolor=RULE, linewidth=0.5, zorder=2))
    axa.text((x0 + x1) / 2, Y_SHIFT, f"{t}  (2 students)", ha="center", va="center",
             fontsize=6.8, color=INK2, zorder=3)
axa.plot([13.5, 13.5], [-0.05, 3.05], color=MUTED, lw=0.7, ls="--", zorder=1)

# implanted, both animals on one row
bar(axa, 9, 11, Y_IMP, H, C_PRE, "pre-sleep 2 h")
bar(axa, 11.0, 12.0, Y_IMP, H, C_MAZE, "#1", fs=6.4)
bar(axa, 12.5, 13.5, Y_IMP, H, C_MAZE, "#2", fs=6.4)
bar(axa, 14, 18, Y_IMP, H, C_POST, "post-sleep 4 h")
axa.text(11.5, Y_IMP - 0.52, "maze, 1 h each", ha="center", va="top",
         fontsize=6.2, color=C_GREY)

# non-implanted, six runs on one row
slots = [9.25, 10.10, 14.25, 15.10, 15.95, 16.80]
for i, s in enumerate(slots):
    bar(axa, s, s + 1 / 3, Y_NON, H, C_TRAIN, f"#{i+1}", fs=6.0)
axa.text(9.72, Y_NON - 0.52, "2 runs", ha="center", va="top", fontsize=6.2, color=C_GREY)
axa.text(15.7, Y_NON - 0.52, "4 runs, 20 min each", ha="center", va="top",
         fontsize=6.2, color=C_GREY)

# maze occupancy
axa.text(8.92, Y_OCC, "HexMaze in use", ha="right", va="center", fontsize=6.6,
         color=C_GREY, style="italic")
for x0 in (11.0, 12.5):
    bar(axa, x0, x0 + 1.0, Y_OCC, 0.32, C_MAZE)
for s in slots:
    bar(axa, s, s + 1 / 3, Y_OCC, 0.32, C_TRAIN)

axa.set_xlim(8.88, 18.3); axa.set_ylim(0.05, 3.85)
axa.set_yticks([Y_SHIFT, Y_IMP, Y_NON])
axa.set_yticklabels(["Staffing", "Implanted #1-2", "Non-implanted #1-6"], fontsize=7)
axa.set_xticks(range(9, 19)); axa.set_xticklabels([f"{h}:00" for h in range(9, 19)], fontsize=7)
for sp in ("top", "right", "left"): axa.spines[sp].set_visible(False)
axa.spines["bottom"].set_color(MUTED)
axa.tick_params(axis="y", length=0, labelcolor=INK2)
axa.tick_params(axis="x", labelsize=7, colors=MUTED, labelcolor=INK2)
ftitle("a", "One recording day, run in two shifts: 2 implanted animals recorded, 6 non-implanted trained", 0.955)

hl = [Rectangle((0, 0), 1, 1, fc=c) for c in (C_PRE, C_MAZE, C_POST, C_TRAIN)]
fig.legend(hl, ["pre-sleep (sleep box)", "HexMaze navigation + recording",
                "post-sleep (sleep box)", "non-implanted training, 20 min"],
           loc="lower center", bbox_to_anchor=(0.565, 0.702), ncol=4, frameon=False,
           fontsize=6.9, handlelength=1.4, columnspacing=1.5, handleheight=0.85,
           labelcolor=INK2)

# ------------------------------------------------------------------ b
axb = panel(0.150, 0.545, 0.830, 0.105)
axb.axvspan(6.55, 13.45, color=FILL, zorder=0)
axb.text(2.5, 2.72, "week 1", ha="center", fontsize=7, color=C_GREY)
axb.text(9.5, 2.72, "week 2", ha="center", fontsize=7, color=C_GREY)
for d in range(5):
    bar(axb, d - 0.45, d + 0.45, 1.95, 0.55, C_MAZE, f"GL1S{d+1}", fs=5.7)
for d in range(7, 12):
    bar(axb, d - 0.45, d + 0.45, 1.95, 0.55, C_MAZE, f"GL2S{d-6}", fs=5.7)
for d, s in [(0, "GL1S1"), (2, "GL1S2"), (3, "GL1S3"), (10, "GL1S4"), (11, "GL1S5")]:
    bar(axb, d - 0.45, d + 0.45, 0.60, 0.55, C_TRAIN, s, fs=5.7)
for x0, x1, t in [(0, 2, "48 h"), (2, 3, "24 h"), (3, 10, "7 d"), (10, 11, "24 h")]:
    axb.annotate("", xy=(x1 - 0.45, 0.10), xytext=(x0 + 0.45, 0.10),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=C_GREY))
    axb.text((x0 + x1) / 2, -0.02, t, ha="center", va="top", fontsize=6.3, color=C_GREY)
axb.set_xlim(-0.85, 13.55); axb.set_ylim(-0.42, 3.0)
axb.set_xticks(range(14))
axb.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] * 2, fontsize=6.4)
axb.set_yticks([1.95, 0.60])
axb.set_yticklabels(["Implanted\n(every day)", "Non-implanted\n(spaced)"], fontsize=6.9)
for s in ("top", "right", "left"): axb.spines[s].set_visible(False)
axb.spines["bottom"].set_color(MUTED)
axb.tick_params(axis="y", length=0, labelcolor=INK2)
axb.tick_params(axis="x", length=2, colors=MUTED, labelcolor=INK2)
ftitle("b", "Session cadence: one session per animal per day. GL = goal location, S = session", 0.660)

# ------------------------------------------------------------------ c
axc = panel(0.150, 0.372, 0.830, 0.080)
blocks = [("GL1", C_GL), ("GL2", C_GL), ("GL3", C_GL), ("GL4", C_GL),
          ("barrier", C_BAR), ("GL5", C_GL), ("barrier", C_BAR), ("GL6", C_GL),
          ("barrier", C_BAR)]
for i, (name, c) in enumerate(blocks):
    bar(axc, i + 0.05, i + 0.95, 1.0, 0.60, c, name, fs=6.6)
    axc.text(i + 0.5, 0.56, "5 sessions", ha="center", va="top", fontsize=5.9, color=C_GREY)
for x0, x1, t, col in [(0, 4, "schema build-up: 4 goal locations", C_GL),
                       (4, 9, "update: barrier and new goal alternate", C_BAR)]:
    axc.plot([x0 + 0.05, x1 - 0.05], [1.62, 1.62], color=col, lw=0.9)
    for xx in (x0 + 0.05, x1 - 0.05):
        axc.plot([xx, xx], [1.55, 1.69], color=col, lw=0.9)
    axc.text((x0 + x1) / 2, 1.76, t, ha="center", fontsize=7, color=col, fontweight="bold")
axc.set_xlim(-0.1, 9.1); axc.set_ylim(0.30, 1.98); axc.axis("off")
ftitle("c", "Full experiment, implanted animal: 1 block = 1 week = 5 sessions", 0.462)

# ------------------------------------------------------------------ d
axd = panel(0.055, 0.055, 0.425, 0.265)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hm_rat_analysis import maze  # the lab's own maze module

G = maze.build_graph()
# regular honeycomb: edges snapped to 0/60/120 deg and whole segment multiples
ideal = maze.idealised_positions(G)
pos = {n: (float(xy[0]), -float(xy[1])) for n, xy in ideal.items()}
goal, start, br = "114", "306", ("121", "302")
H2 = G.copy(); H2.remove_edge(*br)
direct = nx.shortest_path(G, start, goal)
detour = nx.shortest_path(H2, start, goal)

# the maze drawn on fig 1's own maze neutrals, so the same lattice is the same
# colour in both figures: MAZE_LINE corridors, SURFACE-filled nodes
for u, v in G.edges():
    axd.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=MAZE_LINE,
             lw=0.9, zorder=1, solid_capstyle="round")
axd.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
            s=3.4, facecolor=SURFACE, edgecolor=MAZE_LINE, linewidths=0.5, zorder=2)
# The route the rat runs is the maze ACTIVITY, so it is C_MAZE — green here would
# have meant "non-implanted training" one panel up.
axd.plot([pos[n][0] for n in detour], [pos[n][1] for n in detour], color=C_MAZE,
         lw=1.9, zorder=3, solid_capstyle="round")
axd.plot([pos[n][0] for n in direct], [pos[n][1] for n in direct], color=INK,
         lw=1.2, ls=(0, (2.4, 1.8)), zorder=4)
bx = (pos[br[0]][0] + pos[br[1]][0]) / 2; by = (pos[br[0]][1] + pos[br[1]][1]) / 2
dx = pos[br[1]][0] - pos[br[0]][0]; dy = pos[br[1]][1] - pos[br[0]][1]
L = np.hypot(dx, dy); ux, uy = -dy / L, dx / L
axd.plot([bx - ux * 0.16, bx + ux * 0.16], [by - uy * 0.16, by + uy * 0.16],
         color=C_BAR, lw=3.0, zorder=6, solid_capstyle="butt")
for n, col, s, txt in [(goal, C_GL, 60, "G"), (start, INK, 34, "")]:
    axd.scatter([pos[n][0]], [pos[n][1]], s=s, color=col, zorder=7,
                edgecolor=SURFACE, linewidth=0.8)
axd.text(pos[goal][0], pos[goal][1], "G", ha="center", va="center", fontsize=6.2,
         color="white", fontweight="bold", zorder=8)
axd.text(pos[start][0], pos[start][1] + 0.16, "start", ha="center", va="bottom",
         fontsize=6.4, color=INK)
axd.annotate("barrier on the bridge\nnearest the goal", xy=(bx, by),
             xytext=(bx + 0.15, by + 0.95), fontsize=6.5, color=C_BAR, ha="center",
             va="bottom", linespacing=1.25,
             arrowprops=dict(arrowstyle="->", lw=0.8, color=C_BAR,
                             connectionstyle="arc3,rad=-0.25"))
axd.set_aspect("equal"); axd.axis("off")
axd.set_xlim(min(p[0] for p in pos.values()) - 0.25, max(p[0] for p in pos.values()) + 0.25)
axd.set_ylim(min(p[1] for p in pos.values()) - 1.05, max(p[1] for p in pos.values()) + 0.95)
ftitle("d", "Update manipulation", 0.330)
hd = [plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (2.4, 1.8))),
      plt.Line2D([], [], color=C_MAZE, lw=1.9)]
axd.legend(hd, [f"before barrier ({len(direct)-1} hops)",
                f"forced detour ({len(detour)-1} hops)"],
           loc="lower right", bbox_to_anchor=(1.02, -0.02), frameon=False,
           fontsize=6.4, handlelength=1.7, labelspacing=0.35, labelcolor=INK2)

# ------------------------------------------------------------------ e
axe = panel(0.545, 0.055, 0.435, 0.265); axe.axis("off")
axe.add_patch(Rectangle((0, 0), 1, 1, transform=axe.transAxes, facecolor=FILL,
                        edgecolor=RULE, linewidth=0.6))
axe.text(0.055, 0.935, "Throughput per recording day", transform=axe.transAxes,
         fontsize=8.0, fontweight="bold", va="top", color=INK)
items = [("2", "implanted animals recorded per day",
          "2 h pre-sleep, 1 h maze each, 4 h post-sleep"),
         ("6", "non-implanted animals trained per day",
          "20 min each, maze reset between runs"),
         ("12", "non-implanted animals in the pipeline",
          "trained on alternating days, so 12 run in parallel"),
         ("4 + 1", "people run the daily protocol",
          "4 students in 2 shifts, plus myself")]
y = 0.790
for k, t1, t2 in items:
    axe.text(0.075, y, k, transform=axe.transAxes, fontsize=9.0, fontweight="bold",
             color=C_MAZE, va="center", ha="left")
    axe.text(0.235, y, t1, transform=axe.transAxes, fontsize=7.1, va="center",
             color=INK)
    axe.text(0.235, y - 0.083, t2, transform=axe.transAxes, fontsize=6.2,
             color=C_GREY, va="center")
    y -= 0.165
axe.plot([0.055, 0.945], [0.175, 0.175], transform=axe.transAxes, color=RULE, lw=0.6)
axe.text(0.055, 0.045, "Non-implanted animals reach expert level before implantation, giving\n"
                       "a continuous pipeline and a trained reserve if an implant fails.",
         transform=axe.transAxes, fontsize=6.2, color=C_GREY, va="bottom",
         linespacing=1.4, style="italic")

# A bare name lands in OUT_DIR; a path of your own (or an absolute one) bypasses it.
out_stem = Path(sys.argv[1] if len(sys.argv) > 1 else "fig2")
if not out_stem.is_absolute() and out_stem.parent == Path("."):
    out_stem = OUT_DIR / out_stem
out_stem.parent.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "svg", "png"):
    fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=400,
                bbox_inches="tight", pad_inches=0.02)
print("saved %s.{pdf,svg,png}  direct=%d detour=%d"
      % (out_stem, len(direct) - 1, len(detour) - 1))
