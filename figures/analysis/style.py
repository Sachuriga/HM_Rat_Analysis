"""Shared look, and the save helper that puts ONE plot in ONE file.

Every figure here is a single panel written to its own PDF and PNG in
``data.OUT_DIR``, so a plot can be dropped into a slide or a manuscript without
cropping it out of a sheet of panels. Type is in POINTS and floored at
``palette.MIN_PT``: a figure authored at the width it will be placed at keeps its
labels legible, and what gives on a narrow page is the layout, never the type.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

import palette as P                                               # noqa: E402
from analysis import data                                         # noqa: E402

MM = 1 / 25.4
#: One hue per animal, the same two the proposal figures use.
ANIMAL_COLOURS = {"Rat5": P.BLUE, "Rat6": P.ORANGE}
#: Extra x distance opened wherever the goal changes, in slot units.
GAP = 0.85

FS = {"title": 10.0, "label": 9.0, "tick": 8.0, "legend": 8.0, "note": 8.0}
FS = {k: max(P.MIN_PT, v) for k, v in FS.items()}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
})


def new_figure(width_mm=180.0, height_mm=70.0, left_mm=18.0, right_mm=3.0,
               top_mm=7.0, bottom_mm=20.0):
    """A single-panel figure sized in MM, with margins that hold real text.

    Margins are absolute, not fractions: the left one carries a 9 pt y label plus
    its ticks and the bottom one carries rotated slot labels, and neither shrinks
    with the page.
    """
    # No background: the figures are dropped onto slides and posters whose own
    # background is not this cream, and an opaque panel there reads as a pasted
    # rectangle. Everything that needs to be opaque paints itself.
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="none")
    ax = fig.add_axes([left_mm / width_mm, bottom_mm / height_mm,
                       1 - (left_mm + right_mm) / width_mm,
                       1 - (top_mm + bottom_mm) / height_mm])
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P.MUTED)
        ax.spines[s].set_linewidth(P.lw(0.8))
    ax.tick_params(length=2.5, pad=2, colors=P.MUTED, labelcolor=P.INK,
                   labelsize=FS["tick"])
    ax.grid(axis="y", color=P.MUTED, lw=P.lw(0.4), alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    # Kept so an inset can be placed in absolute mm without the call site having
    # to repeat the page size.
    fig.mm_size = (float(width_mm), float(height_mm))
    return fig, ax


def maze_inset(fig, colours, bridge_colour=None, bridge_dashes=None,
               width_mm=34.0, right_mm=2.0, top_mm=1.0, lw=1.0):
    """A mini HexMaze in the figure's top margin, each island in its own colour.

    The four islands are named "left" to "right" in the legend, which says where
    they are only if the reader already knows the maze is a zigzag: island 4 is
    top-left, 3 bottom-left-of-centre, 2 top-right-of-centre, 1 bottom-right.
    Drawing the map in the same colours removes that step. The bridges take the
    bridge line's own colour and dash, so all six series on a figure can be found
    on it.

    Edges rather than nodes: at 34 mm wide a 96-node scatter is a smudge, while
    the edges inside an island fuse into one readable block of colour. The
    geometry is the real one, from ``maze.node_positions_m``, so an island's
    position on the map is where the animal actually ran.

    The y axis is INVERTED. The node coordinates are derived from video pixels,
    where y counts downward from the top of the frame, so plotting them with y
    upward prints the maze mirrored about its long side. Inverting puts it back
    the way round the maze is. x is untouched, and has to be: it is what makes
    island 1 the right-hand one, which is what the legend calls it.

    Each island carries its number, placed clear of the lattice on the outer side
    so it does not sit on the lines.

    Call it AFTER the main axes exist, with a `top_mm` on `new_figure` large
    enough to hold it: it is placed in the margin, never over the data.
    """
    from hm_rat_analysis import maze

    G = maze.build_graph()
    pos = maze.node_positions_m()
    if not pos or G.number_of_edges() == 0:
        return None
    fw, fh = getattr(fig, "mm_size", (180.0, 70.0))
    x0, x1, y0, y1 = maze.MAZE_EXTENT
    pad = 0.55                       # metres of room for the island numbers
    h_mm = width_mm * ((y1 - y0) + 2 * pad) / (x1 - x0)
    ax = fig.add_axes([(fw - right_mm - width_mm) / fw,
                       1 - (top_mm + h_mm) / fh,
                       width_mm / fw, h_mm / fh])
    ax.set_facecolor("none")
    bridges = {frozenset((str(a), str(b))) for a, b in maze.bridge_edges(G)}
    for a, b in G.edges():
        ka, kb = int(a), int(b)
        if ka not in pos or kb not in pos:
            continue
        xs = (pos[ka][0], pos[kb][0])
        ys = (pos[ka][1], pos[kb][1])
        if frozenset((str(a), str(b))) in bridges:
            ax.plot(xs, ys, color=bridge_colour or P.INK, lw=P.lw(lw * 0.75),
                    dashes=bridge_dashes or (2.0, 1.4), zorder=2,
                    solid_capstyle="round")
        else:
            ax.plot(xs, ys, color=colours.get(maze.island_of_node(a)) or P.MUTED,
                    lw=P.lw(lw), zorder=3, solid_capstyle="round")
    # One label per island, on the side away from the middle of the maze so it
    # clears the lattice. Computed in DATA coordinates and then read through the
    # inverted axis, so "outward" stays outward on the page.
    mid = 0.5 * (y0 + y1)
    for k in maze.ISLANDS:
        pts = [pos[n] for n in pos if maze.island_of_node(n) == k]
        if not pts:
            continue
        cx = sum(q[0] for q in pts) / len(pts)
        cy = sum(q[1] for q in pts) / len(pts)
        # The anchor goes outside the island and the text grows AWAY from it.
        # Vertical alignment is resolved on the page, not in data coordinates, so
        # with y inverted an anchor above the island needs va="bottom" to sit
        # above it: "top" would hang the digit back down onto the lattice.
        if cy < mid:                 # upper on the page once y is inverted
            ty, va = min(q[1] for q in pts) - 0.18, "bottom"
        else:
            ty, va = max(q[1] for q in pts) + 0.18, "top"
        ax.text(cx, ty, str(k), ha="center", va=va, fontsize=FS["tick"],
                color=colours.get(k) or P.INK, clip_on=False)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1 + pad, y0 - pad)          # inverted: see the docstring
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def slot_positions(meta, gap=GAP):
    """x per slot, with a gap wherever the goal changes.

    Sessions restart at 1 inside every goal, so a flat 0..N-1 axis puts GL1S4
    next to GL2S1 with nothing to say a new block began.
    """
    pos, x = [], 0.0
    for i, (rep, _s) in enumerate(meta):
        if i and rep != meta[i - 1][0]:
            x += gap
        pos.append(x)
        x += 1.0
    return np.array(pos, float)


def first_exposure_slots(slots):
    """The slots where a goal is met for the FIRST time.

    ``GL2S1`` on an unsplit day, but ``GL2S1b`` where the day is split: on a split
    day the "a" trials still run to the PREVIOUS goal, so banding them would mark
    the old schema as if it were the new one.
    """
    out = set()
    for s in slots:
        if s.startswith("hab"):
            continue
        if s.endswith("S1b") or s.endswith("S1"):
            out.add(s)
    return out


def slot_axis(ax, slots, pos, band_slots=None, goals=None, animals=()):
    """GL{goal}S{session} on x, rotated, with `band_slots` shaded.

    `band_slots` defaults to the first exposure to each goal; pass an explicit set
    to band something else, or an empty set for none.

    `goals` is ``{slot: node}`` from :func:`data.slot_goals`. Where it is given
    the goal node goes on a second line of each tick, because "GL3" names the
    third goal without saying where it was: the two animals ran the same goal
    NUMBERS to different nodes, so a reader comparing this figure to the maze map
    needs the node and not the number. `animals` names the order the nodes are
    joined in, and goes in the axis title so the reader knows which is which.
    """
    band = first_exposure_slots(slots) if band_slots is None else set(band_slots)
    goals = goals or {}
    labels = [f"{s}\n{goals[s]}" if goals.get(s) else s for s in slots]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=FS["tick"], color=P.INK)
    if goals:
        who = ", ".join(animals) if animals else "goal node"
        ax.set_xlabel(f"session / goal node ({who})" if animals
                      else "session / goal node",
                      fontsize=FS["label"], color=P.INK, labelpad=2)
    ax.set_xlim(pos[0] - 0.8, pos[-1] + 0.8)
    for x, s in zip(pos, slots):
        if s in band:
            ax.axvspan(x - 0.46, x + 0.46, color=P.BLUE, alpha=0.10, lw=0, zorder=0)


def legend(ax, **kw):
    kw.setdefault("fontsize", FS["legend"])
    return ax.legend(frameon=False, labelcolor=P.INK, handlelength=1.2,
                     borderpad=0.15, **kw)


def save(fig, name, out_dir=None, dpi=600):
    """Write ONE plot to its own PDF and PNG, and say where it went.

    Both are written with a TRANSPARENT background, so the page or slide behind
    them shows through. Note what that does and does not cover: the figure and
    axes panels are clear, but anything drawn in an explicit light colour is
    still drawn, and on a dark background it stays light. That is the white ring
    around each marker, which is there to keep two animals' points apart where
    they coincide, and the white counts inside the bars, which sit on the bars
    rather than on the background.
    """
    out = data.OUT_DIR if out_dir is None else out_dir
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, extra in ((".pdf", {}), (".png", dict(dpi=dpi))):
        p = (out / name).with_suffix(suffix)
        fig.savefig(p, transparent=True, **extra)
        written.append(p)
    plt.close(fig)
    for p in written:
        print(f"  wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
    return written
