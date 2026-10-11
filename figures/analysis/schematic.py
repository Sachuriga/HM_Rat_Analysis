"""Schematics that explain the task rather than report a result.

Three figures, drawn in the same style and colours as the data figures so a
reader moving between them does not have to relearn anything:

* :func:`schematic_maze` — what the apparatus is.
* :func:`schematic_trial` — what one trial is, and what the performance measure
  on every other figure actually measures.
* :func:`schematic_schedule` — how the sessions are arranged and when the goal
  changes.

The layout is the IDEALISED honeycomb from :func:`maze.idealised_positions`,
not the measured node positions. The measured ones are what the analysis uses,
and they wobble by up to 0.15 m, which is correct for a rate map and wrong for a
diagram: a schematic that shows a slightly irregular lattice invites the reader
to look for meaning in the irregularity. Distances quoted in the captions come
from the MEASURED positions, so the numbers stay true to the maze.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from hm_rat_analysis import behaviour, maze

from analysis import data, style
from analysis.style import ISLAND_COLOURS, ISLAND_DASH, lw as _lw

#: The goal nodes each animal was run to, in order. From the behavioural table
#: (``data.slot_goals``); hard-coded here only so the schematic can be drawn
#: without touching the server.
GOAL_NODES = {"Rat5": (410, 314, 107, 219), "Rat6": (204, 109, 318, 421)}

#: The example trial on the trial schematic. Rat6, 1 July 2026, trial 5: 24 steps
#: against an optimum of 14, from island 4 to island 1.
#:
#: Chosen on two criteria. It RETRACES NOTHING: every one of the 24 steps is a
#: different edge, so the inefficiency is visible as a shape. The first example
#: tried here was one where the animal went back and forth over 210-211-212, and
#: the six wasted steps drew on top of each other and vanished. And the two
#: routes differ in a way worth explaining: the shortest one crosses the 1.99 m
#: diagonal bridge, while the animal took the chain of short bridges around the
#: zigzag. That is the asymmetry the maze schematic points at, in one trial.
EXAMPLE = {"animal": "Rat6", "date": "20260701", "trial": 5,
           "start": 407, "goal": 109}


def _maze_axes(ax, pos, pad=0.72):
    """Equal aspect, no frame, y inverted to match the maze's real orientation.

    Limits come from the NODES plus `pad`, not from ``maze.MAZE_EXTENT``. The
    extent is the video frame and is wider than the maze on every side, so
    fitting to it both wastes a third of the panel and leaves the island labels,
    which sit outside the lattice, off the end of the axes.
    """
    xs = [q[0] for q in pos.values()]
    ys = [q[1] for q in pos.values()]
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)   # inverted: node y is video pixels
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_maze(ax, pos=None, G=None, colours=None, lw=1.6, node_ms=0.0,
              bridge_colour=None, bridge_dashes=None, faint=None):
    """The maze itself: island edges in colour, bridges dashed.

    `faint` draws everything in one muted colour instead, for a figure whose
    subject is drawn on top of the maze rather than the maze itself.
    """
    G = maze.build_graph() if G is None else G
    pos = maze.idealised_positions(G) if pos is None else pos
    colours = ISLAND_COLOURS if colours is None else colours
    bridges = {frozenset((str(a), str(b))) for a, b in maze.bridge_edges(G)}
    for a, b in G.edges():
        if a not in pos or b not in pos:
            continue
        xs = (pos[a][0], pos[b][0])
        ys = (pos[a][1], pos[b][1])
        if frozenset((str(a), str(b))) in bridges:
            ax.plot(xs, ys, color=faint or bridge_colour or style.P.INK,
                    lw=_lw(lw * 0.8), dashes=bridge_dashes or (2.8, 1.8),
                    zorder=2, solid_capstyle="round")
        else:
            c = faint or colours.get(maze.island_of_node(a)) or style.P.MUTED
            ax.plot(xs, ys, color=c, lw=_lw(lw), zorder=3,
                    solid_capstyle="round")
    if node_ms:
        xs = [pos[n][0] for n in G.nodes() if n in pos]
        ys = [pos[n][1] for n in G.nodes() if n in pos]
        cs = [faint or colours.get(maze.island_of_node(n)) or style.P.MUTED
              for n in G.nodes() if n in pos]
        ax.scatter(xs, ys, s=node_ms ** 2, c=cs, zorder=4, linewidths=_lw(0.6),
                   edgecolors="white")
    return pos, G


def _island_numbers(ax, pos, G, colours=None, dy=0.30, fs=None):
    """The island number, outside each island on the side away from the middle."""
    colours = ISLAND_COLOURS if colours is None else colours
    _x0, _x1, y0, y1 = maze.MAZE_EXTENT
    mid = 0.5 * (y0 + y1)
    for k in maze.ISLANDS:
        pts = [pos[n] for n in pos if maze.island_of_node(n) == k]
        if not pts:
            continue
        cx = sum(q[0] for q in pts) / len(pts)
        cy = sum(q[1] for q in pts) / len(pts)
        # Alignment is resolved on the PAGE, so with y inverted an anchor above
        # the island needs va="bottom" to stay above it.
        if cy < mid:
            ty, va = min(q[1] for q in pts) - dy, "bottom"
        else:
            ty, va = max(q[1] for q in pts) + dy, "top"
        ax.text(cx, ty, f"island {k}", ha="center", va=va,
                fontsize=fs or style.FS["tick"], color=colours.get(k),
                clip_on=False)


def _scale_bar(ax, length_m=1.0, x=None, y=None, label=None):
    """A metre bar, because a honeycomb gives no sense of size on its own."""
    xlo, xhi = ax.get_xlim()
    lo, hi = ax.get_ylim()                      # lo > hi: the axis is inverted
    x = (xhi - 1.25 * length_m) if x is None else x
    y = (lo - 0.06 * (lo - hi)) if y is None else y
    ax.plot([x, x + length_m], [y, y], color=style.P.INK, lw=_lw(1.4),
            solid_capstyle="butt", zorder=6)
    ax.text(x + length_m / 2, y - 0.12, label or f"{length_m:g} m",
            ha="center", va="bottom", fontsize=style.FS["tick"],
            color=style.P.INK)


def schematic_maze(trials_unused=None, name="schematic_maze", si=None):
    """The apparatus: four islands, five bridges, ninety-six nodes.

    The islands are drawn in the colours every other figure uses for them, and
    numbered, so this one diagram is the key to all of them. Bridges are dashed
    and labelled with their measured length: three are a single step longer than
    a within-island step, and two cut the diagonal at nearly three times that.
    That asymmetry is the reason a detour exists at all — the long bridges make
    some routes much shorter than they look.

    The goal nodes are marked with a ring per animal, in that animal's colour.
    The two animals never shared a goal, which is why the figures are drawn per
    animal.
    """
    fig, ax = style.new_figure(width_mm=180.0, height_mm=108.0, left_mm=3.0,
                               right_mm=3.0, top_mm=3.0, bottom_mm=4.0)
    pos, G = draw_maze(ax, node_ms=3.4)
    _maze_axes(ax, pos)
    _island_numbers(ax, pos, G, dy=0.30)

    # Bridge lengths from the MEASURED positions, not the drawn honeycomb.
    real = maze.node_positions_m()
    for a, b in maze.bridge_edges(G):
        if a not in pos or b not in pos:
            continue
        d = float(np.hypot(*(np.array(real[int(a)]) - np.array(real[int(b)]))))
        mx = 0.5 * (pos[a][0] + pos[b][0])
        my = 0.5 * (pos[a][1] + pos[b][1])
        ax.text(mx, my - 0.12, f"{d:.2f} m", ha="center", va="bottom",
                fontsize=style.FS["tick"] - 1, color=style.P.INK,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                          alpha=0.85), zorder=5)

    # Goal nodes, one ring per animal.
    for animal, nodes in GOAL_NODES.items():
        colour = style.ANIMAL_COLOURS.get(animal, style.P.INK)
        xs = [pos[str(n)][0] for n in nodes if str(n) in pos]
        ys = [pos[str(n)][1] for n in nodes if str(n) in pos]
        ax.scatter(xs, ys, s=9.5 ** 2, facecolors="none", edgecolors=colour,
                   linewidths=_lw(1.6), zorder=6,
                   label=f"{animal} goals: " + ", ".join(str(n) for n in nodes))
    ax.plot([], [], "-", color=style.P.INK, dashes=(2.8, 1.8), lw=_lw(1.3),
            label="bridge (3 short, 2 long)")
    _scale_bar(ax)
    style.legend(ax, loc="upper left", ncol=1)
    spacing = _within_island_spacing(G, real)
    print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"4 islands + 5 bridges. Within-island step {spacing:.2f} m; bridges "
          f"0.68, 0.69, 0.70, 1.86, 1.99 m. Layout is the idealised honeycomb; "
          f"all distances are from the measured positions.")
    return style.save(fig, name)


def _within_island_spacing(G, real):
    """Median length of an edge that stays on one island, in metres."""
    bridges = {frozenset((str(a), str(b))) for a, b in maze.bridge_edges(G)}
    d = [float(np.hypot(*(np.array(real[int(a)]) - np.array(real[int(b)]))))
         for a, b in G.edges() if frozenset((str(a), str(b))) not in bridges]
    return float(np.median(d)) if d else float("nan")


def _example_path(root=None):
    """``(sequence, start, goal)`` for :data:`EXAMPLE`, read from the sheet."""
    root = data.ROOT if root is None else root
    hits = [p for p in root.glob(f"*/{EXAMPLE['date']}")
            if p.parent.name.lower().startswith(EXAMPLE["animal"].lower())]
    if not hits:
        return None
    df, _src = behaviour._perf_meta_frame(hits[0])
    if df is None or "paths" not in df.columns:
        return None
    row = df.iloc[EXAMPLE["trial"] - 1]
    seq = [int(v) for v in str(row["paths"]).split(",")
           if v.strip().lstrip("-").isdigit()]
    return seq or None


def schematic_trial(trials_unused=None, name="schematic_trial", si=None):
    """One trial, and what the performance measure on the other figures means.

    A real trial, not a drawn one: Rat6, 1 July 2026, trial 5. The animal started
    at 407, on island 4, and the goal was 109, on island 1. The shortest route is
    fourteen steps and crosses the long diagonal bridge; the animal took
    twenty-four, following the short bridges around the zigzag instead. The
    measure every detour figure plots is log10(24/14) = 0.23.

    Why a log ratio and not a difference: ten extra steps on a fourteen-step
    journey is a different failure from ten extra on a forty-step one, and a
    ratio says so while a difference does not. Why base 10: 0.30 is exactly twice
    the optimum, which is a number a reader can hold.

    Two choices in the metric are visible here and are not cosmetic. The optimum
    is measured from the FIRST NODE OF THE RECORDED PATH, not from the
    ``Start_Nodes`` cell, because the two disagree on 6% of trials and the
    recorded one is where the animal actually was. And the path is counted up to
    the FIRST arrival at the goal, so an animal that reaches the goal and then
    wanders is not charged for the wandering.
    """
    seq = _example_path()
    fig, ax = style.new_figure(width_mm=180.0, height_mm=108.0, left_mm=3.0,
                               right_mm=3.0, top_mm=3.0, bottom_mm=4.0)
    pos, G = draw_maze(ax, faint=style.P.MUTED, lw=1.1, node_ms=2.6)
    _maze_axes(ax, pos)
    if seq is None:
        ax.text(0.5, 0.5, f"example session not found under {data.ROOT}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=style.FS["note"], color=style.P.MUTED)
        print(f"  {name}: example session missing; drew the maze only.")
        return style.save(fig, name)

    start, goal = str(seq[0]), str(EXAMPLE["goal"])
    walked = seq[:seq.index(int(goal)) + 1]
    best = nx.shortest_path(G, start, goal)

    def line(nodes, **kw):
        xs = [pos[str(n)][0] for n in nodes if str(n) in pos]
        ys = [pos[str(n)][1] for n in nodes if str(n) in pos]
        ax.plot(xs, ys, **kw)

    line(best, color=style.P.GREEN_DARK, lw=_lw(4.2), alpha=0.35, zorder=4,
         solid_capstyle="round",
         label=f"shortest route: {len(best) - 1} steps")
    line(walked, color=style.P.RED, lw=_lw(1.8), zorder=5,
         solid_capstyle="round", solid_joinstyle="round",
         label=f"route taken: {len(walked) - 1} steps")
    for node, mark, lab in ((start, "o", "start"), (goal, "*", "goal")):
        ax.plot(pos[node][0], pos[node][1], mark,
                ms=7.5 if mark == "o" else 13.0, color=style.P.INK,
                mec="white", mew=_lw(1.1), zorder=7, label=f"{lab} ({node})")
    ratio = np.log10((len(walked) - 1) / (len(best) - 1))
    ax.text(0.015, 0.03,
            f"path inefficiency  $\\log_{{10}}$({len(walked) - 1} / "
            f"{len(best) - 1}) = {ratio:.2f}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=style.FS["label"], color=style.P.INK)
    _scale_bar(ax)
    style.legend(ax, loc="upper left", ncol=1)
    print(f"  {name}: {EXAMPLE['animal']} {EXAMPLE['date']} trial "
          f"{EXAMPLE['trial']}, {start} -> {goal}; {len(best) - 1} optimal, "
          f"{len(walked) - 1} taken, log10 ratio {ratio:.3f}. The optimum is "
          f"measured from the first RECORDED node and the path is cut at the "
          f"first arrival at the goal.")
    return style.save(fig, name)


#: Trial types as they appear in ``Trial_Type``, with what is known about each.
#: Type 6 is left as its number: it opens every second session of a new goal and
#: nothing in the sheets says what it is, so naming it here would be inventing a
#: protocol.
TRIAL_TYPES = {1: "goal trial", 4: "free roaming", 5: "free roaming,\\ngoal switched",
               6: "type 6"}

TYPE_COLOURS = {1: style.P.BLUE, 4: style.P.MUTED, 5: style.P.RED,
                6: style.P.AMBER_INK}

#: Sessions per goal, and the within-session order of trial types. Both read off
#: the sheets rather than assumed: see the printed note under the figure.
SESSIONS_PER_GOAL = {1: 4, 2: 5, 3: 5, 4: 5}
SESSION_SHAPES = {
    "plain":  [(4, 1), (1, 9), (4, 1), (1, 9), (4, 1)],
    "switch": [(4, 1), (1, 9), (5, 1), (1, 12), (4, 1)],
    "second": [(6, 1), (1, 9), (4, 1), (1, 10), (4, 1)],
}


def _strip(ax, shape, y, h, x0=0.0, width=1.0, fs=None, label_counts=True):
    """One session drawn as a left-to-right strip of its periods."""
    total = sum(n for _t, n in shape)
    x = x0
    for t, n in shape:
        w = width * n / total
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=TYPE_COLOURS[t],
                                   alpha=0.30 if t == 1 else 0.75, lw=0,
                                   zorder=3))
        if label_counts and t == 1:
            ax.text(x + w / 2, y + h / 2, f"{n} goal\ntrials", ha="center",
                    va="center", fontsize=fs or style.FS["tick"] - 1,
                    color=style.P.INK, zorder=5)
        x += w
    return x


def schematic_schedule(trials_unused=None, name="schematic_schedule", si=None):
    """How the sessions are arranged, and where the goal changes.

    Top: the four goals. The first ran for four sessions and the rest for five,
    one session a day. Bottom: what a session is made of, in the three shapes the
    sheets actually contain.

    The middle period is the one that matters for reading the other figures. On
    the first session of a NEW goal it is the period across which the goal is
    moved, so the goal trials before it are the last retrievals of the old goal
    and those after it are the first encodings of the new one. That is why the
    first session of each goal appears twice on every axis in this folder, as
    GL2S1a and GL2S1b: one session, two tasks. A figure that averaged them would
    average a well-learned route with a route the animal had never run.

    Counts in the strips are typical, not fixed: a session carries 6 to 32 goal
    trials, and the printed note gives the real range.
    """
    fig, ax = style.new_figure(width_mm=180.0, height_mm=100.0, left_mm=4.0,
                               right_mm=4.0, top_mm=4.0, bottom_mm=4.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ---- top: the goals and their sessions ----
    label_w = 0.105                      # room for "goal 1" at the left
    row_h, row_gap = 0.080, 0.026
    y = 0.985 - row_h
    w = (1.0 - label_w) / 5 - 0.010
    free_slot = None                  # goal 1 has four sessions, so S5 is empty
    switch_at = None
    for goal, n_ses in SESSIONS_PER_GOAL.items():
        colour = [style.P.BLUE, style.P.GREEN_DARK, style.P.AMBER_INK,
                  style.P.RED][goal - 1]
        ax.text(label_w - 0.014, y + row_h / 2, f"goal {goal}", ha="right",
                va="center", fontsize=style.FS["label"], color=colour)
        if n_ses < 5:
            free_slot = (label_w + 4 * (w + 0.010) + w / 2, y + row_h / 2)
        for ses in range(1, n_ses + 1):
            x = label_w + (ses - 1) * (w + 0.010)
            ax.add_patch(plt.Rectangle((x, y), w, row_h, facecolor=colour,
                                       alpha=0.22, lw=0, zorder=3))
            ax.text(x + w / 2, y + row_h / 2, f"S{ses}", ha="center",
                    va="center", fontsize=style.FS["tick"], color=style.P.INK,
                    zorder=5)
            if goal > 1 and ses == 1:
                # The divider sits INSIDE the box, because the goal moves part
                # way through that session rather than between two of them.
                ax.plot([x + w / 2] * 2, [y, y + row_h], color=style.P.RED,
                        lw=_lw(1.5), dashes=(2.0, 1.4), zorder=6)
                if goal == max(SESSIONS_PER_GOAL):
                    switch_at = (x + w / 2, y)      # bottom of the LAST row
        y -= row_h + row_gap
    # Below the block with a SHORT leader to the nearest divider. Inside the
    # block there is only goal 1's empty fifth slot, and a leader from there
    # crosses two rows of boxes to reach the thing it points at.
    if switch_at:
        ax.annotate("the goal moves part way\nthrough this session", switch_at,
                    (switch_at[0] + 0.012, y + row_gap - 0.004), ha="left",
                    va="top", fontsize=style.FS["tick"] - 1, color=style.P.RED,
                    zorder=7,
                    arrowprops=dict(arrowstyle="-", color=style.P.RED,
                                    lw=_lw(0.8), shrinkA=1, shrinkB=1))
    ax.text(1.0, y + row_gap - 0.004, "one session a day", ha="right", va="top",
            fontsize=style.FS["tick"], color=style.P.MUTED)

    # ---- bottom: the three session shapes ----
    y -= 0.055
    strip_h = 0.072
    for tag, title in (("switch", "session 1 of a new goal"),
                       ("second", "session 2"),
                       ("plain", "sessions 3 to 5")):
        ax.text(0.0, y, title, ha="left", va="bottom",
                fontsize=style.FS["tick"], color=style.P.INK)
        y -= strip_h + 0.006
        _strip(ax, SESSION_SHAPES[tag], y, strip_h)
        if tag == "switch":
            total = sum(n for _t, n in SESSION_SHAPES[tag])
            mid = sum(n for _t, n in SESSION_SHAPES[tag][:2]) / total
            for cx, lab in ((mid / 2, "old goal"), ((mid + 1) / 2, "new goal")):
                ax.text(cx, y - 0.006, lab, ha="center", va="top",
                        fontsize=style.FS["tick"] - 1, color=style.P.MUTED)
            y -= 0.030
        y -= 0.042

    for t in (1, 4, 5, 6):
        ax.add_patch(plt.Rectangle((0, -1), 0, 0, facecolor=TYPE_COLOURS[t],
                                   alpha=0.30 if t == 1 else 0.75, lw=0,
                                   label=TRIAL_TYPES[t].replace("\\n", " ")))
    style.legend(ax, loc="lower center", ncol=4)
    _report_schedule(name)
    return style.save(fig, name)


def _report_schedule(name):
    """Print the real counts the strips are a simplification of."""
    try:
        t = data.trials()
    except Exception as e:                                    # noqa: BLE001
        print(f"  {name}: schedule drawn from the documented shape; "
              f"could not read the trial table to check it ({e}).")
        return
    # Habituation is out of the range: it carries a single scored trial, so
    # leaving it in reports the per-session minimum as 1.
    t = t[t["repeat"] > data.HABITUATION_REPEAT]
    g = t[t["trial_type"] == data.GOAL_TRIAL_TYPE]
    per = g.groupby(["animal", "repeat", "session"]).size()
    roam = t[t["trial_type"] == 4].groupby(["animal", "repeat", "session"]).size()
    print(f"  {name}: goal trials per session {per.min()}-{per.max()} "
          f"(median {per.median():.0f}); free-roaming periods per session "
          f"{roam.min()}-{roam.max()}. The strips show a typical session, not a "
          f"fixed one. Type 6 opens the SECOND session of each new goal and is "
          f"drawn unnamed: nothing in the sheets says what it is.")


def maze_map(out_dir=None, name="hexmaze_map", width_mm=297.0, height_mm=210.0,
             idealised=True, node_ms=19.0, label_pt=7.0):
    """A printable map of the maze with every node numbered.

    Not a schematic: this one exists to be looked at next to the apparatus, so
    the node id is the point and everything else serves it. A4 landscape, which
    is the smallest page that fits 96 three-digit labels at a readable size.

    `idealised` draws the regular honeycomb from the graph's topology. The
    measured node positions wobble by up to 0.15 m, and on a map that reads as
    information rather than as measurement error; the two differ in how a node
    LOOKS, never in which nodes it joins. Pass False for the measured layout.

    Saves to `out_dir` (default the Desktop) as PDF and PNG, and returns the
    paths. Background is transparent, as everything else in this folder is; on
    paper and in any viewer with a white page it prints white.
    """
    out = Path(out_dir) if out_dir is not None else Path.home() / "Desktop"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = style.new_figure(width_mm=width_mm, height_mm=height_mm,
                               left_mm=4.0, right_mm=4.0, top_mm=4.0,
                               bottom_mm=4.0)
    G = maze.build_graph()
    pos = (maze.idealised_positions(G) if idealised
           else {str(k): v for k, v in maze.node_positions_m().items()})
    draw_maze(ax, pos=pos, G=G, lw=1.6)
    _maze_axes(ax, pos, pad=0.62)

    real = maze.node_positions_m()
    for a, b in maze.bridge_edges(G):
        if a not in pos or b not in pos:
            continue
        d = float(np.hypot(*(np.array(real[int(a)]) - np.array(real[int(b)]))))
        mx = 0.5 * (pos[a][0] + pos[b][0])
        my = 0.5 * (pos[a][1] + pos[b][1])
        ax.text(mx, my - 0.10, f"{d:.2f} m", ha="center", va="bottom",
                fontsize=style.FS["tick"], color=style.P.INK,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                          alpha=0.9), zorder=6)

    # Every node, with its id inside it. The circle has to be wide enough to
    # hold three digits in its WHITE interior: the first try used a 7 pt marker
    # and island-coloured text, and the outer digits landed on the same-coloured
    # rim, where they were invisible. Only the middle digit of each id could be
    # read. So: a circle sized from the label, a thin rim, and dark text.
    for n, (x, y) in pos.items():
        colour = ISLAND_COLOURS.get(maze.island_of_node(n)) or style.P.MUTED
        ax.plot(x, y, "o", ms=node_ms, mfc="white", mec=colour,
                mew=_lw(1.3), zorder=7)
        ax.text(x, y, str(n), ha="center", va="center", fontsize=label_pt,
                color=style.P.INK, zorder=8)
    _island_numbers(ax, pos, G, dy=0.26, fs=style.FS["label"] + 1)
    _scale_bar(ax)

    written = []
    for suffix, extra in ((".pdf", {}), (".png", dict(dpi=600))):
        f = (out / name).with_suffix(suffix)
        fig.savefig(f, transparent=True, **extra)
        written.append(f)
    plt.close(fig)
    spacing = _within_island_spacing(G, real)
    print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. "
          f"Within-island step {spacing:.2f} m; bridges 0.68, 0.69, 0.70, 1.86, "
          f"1.99 m. Layout is the "
          + ("idealised honeycomb" if idealised else "measured positions")
          + "; the bridge lengths are measured either way.")
    for f in written:
        print(f"  wrote {f}  ({f.stat().st_size / 1024:.0f} KB)")
    return written
