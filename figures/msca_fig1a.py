"""MSCA proposal, Figure 1a: simultaneously recorded CA1 place cells in the HexMaze.

Every unit's spikes are drawn on the animal's path, overlaid on ONE idealised
maze, each unit in its own hue, with its place fields outlined and the goal node
marked. Autocorrelograms sit small above the maze in the matching hue, so the
spike-train signature and the spatial map belong together by colour alone. No
metric text — the numbers belong in the caption.

Firing rate is encoded as SPREAD, not shade: each spike is displaced by a Gaussian
whose width scales with the local rate, so the cloud hugs the path where the cell
is quiet and blooms where it fires hardest. The jitter is cosmetic — every rate,
field and metric is computed from the unjittered map.

Hues are raw samples at equal steps along the full Turbo colormap, unmodified.

Usage:
    python figures/msca_fig1a.py --nwb <session.nwb> --units 10 13 17 25 149 \
        [--out fig1a] [--bin-cm 5] [--sigma 1.0] [--speed 0.025] [--max-jitter 0.10]
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Text stays TEXT in the PDF/SVG (TrueType, not outlines), so the figure can be
# opened in a vector editor and its labels retyped rather than traced.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.patheffects as pe                               # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
import networkx as nx                                             # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib import colormaps                                  # noqa: E402
from pynwb import NWBHDF5IO                                       # noqa: E402

from hm_rat_analysis import maze, nwb as nwbio, place_fields as PF  # noqa: E402
from hm_rat_analysis import spike_metrics as SM                     # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"
MAZE_LINE = "#c9c8c2"

ACG_FAST = (0.020, 0.0005, "± 20 ms")
ACG_SLOW = (0.500, 0.005, "± 500 ms")

#: Type sizes for panel a, in POINTS. Points are absolute on the page, so these do
#: NOT scale with the figure: a 15-inch poster and a 17 cm A4 column want the same
#: 8-11 pt text, and only the artwork around it changes size. PRINT_FONT is that
#: band; FONT is the original poster sizing, kept as this script's own default so
#: the standalone figure is unchanged.
FONT = {"unit": 15, "goal": 13, "scale_bar": 11, "acg_unit": 11,
        "acg_window": 8.5, "header": 10}
PRINT_FONT = {"unit": 11, "goal": 10, "scale_bar": 9, "acg_unit": 8,
              "acg_window": 8, "header": 9}

#: Where the "goal N" caption sits relative to the star, in METRES: +x is right,
#: +y is DOWN on the page (the maze is drawn y-inverted, see panel_maze). The
#: default drops it into the open hexagon below-right of the goal, because the
#: corridors around a goal node are exactly where the place fields are and a
#: caption laid over them hides the data it is pointing at.
GOAL_LABEL_OFFSET = (0.62, 1.20)


# Named palettes. "turbo" is sampled at equal steps for however many units there
# are; the fixed sets below keep a unit on the same colour whatever the count.
#
# rgbkym  the printer primaries as asked. Normal-vision separation is fine (worst
#         pair dE 19.3) but pure green and pure yellow sit at 1.34:1 and 1.05:1
#         against a white page — as small dots they disappear, and under
#         protanopia the two merge (dE 3.5).
# rgbkym-dark  same six identities, green and yellow stepped down in lightness
#         only (hue and saturation untouched) until they clear 3:1 on white.
FIXED_PALETTES = {
    "rgbkym": ["#ff0000", "#00ff00", "#0000ff", "#000000", "#ffff00", "#ff00ff"],
    "rgbkym-dark": ["#ff0000", "#00a900", "#0000ff", "#000000", "#969600", "#ff00ff"],
}
DEFAULT_PALETTE = "turbo"


def hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def unit_hues(n, palette=DEFAULT_PALETTE):
    """`n` colours for `n` units, from the named palette."""
    if palette in FIXED_PALETTES:
        cols = FIXED_PALETTES[palette]
        if n > len(cols):
            raise ValueError(f"palette {palette!r} has {len(cols)} colours, "
                             f"but {n} units were requested")
        return [hex_to_rgb(c) for c in cols[:n]]
    turbo = colormaps["turbo"]
    return [turbo(p)[:3] for p in np.linspace(0.0, 1.0, n)]


def to_hex(rgb):
    return "#%02x%02x%02x" % tuple(round(max(0, min(1, v)) * 255) for v in rgb)


def maze_bbox(nodes_m, margin=0.28):
    if not nodes_m:
        return maze.MAZE_EXTENT
    xs = [p[0] for p in nodes_m.values()]
    ys = [p[1] for p in nodes_m.values()]
    return (min(xs) - margin, max(xs) + margin,
            min(ys) - margin, max(ys) + margin)


def rate_rgba(rate, rgb, gamma=2.0, max_alpha=0.92):
    """One unit's whole rate map as a flat-hue RGBA layer.

    The full map is drawn, not an extracted field: every visited bin appears, with
    opacity proportional to rate / peak. Hue is constant within a unit — with five
    maps stacked, a hue that also varied with rate would be unreadable.

    `gamma` > 1 thins the low-rate wash where maps overlap without introducing a
    hard cutoff; 1.0 is the plain rate map.
    """
    filled = rate.filled(0.0) if np.ma.isMaskedArray(rate) else np.asarray(rate)
    peak = float(filled.max())
    if peak <= 0:
        return None, None
    norm = np.clip(filled / peak, 0, 1)
    if gamma <= 0:
        # norm**0 is 1 even where norm is 0, which would paint unvisited-but-in-range
        # bins solid. Flat opacity wherever the cell actually fired is the sane
        # reading of gamma = 0 — and it discards rate entirely, by construction.
        alpha = np.where(norm > 0, max_alpha, 0.0)
    else:
        alpha = norm ** gamma * max_alpha
    if np.ma.isMaskedArray(rate):
        alpha[np.ma.getmaskarray(rate)] = 0.0

    rgba = np.zeros(filled.shape + (4,))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = alpha
    return rgba, norm


def spikes_on_path(x, y, t, spike_times, rate, extent, speed_thresh=0.025,
                   max_jitter_m=0.10, rng=None):
    """Spike positions along the animal's path, scattered in proportion to rate.

    A plain spikes-on-path plot saturates: once a place field has a few hundred
    spikes they pile onto the same few centimetres of trajectory and the densest
    part looks the same as a merely busy one. Displacing each spike by a Gaussian
    whose width scales with the local firing rate turns rate into *spread* — the
    cloud is tightest on the path where the cell is quiet and blooms widest where
    it fires hardest.

    The jitter is cosmetic: it moves where a spike is drawn, never which bin it
    was counted in. Rate, fields and every metric come from the unjittered map.

    Returns ``(x_jittered, y_jittered, local_rate_fraction)``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    st = np.asarray(spike_times, float)
    if st.size == 0:
        return np.array([]), np.array([]), np.array([])

    good = np.isfinite(x) & np.isfinite(y)
    xg, yg, tg = x[good], y[good], t[good]
    if xg.size < 2:
        return np.array([]), np.array([]), np.array([])

    # speed at each position sample, and each spike's speed by interpolation —
    # the same gate the rate map uses, so the two show the same spikes
    speed = np.zeros_like(xg)
    d = np.hypot(np.diff(xg), np.diff(yg))
    dts = np.diff(tg)
    speed[1:] = d / np.where(dts > 0, dts, np.inf)

    sx = np.interp(st, tg, xg, left=np.nan, right=np.nan)
    sy = np.interp(st, tg, yg, left=np.nan, right=np.nan)
    sv = np.interp(st, tg, speed, left=0.0, right=0.0)
    keep = np.isfinite(sx) & np.isfinite(sy)
    if speed_thresh > 0:
        keep &= sv > speed_thresh
    sx, sy = sx[keep], sy[keep]
    if sx.size == 0:
        return np.array([]), np.array([]), np.array([])

    # local rate at each spike, as a fraction of this unit's peak
    filled = rate.filled(0.0) if np.ma.isMaskedArray(rate) else np.asarray(rate)
    peak = float(filled.max())
    x0, x1, y0, y1 = extent
    ny, nx = filled.shape
    ix = np.clip(((sx - x0) / (x1 - x0) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((sy - y0) / (y1 - y0) * ny).astype(int), 0, ny - 1)
    frac = (filled[iy, ix] / peak) if peak > 0 else np.zeros(sx.shape)

    sigma = max_jitter_m * frac
    return (sx + rng.normal(0, 1, sx.shape) * sigma,
            sy + rng.normal(0, 1, sy.shape) * sigma,
            frac)


def field_mask(rate, field_frac=0.30, min_field_bins=3, min_peak_hz=0.5):
    """Boolean mask of the unit's place fields, for the outline.

    The fill shows the whole rate map; this only says where the fields are. It is
    the package's place-field definition — connected regions above `field_frac` of
    the peak, at least `min_field_bins` bins, in-field peak at least
    `min_peak_hz` — so the outline means the same thing as the number in the
    metrics table.
    """
    fields = PF.place_fields(rate, field_frac=field_frac, min_peak_hz=min_peak_hz,
                             min_field_bins=min_field_bins)
    if not fields:
        return None
    mask = np.zeros(fields[0].shape, dtype=bool)
    for comp in fields:
        mask |= comp
    return mask


def peak_anchor(norm, extent):
    """The unit's peak bin, in maze coordinates — where its number points."""
    if norm is None or not np.isfinite(norm).any() or norm.max() <= 0:
        return None
    iy, ix = np.unravel_index(int(np.argmax(norm)), norm.shape)
    ny, nx = norm.shape
    x0, x1, y0, y1 = extent
    return (x0 + (ix + 0.5) * (x1 - x0) / nx,
            y0 + (iy + 0.5) * (y1 - y0) / ny)


def place_labels(layers, bbox, min_sep=0.66, offset=0.60):
    """Put each unit's number beside its peak, then push apart any that collide."""
    mid_y = (bbox[2] + bbox[3]) / 2
    for lay in layers:
        if lay["anchor"] is None:
            lay["label_xy"] = None
            continue
        ax_, ay_ = lay["anchor"]
        lay["label_xy"] = (ax_, ay_ + (offset if ay_ < mid_y else -offset))

    placed = [l for l in layers if l["label_xy"] is not None]
    for _ in range(80):
        moved = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                (xi, yi), (xj, yj) = placed[i]["label_xy"], placed[j]["label_xy"]
                dx, dy = xj - xi, yj - yi
                d = float(np.hypot(dx, dy))
                if d >= min_sep:
                    continue
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                push = (min_sep - d) / 2 + 0.01
                ux, uy = dx / d, dy / d
                placed[i]["label_xy"] = (xi - ux * push, yi - uy * push)
                placed[j]["label_xy"] = (xj + ux * push, yj + uy * push)
                moved = True
        if not moved:
            break

    for lay in placed:
        lx, ly = lay["label_xy"]
        lay["label_xy"] = (float(np.clip(lx, bbox[0] + 0.25, bbox[1] - 0.25)),
                           float(np.clip(ly, bbox[2] + 0.25, bbox[3] - 0.25)))


def break_jumps(xm, ym, max_step_m=0.45):
    """Trajectory with NaN inserted at tracking teleports, so the drawn path does
    not spray straight streaks across the maze."""
    x, y = np.asarray(xm, float).copy(), np.asarray(ym, float).copy()
    step = np.hypot(np.diff(x), np.diff(y))
    cut = np.flatnonzero(step > max_step_m) + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)


def panel_maze(ax, layers, path_xy, goal_xy, goal_node, bbox, spike_size=3.2,
               rasterize=True, goal_label_offset=GOAL_LABEL_OFFSET,
               scale=1.0, fonts=None):
    """The maze, the trajectory and every unit's spikes.

    `rasterize` keeps the two heavy layers — the trajectory and the spike clouds,
    tens of thousands of marks — as an embedded image inside the vector page. Turn
    it off for a fully editable vector figure, at the cost of a much larger file.

    `scale` shrinks the ARTWORK — line widths, marker sizes, halos — for a smaller
    page; it deliberately leaves `fonts` alone, because points are absolute and a
    figure printed at 17 cm needs the same 8-11 pt text as one printed at 38 cm.
    Marker areas scale as scale^2, since `s` is an area in pt^2 and only then does
    a dot keep its size relative to the maze.
    """
    fonts = FONT if fonts is None else fonts
    lw = lambda w: w * scale                  # noqa: E731 - widths are linear
    area = lambda s: s * scale ** 2           # noqa: E731 - marker sizes are areas

    G = maze.build_graph()
    pos = maze.idealised_positions(G)       # regular lattice, not measured wobble
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=MAZE_LINE, lw=lw(2.4), zorder=1, solid_capstyle="round")
    pts = np.array(list(pos.values()))
    ax.scatter(pts[:, 0], pts[:, 1], s=area(15), facecolor=SURFACE,
               edgecolor=MAZE_LINE, linewidths=lw(1.0), zorder=2)

    if path_xy is not None:
        px, py = break_jumps(*path_xy)
        ax.plot(px, py, color="#b7b6b0", lw=lw(0.35), alpha=0.9, zorder=3,
                rasterized=rasterize, solid_joinstyle="round")

    for i, lay in enumerate(layers):
        if lay.get("spikes") is not None and len(lay["spikes"][0]):
            sx, sy = lay["spikes"]
            ax.scatter(sx, sy, s=max(0.45, area(spike_size)), c=[to_hex(lay["rgb"])],
                       linewidths=0, alpha=0.75, zorder=4 + i, rasterized=rasterize)
        if lay.get("mask") is not None:
            ax.contour(lay["mask"].astype(float), levels=[0.5],
                       extent=maze.MAZE_EXTENT, origin="lower",
                       colors=[to_hex(lay["rgb"])],
                       linewidths=lw(1.6), linestyles="solid", zorder=4 + i + 0.5)

    if goal_xy:
        # white rim so the star still reads where it sits on top of spikes
        ax.scatter(*goal_xy, s=area(1150), marker="*", facecolor=INK,
                   edgecolor=SURFACE, linewidths=lw(2.0), zorder=21)
        dx, dy = goal_label_offset
        ax.annotate(f"goal {goal_node}", xy=goal_xy,
                    xytext=(goal_xy[0] + dx, goal_xy[1] + dy),
                    ha="center", color=INK, fontsize=fonts["goal"],
                    fontweight="bold", zorder=22,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=lw(1.4),
                                    shrinkA=0, shrinkB=15 * scale))

    halo = [pe.withStroke(linewidth=lw(3.4), foreground=SURFACE)]
    for lay in layers:
        if lay.get("label_xy") is None:
            continue
        hue = to_hex(lay["rgb"])
        ax.annotate(str(lay["cid"]), xy=lay["anchor"], xytext=lay["label_xy"],
                    ha="center", va="center", fontsize=fonts["unit"],
                    fontweight="bold", color=hue, zorder=25, path_effects=halo,
                    arrowprops=dict(arrowstyle="-", lw=lw(1.2), shrinkA=2, shrinkB=2,
                                    color=hue, alpha=0.75))

    xmin, xmax, ymin, ymax = bbox
    # y is inverted: tracker positions are video-pixel coordinates, where y grows
    # downwards. Drawing them y-up flips the maze relative to the room.
    x0, y0 = xmin + 0.12, ymax - 0.12
    ax.plot([x0, x0 + 1], [y0, y0], color=INK, lw=lw(3.0), solid_capstyle="butt",
            zorder=26)
    ax.text(x0 + 0.5, y0 - 0.09, "1 m", ha="center", va="bottom",
            fontsize=fonts["scale_bar"], color=INK, fontweight="bold")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_acg(ax, counts, centres, window_ms, rgb, title=None, row_label=None,
              fontsize=13, label_size=9.5, scale=1.0):
    width = (centres[1] - centres[0]) if len(centres) > 1 else 1.0
    ax.bar(centres, counts, width=width, color=to_hex(rgb), linewidth=0)
    ax.set_xlim(-window_ms, window_ms)
    ax.set_ylim(0, max(1, counts.max() * 1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)                 # sits over the maze when inset
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.8 * scale)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=2,
                     color=to_hex(rgb))
    if row_label:
        ax.annotate(row_label, xy=(0, 0.45), xycoords="axes fraction",
                    xytext=(-5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=label_size, color=INK2)


def empty_corner_boxes(G, pos, bbox, clearance=0.22, grid=(240, 140)):
    """The two empty corners of the maze, as (x, y, w, h) in axes fraction.

    The maze is a zigzag, so roughly a fifth of its bounding box is unused. Rather
    than eyeballing where, rasterise the edges and grow the largest empty
    rectangle from each corner. Returned in axes fraction with the y axis already
    inverted, so they can be handed straight to ``ax.inset_axes``.
    """
    x0, x1, y0, y1 = bbox
    nx_, ny_ = grid
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx_), np.linspace(y0, y1, ny_))
    occ = np.zeros((ny_, nx_), bool)
    for u, v in G.edges():
        ax_, ay_ = pos[u]
        bx_, by_ = pos[v]
        dx, dy = bx_ - ax_, by_ - ay_
        L2 = dx * dx + dy * dy
        if L2 == 0:
            continue
        s = np.clip(((gx - ax_) * dx + (gy - ay_) * dy) / L2, 0, 1)
        occ |= np.hypot(gx - (ax_ + s * dx), gy - (ay_ + s * dy)) < clearance

    def grow(flipy, flipx):
        o = occ[::-1, :] if flipy else occ
        o = o[:, ::-1] if flipx else o
        best = (0, 0, 0)
        for h in range(1, ny_):
            w = 0
            while w + 1 <= nx_ and not o[:h, :w + 1].any():
                w += 1
            if w * h > best[0]:
                best = (w * h, w, h)
        return best[1] / nx_, best[2] / ny_

    # data bottom-left shows top-left once y is inverted, and vice versa
    w_tl, h_tl = grow(flipy=False, flipx=False)
    w_br, h_br = grow(flipy=True, flipx=True)
    return [(0.0, 1.0 - h_tl, w_tl, h_tl),
            (1.0 - w_br, 0.0, w_br, h_br)]


def inset_correlograms(ax, layers, boxes, pad=0.018, scale=1.0, fonts=None):
    """Tuck each unit's correlograms into the maze's empty corners.

    Units are split between the two boxes; each gets one row holding its two
    windows side by side, with its number in its own colour at the left.
    """
    fonts = FONT if fonts is None else fonts
    n = len(layers)
    half = (n + 1) // 2
    for bi, (box, group) in enumerate(zip(boxes, (layers[:half], layers[half:]))):
        if not group:
            continue
        bx, by, bw, bh = box
        bx, by = bx + pad, by + pad
        bw, bh = bw - 2 * pad, bh - 2 * pad
        row_h = bh / len(group)
        lab_w = 0.16 * bw                       # room for the unit number
        cell_w = (bw - lab_w) / 2

        for k, lay in enumerate(group):
            # rows fill downwards inside the box
            ry = by + bh - (k + 1) * row_h
            ax.text(bx + lab_w * 0.72, ry + row_h * 0.45, str(lay["cid"]),
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=fonts["acg_unit"], fontweight="bold",
                    color=to_hex(lay["rgb"]), zorder=30)
            for j, (counts, centres, window) in enumerate(
                    ((*lay["c_fast"], ACG_FAST[0] * 1000),
                     (*lay["c_slow"], ACG_SLOW[0] * 1000))):
                sub = ax.inset_axes([bx + lab_w + j * cell_w,
                                     ry + row_h * 0.12,
                                     cell_w * 0.88, row_h * 0.74])
                sub.set_zorder(30)
                panel_acg(sub, counts, centres, window, lay["rgb"], scale=scale)
                # The two windows are named ONCE, over the first box. The second box
                # holds the same two columns in the same order, and its top edge is
                # where a unit's own label often lands — two captions there collide
                # with it on a small page and say nothing new.
                if k == 0 and bi == 0:
                    ax.text(bx + lab_w + (j + 0.44) * cell_w, by + bh + 0.006,
                            ACG_FAST[2] if j == 0 else ACG_SLOW[2],
                            transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=fonts["acg_window"], color=INK2, zorder=30)


def load_session(nwb_path, units, bin_cm=5.0, min_occ=0.25, sigma=1.0,
                 speed_thresh=0.025, gamma=2.0, field_frac=0.30,
                 min_field_bins=3, max_jitter=0.10, palette=DEFAULT_PALETTE):
    """Everything panel a needs from one session NWB, drawn by :func:`draw_panel_a`.

    Kept apart from the drawing so the panel can be composed into a larger figure
    (``figures/msca_fig1.py``) without reopening the NWB or duplicating any of the
    rate-map conventions.
    """
    io = NWBHDF5IO(str(nwb_path), "r", load_namespaces=True)
    try:
        nwb = io.read()
        pos = nwbio.load_position(nwb)
        if pos is None:
            raise ValueError("no Behavior/Position in this NWB")
        x, y, t = pos
        xm, ym = maze.warp_to_idealised(x / maze.SCALE_X, y / maze.SCALE_Y)
        dt = float(np.median(np.diff(t)))

        desc = str(nwb.session_description)
        m = re.search(r"Goal_Node\s+(\d+)", desc)
        goal_node = int(m.group(1)) if m else None
        animal = nwb.subject.subject_id if nwb.subject is not None else "?"
        date = str(nwb.session_id)

        nodes_m = {int(k): v for k, v in maze.idealised_positions().items()}
        goal_xy = nodes_m.get(goal_node) if goal_node else None
        if goal_node and goal_xy is None:
            print(f"WARNING: goal node {goal_node} is not in the node table; not drawn.")

        udf = nwb.units.to_dataframe()
        bins = (int(round((maze.MAZE_EXTENT[1] - maze.MAZE_EXTENT[0]) / (bin_cm / 100))),
                int(round((maze.MAZE_EXTENT[3] - maze.MAZE_EXTENT[2]) / (bin_cm / 100))))

        hues = unit_hues(len(units), palette)
        layers = []
        for i, cid in enumerate(units):
            sel = udf[udf["phy_cluster_id"] == cid]
            if sel.empty:
                raise ValueError(
                    f"unit {cid} is not in this NWB (phy_cluster_id runs "
                    f"{int(udf['phy_cluster_id'].min())}..{int(udf['phy_cluster_id'].max())})")
            st = np.asarray(sel.iloc[0]["spike_times"], dtype=float)

            rate, occ, _ = PF.make_rate_map(xm, ym, t, st, maze.MAZE_EXTENT, bins, dt,
                                            sigma, speed_thresh=speed_thresh,
                                            return_occ=True)
            rate = np.ma.masked_where(occ.filled(0) < min_occ, rate)
            _, norm = rate_rgba(rate, hues[i], gamma=gamma)
            mask = field_mask(rate, field_frac=field_frac,
                              min_field_bins=min_field_bins)
            jx, jy, _frac = spikes_on_path(
                xm, ym, t, st, rate, maze.MAZE_EXTENT,
                speed_thresh=speed_thresh, max_jitter_m=max_jitter,
                rng=np.random.default_rng(1234 + i))
            if norm is None:
                print(f"WARNING: unit {cid} has no bins clearing the {min_occ:.2f} s "
                      f"occupancy floor ({len(st)} spikes) — nothing to draw for it.")
            layers.append(dict(cid=cid, rgb=hues[i], norm=norm,
                               mask=mask, spikes=(jx, jy),
                               anchor=peak_anchor(norm, maze.MAZE_EXTENT),
                               c_fast=SM.autocorrelogram(st, *ACG_FAST[:2]),
                               c_slow=SM.autocorrelogram(st, *ACG_SLOW[:2])))
    finally:
        io.close()

    bbox = maze_bbox(nodes_m)
    place_labels(layers, bbox)
    return dict(layers=layers, path=(xm, ym), goal_xy=goal_xy, goal_node=goal_node,
                bbox=bbox, animal=animal, date=date)


def draw_panel_a(ax, data, rasterize=True, goal_label_offset=GOAL_LABEL_OFFSET,
                 scale=1.0, fonts=None):
    """Panel a \u2014 every unit's spikes on the idealised maze, correlograms inset."""
    panel_maze(ax, data["layers"], data["path"], data["goal_xy"], data["goal_node"],
               data["bbox"], rasterize=rasterize,
               goal_label_offset=goal_label_offset, scale=scale, fonts=fonts)
    G = maze.build_graph()
    boxes = empty_corner_boxes(G, maze.idealised_positions(G), data["bbox"])
    inset_correlograms(ax, data["layers"], boxes, scale=scale, fonts=fonts)


def build(nwb_path, units, out_stem, bin_cm=5.0, min_occ=0.25, sigma=1.0,
          speed_thresh=0.025, gamma=2.0, field_frac=0.30,
          min_field_bins=3, max_jitter=0.10, palette=DEFAULT_PALETTE,
          goal_label_offset=GOAL_LABEL_OFFSET):
    data = load_session(nwb_path, units, bin_cm=bin_cm, min_occ=min_occ, sigma=sigma,
                        speed_thresh=speed_thresh, gamma=gamma, field_frac=field_frac,
                        min_field_bins=min_field_bins, max_jitter=max_jitter,
                        palette=palette)
    bbox = data["bbox"]

    width = 15.0
    left, right, top, bottom = 0.005, 0.995, 0.965, 0.01
    maze_aspect = (bbox[1] - bbox[0]) / (bbox[3] - bbox[2])
    axes_w = width * (right - left)
    height = (axes_w / maze_aspect) / (top - bottom)

    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    draw_panel_a(ax, data, goal_label_offset=goal_label_offset)

    fig.text(left + 0.004, 0.982, f"Rat {data['animal']}  \u00b7  {data['date']}",
             fontsize=10, color=INK2)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")
    fig.savefig(pdf, facecolor=SURFACE)
    fig.savefig(png, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nwb", required=True)
    ap.add_argument("--units", type=int, nargs="+", required=True)
    ap.add_argument("--out", default="fig1a")
    ap.add_argument("--bin-cm", type=float, default=5.0)
    ap.add_argument("--min-occ", type=float, default=0.25,
                    help="blank rate-map bins visited for less than this (s)")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian smoothing sigma, in bins")
    ap.add_argument("--speed", type=float, default=0.025,
                    help="speed threshold in m/s (default 0.025 = 2.5 cm/s)")
    ap.add_argument("--field-frac", type=float, default=0.30,
                    help="outline encloses bins above this fraction of the peak")
    ap.add_argument("--min-field-bins", type=int, default=3,
                    help="do not outline connected regions smaller than this")
    ap.add_argument("--palette", default=DEFAULT_PALETTE,
                    choices=["turbo"] + sorted(FIXED_PALETTES),
                    help="unit colours (default: equal steps along Turbo)")
    ap.add_argument("--goal-label", type=float, nargs=2, metavar=("DX", "DY"),
                    default=list(GOAL_LABEL_OFFSET),
                    help="where the 'goal N' caption sits relative to the star, in "
                         "metres: +DX right, +DY DOWN the page (default "
                         f"{GOAL_LABEL_OFFSET[0]:g} {GOAL_LABEL_OFFSET[1]:g})")
    ap.add_argument("--max-jitter", type=float, default=0.10,
                    help="spike scatter at a unit's peak rate, in metres")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="opacity exponent; >1 thins the low-rate wash under overlap")
    a = ap.parse_args(argv)
    build(a.nwb, a.units, a.out, bin_cm=a.bin_cm, min_occ=a.min_occ, sigma=a.sigma,
          speed_thresh=a.speed, gamma=a.gamma, field_frac=a.field_frac,
          min_field_bins=a.min_field_bins, max_jitter=a.max_jitter,
          palette=a.palette, goal_label_offset=tuple(a.goal_label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
