"""HexMaze geometry: the node table, the connectivity graph, and shortest paths.

Two coordinate systems are in play and mixing them is the easiest mistake to make
here:

* **pixels** — what the tracker writes into the logs and ``node_list_new.csv``.
  The maze graph is built in pixels, because its adjacency threshold is a pixel
  distance.
* **metres** — pixels divided by :data:`SCALE_X` / :data:`SCALE_Y`. Every plot and
  every place-field metric uses metres, so the maze always occupies the same
  :data:`MAZE_EXTENT` box and sessions are directly comparable.

Columns are suffixed accordingly: ``x``/``y`` are pixels, ``x_m``/``y_m`` metres.
"""

from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Pixel -> metre scaling and the fixed maze frame, matching the tracker's
# plot_trials.py so rate maps and trajectories share one coordinate system.
SCALE_X = 2352 / 2 / 9
SCALE_Y = 1424 / 2 / 5
MAZE_EXTENT = (0.0, 9.0, 0.0, 5.0)

NODE_CSV = Path(__file__).resolve().parent / "data" / "node_list_new.csv"

# Nodes closer than this (pixels) are treated as connected.
ADJACENCY_THRESHOLD_PX = 65
# Corner nodes that are not part of the maze proper.
EXCLUDED_NODES = ("501", "502")
# Real maze connections the distance threshold alone does not recover.
MANUAL_EDGES = (
    ("121", "302"), ("324", "401"), ("305", "220"),
    ("404", "223"), ("201", "124"), ("224", "218"),
)


#: The maze is four hexagonal islands joined by three long bridges, and the node
#: id says which island a node is on: 1xx, 2xx, 3xx, 4xx. They are genuinely
#: separate blocks of space, about 2 m square each, with 0.36 m between
#: neighbouring nodes inside an island and 0.68 m across a bridge.
ISLANDS = (1, 2, 3, 4)

#: The corridors between islands, pooled into ONE region. There are five of them:
#: three short ones (0.68-0.70 m) joining neighbours along the zigzag, and two long
#: ones (1.86 and 1.99 m) cutting the diagonal. Pooled because a single short
#: bridge is about fourteen 2.5 cm bins and under 1% of a session's occupancy,
#: which is not enough for a Skaggs estimate; the five together are ~194 bins and
#: ~7%, which is.
BRIDGE = 5

#: Every region ``bin_islands`` can return, besides -1 for none.
REGIONS = ISLANDS + (BRIDGE,)

#: How far a rate-map bin may sit from the nearest node and still count as being
#: on that node's island, in METRES. 0.25 m is 0.7 of the within-island node
#: spacing: it covers the corridors inside an island and cuts the middle of each
#: bridge, which belongs to no island. Bins beyond it are reported separately
#: rather than folded into whichever island happens to be nearer.
ISLAND_BIN_RADIUS_M = 0.25


def island_of_node(node_id):
    """Which island a node is on, from its id: ``312`` -> 3."""
    return int(node_id) // 100


def bridge_edges(G=None):
    """The maze edges that join two different islands, as ``(a, b)`` id strings."""
    G = build_graph() if G is None else G
    return tuple((a, b) for a, b in G.edges()
                 if island_of_node(a) != island_of_node(b))


def _segment_distance(pts, a, b):
    """Distance from each row of `pts` to the segment a-b."""
    ab = b - a
    l2 = float((ab ** 2).sum())
    if l2 <= 0:
        return np.linalg.norm(pts - a, axis=1)
    s = np.clip((pts - a) @ ab / l2, 0.0, 1.0)
    return np.linalg.norm(pts - (a + s[:, None] * ab), axis=1)


def bin_islands(extent, bins, radius_m=ISLAND_BIN_RADIUS_M):
    """Region index per rate-map bin, as an int array shaped like a rate map.

    1-4 for the islands, :data:`BRIDGE` for any of the five inter-island
    corridors, ``-1`` for a bin that is near neither. A bin goes to an island when
    it is within `radius_m` of one of that island's nodes, and to the bridge
    region only if it is not: the island claim comes first, so the ends of a
    bridge belong to the islands they leave and only the span between them is
    bridge.

    The array is indexed ``[iy, ix]`` to match the maps ``place_fields`` builds.
    """
    x0, x1, y0, y1 = extent
    nx, ny = bins
    cx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    cy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    gx, gy = np.meshgrid(cx, cy)                      # both [ny, nx]
    nt = node_table()
    pos = nt[["x_m", "y_m"]].to_numpy()
    isl = np.array([island_of_node(i) for i in nt["id"]], int)
    keep = np.isin(isl, ISLANDS)
    pos, isl = pos[keep], isl[keep]
    d2 = ((gx[..., None] - pos[:, 0]) ** 2 + (gy[..., None] - pos[:, 1]) ** 2)
    out = isl[d2.argmin(axis=-1)]
    far = np.sqrt(d2.min(axis=-1)) > radius_m
    out[far] = -1

    # Whatever is left over but still lies along a cross-island corridor.
    nt_idx = nt.set_index("id_str")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    best = np.full(pts.shape[0], np.inf)
    for a, b in bridge_edges():
        pa = nt_idx.loc[a, ["x_m", "y_m"]].to_numpy(float)
        pb = nt_idx.loc[b, ["x_m", "y_m"]].to_numpy(float)
        best = np.minimum(best, _segment_distance(pts, pa, pb))
    on_bridge = (best <= radius_m).reshape(ny, nx)
    out[far & on_bridge] = BRIDGE
    return out


@lru_cache(maxsize=1)
def node_table():
    """Node coordinates as a DataFrame: ``id, x, y`` (pixels), ``id_str``, and
    ``x_m, y_m`` (metres). Empty DataFrame if the CSV is missing.

    Cached — callers must not mutate the result; use ``.copy()`` if they need to.
    """
    cols = ["id", "x", "y", "id_str", "x_m", "y_m"]
    if not NODE_CSV.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(NODE_CSV, header=None, names=["id", "x", "y"])
    df["id_str"] = df["id"].astype(int).astype(str)
    df["x_m"] = df["x"] / SCALE_X
    df["y_m"] = df["y"] / SCALE_Y
    return df


def node_positions_m():
    """``{node_id: (x_m, y_m)}`` in metres. ``{}`` if the node table is missing."""
    df = node_table()
    if df.empty:
        return {}
    return {int(r.id): (r.x_m, r.y_m) for r in df.itertuples()}


@lru_cache(maxsize=1)
def build_graph():
    """Maze connectivity graph in PIXEL coordinates, nodes keyed by id string.

    Edges are node pairs within :data:`ADJACENCY_THRESHOLD_PX`, weighted by their
    euclidean distance, plus :data:`MANUAL_EDGES`. Each node carries a ``pos``
    attribute (pixels). Cached — treat the returned graph as read-only.
    """
    df = node_table()
    G = nx.Graph()
    if df.empty:
        return G

    pos = {}
    for r in df.itertuples():
        G.add_node(r.id_str, pos=(r.x, r.y))
        pos[r.id_str] = np.array([r.x, r.y], dtype=float)

    ids = df["id_str"].tolist()
    dists = squareform(pdist(df[["x", "y"]].values))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if dists[i, j] < ADJACENCY_THRESHOLD_PX:
                G.add_edge(ids[i], ids[j], weight=dists[i, j])

    for n in EXCLUDED_NODES:
        if n in G:
            G.remove_node(n)

    for u, v in MANUAL_EDGES:
        if u in G and v in G:
            G.add_edge(u, v, weight=float(np.linalg.norm(pos[u] - pos[v])))
    return G


def idealised_positions(G=None, edge_len_m=None):
    """Regular honeycomb coordinates for the maze nodes, in metres.

    ``node_list_new.csv`` holds *measured* positions, so the drawn maze wobbles.
    The real maze is a honeycomb: every edge runs at 0/60/120 degrees and spans a
    whole number of ~0.4 m segments. This rebuilds the layout from the graph's own
    topology — walk the edges from a root, snapping each one to the nearest
    canonical direction and to a whole number of segments — then rigidly aligns
    the result (translation only, so no scale or rotation is introduced) onto the
    measured positions.

    Returns ``{node_id_str: (x_m, y_m)}``. Because the fit is translation-only,
    the result stays registered to the measured frame and anything already in
    maze coordinates — a rate map, a trajectory — still overlays correctly.
    """
    G = build_graph() if G is None else G
    if G.number_of_nodes() == 0:
        return {}

    real = {n: np.array([p[0] / SCALE_X, p[1] / SCALE_Y], float)
            for n, p in nx.get_node_attributes(G, "pos").items()}

    lengths = np.array([np.linalg.norm(real[v] - real[u]) for u, v in G.edges()])
    if edge_len_m is None:
        # the short edges are the lattice unit; long connectors are multiples
        edge_len_m = float(np.median(lengths[lengths < 2 * np.median(lengths)]))

    dirs = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
                     for a in range(0, 360, 60)])

    ideal = {}
    for component in nx.connected_components(G):
        root = min(component)
        ideal[root] = np.zeros(2)
        for parent, child in nx.bfs_edges(G.subgraph(component), root):
            d = real[child] - real[parent]
            dist = float(np.linalg.norm(d))
            if dist == 0:
                ideal[child] = ideal[parent].copy()
                continue
            unit = dirs[int(np.argmax(dirs @ (d / dist)))]
            steps = max(1, int(round(dist / edge_len_m)))
            ideal[child] = ideal[parent] + unit * edge_len_m * steps

    # translation-only alignment back onto the measured frame
    nodes = list(ideal)
    shift = (np.mean([real[n] for n in nodes], axis=0)
             - np.mean([ideal[n] for n in nodes], axis=0))
    return {n: tuple(ideal[n] + shift) for n in nodes}


def warp_to_idealised(x_m, y_m, G=None, smoothing=0.0):
    """Map measured maze coordinates into the idealised frame.

    Drawing the idealised maze under a rate map built in measured coordinates
    leaves the two ~9 cm apart — a quarter of a corridor — so the firing lands
    beside the corridor instead of on it. This carries the positions through the
    same node-to-node correspondence (thin-plate spline through every node), so
    the map and the drawn maze agree.

    NaNs pass through as NaNs. Returns ``(x_ideal, y_ideal)``.
    """
    from scipy.interpolate import RBFInterpolator

    G = build_graph() if G is None else G
    ideal = idealised_positions(G)
    real = {n: (p[0] / SCALE_X, p[1] / SCALE_Y)
            for n, p in nx.get_node_attributes(G, "pos").items()}
    nodes = [n for n in ideal if n in real]
    if not nodes:
        return np.asarray(x_m, float), np.asarray(y_m, float)

    src = np.array([real[n] for n in nodes])
    dst = np.array([ideal[n] for n in nodes])
    spline = RBFInterpolator(src, dst, kernel="thin_plate_spline",
                             smoothing=smoothing)

    x = np.asarray(x_m, float)
    y = np.asarray(y_m, float)
    out_x = np.full(x.shape, np.nan)
    out_y = np.full(y.shape, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        warped = spline(np.column_stack([x[ok], y[ok]]))
        out_x[ok], out_y[ok] = warped[:, 0], warped[:, 1]
    return out_x, out_y


def idealisation_residuals(G=None):
    """Per-node distance (m) between the idealised and measured layouts.

    Small values mean the idealised maze can stand in for the measured one without
    pulling the drawing off the data.
    """
    G = build_graph() if G is None else G
    ideal = idealised_positions(G)
    real = {n: np.array([p[0] / SCALE_X, p[1] / SCALE_Y], float)
            for n, p in nx.get_node_attributes(G, "pos").items()}
    return {n: float(np.linalg.norm(np.asarray(ideal[n]) - real[n])) for n in ideal}


def shortest_path_segments(G, start_node, end_node, weight_mode="weight"):
    """Every shortest path from `start_node` to `end_node`, as drawable segments.

    `weight_mode` is ``"weight"`` for physical distance or ``None`` for hop count.
    Returns ``(paths, label, metric)`` where `paths` is a list (one entry per tied
    shortest path) of ``[(p1, p2), ...]`` pixel-coordinate segments, and `metric`
    is the path length in the chosen mode (0.0 when there is no path).
    """
    if start_node not in G or end_node not in G:
        return [], "Node not found", 0.0
    try:
        if not nx.has_path(G, start_node, end_node):
            return [], "No Path", 0.0
        metric = nx.shortest_path_length(G, start_node, end_node, weight=weight_mode)
        pos = nx.get_node_attributes(G, "pos")
        paths = [[(pos[p[i]], pos[p[i + 1]]) for i in range(len(p) - 1)]
                 for p in nx.all_shortest_paths(G, start_node, end_node, weight=weight_mode)]
    except nx.NetworkXNoPath:
        return [], "No Path", 0.0
    label = f"{'Dist' if weight_mode == 'weight' else 'Hops'}: {metric:.1f} (N={len(paths)})"
    return paths, label, float(metric)
