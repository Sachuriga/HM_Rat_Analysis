"""Per-session behavioural report: one page per trial, plus aggregates.

Reads one session folder's tracker outputs (see :mod:`hm_rat_analysis.behaviour`)
and writes, next to them:

* ``<stem>_analysis_final.pdf``   — metadata page, one page per trial, aggregates
* ``<stem>_trial_metrics.csv``    — tidy per-trial metrics, one row per trial
* ``<stem>_all_plot_data.pkl``    — every plotted series, for further analysis

Each trial is scored against the maze graph by comparing the route the animal
actually took to the shortest one available:

    score = log(optimal / actual)

so 0 is a perfect route and more negative is more wandering. It is computed twice
— over physical distance (``dist_log_score``) and over hop count
(``hops_log_score``) — because a rat can take a topologically direct route that is
physically long, or vice versa.

Usage:
    hm-trial-report -o <session folder>
"""

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import networkx as nx                                             # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages              # noqa: E402
from matplotlib.collections import LineCollection                 # noqa: E402

from .. import behaviour, maze                                    # noqa: E402

#: Speed smoothing windows (seconds) stored for every trial.
SMOOTH_WINDOWS_S = (0.5, 1.0, 2.0, 5.0)
#: Window (seconds) for the "raw" lightly-smoothed speed trace.
RAW_SMOOTH_S = 0.4
#: A goal counts as reached when the animal comes within this many PIXELS of it.
GOAL_RADIUS_PX = 50
#: If the trial ends further than this (metres) from the goal, the trajectory is
#: extended to the goal so the drawn path and the scored path agree.
GOAL_SNAP_M = 0.5
#: Occupancy histogram resolution over MAZE_EXTENT.
OCCUPANCY_BINS = (50, 30)
#: Speed colour scale ceiling (m/s) for the trajectory plots.
SPEED_VMAX = 1.0


def _goal_position(nodes, goal_node):
    """``(x_m, y_m, x_px, y_px)`` for `goal_node`, or four Nones if unknown."""
    if not goal_node or nodes.empty:
        return None, None, None, None
    row = nodes[nodes["id_str"] == goal_node]
    if row.empty:
        return None, None, None, None
    r = row.iloc[0]
    return r["x_m"], r["y_m"], r["x"], r["y"]


def analyse_trial(trial_id, xy, node_sequence, goal_node, nodes, graph):
    """Metrics and drawable series for one trial.

    `xy` is the trial's N x 2 PIXEL trajectory. Returns a dict holding both the
    scalar metrics and the arrays the trial page draws.
    """
    x_px, y_px = xy[:, 0], xy[:, 1]
    x_m, y_m = x_px / maze.SCALE_X, y_px / maze.SCALE_Y
    speed = behaviour.compute_speed_from_xy(x_m, y_m, behaviour.FS)
    dt = 1.0 / behaviour.FS

    gx_m, gy_m, gx_px, gy_px = _goal_position(nodes, goal_node)

    # Did the animal actually visit the goal, and when first?
    goal_reached, first_visit = False, -1
    if gx_px is not None:
        near = np.where((x_px - gx_px) ** 2 + (y_px - gy_px) ** 2 < GOAL_RADIUS_PX ** 2)[0]
        if near.size:
            goal_reached, first_visit = True, int(near[0])

    # If it never got there, extend the drawn path to the goal so the picture and
    # the distance score describe the same route.
    x_plot, y_plot = x_m.copy(), y_m.copy()
    appended = False
    if gx_m is not None and not goal_reached:
        if np.hypot(x_plot[-1] - gx_m, y_plot[-1] - gy_m) > GOAL_SNAP_M:
            x_plot = np.append(x_plot, gx_m)
            y_plot = np.append(y_plot, gy_m)
            appended = True

    raw_window = max(1, int(round(RAW_SMOOTH_S * behaviour.FS)))
    speed_raw = behaviour.moving_average(speed, raw_window)
    speed_smooth = {w: behaviour.moving_average(speed, max(1, int(round(w * behaviour.FS))))
                    for w in SMOOTH_WINDOWS_S}

    # Actual distance travelled: to the first goal visit if there was one, else the
    # whole trial (plus the snap segment, when one was added).
    if goal_reached and first_visit > 0:
        actual_dist = behaviour.compute_path_length(x_px[:first_visit + 1], y_px[:first_visit + 1])
        score_note = "(Start->FirstGoal)"
    else:
        actual_dist = behaviour.compute_path_length(x_px, y_px)
        score_note = "(Full Path)"
        if appended:
            actual_dist += float(np.hypot(x_px[-1] - gx_px, y_px[-1] - gy_px))

    # Hop count from the node sequence the tracker logged.
    actual_hops, start_node = 0, None
    path_nodes = [t.strip() for t in node_sequence.split(",") if t.strip()] if node_sequence else []
    if path_nodes:
        start_node = path_nodes[0]
        if goal_node in path_nodes:
            actual_hops = path_nodes.index(goal_node)
            if "FirstGoal" not in score_note:
                score_note = "(Start->GoalNode)"
        else:
            actual_hops = max(0, len(path_nodes) - 1)

    optimal_dist = optimal_hops = np.nan
    dist_score = hops_score = np.nan
    segments_dist, segments_hops = [], []
    if graph and start_node and goal_node:
        try:
            optimal_dist = nx.shortest_path_length(graph, start_node, goal_node, weight="weight")
            dist_score = np.log(optimal_dist / actual_dist) if actual_dist > 0 else np.nan

            optimal_hops = nx.shortest_path_length(graph, start_node, goal_node, weight=None)
            if actual_hops > 0:
                hops_score = np.log(optimal_hops / actual_hops)
            elif optimal_hops == 0:
                hops_score = 0.0

            segments_dist, _, _ = maze.shortest_path_segments(graph, start_node, goal_node, "weight")
            segments_hops, _, _ = maze.shortest_path_segments(graph, start_node, goal_node, None)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return {
        "trial_id": trial_id,
        "goal_node": goal_node,
        "start_node": start_node,
        "x_m": x_m, "y_m": y_m, "x_plot": x_plot, "y_plot": y_plot,
        "speed": speed, "speed_raw": speed_raw, "speed_smooth": speed_smooth,
        "time_s": np.arange(len(speed)) * dt,
        "normalized_time": np.linspace(0, 1, len(speed)) if len(speed) > 1 else np.array([0.0]),
        "appended": appended,
        "goal_xy_m": (gx_m, gy_m),
        "goal_reached": goal_reached,
        "actual_dist_px": actual_dist, "optimal_dist_px": optimal_dist,
        "actual_hops": actual_hops, "optimal_hops": optimal_hops,
        "dist_log_score": dist_score, "hops_log_score": hops_score,
        "score_note": score_note,
        "segments_dist": segments_dist, "segments_hops": segments_hops,
        "node_sequence": node_sequence or "",
        "duration_s": len(speed) * dt,
        "avg_speed": float(np.mean(speed)) if len(speed) else 0.0,
        "median_speed": float(np.median(speed)) if len(speed) else 0.0,
    }


def _style_maze_axes(axes, nodes):
    """Put every spatial panel in the same metre frame with the nodes faintly drawn."""
    xmin, xmax, ymin, ymax = maze.MAZE_EXTENT
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)          # y inverted: image convention
        ax.set_aspect("equal")
        if not nodes.empty:
            ax.scatter(nodes["x_m"], nodes["y_m"], c="none", edgecolors="grey", alpha=0.3)


def _plot_trial_page(pdf, tr, nodes):
    """One PDF page for a single trial."""
    dt = 1.0 / behaviour.FS
    fig = plt.figure(figsize=(12, 23))
    gs = fig.add_gridspec(6, 2, height_ratios=[0.3, 1, 1, 1, 0.6, 0.6])

    d_msg = f"{tr['dist_log_score']:.3f}" if np.isfinite(tr["dist_log_score"]) else "N/A"
    h_msg = f"{tr['hops_log_score']:.3f}" if np.isfinite(tr["hops_log_score"]) else "N/A"
    ax_txt = fig.add_subplot(gs[0, :])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5,
                f"Trial {tr['trial_id']} | Goal: {tr['goal_node']}\n"
                f"Score(Dist): {d_msg} | Score(Hops): {h_msg} {tr['score_note']}",
                ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round", fc="#f0f0f0"))

    # trajectory coloured by speed, and by time
    speed_vis = np.clip(tr["speed_raw"], None, SPEED_VMAX)
    if tr["appended"]:
        speed_vis = np.append(speed_vis, 0.0)
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_speed.scatter(tr["x_plot"], tr["y_plot"], c=speed_vis, s=10,
                     vmax=SPEED_VMAX, cmap="hot", rasterized=True)
    ax_speed.set_title("Speed Track")

    ax_time = fig.add_subplot(gs[1, 1])
    if len(tr["x_plot"]) > 1:
        pts = np.column_stack([tr["x_plot"], tr["y_plot"]])
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="cool",
                            norm=plt.Normalize(0, 1), rasterized=True)
        lc.set_array(np.linspace(0, 1, len(pts) - 1))
        ax_time.add_collection(lc)
    ax_time.set_title("Time Evolution")

    # shortest routes: physical distance and hop count
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_hops = fig.add_subplot(gs[2, 1])
    for ax, segs, colour, title in ((ax_dist, tr["segments_dist"], "b", "Shortest (distance)"),
                                    (ax_hops, tr["segments_hops"], "purple", "Shortest (hops)")):
        for path in segs:
            for p1, p2 in path:
                ax.plot([p1[0] / maze.SCALE_X, p2[0] / maze.SCALE_X],
                        [p1[1] / maze.SCALE_Y, p2[1] / maze.SCALE_Y],
                        colour, alpha=0.4, lw=3)
        ax.set_title(title)

    # occupancy: relative, and seconds per bin
    xmin, xmax, ymin, ymax = maze.MAZE_EXTENT
    H, _, _ = np.histogram2d(tr["x_m"], tr["y_m"], bins=list(OCCUPANCY_BINS),
                             range=[[xmin, xmax], [ymin, ymax]])
    H = H.T
    ax_rel = fig.add_subplot(gs[3, 0])
    ax_sec = fig.add_subplot(gs[3, 1])
    ax_rel.imshow(np.ma.masked_where(H == 0, H / (H.sum() or 1)),
                  extent=[xmin, xmax, ymax, ymin], cmap="jet", aspect="auto")
    ax_rel.set_title("Occupancy (relative)")
    ax_sec.imshow(np.ma.masked_where(H == 0, H * dt),
                  extent=[xmin, xmax, ymax, ymin], cmap="jet", aspect="auto", vmax=5)
    ax_sec.set_title("Occupancy (seconds)")

    # speed against real time and against normalised time
    ax_t = fig.add_subplot(gs[4, :])
    ax_n = fig.add_subplot(gs[5, :])
    for ax, xs, xlabel in ((ax_t, tr["time_s"], "time (s)"),
                           (ax_n, tr["normalized_time"], "normalised time")):
        ax.plot(xs, tr["speed_raw"], color="gray", alpha=0.3, label=f"{RAW_SMOOTH_S}s")
        ax.plot(xs, tr["speed_smooth"][0.5], label="0.5s")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("speed (m/s)")
        ax.legend(loc="upper right", fontsize=8)

    _style_maze_axes([ax_speed, ax_time, ax_dist, ax_hops, ax_rel, ax_sec], nodes)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_metadata_page(pdf, meta):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.axis("off")
    txt = "SESSION METADATA\n\n" + "\n".join(f"{k:<20}: {str(v)[:80]}" for k, v in meta.items())
    ax.text(0.1, 0.8, txt, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _plot_aggregate_page(pdf, trials):
    """Pooled speed distribution across every trial in the session."""
    speeds = [tr["speed_raw"] for tr in trials if len(tr["speed_raw"])]
    if not speeds:
        return
    fig, ax = plt.subplots()
    ax.hist(np.concatenate(speeds), bins=50, density=True, histtype="step", color="k")
    ax.set_title("Aggregate Speed Distribution")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("density")
    pdf.savefig(fig)
    plt.close(fig)


def _serialise_segments(paths):
    """Shortest-path segments as plain lists, for the pickle."""
    return [[[float(p[0]), float(p[1])] for p in seg] for path in paths for seg in path]


def _plot_data_frame(trials):
    """One-row DataFrame holding every plotted series, keyed by column.

    Kept in the original shape (one row, each cell a list-of-lists) so notebooks
    that already read these pickles keep working.
    """
    store = {
        "trial_ids": [tr["trial_id"] for tr in trials],
        "raw_x_scaled": [tr["x_plot"].tolist() for tr in trials],
        "raw_y_scaled": [tr["y_plot"].tolist() for tr in trials],
        "speed_raw_smoothed": [tr["speed_raw"].tolist() for tr in trials],
        "time_seconds": [tr["time_s"].tolist() for tr in trials],
        "normalized_time": [tr["normalized_time"].tolist() for tr in trials],
        "stitched_time_seconds": [np.asarray(tr["stitched_time"]).tolist() for tr in trials],
        "physical_score_val": [tr["dist_log_score"] for tr in trials],
        "hops_score_val": [tr["hops_log_score"] for tr in trials],
        "path_physical_segments": [_serialise_segments(tr["segments_dist"]) for tr in trials],
        "path_topological_segments": [_serialise_segments(tr["segments_hops"]) for tr in trials],
        "node_sequence_str": [tr["node_sequence"] for tr in trials],
    }
    for w in SMOOTH_WINDOWS_S:
        store[f"speed_{str(w).replace('.', '_')}s"] = [tr["speed_smooth"][w].tolist() for tr in trials]
    # column order matches the pre-package pickles
    order = ["trial_ids", "raw_x_scaled", "raw_y_scaled", "speed_raw_smoothed",
             "speed_0_5s", "speed_1_0s", "speed_2_0s", "speed_5_0s",
             "time_seconds", "normalized_time", "stitched_time_seconds",
             "physical_score_val", "hops_score_val",
             "path_physical_segments", "path_topological_segments", "node_sequence_str"]
    return pd.DataFrame([store])[order]


def metrics_table(trials):
    """Tidy per-trial metrics, one row per trial."""
    cols = ["trial_id", "goal_node", "start_node", "avg_speed", "median_speed",
            "dist_log_score", "hops_log_score", "actual_dist_px", "optimal_dist_px",
            "actual_hops", "optimal_hops", "duration_s", "goal_reached", "score_note"]
    return pd.DataFrame([{c: tr[c] for c in cols} for tr in trials], columns=cols)


def build_report(work_dir, out_dir=None):
    """Analyse one session folder and write the PDF, metrics CSV and pickle.

    Returns ``(metrics_table, plot_data_frame)``.
    """
    session = behaviour.load_session(work_dir)
    out_dir = Path(out_dir) if out_dir else session["work_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = maze.node_table()
    graph = maze.build_graph()
    if nodes.empty:
        print("Warning: node table unavailable; routes will not be scored.")

    trials = []
    for row in session["trials"].itertuples(index=False):
        if row.xy.size == 0:
            continue
        tr = analyse_trial(row.trial_id, row.xy,
                           session["node_sequences"].get(row.trial_id, ""),
                           session["goal_node"], nodes, graph)
        tr["stitched_time"] = row.stitched_time
        trials.append(tr)

    if not trials:
        raise ValueError(f"No trials with position data in {work_dir}")

    pdf_path = out_dir / f"{session['stem']}_analysis_final.pdf"
    print(f"Generating PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        if session["meta"]:
            _plot_metadata_page(pdf, session["meta"])
        for tr in trials:
            _plot_trial_page(pdf, tr, nodes)
        _plot_aggregate_page(pdf, trials)

    metrics = metrics_table(trials)
    csv_path = out_dir / f"{session['stem']}_trial_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    print(f"Wrote per-trial metrics: {csv_path}")

    plot_data = _plot_data_frame(trials)
    pkl_path = out_dir / f"{session['stem']}_all_plot_data.pkl"
    plot_data.to_pickle(pkl_path)
    print(f"Wrote compiled plot data: {pkl_path}")
    return metrics, plot_data


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-session behavioural report from tracker logs.")
    ap.add_argument("-o", "--output", dest="work_dir", required=True,
                    help="session folder containing the .log files")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the report (default: the session folder)")
    args = ap.parse_args(argv)
    if not Path(args.work_dir).exists():
        print(f"Error: {args.work_dir} does not exist.")
        return 1
    try:
        build_report(args.work_dir, args.out_dir)
    except Exception as e:
        print(f"[trial-report] Failed: {e}")
        traceback.print_exc()
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
