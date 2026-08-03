"""End-to-end: both reports run on a synthetic session and produce sane output."""

import numpy as np
import pandas as pd
import pytest

from hm_rat_analysis.reports import session_summary, trial_report


# ------------------------------------------------------- behavioural report
def test_trial_report_writes_all_three_outputs(session_dir, tmp_path):
    metrics, plot_data = trial_report.build_report(session_dir, out_dir=tmp_path)

    stem = "20210612_Rat5"
    assert (tmp_path / f"{stem}_analysis_final.pdf").stat().st_size > 0
    assert (tmp_path / f"{stem}_trial_metrics.csv").exists()
    assert (tmp_path / f"{stem}_all_plot_data.pkl").exists()

    assert list(metrics["trial_id"]) == [1, 2, 3, 4]
    assert (metrics["goal_node"] == "217").all()
    # every trial has a start node, so both scores resolve
    assert metrics["dist_log_score"].notna().all()
    # scores are log(optimal/actual): never above 0, since actual >= optimal
    assert (metrics["dist_log_score"] <= 1e-9).all()
    assert (metrics["actual_dist_px"] >= metrics["optimal_dist_px"]).all()


def test_trial_report_goal_detection_matches_the_fixture(session_dir, tmp_path):
    """Odd trials in the fixture end on the goal node; even trials are shifted away."""
    metrics, _ = trial_report.build_report(session_dir, out_dir=tmp_path)
    reached = dict(zip(metrics["trial_id"], metrics["goal_reached"]))
    assert reached[1] and reached[3]
    assert not reached[2] and not reached[4]
    notes = dict(zip(metrics["trial_id"], metrics["score_note"]))
    assert notes[1] == "(Start->FirstGoal)"
    assert notes[2] == "(Full Path)"


def test_plot_data_pickle_keeps_its_historical_columns(session_dir, tmp_path):
    """Notebooks read these pickles by column name — the schema is a contract."""
    _, plot_data = trial_report.build_report(session_dir, out_dir=tmp_path)
    assert list(plot_data.columns) == [
        "trial_ids", "raw_x_scaled", "raw_y_scaled", "speed_raw_smoothed",
        "speed_0_5s", "speed_1_0s", "speed_2_0s", "speed_5_0s",
        "time_seconds", "normalized_time", "stitched_time_seconds",
        "physical_score_val", "hops_score_val",
        "path_physical_segments", "path_topological_segments", "node_sequence_str"]
    row = plot_data.iloc[0]
    assert row["trial_ids"] == [1, 2, 3, 4]
    # every per-trial series has one entry per trial, and matching lengths
    for trial in range(4):
        n = len(row["raw_x_scaled"][trial])
        assert len(row["raw_y_scaled"][trial]) == n
        assert len(row["speed_raw_smoothed"][trial]) in (n, n - 1)
        assert np.isfinite(row["stitched_time_seconds"][trial]).all()


def _walk_shortest_path(start, goal, per_edge=40):
    """A dense trajectory (pixels) following the graph's own shortest path."""
    import networkx as nx
    from hm_rat_analysis import maze
    graph = maze.build_graph()
    path = nx.shortest_path(graph, start, goal, weight="weight")
    pos = nx.get_node_attributes(graph, "pos")
    legs = []
    for u, v in zip(path[:-1], path[1:]):
        (x0, y0), (x1, y1) = pos[u], pos[v]
        legs.append(np.column_stack([np.linspace(x0, x1, per_edge),
                                     np.linspace(y0, y1, per_edge)]))
    return np.vstack(legs), path


def test_analyse_trial_scores_an_optimal_route_near_zero():
    """Walking the graph's own shortest path scores log(optimal/actual) ~ 0."""
    from hm_rat_analysis import maze
    nodes, graph = maze.node_table(), maze.build_graph()
    start, goal = "101", "217"
    xy, path = _walk_shortest_path(start, goal)

    tr = trial_report.analyse_trial(1, xy, ", ".join(path), goal, nodes, graph)
    assert tr["goal_reached"]
    assert tr["dist_log_score"] == pytest.approx(0.0, abs=0.1)
    assert tr["hops_log_score"] == pytest.approx(0.0, abs=1e-9)


def test_a_wandering_route_scores_below_an_optimal_one():
    from hm_rat_analysis import maze
    nodes, graph = maze.node_table(), maze.build_graph()
    start, goal = "101", "217"
    xy, path = _walk_shortest_path(start, goal)

    direct = trial_report.analyse_trial(1, xy, ", ".join(path), goal, nodes, graph)
    # same endpoints, but with a detour bolted on before the goal is approached
    detour = np.vstack([xy[:len(xy) // 2], xy[:len(xy) // 2][::-1], xy])
    wander = trial_report.analyse_trial(2, detour, ", ".join(path), goal, nodes, graph)

    assert wander["actual_dist_px"] > direct["actual_dist_px"]
    assert wander["dist_log_score"] < direct["dist_log_score"]


def test_goal_radius_dominates_for_adjacent_nodes():
    """Neighbouring nodes are ~61 px apart while GOAL_RADIUS_PX is 50, so the goal
    registers almost immediately and the distance score is NOT ~0. Pinned because
    it makes short-hop trials look implausibly efficient."""
    from hm_rat_analysis import maze
    nodes, graph = maze.node_table(), maze.build_graph()
    a = nodes[nodes["id_str"] == "101"].iloc[0]
    b = nodes[nodes["id_str"] == "102"].iloc[0]
    assert np.hypot(a.x - b.x, a.y - b.y) < 2 * trial_report.GOAL_RADIUS_PX

    n = 60
    xy = np.column_stack([np.linspace(a.x, b.x, n), np.linspace(a.y, b.y, n)])
    tr = trial_report.analyse_trial(1, xy, "101, 102", "102", nodes, graph)
    assert tr["goal_reached"]
    assert tr["dist_log_score"] > 1.0


def test_trial_report_cli(session_dir, tmp_path):
    rc = trial_report.main(["-o", str(session_dir), "--out-dir", str(tmp_path)])
    assert rc == 0
    assert list(tmp_path.glob("*_analysis_final.pdf"))


def test_trial_report_cli_missing_folder_returns_error(tmp_path):
    assert trial_report.main(["-o", str(tmp_path / "nope")]) == 1


# ----------------------------------------------------------- session summary
def test_session_summary_end_to_end(nwb_root):
    # min_spikes is relaxed: the synthetic units are far below the 100-spike floor
    # the real report uses, and this test is about the plumbing, not the statistics.
    pdf, xlsx = session_summary.run(nwb_root, min_spikes=0)

    # the outputs are stamped with the parameters that produced them, so a 2.5 cm
    # run cannot overwrite a 5 cm one
    assert pdf.name == "session_summary_bin2.5cm_sm5cm_occ0.30s.pdf"
    assert pdf.stat().st_size > 0 and xlsx.exists()

    sessions = pd.read_excel(xlsx, "sessions")
    assert len(sessions) == 4
    assert set(sessions["animal"]) == {"Rat5", "Rat6"}
    assert (sessions["n_good"] == 6).all() and (sessions["n_mua"] == 2).all()
    # place-field metrics were actually computed, not silently skipped
    assert sessions["spatial_info"].notna().all()
    assert sessions["n_fields"].notna().all()
    # the fixture's goal node is real, so field-to-goal distances resolve
    assert sessions["field_goal_m"].notna().all()
    # the analysis parameters and the trial-window provenance are on every row
    assert (sessions["bin_cm"] == 2.5).all() and (sessions["smooth_cm"] == 5.0).all()
    assert (sessions["sigma_bins"] == 2.0).all()
    # the fixture has no coordinate/seconds files, so build_trials would fall back
    # to RecordingMeta times: the report must say so rather than trust them
    assert (sessions["trial_windows"] == "recordingmeta_fallback").all()

    units = pd.read_excel(xlsx, "units")
    assert len(units) == 24                       # 6 good units x 4 sessions
    assert set(units["cell_type"]) <= {"pyramidal", "interneuron"}
    # exposure columns travel with every unit row, or no confound can be checked
    for col in ("n_spikes_epoch", "epoch_dur_s", "occ_total_s", "n_valid_bins"):
        assert col in units.columns
    pf = units[units["cell_type"] == "pyramidal"]
    assert (pf["n_spikes_epoch"] > 0).any()

    notes = pd.read_excel(xlsx, "notes").set_index("metric")["note"]
    assert notes["spatial_info"] == "confounded_with_spike_count"
    assert notes["spatial_info_matched"] in ("", None) or pd.isna(notes["spatial_info_matched"])


def _parked_session(n_trials=10, run_n=300, park_n=900, dt=1 / 30.0):
    """A session with the real inter-trial artifact: between trials the rat is
    carried out and the series parks on the sentinel pixel (447, 303), reached by a
    single-frame teleport in and out."""
    from hm_rat_analysis import maze
    sx, sy = 447 / maze.SCALE_X, 303 / maze.SCALE_Y
    xs, ys, ts, wins, iti_spikes = [], [], [], [], []
    now = 0.0
    for _ in range(n_trials):
        tr = np.arange(run_n) * dt + now
        s = np.linspace(0, 1, run_n)
        xs.append(1.0 + 6.0 * s); ys.append(0.6 + 3.4 * s); ts.append(tr)
        wins.append((tr[0], tr[-1]))
        now = tr[-1] + dt
        tg = np.arange(park_n) * dt + now
        xs.append(np.full(park_n, sx)); ys.append(np.full(park_n, sy)); ts.append(tg)
        iti_spikes.append(tg[::10])            # a unit that fires ONLY off the maze
        now = tg[-1] + dt
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ts),
            wins, np.concatenate(iti_spikes), (sx, sy))


def test_inter_trial_parking_never_reaches_the_rate_map():
    """The sentinel pixel converts to maze node 318 — a point ON the graph, in a
    corridor the rat really runs — so no geometric or occupancy filter can remove
    it. Only the trial windows can, and the spikes must be windowed with them."""
    from hm_rat_analysis import maze, place_fields
    x, y, t, wins, st, sentinel = _parked_session()
    dt = 1 / 30.0
    windows = session_summary._merge_windows(wins)
    px, py, pt, spd = session_summary._prep_positions(x, y, t, windows, dt)
    kept = st[session_summary._in_windows(st, windows)]
    assert kept.size == 0, "every spike here was emitted between trials"

    bins = (360, 200)
    m, rate, _ = place_fields.place_field_metrics(
        px, py, pt, kept, maze.MAZE_EXTENT, bins, dt, sigma=2.0, speed_thresh=0.02,
        speed=spd, min_occ_s=0.30)
    ix = int((sentinel[0] - maze.MAZE_EXTENT[0]) / 9.0 * bins[0])
    iy = int((sentinel[1] - maze.MAZE_EXTENT[2]) / 5.0 * bins[1])
    assert np.ma.getmaskarray(rate)[iy, ix], "the parked bin must not be a valid bin"
    assert m["n_spikes_epoch"] == 0


def test_session_summary_on_empty_root_is_a_no_op(tmp_path, capsys):
    session_summary.run(tmp_path)
    assert "No .nwb files found" in capsys.readouterr().out
    assert not list(tmp_path.glob("session_summary*.pdf"))


def test_session_summary_cli(nwb_root):
    assert session_summary.main(["--root", str(nwb_root)]) == 0


# ------------------------------------------------- per-trial map divergence
def test_trial_divergence_scores_every_trial_against_both_references():
    """The two references answer different questions and both must reach the
    table: internal consistency, and drift from a goal-independent baseline."""
    import numpy as np
    from hm_rat_analysis import place_fields as PF
    from hm_rat_analysis.reports import session_summary as SS

    dt, n = 1 / 30.0, 12000
    t = np.arange(n) * dt
    s = np.abs(((np.arange(n) / 900.0) % 2.0) - 1.0)
    x, y = 1.0 + 6.0 * s, 0.6 + 3.4 * s
    spikes = t[np.hypot(x - 4.0, y - 2.0) < 0.35]
    wins = [(k * 40.0, (k + 1) * 40.0 - 1.0) for k in range(int(t[-1] // 40.0))]
    occs = [PF.occupancy_parts(x, y, t, (0, 9, 0, 5), (180, 100), dt, 1.0, t0=a, t1=b)
            for a, b in wins]
    udf = pd.DataFrame({"spike_times": [spikes], "phy_cluster_id": [7]})
    types = [1] * len(occs); types[0] = 4        # one free-roaming trial

    div = SS._trial_divergence(occs, types, udf, [0], {"animal": "Rat5"},
                               sigma=1.0, min_occ_s=0.30)
    assert len(div) == len(occs)
    for ref in SS.KL_REFERENCES:
        assert f"kl_{ref}_per_bin" in div and f"kl_{ref}_n_bins" in div
        assert div[f"kl_{ref}_per_bin"].notna().any(), f"{ref} produced nothing"
    # the free-roaming trial is scored too — it is the control, not a gap
    assert (div["trial_type"] == 4).sum() == 1

    slopes = SS._kl_unit_slopes(div, {"animal": "Rat5"})
    assert len(slopes) == 1 and slopes.loc[0, "unit_id"] == 7
    # goal trials only: the free-roaming trial must not enter the cumulative curve
    assert slopes.loc[0, "n_trials"] == len(occs) - 1


def test_auto_si_match_picks_one_count_that_every_session_can_supply():
    """A per-session threshold would be wrong: SI at 300 spikes is not comparable
    with SI at 100, so the count has to be global. What the auto mode equalises is
    the FRACTION of cells contributing, which a fixed count does not."""
    import numpy as np
    from hm_rat_analysis.reports import session_summary as SS

    rich = np.array([2000, 1500, 1200, 900, 800, 700, 600, 500, 400, 350], float)
    thin = np.array([600, 400, 300, 220, 180, 150, 120, 100, 90, 80], float)
    n, frac = SS.choose_si_match_n([rich, thin], target_frac=0.9)
    assert n is not None
    # every session keeps at least the target fraction, thin one included
    for c in (rich, thin):
        assert (c >= n).mean() >= 0.9 - 1e-9
    assert frac >= 0.9
    # the thin session is what binds it, and the count lands far below the fixed
    # 300 that keeps under half of that session's cells
    assert n < np.median(thin)
    assert (thin >= 300).mean() < 0.5

    # a stricter target can only lower the count
    n2, _ = SS.choose_si_match_n([rich, thin], target_frac=1.0)
    assert n2 <= n
    assert SS.choose_si_match_n([], target_frac=0.9)[0] is None
