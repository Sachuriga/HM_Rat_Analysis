import numpy as np
import pytest

from hm_rat_analysis import behaviour


# ---------------------------------------------------------------- kinematics
def test_parse_video_to_seconds():
    assert behaviour.parse_video_to_seconds("01:02:03.500") == pytest.approx(3723.5)
    assert behaviour.parse_video_to_seconds("") is None
    assert behaviour.parse_video_to_seconds("nonsense") is None


def test_moving_average_preserves_length_and_mean_level():
    a = np.arange(20, dtype=float)
    for k in (1, 3, 8):
        out = behaviour.moving_average(a, k)
        assert out.shape == a.shape
    # a constant signal survives smoothing untouched (edge padding matters here)
    const = np.full(15, 4.0)
    np.testing.assert_allclose(behaviour.moving_average(const, 5), const)
    assert behaviour.moving_average(np.array([]), 3).size == 0


def test_compute_speed_from_xy_on_constant_velocity():
    fs = 30.0
    t = np.arange(100) / fs
    x, y = 2.0 * t, np.zeros_like(t)          # 2 m/s along x
    speed = behaviour.compute_speed_from_xy(x, y, fs)
    np.testing.assert_allclose(speed, 2.0)


def test_compute_path_length():
    x = np.array([0.0, 3.0, 3.0])
    y = np.array([0.0, 0.0, 4.0])
    assert behaviour.compute_path_length(x, y) == pytest.approx(7.0)
    assert behaviour.compute_path_length(np.array([1.0]), np.array([1.0])) == 0.0


def test_nearest_stitched_times_picks_the_closer_neighbour():
    ref_ts = np.array([0.0, 10.0, 20.0])
    ref_sec = np.array([100.0, 110.0, 120.0])
    got = behaviour.nearest_stitched_times([-5.0, 4.0, 6.0, 99.0], ref_ts, ref_sec)
    np.testing.assert_allclose(got, [100.0, 100.0, 110.0, 120.0])


def test_nearest_stitched_times_with_empty_reference_is_all_nan():
    got = behaviour.nearest_stitched_times([1.0, 2.0], np.array([]), np.array([]))
    assert np.isnan(got).all()


# ------------------------------------------------------------------- parsing
def test_parse_logs_and_trial_ids(session_dir):
    files = behaviour.find_session_files(session_dir)
    assert files["log"] and files["txt"] and files["meta"]

    df = behaviour.parse_logs(files["log"])
    assert not df.empty
    assert set(df["event"]) >= {"rat_position", "recording_start"}
    positions = df[df["event"] == "rat_position"]
    assert positions["x"].notna().all() and positions["y"].notna().all()

    tagged = behaviour.assign_trial_ids(df)
    assert sorted(tagged["trial_id"].unique().tolist()) == [1, 2, 3, 4]
    # a trial marker applies to the rows that follow it
    first_marker = tagged[tagged["event"] == "recording_start"].index[0]
    assert tagged.loc[first_marker, "trial_id"] == 1


def test_parse_node_sequences(session_dir):
    txt = behaviour.find_session_files(session_dir)["txt"][0]
    seqs = behaviour.parse_node_sequences(txt)
    assert sorted(seqs) == [1, 2, 3, 4]
    assert all(s and "," in s for s in seqs.values())


def test_load_session_meta_returns_goal_as_string(session_dir):
    meta, goal = behaviour.load_session_meta(session_dir)
    assert meta["Rat"] == 5
    assert goal == "217", "graph nodes are keyed by string, so the goal must be too"


def test_load_time_reference_and_session(session_dir):
    ref_ts, ref_sec = behaviour.load_time_reference(session_dir)
    assert ref_ts is not None and len(ref_ts) == len(ref_sec)

    session = behaviour.load_session(session_dir)
    assert session["goal_node"] == "217"
    assert len(session["trials"]) == 4
    for row in session["trials"].itertuples(index=False):
        assert row.xy.ndim == 2 and row.xy.shape[1] == 2
        assert len(row.stitched_time) == len(row.xy)
        assert np.isfinite(row.stitched_time).all()


def test_trial_trajectories_drops_the_tail(session_dir):
    files = behaviour.find_session_files(session_dir)
    df = behaviour.assign_trial_ids(behaviour.parse_logs(files["log"]))
    full = behaviour.trial_trajectories(df, drop_tail_samples=0)
    trimmed = behaviour.trial_trajectories(df, drop_tail_samples=5)
    for a, b in zip(full.itertuples(index=False), trimmed.itertuples(index=False)):
        assert len(b.xy) == len(a.xy) - 5


def test_load_session_without_logs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        behaviour.load_session(tmp_path)


# ------------------------------------------------------------ task performance
def _perf_meta(tmp_path, rows):
    import pandas as pd
    pd.DataFrame(rows).to_excel(tmp_path / "RecordingMeta.xlsx", index=False)
    return tmp_path


def test_trial_performance_scores_an_optimal_run_at_zero(tmp_path):
    """log10(shortest/actual) is 0 when the rat walked the graph's own shortest
    path, and negative for every detour — so the sign alone says 'suboptimal'."""
    import networkx as nx
    from hm_rat_analysis import maze

    G = maze.build_graph()
    start, goal = "224", "217"
    route = nx.shortest_path(G, start, goal)
    detour = route[:1] + [n for n in nx.shortest_path(G, start, route[1])] + route[1:]

    d = _perf_meta(tmp_path, [
        {"Trial_Type": 1, "Start_Nodes": int(start), "Goal_Node": int(goal),
         "paths": ", ".join(route)},
        {"Trial_Type": 1, "Start_Nodes": int(start), "Goal_Node": int(goal),
         "paths": ", ".join(detour)},
    ])
    tp = behaviour.trial_performance(d)
    assert list(tp["trial"]) == [1, 2]
    assert tp.loc[0, "performance"] == pytest.approx(0.0)
    assert tp.loc[1, "performance"] < 0
    assert tp.loc[0, "actual_hops"] == tp.loc[0, "shortest_hops"] == len(route) - 1


def test_trial_performance_stops_at_the_first_goal_visit(tmp_path):
    """A rat that reaches the goal and keeps walking has still solved the trial."""
    import networkx as nx
    from hm_rat_analysis import maze

    G = maze.build_graph()
    start, goal = "224", "217"
    route = nx.shortest_path(G, start, goal)
    overshoot = route + [n for n in G.neighbors(goal)][:2]
    d = _perf_meta(tmp_path, [{"Trial_Type": 1, "Start_Nodes": int(start),
                               "Goal_Node": int(goal), "paths": ", ".join(overshoot)}])
    assert behaviour.trial_performance(d).loc[0, "performance"] == pytest.approx(0.0)


def test_trial_performance_without_a_metadata_sheet_is_empty_not_an_error(tmp_path):
    tp = behaviour.trial_performance(tmp_path)
    assert tp.empty and "performance" in tp.columns
