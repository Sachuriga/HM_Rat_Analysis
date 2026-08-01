import networkx as nx
import numpy as np
import pytest

from hm_rat_analysis import maze


def test_node_table_has_both_coordinate_systems():
    df = maze.node_table()
    assert not df.empty
    assert {"id", "x", "y", "id_str", "x_m", "y_m"} <= set(df.columns)
    # metres are pixels / scale, and land inside the maze box
    np.testing.assert_allclose(df["x_m"], df["x"] / maze.SCALE_X)
    np.testing.assert_allclose(df["y_m"], df["y"] / maze.SCALE_Y)
    xmin, xmax, ymin, ymax = maze.MAZE_EXTENT
    assert df["x_m"].between(xmin, xmax).all()
    assert df["y_m"].between(ymin, ymax).all()


def test_node_ids_are_the_documented_range():
    """Goal nodes come from metadata as ints; a caller passing 1..N gets nothing,
    so pin the real id range to keep that failure mode visible."""
    ids = sorted(maze.node_positions_m())
    assert ids[0] == 101 and ids[-1] == 502
    assert len(ids) == 98


def test_graph_excludes_corner_nodes_and_adds_manual_edges():
    G = maze.build_graph()
    for n in maze.EXCLUDED_NODES:
        assert n not in G
    for u, v in maze.MANUAL_EDGES:
        assert G.has_edge(u, v)
    assert nx.is_connected(G), "every maze node should be reachable"


def test_graph_edges_are_short_and_weighted_by_distance():
    G = maze.build_graph()
    pos = nx.get_node_attributes(G, "pos")
    manual = {frozenset(e) for e in maze.MANUAL_EDGES}
    for u, v, w in G.edges(data="weight"):
        expected = np.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1])
        assert w == pytest.approx(expected)
        if frozenset((u, v)) not in manual:
            assert w < maze.ADJACENCY_THRESHOLD_PX


def test_shortest_path_segments_distance_and_hops():
    G = maze.build_graph()
    start, end = "101", "217"
    segs, label, metric = maze.shortest_path_segments(G, start, end, "weight")
    assert segs and metric > 0 and label.startswith("Dist")
    # each entry is one tied path, made of (p1, p2) point pairs
    assert all(len(seg) == 2 for path in segs for seg in path)

    hop_segs, hop_label, hops = maze.shortest_path_segments(G, start, end, None)
    assert hop_label.startswith("Hops")
    # hop count equals the number of segments along one shortest path
    assert hops == len(hop_segs[0])


def test_shortest_path_segments_handles_unknown_nodes():
    G = maze.build_graph()
    segs, label, metric = maze.shortest_path_segments(G, "101", "9999", "weight")
    assert segs == [] and metric == 0.0 and label == "Node not found"


def test_node_table_is_cached_but_callers_get_the_same_object():
    assert maze.node_table() is maze.node_table()
    assert maze.build_graph() is maze.build_graph()


def test_idealised_layout_is_a_regular_lattice():
    """Every edge should be one of the six lattice directions and a whole number
    of segments long — that is what makes the drawn maze look clean."""
    G = maze.build_graph()
    ideal = maze.idealised_positions(G)
    assert set(ideal) == set(G.nodes)

    lengths, angles = [], []
    for u, v in G.edges():
        d = np.array(ideal[v]) - np.array(ideal[u])
        lengths.append(float(np.hypot(*d)))
        angles.append(np.degrees(np.arctan2(d[1], d[0])) % 60)

    for a in angles:
        assert min(a, 60 - a) == pytest.approx(0, abs=1e-6), \
            "edges must run at multiples of 60 degrees"

    unit = min(lengths)                     # the lattice step is fitted, not fixed
    assert 0.3 < unit < 0.5
    for length in lengths:
        steps = length / unit
        assert steps == pytest.approx(round(steps), abs=1e-6), \
            "edge lengths must be whole multiples of the lattice unit"


def test_idealised_layout_stays_registered_to_the_measured_one():
    """The fit is translation-only, so the idealised maze must not drift away from
    the measured coordinates a rate map is built in."""
    res = np.array(list(maze.idealisation_residuals().values()))
    assert res.max() < 0.25, "no node should move more than a quarter of an edge"
    assert np.median(res) < 0.15


def test_warp_maps_measured_nodes_onto_idealised_nodes():
    G = maze.build_graph()
    ideal = maze.idealised_positions(G)
    df = maze.node_table()
    df = df[df["id_str"].isin(ideal)]
    wx, wy = maze.warp_to_idealised(df["x_m"].to_numpy(), df["y_m"].to_numpy())
    target = np.array([ideal[s] for s in df["id_str"]])
    # the spline interpolates the correspondences exactly
    np.testing.assert_allclose(np.column_stack([wx, wy]), target, atol=1e-6)


def test_warp_passes_nans_through():
    wx, wy = maze.warp_to_idealised(np.array([np.nan, 4.0]), np.array([2.0, np.nan]))
    assert np.isnan(wx[0]) and np.isnan(wy[0])
    assert np.isnan(wx[1]) and np.isnan(wy[1])
