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
