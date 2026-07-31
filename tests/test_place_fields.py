import numpy as np
import pytest

from hm_rat_analysis import maze, place_fields

EXTENT = maze.MAZE_EXTENT
BINS = (90, 50)          # 10 cm bins over a 9 x 5 m maze
DT = 1 / 30.0


def _sweep(n=9000):
    """A trajectory that covers the maze in raster sweeps, in metres."""
    t = np.arange(n) * DT
    rows = 25
    x = np.linspace(0, 9, n // rows).tolist() * rows
    x = np.array((x + x[:n - len(x)])[:n])
    y = np.repeat(np.linspace(0.2, 4.8, rows), n // rows)[:n]
    if y.size < n:
        y = np.concatenate([y, np.full(n - y.size, y[-1])])
    return x, y, t


def test_rate_map_is_masked_off_the_path():
    x, y, t = _sweep()
    rate, extent = place_fields.make_rate_map(x, y, t, np.array([]), EXTENT, BINS, DT, 0)
    assert extent == EXTENT
    assert np.ma.is_masked(rate)
    # no spikes anywhere -> every visited bin is zero, unvisited bins are masked
    assert rate.count() > 0
    assert float(rate.max()) == 0.0


def test_rate_map_too_few_samples_returns_none():
    rate, extent = place_fields.make_rate_map(
        np.array([1.0]), np.array([1.0]), np.array([0.0]),
        np.array([]), EXTENT, BINS, DT, 0)
    assert rate is None and extent == EXTENT


def test_place_field_metrics_localise_a_single_field():
    x, y, t = _sweep()
    centre = (4.5, 2.5)
    d = np.hypot(x - centre[0], y - centre[1])
    spikes = t[d < 0.4]                       # fires only near the maze centre
    assert spikes.size > 20

    m, rate, _ = place_fields.place_field_metrics(
        x, y, t, spikes, EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0,
        goal_xy=centre)
    assert m["n_fields"] >= 1
    assert m["spatial_info"] > 1.0, "a compact field carries high spatial information"
    assert m["selectivity"] > 2.0
    # the field sits on the goal, so the distance to it is small
    assert m["field_goal_m"] == pytest.approx(0.0, abs=0.6)


def test_place_field_metrics_without_goal_gives_nan_distances():
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    m, _, _ = place_fields.place_field_metrics(
        x, y, t, t[d < 0.4], EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0,
        goal_xy=None)
    assert m["n_fields"] >= 1                 # fields are still found
    for key in ("field_goal_m", "field_goal_largest_m", "field_goal_smallest_m"):
        assert np.isnan(m[key]), "no goal node -> distances are NaN, not an error"


def test_place_field_metrics_with_no_spikes_is_empty_not_an_error():
    x, y, t = _sweep()
    m, _, _ = place_fields.place_field_metrics(
        x, y, t, np.array([]), EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0)
    assert m["n_fields"] == 0
    assert np.isnan(m["spatial_info"]) or m["spatial_info"] == 0.0


def test_place_fields_respects_minimum_size():
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    _, rate, _ = place_fields.place_field_metrics(
        x, y, t, t[d < 0.4], EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0)
    many = place_fields.place_fields(rate, min_field_bins=1)
    few = place_fields.place_fields(rate, min_field_bins=10_000)
    assert len(many) >= 1 and few == []


def test_place_field_mask_picks_the_peak_region():
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    _, rate, _ = place_fields.place_field_metrics(
        x, y, t, t[d < 0.4], EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0)
    mask = place_fields.place_field_mask(rate)
    assert mask is not None and mask.any()
    peak_idx = np.unravel_index(np.ma.argmax(rate), rate.shape)
    assert mask[peak_idx], "the mask must contain the peak bin"
