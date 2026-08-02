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
    """A silent cell yields NaN for EVERY metric, n_fields included.

    n_fields used to come back as a finite 0 here while the other metrics were
    NaN, so the "# place fields" panel was averaged over a larger population than
    every other panel — and the gap grew as sessions got shorter, moving the panel
    with no change in the cells. The exposure counters are still reported.
    """
    x, y, t = _sweep()
    m, _, _ = place_fields.place_field_metrics(
        x, y, t, np.array([]), EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0)
    assert np.isnan(m["n_fields"])
    assert np.isnan(m["spatial_info"])
    assert m["n_spikes_epoch"] == 0 and m["n_valid_bins"] > 0


def test_place_fields_minimum_size_is_an_area_not_a_bin_count():
    """The size floor is cm^2, so it means the same thing at any bin size."""
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    _, rate, _ = place_fields.place_field_metrics(
        x, y, t, t[d < 0.4], EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0)
    area = np.prod(place_fields.bin_size_cm(EXTENT, BINS))     # 100 cm^2 at 10 cm
    many = place_fields.place_fields(rate, min_field_cm2=area, bin_area_cm2=area)
    few = place_fields.place_fields(rate, min_field_cm2=1e6, bin_area_cm2=area)
    assert len(many) >= 1 and few == []


def test_place_fields_still_accepts_the_deprecated_bin_count():
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    _, rate, _ = place_fields.place_field_metrics(
        x, y, t, t[d < 0.4], EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0)
    with pytest.warns(DeprecationWarning):
        many = place_fields.place_fields(rate, min_field_bins=1)
    with pytest.warns(DeprecationWarning):
        few = place_fields.place_fields(rate, min_field_bins=10_000)
    assert len(many) >= 1 and few == []


def test_field_count_survives_halving_the_bin_size():
    """The regression guard for the 2.5 cm artifact.

    One field on a DIAGONAL corridor — the shape the rat actually runs. With
    4-connectivity, a bin-count size floor or a raw-occupancy validity rule, the
    same cell reports a different number of differently sized fields at 2.5 cm
    than at 5 cm purely from the grid.
    """
    n = 12000
    t = np.arange(n) * DT
    s = np.abs(((np.arange(n) / 900.0) % 2.0) - 1.0)          # back and forth
    x = 1.0 + 6.0 * s
    y = 0.6 + 3.4 * s                                          # 30-degree diagonal
    d = np.hypot(x - 4.0, y - 2.0)
    spikes = t[d < 0.35]
    assert spikes.size > 200

    out = {}
    for bins, sigma in (((180, 100), 1.0), ((360, 200), 2.0)):  # 5 cm and 2.5 cm, 5 cm kernel
        m, _, _ = place_fields.place_field_metrics(
            x, y, t, spikes, EXTENT, bins, DT, sigma=sigma, speed_thresh=0.0,
            goal_xy=(4.0, 2.0), min_field_cm2=60.0, min_occ_s=0.30)
        out[bins] = m
    coarse, fine = out[(180, 100)], out[(360, 200)]
    assert coarse["n_fields"] == fine["n_fields"] == 1
    assert fine["field_goal_largest_m"] == pytest.approx(
        coarse["field_goal_largest_m"], abs=0.15)
    # The field's AREA is not bin-size invariant on a 1-D track and cannot be: the
    # corridor is narrower than one bin at either resolution, so the covered area
    # is (track length) x (bin width) and halves with the bin. What is invariant is
    # the field's LENGTH along the track, which is what the 60 cm^2 floor is chosen
    # against (60 cm^2 = 12 cm of track at 5 cm bins, 24 cm at 2.5 cm) — so quote
    # field extent in track cm, never in cm^2, when comparing across resolutions.
    length_cm = {b: m["field_size_largest_m2"] * 1e4 / place_fields.bin_size_cm(EXTENT, b)[0]
                 for b, m in out.items()}
    assert length_cm[(360, 200)] == pytest.approx(length_cm[(180, 100)], rel=0.25)


def test_spike_count_matched_spatial_information():
    """The count-matched SI is defined only above the matching count, and is
    reproducible for a given seed — it is the one spatial-tuning number in the
    report that two sessions of different length can be compared on."""
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    spikes = t[d < 0.6]
    assert spikes.size > 150

    kw = dict(sigma=1.0, speed_thresh=0.0, si_match_n=150, si_match_repeats=5, seed=7)
    m, _, _ = place_fields.place_field_metrics(x, y, t, spikes, EXTENT, BINS, DT, **kw)
    again, _, _ = place_fields.place_field_metrics(x, y, t, spikes, EXTENT, BINS, DT, **kw)
    assert np.isfinite(m["spatial_info_matched"])
    assert m["spatial_info_matched"] == again["spatial_info_matched"]

    thin, _, _ = place_fields.place_field_metrics(
        x, y, t, spikes[:80], EXTENT, BINS, DT, **kw)
    assert np.isnan(thin["spatial_info_matched"]), "fewer spikes than the match count"
    assert np.isfinite(thin["spatial_info"]), "the raw (biased) value is still reported"


def test_field_to_goal_carries_its_occupancy_matched_null():
    """A field-to-goal distance is bounded by the ground the animal covered, so the
    null — the mean goal distance of that epoch's own occupancy — travels with it."""
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    m, _, _ = place_fields.place_field_metrics(
        x, y, t, t[d < 0.4], EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0,
        goal_xy=(4.5, 2.5))
    assert m["field_goal_null_m"] > m["field_goal_m"], "the field is nearer than chance"
    assert m["field_goal_over_null"] == pytest.approx(
        m["field_goal_m"] / m["field_goal_null_m"])


def test_place_field_mask_picks_the_peak_region():
    x, y, t = _sweep()
    d = np.hypot(x - 4.5, y - 2.5)
    _, rate, _ = place_fields.place_field_metrics(
        x, y, t, t[d < 0.4], EXTENT, BINS, DT, sigma=1.0, speed_thresh=0.0)
    mask = place_fields.place_field_mask(rate)
    assert mask is not None and mask.any()
    peak_idx = np.unravel_index(np.ma.argmax(rate), rate.shape)
    assert mask[peak_idx], "the mask must contain the peak bin"
