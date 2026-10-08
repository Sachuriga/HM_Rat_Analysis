"""Figure 1's summary strip: the panels read the summary tables, so the contract
tested here is the one between hm-session-summary's sheets and the figure."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import pytest                                                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import msca_fig1 as FIG                                           # noqa: E402


@pytest.fixture
def summary():
    """Two animals x three sessions, with the columns the strip reads."""
    rng = np.random.default_rng(0)
    sessions, units, trials = [], [], []
    for animal in ("Rat5", "Rat6"):
        for i, date in enumerate(("20260622", "20260623", "20260624")):
            sessions.append({"animal": animal, "date": date, "repeat": 1,
                             "session": i + 1, "n_good": 40 + i, "n_mua": 60 - i,
                             "performance_med": -0.2 + 0.05 * i,
                             "performance_n_trials": 10, "bin_cm": 2.5,
                             "smooth_cm": 5.0, "sigma_bins": 2.0, "min_occ_s": 0.3,
                             "field_frac": 0.3, "min_field_cm": 15.0,
                             "speed_thresh_ms": 0.02, "min_spikes": 100})
            for u in range(12):
                units.append({"animal": animal, "date": date, "repeat": 1,
                              "session": i + 1, "session_label": f"R1S{i + 1}",
                              "epoch": "whole", "unit_id": u,
                              "spatial_info": float(rng.gamma(2, 0.4)),
                              "spatial_info_matched": float(rng.gamma(2, 0.3)),
                              "n_fields": int(rng.integers(1, 4)),
                              "field_size_mean_cm": float(rng.gamma(6, 6)),
                              "stability": float(rng.uniform(-0.2, 0.9))})
            for k in range(10):
                trials.append({"animal": animal, "date": date, "repeat": 1,
                               "session": i + 1, "trial": k + 1, "trial_type": 1,
                               "performance": float(-rng.random() * 0.4)})
    return {"sessions": pd.DataFrame(sessions), "units": pd.DataFrame(units),
            "trials": pd.DataFrame(trials)}


def _strip(summary, **kw):
    # ALL_STRIP, not STRIP: the figure now draws only the first two panels, but the
    # capping, missing-column and per-animal behaviour these tests cover is the same
    # code for all six, and it is still reachable through `strip=`.
    kw.setdefault("strip", FIG.ALL_STRIP)
    fig, axes = plt.subplots(1, len(FIG.ALL_STRIP), figsize=(15, 2))
    stamp, si_col, capped = FIG.summary_strip(list(axes), summary, **kw)
    return fig, list(axes), stamp, si_col, capped


def test_every_panel_draws_something(summary):
    fig, axes, stamp, _, _capped = _strip(summary, animal="Rat6")
    try:
        # bars in b, dots + a median line in c-g: no panel may come back empty,
        # which is what a renamed summary column would silently produce
        assert len(axes[0].patches) > 0, "panel b drew no bars"
        for ax, (key, letter, *_rest) in list(zip(axes, FIG.ALL_STRIP))[1:]:
            assert ax.collections, f"panel {letter} ({key}) drew no measurements"
            assert ax.lines, f"panel {letter} ({key}) drew no median line"
        assert "bin 2.50 cm" in stamp
    finally:
        plt.close(fig)


def _bars(ax):
    """The unit-yield bars, without the shaded first-session bands.

    ``ax.axvspan`` puts a Rectangle into ``ax.patches``, the same list the bars
    live in, so a test that counts patches counts the bands too.
    """
    return [p for p in ax.patches if p.get_label() != FIG.BAND_LABEL]


def test_one_animal_is_selected_not_pooled(summary):
    """Panels b-g must show the animal panel a came from, or the unit counts of two
    rats land in one bar and the medians mix two populations."""
    fig, axes, _, _, _capped = _strip(summary, animal="Rat6")
    try:
        heights = sorted({round(p.get_height()) for p in _bars(axes[0])})
        assert heights == [40, 41, 42, 58, 59, 60], "counts are not one animal's"
        assert len(axes[0].get_xticks()) == 3
    finally:
        plt.close(fig)


def test_both_animals_share_the_axis_without_being_pooled(summary):
    """Animals are not recorded on the same days, so R{repeat}S{session} is the only
    slot they can share — but they stay separate series on it: one line hiding a
    disagreement between two rats is worse than a busier panel."""
    fig, axes, _, _, _capped = _strip(summary)                 # default: every animal
    try:
        assert len(axes[0].get_xticks()) == 3, "three session slots, not six"
        # panel b: a good and an MUA bar per animal per slot, never summed
        bars = _bars(axes[0])
        assert len(bars) == 2 * 2 * 3
        assert round(bars[0].get_height()) == 40, "Rat5's own count"
        # panels c-g: one median line (plus its marker line) per animal
        assert {l.get_label() for l in axes[2].lines} >= {"Rat5", "Rat6"}
    finally:
        plt.close(fig)


def test_sessions_are_grouped_by_repeat_with_a_gap_between_blocks(summary):
    """Session numbers restart inside each repeat, so R1S4 sits next to R2S1 with
    nothing to say a new block began — the gap is what makes the axis readable."""
    s = {k: v.copy() for k, v in summary.items()}
    for name in ("sessions", "units", "trials"):      # 3 sessions -> repeats 1,1,2
        s[name]["repeat"] = np.where(s[name]["session"] == 3, 2, 1)
    _sess, keys, meta, _animals = FIG.session_axis(s)
    pos = FIG.slot_positions(meta)
    assert keys == ["R1S1", "R1S2", "R2S3"]
    assert pos[1] - pos[0] == pytest.approx(1.0), "same repeat: adjacent"
    assert pos[2] - pos[1] > 1.0, "repeat changes: a gap opens"
    assert FIG.repeat_groups(meta, pos) == [(1, pos[0], pos[1]), (2, pos[2], pos[2])]

    fig, axes, _, _, _capped = _strip(s)
    try:                                              # ticks carry the session only
        assert [t.get_text() for t in axes[0].get_xticklabels()] == ["1", "2", "3"]
        labels = " ".join(t.get_text() for t in axes[0].texts)
        assert "goal 1" in labels and "goal 2" in labels
    finally:
        plt.close(fig)


def test_blocks_are_named_by_goal_and_repeat_zero_is_habituation(summary):
    """A block of sessions is one goal location; repeat 0 predates any goal."""
    assert FIG.group_label(0) == "habituation"
    assert FIG.group_label(2) == "goal 2"

    s = {k: v.copy() for k, v in summary.items()}
    for name in ("sessions", "units", "trials"):
        s[name]["repeat"] = np.where(s[name]["session"] == 1, 0, 1)
    fig, axes, _, _, _capped = _strip(s)
    try:
        labels = " ".join(t.get_text() for t in axes[0].texts)
        assert "habituation" in labels and "goal 1" in labels
        assert "goal 0" not in labels
    finally:
        plt.close(fig)


def test_the_cap_never_hides_a_point_that_was_drawn(summary):
    """Panels c-g draw session MEANS, not individual cells, so a mean above the
    nominal cap is a result and not a tail: the axis has to grow past the cap to
    hold it. An axis capped below a drawn point would show a flat series where the
    data has a spike, which is the one failure a cap must not cause."""
    s = {k: v.copy() for k, v in summary.items()}
    # a tail big enough to drag the session means it belongs to above the cap
    s["units"].loc[s["units"].index[:5], "field_size_mean_cm"] = 500.0
    cap = FIG.YMAX["field_size_mean_cm"]
    fig, axes, _, _si_col, capped = _strip(s)
    try:
        f = axes[4]
        top = f.get_ylim()[1]
        ys = np.concatenate([_median_line(f, a).get_ydata() for a in ("Rat5", "Rat6")])
        assert np.nanmax(ys) > cap, "fixture did not push a mean above the cap"
        assert np.nanmax(ys) <= top, "a drawn point is above the axis"
        assert top > cap, "the axis did not grow to hold it"
        # Nothing was hidden, so there is nothing for the caption to report.
        assert "Field size" not in capped
        assert not [t for t in f.texts if "above" in t.get_text()]
    finally:
        plt.close(fig)


def test_the_cap_is_the_top_when_every_mean_fits_inside_it(summary):
    """Where the cap binds it is the top exactly, so the panel keeps the round
    number the caption quotes instead of a padded one just past it.

    With means rather than individual cells the autoscaled span is usually well
    inside the cap and the cap does nothing; the one case it still decides is
    this one, where the series runs close under the cap and the padding would
    otherwise carry the axis over it.
    """
    s = {k: v.copy() for k, v in summary.items()}
    cap = FIG.YMAX["field_size_mean_cm"]
    s["units"]["field_size_mean_cm"] = cap - 2.0
    fig, axes, _, _, capped = _strip(s)
    try:
        assert axes[4].get_ylim()[1] == pytest.approx(cap)
        assert "Field size" not in capped, "nothing is above a cap that holds"
    finally:
        plt.close(fig)


def test_more_animals_than_separable_hues_is_refused(summary):
    """A fifth hue is indistinguishable from an existing one under CVD, so the
    strip refuses rather than quietly drawing two animals in the same colour."""
    extra = []
    for i in range(3):
        d = summary["sessions"].copy()
        d["animal"] = f"Rat{20 + i}"
        extra.append(d)
    summary["sessions"] = pd.concat([summary["sessions"]] + extra, ignore_index=True)
    with pytest.raises(ValueError, match="colour-vision"):
        FIG.session_axis(summary)


def test_spatial_information_prefers_the_count_matched_column(summary):
    fig, _, _, si_col, _capped = _strip(summary, animal="Rat6")
    plt.close(fig)
    assert si_col == "spatial_info_matched"
    fig, _, _, si_col, _capped = _strip(summary, animal="Rat6", si_mode="raw")
    plt.close(fig)
    assert si_col == "spatial_info", "--si raw must not be silently upgraded"


def test_a_summary_without_the_new_columns_says_so_instead_of_failing(summary):
    """An xlsx written before stability existed still draws — panels f and g say
    what is missing rather than the figure failing at the last step."""
    summary["units"] = summary["units"].drop(columns=["stability",
                                                      "field_size_mean_cm"])
    fig, axes, _, _, _capped = _strip(summary, animal="Rat6")
    try:
        texts = " ".join(t.get_text() for ax in axes for t in ax.texts)
        assert "stability" in texts and "field_size_mean_cm" in texts
    finally:
        plt.close(fig)


def _median_line(ax, animal):
    """The animal's median line — by LABEL, since the axis also carries unlabelled
    lines for the goal-block rules under it."""
    return next(l for l in ax.lines if l.get_label() == animal)


def test_missing_per_trial_sheet_falls_back_to_session_medians(summary):
    del summary["trials"]
    fig, axes, _, _, _capped = _strip(summary, animal="Rat6")
    try:
        ys = _median_line(axes[1], "Rat6").get_ydata()
        assert np.isfinite(ys).all() and (ys < 0).all()
    finally:
        plt.close(fig)


def test_load_summary_reads_the_sheets_it_needs(summary, tmp_path):
    p = tmp_path / "session_summary_bin2.5cm_sm5cm_occ0.30s.xlsx"
    with pd.ExcelWriter(p) as xw:
        for name, df in summary.items():
            df.to_excel(xw, index=False, sheet_name=name)
    for arg in (p, tmp_path):                 # the file, or the folder holding it
        got = FIG.load_summary(arg)
        assert {"sessions", "units", "trials"} <= set(got)
        assert len(got["sessions"]) == 6


def test_numeric_dates_from_the_workbook_still_match(summary):
    """YYYYMMDD is all digits, so a reader may hand it back as int64 (or float64
    with one blank row) — the join must survive both, or every panel comes back
    empty with nothing raised."""
    for cast in ("int64", "float64"):
        s = {k: v.copy() for k, v in summary.items()}
        for name in ("sessions", "units", "trials"):
            s[name]["date"] = s[name]["date"].astype(cast)
        fig, axes, _, _, _capped = _strip(s, animal="Rat6")
        try:
            assert axes[3].collections, f"no per-unit points with {cast} dates"
            assert np.isfinite(axes[3].lines[0].get_ydata()).all()
        finally:
            plt.close(fig)
