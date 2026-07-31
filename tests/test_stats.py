import numpy as np
import pandas as pd
import pytest

from hm_rat_analysis import stats


def test_holm_is_monotone_and_bounded():
    p = [0.01, 0.04, 0.03]
    adj = stats.holm(p)
    assert len(adj) == 3
    assert (adj <= 1.0).all()
    assert (adj >= np.array(p)).all(), "adjustment never lowers a p-value"
    # smallest raw p gets the largest multiplier (m=3)
    assert adj[0] == pytest.approx(0.03)


def test_holm_caps_at_one():
    assert (stats.holm([0.9, 0.95]) <= 1.0).all()


def test_oneway_anova_separates_shifted_groups():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 60), rng.normal(4, 1, 60)
    F, p, k, N = stats.oneway_anova({"a": a, "b": b})
    assert k == 2 and N == 120
    assert F > 10 and p < 1e-6


def test_oneway_anova_undefined_with_one_usable_group():
    F, p, k, N = stats.oneway_anova({"a": np.arange(5.0), "b": np.array([1.0])})
    assert np.isnan(F) and np.isnan(p)
    assert k == 1


def _units_frame():
    rng = np.random.default_rng(1)
    rows = []
    for label, shift in (("R1S1", 0.0), ("R1S2", 3.0)):
        for v in rng.normal(shift, 1, 30):
            rows.append({"session_label": label, "epoch": "whole", "spatial_info": v})
    rows.append({"session_label": "R1S1", "epoch": "after", "spatial_info": 99.0})
    return pd.DataFrame(rows)


def test_groups_by_uses_only_whole_and_before_epochs():
    groups = stats.groups_by(_units_frame(), "session_label", "spatial_info")
    assert set(groups) == {"R1S1", "R1S2"}
    assert len(groups["R1S1"]) == 30, "the 'after' epoch row must be excluded"
    assert 99.0 not in groups["R1S1"]


def test_groups_by_order_fills_missing_groups():
    groups = stats.groups_by(_units_frame(), "session_label", "spatial_info",
                             order=["R1S1", "R1S2", "R9S9"])
    assert list(groups) == ["R1S1", "R1S2", "R9S9"]
    assert groups["R9S9"].size == 0


def test_posthoc_rows_are_tagged_and_corrected():
    rows = stats.posthoc(stats.groups_by(_units_frame(), "session_label", "spatial_info"),
                         scope="Rat5", metric="spatial_info")
    assert len(rows) == 1
    r = rows[0]
    assert r["scope"] == "Rat5" and r["metric"] == "spatial_info"
    assert r["n1"] == 30 and r["n2"] == 30
    assert r["p_holm"] >= r["p_raw"]
    assert r["sig"] == "*", "a 3-sigma separation should survive correction"
