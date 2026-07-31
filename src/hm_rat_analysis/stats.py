"""Group statistics used by the cross-session summary.

One-way ANOVA across session groups plus Holm-corrected pairwise Welch t-tests.
Kept separate from the report so the tests can exercise the maths directly.
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind


def groups_by(units, group_col, value_col, order=None):
    """``{group -> finite values}`` for a metric, using the whole/before epoch rows
    (one value per neuron per session, matching the plotted line)."""
    d = units[units["epoch"].isin(["whole", "before"])]
    out = {}
    for g, sub in d.groupby(group_col):
        v = pd.to_numeric(sub[value_col], errors="coerce").to_numpy()
        out[str(g)] = v[np.isfinite(v)]
    if order is not None:
        out = {str(k): out.get(str(k), np.array([])) for k in order}
    return out


def oneway_anova(groups):
    """One-way ANOVA across groups (dict or list). Returns (F, p, k, N).

    Groups with fewer than 2 values are dropped; with fewer than 2 groups left the
    test is undefined and (nan, nan, k, N) comes back.
    """
    gs = [g for g in (groups.values() if isinstance(groups, dict) else groups) if len(g) >= 2]
    if len(gs) < 2:
        return np.nan, np.nan, len(gs), int(sum(len(g) for g in gs))
    F, p = f_oneway(*gs)
    return float(F), float(p), len(gs), int(sum(len(g) for g in gs))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        prev = max(prev, min((m - rank) * pvals[idx], 1.0))
        adj[idx] = prev
    return adj


def posthoc(groups, scope, metric):
    """Pairwise Welch t-tests between session groups with Holm correction.

    Returns one row dict per pair, tagged with `scope` and `metric` so rows from
    several calls can be concatenated into one table.
    """
    keys = [k for k, v in groups.items() if len(v) >= 2]
    pairs, praw = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = groups[keys[i]], groups[keys[j]]
            tv, p = ttest_ind(a, b, equal_var=False)
            pairs.append((keys[i], keys[j], len(a), len(b), float(np.mean(a)),
                          float(np.mean(b)), float(tv), float(p)))
            praw.append(p)
    padj = holm(praw) if praw else []
    rows = []
    for (l1, l2, n1, n2, m1, m2, tv, p), pa in zip(pairs, padj):
        rows.append({"scope": scope, "metric": metric, "group1": l1, "group2": l2,
                     "n1": n1, "n2": n2, "mean1": m1, "mean2": m2, "t": tv,
                     "p_raw": p, "p_holm": float(pa), "sig": "*" if pa < 0.05 else ""})
    return rows
