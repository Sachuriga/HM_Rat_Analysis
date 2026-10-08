"""The tidy per-trial table every figure in this folder reads.

One row per trial across every session on the server, with the corrected path
inefficiency from :func:`hm_rat_analysis.behaviour.trial_detour`. Built once and
cached as CSV next to the figures, because the 40 RecordingMeta sheets take about
ten seconds to walk and a figure should be instant to redraw.

Slot labels are GL{goal}S{session} throughout, never the session number alone:
a bare "3" is ambiguous across goals, and these figures are read on their own
rather than beside a block rule.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from hm_rat_analysis import behaviour, maze

#: Where the sessions live, and where the figures go. Both overridable, so the
#: same code runs against a local copy when the share is not mounted.
ROOT = Path(os.environ.get(
    "HM_TASK_ROOT", "/Volumes/genzel/Rat/HM/Rat_HM_Neuron/task"))
OUT_DIR = Path(os.environ.get("HM_FIG_DIR", ROOT / "figures"))

#: repeat 0 predates any goal. Dropped from the figures: with no goal there is no
#: optimal route, so a path efficiency computed there measures nothing.
HABITUATION_REPEAT = 0

#: The goal trial. Everything else in Trial_Type is a long free-roaming period or
#: a probe, and a route efficiency over ten minutes of free roaming is not a route
#: efficiency: type 5 runs at a median detour of 0.99, a ten-fold route.
GOAL_TRIAL_TYPE = 1

#: An unreached trial with this many nodes or fewer is a recording that stopped,
#: not a rat that searched: the unreached trials split at a clean gap between
#: three nodes and seven. Those are missing data and leave the success rate
#: entirely; a rat that walked 52 nodes without finding the goal did not.
TRACKING_MIN_NODES = 4

#: The free-roaming period that introduces the NEW goal, and the only trial type
#: that appears exactly once in each of the six (goal > 1, session 1) sessions.
#: The goal node switches across it, so the goal trials before it are the last
#: retrievals of the OLD goal and those after it are the first encodings of the
#: new one. Splitting there is what makes that visible.
GOAL_SWITCH_TRIAL_TYPE = 5


def slot_label(repeat, session, phase=""):
    """``GL3S2``, ``GL3S1a`` / ``GL3S1b`` where the day is split, ``habS1`` before
    any goal was set.

    `phase` is "a" for the goal trials before the goal switched that day and "b"
    for those after it; it is empty on every other session.
    """
    rep = int(repeat)
    stem = "hab" if rep == HABITUATION_REPEAT else f"GL{rep}"
    return f"{stem}S{int(session)}{phase}"


def _phase_column(t):
    """"a"/"b" per trial, split at the goal-switch free roam; "" where there is none.

    Only the six (goal > 1, session 1) sessions carry a switch, so everywhere else
    this is empty and the slot keeps its plain name.
    """
    sw = t.index[t["trial_type"] == GOAL_SWITCH_TRIAL_TYPE]
    if len(sw) != 1:
        return pd.Series([""] * len(t), index=t.index)
    k = int(t.loc[sw[0], "trial"])
    return pd.Series(np.where(t["trial"] < k, "a",
                              np.where(t["trial"] > k, "b", "")), index=t.index)


def _sessions(root):
    """``(animal, date, folder)`` for every session folder under `root`."""
    for animal_dir in sorted(p for p in root.iterdir()
                             if p.is_dir() and p.name.lower().startswith("rat")):
        animal = animal_dir.name.split("_")[0].capitalize()
        for d in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
            yield animal, d.name, d


def build_trials(root=None, graph=None):
    """Walk every session and return the per-trial table."""
    root = ROOT if root is None else Path(root)
    G = maze.build_graph() if graph is None else graph
    frames = []
    for animal, date, d in _sessions(root):
        t = behaviour.trial_detour(d, graph=G)
        if t.empty:
            continue
        meta, _ = behaviour._perf_meta_frame(d)
        first = meta.iloc[0]
        rep = int(first.get("Repeat", 0) or 0)
        ses = int(first.get("Session", 0) or 0)
        phase = _phase_column(t) if rep > HABITUATION_REPEAT and ses == 1 else ""
        t = t.assign(animal=animal, date=date, repeat=rep, session=ses, phase=phase)
        frames.append(t)
    if not frames:
        raise FileNotFoundError(f"no session metadata under {root}")
    out = pd.concat(frames, ignore_index=True)
    out["slot"] = [slot_label(r, s, p) for r, s, p
                   in zip(out["repeat"], out["session"], out["phase"])]
    return out


def trials(root=None, cache=True, rebuild=False):
    """The per-trial table, from cache when one is there.

    The cache is a plain CSV so it can be opened in Excel and mailed to anyone
    who asks where a number came from.
    """
    root = ROOT if root is None else Path(root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OUT_DIR / "trials_detour.csv"
    if cache and not rebuild and cache_file.exists():
        return pd.read_csv(cache_file)
    out = build_trials(root)
    if cache:
        out.to_csv(cache_file, index=False)
    return out


def slot_order(df):
    """``(slots, meta)`` ordered by goal, session, then phase; one entry per slot."""
    o = (df[["slot", "repeat", "session", "phase"]].fillna({"phase": ""})
         .drop_duplicates("slot").sort_values(["repeat", "session", "phase"]))
    return list(o["slot"]), list(zip(o["repeat"].astype(int),
                                     o["session"].astype(int)))


def goal_trials(df):
    """The goal trials a figure may draw from, ranked within their block.

    Habituation is out (no goal, so no optimal route) and so is every
    free-roaming period (a route efficiency over ten minutes of roaming is not a
    route efficiency: the goal-switch roam runs at a median detour of 0.99).

    Rows whose detour is NaN are KEPT here, because a trial the rat ran and failed
    to finish is data: it is :func:`scorable` that drops them, and a figure about
    first trials has to be able to see them. Rank is over all goal trials in the
    block, so "trial 1" stays the first trial the rat actually ran.
    """
    out = df[df["repeat"] > HABITUATION_REPEAT]
    out = out[out["trial_type"] == GOAL_TRIAL_TYPE]
    out = out.sort_values(["animal", "slot", "trial"]).copy()
    out["trial_rank"] = out.groupby(["animal", "slot"]).cumcount() + 1
    # A trial counts towards success only if we can tell what the rat did.
    out["tracking_ok"] = (out["reached"].astype(bool)
                          | (out["path_nodes"] >= TRACKING_MIN_NODES))
    return out


def completion(df):
    """Per block and animal: how often the goal was reached, and when first.

    Returns a frame indexed by ``(animal, slot)`` with ``n`` scoreable trials,
    ``success_rate``, ``first_success`` (the rank of the first trial that reached
    the goal, among trials we can score) and ``failures``.

    Trials whose recording stopped are out of BOTH the numerator and the
    denominator: counting them as failures would make this a figure about
    tracking. How many were dropped comes back in `notes` so the caption can
    say so.
    """
    g = goal_trials(df)
    dropped = int((~g["tracking_ok"]).sum())
    ok = g[g["tracking_ok"]].copy()
    ok["rank_ok"] = ok.groupby(["animal", "slot"]).cumcount() + 1
    grp = ok.groupby(["animal", "slot"])
    out = grp.agg(n=("reached", "size"),
                  success_rate=("reached", "mean"),
                  failures=("reached", lambda v: int((~v.astype(bool)).sum())))
    hit = ok[ok["reached"].astype(bool)].groupby(["animal", "slot"])["rank_ok"].min()
    out["first_success"] = hit
    notes = {"goal_trials": len(g), "scored": len(ok),
             "tracking_dropouts": dropped,
             "blocks_with_a_failure": int((out["failures"] > 0).sum()),
             "blocks_needing_more_than_one": int((out["first_success"] > 1).sum()),
             "blocks_never_solved": int(out["first_success"].isna().sum())}
    return out.reset_index(), notes


def scorable(df):
    """Rows that carry a real detour, and the counts of what was dropped.

    Returns ``(kept, notes)``. `notes` is what the caption has to say: a figure
    that silently drops 4% of trials is a figure that overstates performance.
    """
    df = goal_trials(df)
    n = len(df)
    unreached = int((~df["reached"].astype(bool)).sum())
    kept = df[df["detour"].notna()].copy()
    # Rank among the trials that SUCCEEDED, alongside the rank among all of them.
    # The first-trial figures count in successes: the first trial of a block is
    # often a trial the rat failed, and a figure of route efficiency can only
    # speak about routes that reached the goal. `trial_rank` is kept so the
    # figures can say HOW FAR into the block each point actually came from, and
    # `trials_to_first_success` carries the failures themselves.
    kept["success_rank"] = kept.groupby(["animal", "slot"]).cumcount() + 1
    # Only tracking that skipped a node can beat the optimum; clamp, do not drop,
    # or the best trials are the ones that leave the figure.
    neg = int((kept["detour"] < -1e-9).sum())
    kept["detour"] = kept["detour"].clip(lower=0.0)
    notes = {"goal_trials": n, "scored": len(kept),
             "goal_never_reached": unreached, "clamped_below_zero": neg}
    return kept, notes
