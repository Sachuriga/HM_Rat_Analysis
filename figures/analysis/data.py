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
    return out


def scorable(df):
    """Rows that carry a real detour, and the counts of what was dropped.

    Returns ``(kept, notes)``. `notes` is what the caption has to say: a figure
    that silently drops 4% of trials is a figure that overstates performance.
    """
    df = goal_trials(df)
    n = len(df)
    unreached = int((~df["reached"].astype(bool)).sum())
    kept = df[df["detour"].notna()].copy()
    # Only tracking that skipped a node can beat the optimum; clamp, do not drop,
    # or the best trials are the ones that leave the figure.
    neg = int((kept["detour"] < -1e-9).sum())
    kept["detour"] = kept["detour"].clip(lower=0.0)
    notes = {"goal_trials": n, "scored": len(kept),
             "goal_never_reached": unreached, "clamped_below_zero": neg}
    return kept, notes
