"""Skaggs spatial information computed separately inside each of the four islands.

The HexMaze is four hexagonal blocks joined by bridges (see
``maze.bin_islands``). A cell's whole-session spatial information mixes all four:
a cell with one field on one island scores as informative because the other three
islands are silent, which is information about WHICH island rather than about
where on it. Scoring each island on its own asks the narrower question, and the
four values together say whether a cell carries position everywhere or only in
one block of space.

Everything about how a rate map is built — trial windows, the boxcar, the speed
gate, the occupancy prior — is taken from ``reports.session_summary`` rather than
restated here, so these numbers and the cross-session summary cannot disagree.
Occupancy is computed ONCE per segment and reused across cells; only the spike
half is per cell.

Two things make the raw per-region number unsafe to read on its own, and both are
carried in the returned columns rather than left to the reader to remember:

* **Exposure.** Skaggs bits/spike is biased upward at low spike counts, and the
  bridges collect about a fifth of an island's spikes. ``si_m*`` are the same
  estimate thinned to a fixed count, which is what a comparison has to use.
* **The goal switch.** Three days carry two goals, split by a free-roaming
  period. Those days are returned three times: once whole (``phase = ""``) and
  once for each side of the switch, so the figures can use the same 22 blocks the
  behavioural ones use.

    from hm_rat_analysis import islands
    df = islands.session_island_si("…/Rat6_20260708.nwb")
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from . import maze, nwb as nwbio, place_fields as PF
from .reports import session_summary as SS

#: Only well-isolated pyramidal cells. Interneurons do not carry place fields and
#: MUA is a mixture, so neither has a spatial information worth comparing.
QUALITY = "good"
CELL_TYPE = "pyramidal"

#: Speed gate, m/s. The rest of the pipeline runs at 0.02 m/s, which on this
#: tracking is barely a gate at all: it keeps 48% of the frames of a session, so
#: a third of the remaining occupancy is the animal grooming, rearing or sitting
#: at the goal. Those samples have no position to code and they land in a handful
#: of bins, which is where a spurious peak in a rate map comes from. At 0.05 m/s
#: the whole-maze occupancy of a session drops from 453 s to 305 s and the median
#: spike count from 1558 to 1248, while the BRIDGES lose almost nothing (11 s and
#: 75 -> 73 spikes): the animal crosses a bridge at speed, so the gate removes
#: slow island time and leaves the corridors as they were.
#:
#: This differs from the cross-session summary on purpose, so the two sets of
#: numbers are not interchangeable. The gate is the first thing to state when
#: quoting a value from here.
SPEED_THRESH = 0.05

#: Spikes a cell-block needs to be scored at all. Zero: every cell that fired
#: inside the window gets a value.
#:
#: Read the raw numbers with that in mind. Skaggs bits/spike is biased upward at
#: low counts and the bias is steep, so a cell with a handful of spikes in a
#: region scores high for having a handful of spikes. With no floor those cells
#: are in the median, and the regions with the least exposure are flattered
#: most: the bridges carry a median of about 34 spikes. The ``si_m*`` columns are
#: the same estimate thinned to a common count and are what a comparison has to
#: use; the floor is gone, the arithmetic that made it necessary is not.
MIN_SPIKES = 0

#: Trial types, as in ``figures/analysis/data.py``. Type 5 is the free-roaming
#: period that introduces the new goal and appears exactly once on each day the
#: goal changed; type 1 is a goal trial, and its ``Goal_Node`` is what the x axis
#: of these figures is labelled with.
GOAL_TRIAL_TYPE = 1
GOAL_SWITCH_TRIAL_TYPE = 5

#: Spike counts to thin each estimate to. A region's figure uses the highest
#: level most of its cell-sessions can afford: 400 is for the whole-session value
#: (median 1206), 200 for the islands (225-352) and 40 for the bridges (45).
#: Below 40 the estimate is mostly bias, above 400 most sessions drop out.
SI_MATCH_LEVELS = (40, 100, 200, 400)
SI_MATCH_REPEATS = 8


def _phase_segments(trials):
    """``[(phase, trials)]``: the whole session, plus each side of a goal switch.

    A day with exactly one goal-switch roam carries two goals, and one rate map
    over the whole day mixes the last retrievals of the old goal with the first
    encodings of the new one. Those days come back as three segments; every other
    day as one. The split rule is the one ``data._phase_column`` uses, so a slot
    called ``GL2S1b`` means the same trials in both places.
    """
    segs = [("", list(trials))]
    sw = [i for i, tr in enumerate(trials) if int(tr[0]) == GOAL_SWITCH_TRIAL_TYPE]
    if len(sw) == 1:
        k = sw[0]
        segs += [("a", list(trials[:k])), ("b", list(trials[k + 1:]))]
    return segs


def _goal_node(trials):
    """The modal ``Goal_Node`` over the goal trials of a segment.

    Modal and not first: one session has a single trial typed to node 110 among
    seventeen 109s, which is a data-entry slip in RecordingMeta, and a figure
    labelled from the first row would carry it.
    """
    g = [str(tr[1]) for tr in trials
         if int(tr[0]) == GOAL_TRIAL_TYPE and tr[1] is not None
         and str(tr[1]) not in ("", "nan", "None")]
    if not g:
        return ""
    return pd.Series(g).mode().iat[0]


def session_island_si(nwb_path, bin_cm=2.5, smooth_cm=5.0, speed=SPEED_THRESH,
                      min_occ_s=PF.DEFAULT_MIN_OCC_S, min_spikes=MIN_SPIKES,
                      radius_m=maze.ISLAND_BIN_RADIUS_M,
                      match_levels=SI_MATCH_LEVELS, seed=0):
    """One row per (unit, segment, island) for one session.

    Columns: ``animal``, ``date``, ``phase`` ("" for the whole session, "a"/"b"
    for the two sides of a goal switch), ``goal_node``, ``unit_id``, ``island``
    (1-4 for the islands, ``maze.BRIDGE`` for the corridors pooled, 0 for the
    whole maze), ``spatial_info``, ``si_m40`` … ``si_m400`` (the same estimate
    thinned to that many spikes), ``n_spikes``, ``occ_s``, ``n_valid_bins``.

    `n_spikes` and `occ_s` are the exposure BEHIND each value and are not
    decoration: an island the animal barely visited scores high for that reason
    alone. Any comparison that is not between two matched columns has to carry
    them. There is no spike floor (see :data:`MIN_SPIKES`), so the lowest-count
    rows are the ones most in need of them.

    Positions are gated at :data:`SPEED_THRESH`, which is NOT the gate the
    cross-session summary uses.
    """
    path = str(nwb_path)
    io = NWBHDF5IO(path, "r", load_namespaces=True)
    try:
        nwb = io.read()
        subj = nwb.subject.subject_id if nwb.subject is not None else "?"
        animal = f"Rat{int(subj)}" if str(subj).isdigit() else str(subj)
        date = str(nwb.session_id) if nwb.session_id else ""
        if nwb.units is None:
            return pd.DataFrame()
        udf = nwb.units.to_dataframe()
        ql = (udf["quality_label"].astype(str) if "quality_label" in udf
              else pd.Series(QUALITY, index=udf.index))
        pos = nwbio.load_position(nwb)
        if pos is None:
            return pd.DataFrame()
        t_raw = pos[2]
        dt = float(np.median(np.diff(t_raw))) if t_raw.size > 1 else 1.0 / 30
        trials, windows, _src = SS._trial_windows(Path(path).parent, nwb, t_raw)
        if not windows:
            return pd.DataFrame()
        px, py = pos[0] / maze.SCALE_X, pos[1] / maze.SCALE_Y

        ext = maze.MAZE_EXTENT
        bm = bin_cm / 100.0
        bins = (max(5, int(round((ext[1] - ext[0]) / bm))),
                max(5, int(round((ext[3] - ext[2]) / bm))))
        bx_cm, _ = PF.bin_size_cm(ext, bins)
        sigma = float(smooth_cm) / bx_cm
        isl = maze.bin_islands(ext, bins, radius_m=radius_m)

        # Cell type on the WHOLE session, so a cell is classified the same way in
        # both halves of a split day: classifying on ten trials would let the
        # same unit be pyramidal before the switch and not after it, and the
        # before/after comparison would then be between different cells.
        um = SS._unit_metrics(nwb, udf, windows=windows)
        keep = (ql == QUALITY) & (um["cell_type"] == CELL_TYPE)

        rows = []
        for phase, seg in _phase_segments(trials):
            wins = SS._merge_windows([(float(a), float(b))
                                      for (_tt, _g, _sn, a, b) in seg
                                      if np.isfinite(a) and np.isfinite(b) and b > a])
            if not wins:
                continue
            prep = SS._prep_positions(px, py, t_raw, wins, dt)
            if prep is None:
                continue
            x, y, tt, spd = prep
            occ = PF.occupancy_parts(x, y, tt, ext, bins, dt, sigma,
                                     speed_thresh=speed, speed=spd)
            if occ is None:
                continue
            visited = PF.valid_bins(occ["occ_raw"], occ["occ_s"], sigma, min_occ_s)
            masks = {0: visited}
            for k in maze.REGIONS:
                masks[k] = visited & (isl == k)
            goal = _goal_node(seg)

            for idx in udf.index[keep]:
                st = np.asarray(udf.loc[idx, "spike_times"], dtype=float)
                st = st[SS._in_windows(st, wins)]
                spk_s, sx, sy = PF.spike_parts(occ, st)
                # No spike floor by default; a cell with nothing left after the
                # speed gate still has no rate map, so that one case is out.
                if sx.size == 0 or sx.size < int(min_spikes):
                    continue
                rate = PF.rate_from_maps(spk_s, occ["occ_s"], visited)
                lam = rate.filled(0.0)
                # RAW spike counts per bin, for the per-region exposure. The
                # smoothed map would leak counts across a bridge and credit them
                # to an island the spike did not happen on.
                raw_spk, _, _ = np.histogram2d(sx, sy, **occ["hist_kw"])
                raw_spk = raw_spk.T
                for k, m in masks.items():
                    si, occ_s, nb = PF.spatial_info_in(lam, occ["occ_raw"], m)
                    row = {"animal": animal, "date": date, "phase": phase,
                           "goal_node": goal,
                           "unit_id": int(udf.loc[idx, "unit_id"])
                           if "unit_id" in udf.columns else int(idx),
                           "island": k, "spatial_info": si,
                           "n_spikes": int(raw_spk[m].sum()),
                           "occ_s": float(occ_s), "n_valid_bins": nb}
                    for n in match_levels:
                        row[f"si_m{int(n)}"] = PF.spatial_info_in_matched(
                            occ, sx, sy, m, visited, n,
                            repeats=SI_MATCH_REPEATS, seed=seed)
                    rows.append(row)
        return pd.DataFrame(rows)
    finally:
        io.close()
