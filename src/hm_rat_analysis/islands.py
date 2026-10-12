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

#: The level each region's figures actually use, and the level its count-matched
#: SHUFFLE is run at. Defined here rather than in the figure code because it is a
#: property of the data: it is chosen for coverage, and 40 is the only level the
#: bridges can reach at all. ``plots.ISLAND_MATCH`` is an alias.
SI_MATCH_BY_REGION = {0: 200, 1: 40, 2: 40, 3: 40, 4: 40, maze.BRIDGE: 40}

#: Shuffles behind the COUNT-MATCHED test. Same number as the unmatched one, so
#: the two pass rates are resolved to the same precision and can be compared.
MATCH_SHUFFLES = 500

#: Circular shuffles behind each cell's own null distribution. 500 puts the 95th
#: percentile at the 25th largest of 500, which is stable enough to threshold on;
#: it costs about 0.35 s per cell-block and all six regions share the draws.
SHUFFLES = 500

#: Smallest circular shift, in seconds. Shorter shifts leave the spike train
#: still aligned to the trajectory it came from, so the null would inherit the
#: cell's real field and the test would be against nothing.
MIN_SHIFT_S = 20.0


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
                      match_levels=SI_MATCH_LEVELS, shuffles=SHUFFLES,
                      match_shuffles=MATCH_SHUFFLES,
                      match_by_region=None, trial_types=None,
                      min_shift_s=MIN_SHIFT_S, seed=0):
    """One row per (unit, segment, island) for one session.

    Columns: ``animal``, ``date``, ``phase`` ("" for the whole session, "a"/"b"
    for the two sides of a goal switch), ``goal_node``, ``unit_id``, ``island``
    (1-4 for the islands, ``maze.BRIDGE`` for the corridors pooled, 0 for the
    whole maze), ``spatial_info``, ``si_m40`` … ``si_m400`` (the same estimate
    thinned to that many spikes), ``n_spikes``, ``occ_s``, ``n_valid_bins``, and
    the cell's own shuffle null: ``si_null_p95``, ``si_null_mean``, ``si_null_p``
    and ``sig95``.

    ``sig95_matched`` is the same test with the observed value AND every shuffle
    thinned to :data:`SI_MATCH_BY_REGION` spikes, with ``si_matched``,
    ``si_match_n``, ``si_match_null_p95`` and ``si_match_null_p`` beside it. Use
    it whenever PASS RATES are compared between regions: ``sig95`` holds the
    count fixed per cell, which is fair for that cell but leaves a bridge cell
    (median 20 spikes) with less power than an island cell (106 to 149), so the
    two regions' pass rates are not on the same footing. Matching costs the
    cell-blocks that cannot reach the level, which is 60% of the bridges.

    ``sig95`` is the test to use for one region. The null is built by circularly shifting THIS
    cell's spike train along the trajectory, so the spike count, the occupancy,
    the bin count and the cell's own burst structure are all held fixed and the
    only thing that differs from the observed value is whether the spikes are
    where the animal was when it fired them. A cell above its own 95th
    percentile has beaten the low-count bias as well, which no amount of
    count-matching across conditions can establish.

    `n_spikes` and `occ_s` are the exposure BEHIND each value and are not
    decoration: an island the animal barely visited scores high for that reason
    alone. Any comparison that is not between two matched columns has to carry
    them. There is no spike floor (see :data:`MIN_SPIKES`), so the lowest-count
    rows are the ones most in need of them.

    Positions are gated at :data:`SPEED_THRESH`, which is NOT the gate the
    cross-session summary uses.

    `trial_types` restricts which trials contribute. None, the default, uses all
    of them, which is what every result so far rests on — and two thirds of that
    window time is free roaming: the three ten-minute periods in each session
    come to 991 minutes across the dataset against 545 minutes of goal trials.
    Pass ``(GOAL_TRIAL_TYPE,)`` to score the goal-directed running alone. The
    phase split is taken BEFORE the filter, so dropping the goal-switch roam
    does not destroy the a/b boundary it defines.
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
        match_r = (SI_MATCH_BY_REGION if match_by_region is None
                   else dict(match_by_region))

        # Cell type on the WHOLE session, so a cell is classified the same way in
        # both halves of a split day: classifying on ten trials would let the
        # same unit be pyramidal before the switch and not after it, and the
        # before/after comparison would then be between different cells.
        um = SS._unit_metrics(nwb, udf, windows=windows)
        keep = (ql == QUALITY) & (um["cell_type"] == CELL_TYPE)

        rows = []
        want_tt = None if trial_types is None else {int(q) for q in trial_types}
        for phase, seg in _phase_segments(trials):
            # The phase split already happened, so filtering here cannot destroy
            # the goal-switch boundary that defines a and b.
            wins = SS._merge_windows([(float(a), float(b))
                                      for (_tt, _g, _sn, a, b) in seg
                                      if np.isfinite(a) and np.isfinite(b) and b > a
                                      and (want_tt is None or int(_tt) in want_tt)])
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
            # Priors once per segment, not once per shuffle: the shuffle reads
            # each rate map only inside the masks.
            priors = PF.region_priors(occ["occ_raw"], masks)
            # Cropped windows for the matched shuffle: it rebuilds a map per
            # region per draw, and a full-grid filter spends 86% of its work on
            # bins an island never reads.
            wins_r = (PF.region_windows(occ, masks, priors)
                      if match_shuffles else {})
            goal = _goal_node(seg)

            for idx in udf.index[keep]:
                st = np.asarray(udf.loc[idx, "spike_times"], dtype=float)
                st = st[SS._in_windows(st, wins)]
                spk_s, sx, sy, sidx = PF.spike_parts(occ, st, return_index=True)
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
                null = (PF.shuffled_spatial_info(
                    occ, sidx, priors, n_shuffles=shuffles,
                    min_shift_s=min_shift_s, seed=seed) if shuffles
                    else {k: np.array([]) for k in masks})
                null_m = (PF.shuffled_spatial_info_matched(
                    occ, sidx, masks, wins_r, match_r,
                    n_shuffles=match_shuffles, min_shift_s=min_shift_s,
                    seed=seed) if match_shuffles
                    else {k: np.array([]) for k in masks})
                for k, m in masks.items():
                    si, occ_s, nb = PF.spatial_info_in(lam, occ["occ_raw"], m)
                    row = {"animal": animal, "date": date, "phase": phase,
                           "goal_node": goal,
                           "unit_id": int(udf.loc[idx, "unit_id"])
                           if "unit_id" in udf.columns else int(idx),
                           "island": k, "spatial_info": si,
                           "n_spikes": int(raw_spk[m].sum()),
                           "occ_s": float(occ_s), "n_valid_bins": nb}
                    v = null[k][~np.isnan(null[k])] if k in null else np.array([])
                    if v.size and np.isfinite(si):
                        p95 = float(np.percentile(v, 95))
                        # (hits + 1) / (n + 1): under the null the observed value
                        # is itself one draw, so a p of exactly 0 is not a result
                        # this test can produce.
                        row["si_null_p95"] = p95
                        row["si_null_mean"] = float(v.mean())
                        row["si_null_p"] = float((np.sum(v >= si) + 1) / (v.size + 1))
                        row["sig95"] = bool(si > p95)
                        row["n_null"] = int(v.size)
                    else:
                        row.update({"si_null_p95": np.nan, "si_null_mean": np.nan,
                                    "si_null_p": np.nan, "sig95": False,
                                    "n_null": 0})
                    for n in match_levels:
                        row[f"si_m{int(n)}"] = PF.spatial_info_in_matched(
                            occ, sx, sy, m, visited, n,
                            repeats=SI_MATCH_REPEATS, seed=seed)
                    # The same test with both sides thinned to a common count.
                    want = int(match_r.get(k, 0))
                    obs_m = row.get(f"si_m{want}", np.nan) if want else np.nan
                    vm = null_m[k][~np.isnan(null_m[k])] if k in null_m \
                        else np.array([])
                    row["si_matched"] = obs_m
                    row["si_match_n"] = want
                    if vm.size and np.isfinite(obs_m):
                        p95m = float(np.percentile(vm, 95))
                        row["si_match_null_p95"] = p95m
                        row["si_match_null_p"] = float(
                            (np.sum(vm >= obs_m) + 1) / (vm.size + 1))
                        row["sig95_matched"] = bool(obs_m > p95m)
                        row["n_null_matched"] = int(vm.size)
                    else:
                        row.update({"si_match_null_p95": np.nan,
                                    "si_match_null_p": np.nan,
                                    "sig95_matched": False,
                                    "n_null_matched": 0})
                    rows.append(row)
        return pd.DataFrame(rows)
    finally:
        io.close()


#: Seconds of occupancy every region is thinned to for the dwell-matched test.
#: 100 s keeps 81% of the island blocks (37 of 44 goal-island, 105 of 132 other);
#: the bridges carry a median of 46 s and mostly drop out, which is why this
#: analysis is about the islands.
OCC_TARGET_S = 100.0

#: Shuffles for the dwell-matched test. Fewer than the main one because the
#: thinned trajectory is a third the size, and two tests are run per region.
OCC_SHUFFLES = 300


def _occ_maps(xs, ys, dt, sigma, hist_kw):
    """Raw and smoothed occupancy from an arbitrary set of frames."""
    occ_raw, _, _ = np.histogram2d(xs, ys, **hist_kw)
    occ_raw = occ_raw.T * float(dt)
    occ_s = (PF.gaussian_filter(occ_raw, sigma) if (sigma and sigma > 0)
             else occ_raw)
    return occ_raw, occ_s


def _occ_window(visited, occ_s, sigma, pad_sigma=3.0):
    """Crop to the region plus a kernel margin. Exact: only in-region spikes are
    ever placed, and the value is only ever read in-region."""
    pad = int(np.ceil(pad_sigma * float(sigma))) + 1
    ys, xs = np.nonzero(visited)
    ny, nx = occ_s.shape
    y0, y1 = max(0, ys.min() - pad), min(ny, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(nx, xs.max() + pad + 1)
    win = visited[y0:y1, x0:x1]
    return {"y0": y0, "x0": x0, "h": y1 - y0, "w": x1 - x0,
            "occ_s": occ_s[y0:y1, x0:x1], "flat": np.flatnonzero(win.ravel())}


def _si_in_window(wy, wx, win, p_r, sigma):
    """Skaggs bits/spike from spike bin indices already in window coordinates."""
    h, w = win["h"], win["w"]
    ok = (wy >= 0) & (wy < h) & (wx >= 0) & (wx < w)
    spk = np.bincount(wy[ok] * w + wx[ok], minlength=h * w).reshape(h, w).astype(float)
    if sigma and sigma > 0:
        spk = PF.gaussian_filter(spk, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(win["occ_s"] > 0, spk / win["occ_s"], 0.0)
    lam_r = lam.ravel()[win["flat"]]
    lam_mean = float((p_r * lam_r).sum())
    if lam_mean <= 0:
        return np.nan
    ratio = lam_r / lam_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(lam_r > 0,
                         p_r * ratio * np.log2(np.where(ratio > 0, ratio, 1.0)),
                         0.0)
    return float(np.nansum(terms))


def _occ_row(kx, ky, sp, occ_raw, occ_s, sigma, hist_kw, idx_r, p_r, visited,
             shuffles, match_n, dt, min_shift_s, seed):
    """One cell, on the thinned trajectory: SI, count-matched SI, and both nulls.

    The shuffle slides the spike train along the KEPT frames, so the null sees
    the same dwell time, the same occupancy and the same bin set as the observed
    value, and differs from it only in whether the spikes are where the animal
    was when it fired them.
    """
    (xmin, xmax), (ymin, ymax) = hist_kw["range"]
    nbx, nby = hist_kw["bins"]
    win = _occ_window(visited, occ_s, sigma)
    # Prior restricted to the window, in the window's own flat order.
    full = np.zeros(occ_raw.size, dtype=float)
    full[idx_r] = p_r
    p_win = full.reshape(occ_raw.shape)[win["y0"]:win["y0"] + win["h"],
                                        win["x0"]:win["x0"] + win["w"]]
    p_w = p_win.ravel()[win["flat"]]

    bx = np.clip(((kx - xmin) / (xmax - xmin) * nbx).astype(int), 0, nbx - 1)
    by = np.clip(((ky - ymin) / (ymax - ymin) * nby).astype(int), 0, nby - 1)
    wy_all, wx_all = by - win["y0"], bx - win["x0"]

    n = kx.size
    sp = np.asarray(sp, dtype=int)
    out = {"n_spikes_occ": int(sp.size)}
    out["si_occ"] = _si_in_window(wy_all[sp], wx_all[sp], win, p_w, sigma)

    gen = np.random.default_rng(seed)
    out[f"si_occ_m{match_n}"] = np.nan
    if sp.size >= match_n:
        vals = [_si_in_window(wy_all[sel], wx_all[sel], win, p_w, sigma)
                for sel in (sp[gen.choice(sp.size, match_n, replace=False)]
                            for _ in range(SI_MATCH_REPEATS))]
        out[f"si_occ_m{match_n}"] = float(np.nanmean(vals)) if vals else np.nan

    lo = int(max(1, round(float(min_shift_s) / float(dt))))
    if n <= 2 * lo:
        lo = max(1, n // 10)
    null, null_m = [], []
    for _ in range(int(shuffles)):
        sh = int(gen.integers(lo, max(lo + 1, n - lo)))
        j = (sp + sh) % n
        null.append(_si_in_window(wy_all[j], wx_all[j], win, p_w, sigma))
        if sp.size >= match_n:
            jm = j[gen.choice(j.size, match_n, replace=False)]
            null_m.append(_si_in_window(wy_all[jm], wx_all[jm], win, p_w, sigma))
    for tag, vals, obs in (("", null, out["si_occ"]),
                           ("_matched", null_m, out[f"si_occ_m{match_n}"])):
        v = np.asarray([q for q in vals if np.isfinite(q)], dtype=float)
        if v.size and np.isfinite(obs):
            out[f"si_occ_null_p95{tag}"] = float(np.percentile(v, 95))
            out[f"si_occ_null_p{tag}"] = float((np.sum(v >= obs) + 1) / (v.size + 1))
            out[f"sig95_occ{tag}"] = bool(obs > np.percentile(v, 95))
        else:
            out[f"si_occ_null_p95{tag}"] = np.nan
            out[f"si_occ_null_p{tag}"] = np.nan
            out[f"sig95_occ{tag}"] = False
    return out


def session_island_si_occmatched(nwb_path, bin_cm=2.5, smooth_cm=5.0,
                                 speed=SPEED_THRESH, min_occ_s=PF.DEFAULT_MIN_OCC_S,
                                 radius_m=maze.ISLAND_BIN_RADIUS_M,
                                 target_s=OCC_TARGET_S, shuffles=OCC_SHUFFLES,
                                 match_n=40, min_shift_s=MIN_SHIFT_S, seed=0):
    """Spatial information with every region's DWELL TIME forced equal.

    The main table controls spike count. It does not control how long the animal
    spent in a region, and that is the confound this answers: the animal lingers
    on the island the goal is on (median 239 s against 150 s elsewhere), and
    longer exposure makes a rate map more reliable and a shuffle test easier to
    beat. Stratifying after the fact showed the goal-island effect surviving at
    matched dwell, but stratifying is a comparison between unequal things that
    happen to be near each other. This removes the difference at source.

    Per block and region: take the moving frames inside the region, keep a
    random `target_s` seconds of them, and build the occupancy, the rate maps and
    the shuffle null from THAT and nothing else. Regions with less than
    `target_s` to give are not scored.

    One draw of frames per (block, region), fixed by `seed`, shared by the
    observed value and by every shuffle. The null is then built on the identical
    thinned trajectory, so whichever frames the draw happened to pick is common
    to both sides and cancels in the test.

    Spikes are placed at their nearest kept frame rather than interpolated: the
    thinned trajectory has gaps everywhere by construction, and interpolating
    across them would plant spikes on paths the animal never ran.

    Columns: as the main table for the identifying fields, plus ``occ_target_s``,
    ``si_occ`` (every spike in the thinned window), ``si_occ_m{match_n}``,
    ``sig95_occ`` and ``sig95_occ_matched``, with ``n_spikes_occ`` behind them.
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
        nbx, nby = bins
        hist_kw = {"bins": [nbx, nby], "range": [[ext[0], ext[1]], [ext[2], ext[3]]]}
        n_keep = int(round(float(target_s) / dt))

        um = SS._unit_metrics(nwb, udf, windows=windows)
        keep_u = (ql == QUALITY) & (um["cell_type"] == CELL_TYPE)

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
            move = spd > speed if speed > 0 else np.ones_like(x, dtype=bool)
            mi = np.flatnonzero(move)
            if mi.size < 10:
                continue
            fx, fy, ft = x[mi], y[mi], tt[mi]
            ix = np.clip(((fx - ext[0]) / (ext[1] - ext[0]) * nbx).astype(int),
                         0, nbx - 1)
            iy = np.clip(((fy - ext[2]) / (ext[3] - ext[2]) * nby).astype(int),
                         0, nby - 1)
            goal = _goal_node(seg)
            gen = np.random.default_rng(seed)

            for k in maze.REGIONS:
                region = isl == k
                here = np.flatnonzero(region[iy, ix])
                if here.size < n_keep:
                    continue
                # One draw, kept in time order so a circular shift still slides
                # the spike train ALONG the path rather than through a shuffled
                # bag of frames.
                sel = np.sort(gen.choice(here.size, n_keep, replace=False))
                fsel = here[sel]
                kx, ky, kt = fx[fsel], fy[fsel], ft[fsel]
                occ_raw, occ_s = _occ_maps(kx, ky, dt, sigma, hist_kw)
                visited = PF.valid_bins(occ_raw, occ_s, sigma, min_occ_s) & region
                if not visited.any():
                    continue
                priors = PF.region_priors(occ_raw, {k: visited})
                idx_r, p_r = priors[k]
                if p_r is None or idx_r.size == 0:
                    continue
                kept_set = np.empty(mi.size, dtype=bool)
                kept_set[:] = False
                kept_set[fsel] = True
                pos_in_kept = np.full(mi.size, -1, dtype=int)
                pos_in_kept[fsel] = np.arange(fsel.size)

                for u in udf.index[keep_u]:
                    st = np.asarray(udf.loc[u, "spike_times"], dtype=float)
                    st = st[SS._in_windows(st, wins)]
                    if st.size == 0:
                        continue
                    # Nearest MOVING frame, then only the ones that survived the
                    # draw. No interpolation: the thinned path is all gaps.
                    j = np.clip(np.searchsorted(ft, st), 1, ft.size - 1)
                    near = np.where(np.abs(ft[j] - st) <= np.abs(st - ft[j - 1]),
                                    j, j - 1)
                    near = near[kept_set[near]]
                    if near.size == 0:
                        continue
                    sp = pos_in_kept[near]
                    row = _occ_row(kx, ky, sp, occ_raw, occ_s, sigma, hist_kw,
                                   idx_r, p_r, visited, shuffles, match_n,
                                   dt, min_shift_s, seed)
                    row.update({"animal": animal, "date": date, "phase": phase,
                                "goal_node": goal, "island": k,
                                "unit_id": int(udf.loc[u, "unit_id"])
                                if "unit_id" in udf.columns else int(u),
                                "occ_target_s": float(n_keep * dt),
                                "n_valid_bins_occ": int(visited.sum())})
                    rows.append(row)
        return pd.DataFrame(rows)
    finally:
        io.close()


#: Valid bins every region is scored on. 500 keeps 90% of the island blocks
#: (39 of 44 goal-island, 120 of 132 other). The animal's path is NOT touched:
#: this picks a fixed-size sample of the bins it visited and scores only those.
BIN_TARGET = 500

#: Shuffles for the bin-matched test, and the number of thinning draws averaged
#: into BOTH the observed count-matched value and each shuffle of it.
#:
#: Symmetric on purpose. The main table averages the observed value over 8 draws
#: and compares it against a null whose every entry is a SINGLE draw, which
#: makes the null wider than the statistic it judges and the test conservative:
#: measured on this data the draw-to-draw SD is 0.140 against a null SD of
#: 0.178, so roughly 62% of the null's width is thinning noise the observed
#: value does not carry. Averaging both sides by the same amount removes that.
BIN_SHUFFLES = 200
BIN_MATCH_REPEATS = 4


def _bin_subset(mask, target, gen):
    """A fixed-size random sample of the bins in `mask`. None if too few."""
    idx = np.flatnonzero(mask.ravel())
    if idx.size < int(target):
        return None
    keep = gen.choice(idx.size, int(target), replace=False)
    out = np.zeros(mask.size, dtype=bool)
    out[idx[keep]] = True
    return out.reshape(mask.shape)


def _si_from_spikes(wy, wx, win, p_w, sigma, sel=None):
    """Skaggs over the window's scored bins, from spike bin indices."""
    if sel is not None:
        wy, wx = wy[sel], wx[sel]
    return PF._si_in_window_arrays(wy, wx, win, p_w, sigma)


def session_island_si_binmatched(nwb_path, bin_cm=2.5, smooth_cm=5.0,
                                 speed=SPEED_THRESH,
                                 min_occ_s=PF.DEFAULT_MIN_OCC_S,
                                 radius_m=maze.ISLAND_BIN_RADIUS_M,
                                 target_bins=BIN_TARGET, match_n=40,
                                 shuffles=BIN_SHUFFLES,
                                 repeats=BIN_MATCH_REPEATS,
                                 min_shift_s=MIN_SHIFT_S, seed=0):
    """Spatial information scored on an EQUAL NUMBER OF BINS per region.

    The dwell-matched analysis equalised exposure by throwing away time, which
    also throws away the places the animal went: the occupancy map itself
    shrinks, and a rate map on a smaller map is a different measurement. This
    does not touch the trajectory, the occupancy or the spikes. It picks
    `target_bins` of the bins the animal actually visited in that region, and
    scores only those.

    Why bins. Dwell time and the number of visited bins are nearly the same
    variable here (Spearman 0.87), and it is the BIN COUNT that enters the
    estimate: Skaggs asks how concentrated a cell's firing is across the bins it
    is summed over, and a region offering 886 bins gives more room to look
    concentrated than one offering 548. Matching the spike count does not touch
    that path, which is why it only moved the dwell correlation from +0.72 to
    +0.62. Matching the bins cuts it directly.

    One draw of bins per (block, region), fixed by `seed` and shared by the
    observed value and every shuffle, so which bins were picked cancels.

    Columns mirror the main table with a ``_bm`` suffix, plus ``n_bins_bm`` and
    ``n_spikes_bm`` (spikes landing in the scored bins).
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
        um = SS._unit_metrics(nwb, udf, windows=windows)
        keep_u = (ql == QUALITY) & (um["cell_type"] == CELL_TYPE)

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
            goal = _goal_node(seg)
            gen = np.random.default_rng(seed)
            scored, priors = {}, {}
            for k in maze.ISLANDS:
                sub = _bin_subset(visited & (isl == k), target_bins, gen)
                if sub is None:
                    continue
                scored[k] = sub
                priors[k] = PF.region_priors(occ["occ_raw"], {k: sub})[k]
            if not scored:
                continue
            wins_r = PF.region_windows(occ, scored, priors)

            for u in udf.index[keep_u]:
                st = np.asarray(udf.loc[u, "spike_times"], dtype=float)
                st = st[SS._in_windows(st, wins)]
                spk_s, sx, sy, sidx = PF.spike_parts(occ, st, return_index=True)
                if sx.size == 0:
                    continue
                rate = PF.rate_from_maps(spk_s, occ["occ_s"], visited)
                lam = rate.filled(0.0)
                raw_spk, _, _ = np.histogram2d(sx, sy, **occ["hist_kw"])
                raw_spk = raw_spk.T
                null = PF.shuffled_spatial_info_binmatched(
                    occ, sidx, scored, wins_r, match_n, repeats=repeats,
                    n_shuffles=shuffles, min_shift_s=min_shift_s, seed=seed)
                for k, sub in scored.items():
                    si, occ_s, nb = PF.spatial_info_in(lam, occ["occ_raw"], sub)
                    sim = PF.spatial_info_in_matched(
                        occ, sx, sy, sub, visited, match_n, repeats=repeats,
                        seed=seed)
                    row = {"animal": animal, "date": date, "phase": phase,
                           "goal_node": goal, "island": k,
                           "unit_id": int(udf.loc[u, "unit_id"])
                           if "unit_id" in udf.columns else int(u),
                           "si_bm": si, "si_bm_m": sim,
                           "n_bins_bm": nb, "occ_s_bm": float(occ_s),
                           "n_spikes_bm": int(raw_spk[sub].sum())}
                    for tag, obs, vals in (("", si, null[k]["own"]),
                                           ("_m", sim, null[k]["matched"])):
                        v = np.asarray([q for q in vals if np.isfinite(q)])
                        if v.size and np.isfinite(obs):
                            row[f"si_bm{tag}_null_p95"] = float(np.percentile(v, 95))
                            row[f"si_bm{tag}_null_p"] = float(
                                (np.sum(v >= obs) + 1) / (v.size + 1))
                            row[f"sig95_bm{tag}"] = bool(obs > np.percentile(v, 95))
                        else:
                            row[f"si_bm{tag}_null_p95"] = np.nan
                            row[f"si_bm{tag}_null_p"] = np.nan
                            row[f"sig95_bm{tag}"] = False
                    rows.append(row)
        return pd.DataFrame(rows)
    finally:
        io.close()
