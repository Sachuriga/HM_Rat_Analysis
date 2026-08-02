"""
Cross-session summary per animal.

Scans a folder tree for session NWBs (written by the HM_Tracker_2025 pipeline,
steps w/u), groups them by animal (NWB subject_id), and for each animal plots — as
a function of session date (labelled with Repeat & Session) —:
  - number of GOOD and MUA units
  - among GOOD units, number of pyramidal vs interneuron
  - pyramidal spatial information (Skaggs, bits/spike)
  - pyramidal selectivity (peak/mean rate)
  - pyramidal number of place fields
  - pyramidal mean place-field distance to the goal node (m)

Cell type + waveform metrics are read from the NWB when step u stored them, else
recomputed (:mod:`hm_rat_analysis.spike_metrics`). Place-field metrics are
computed here from the NWB position + units, using the session's dominant goal
node.

Everything the place-field metrics see is restricted to the trial windows from
:func:`hm_rat_analysis.nwb.build_trials` — between trials the rat is carried off
the maze and the position series parks on one sentinel pixel (447, 303), which
converts to maze node 318, i.e. a point ON the graph in a corridor the rat really
runs. No geometric or occupancy filter can tell that apart from a real visit; only
trial timing can. Inside each trial the position is boxcar-smoothed over 400 ms
BEFORE the speed is computed, because the raw trace is integer pixels (7.65 mm in
x, 7.02 mm in y) and cannot express a speed below ~0.20 m/s — a raw gate at
0.05 m/s is a duplicate-frame filter, not a speed gate. The smoothing must never
run across a trial boundary: smoothing the untrimmed session smears each carry-out
teleport into a dozen fictitious off-maze samples.

Several plotted metrics remain confounded with how much data a session contains
(spike counts fall as trials shorten with learning). They are still computed, but
every one carries a machine-readable note from
:data:`hm_rat_analysis.place_fields.METRIC_NOTES`, printed on the panel and
written into the xlsx, and a spike-count-matched spatial-information column is
computed alongside the raw one.

Usage:
    hm-session-summary --root <folder>
        [--bin_cm 2.5] [--smooth_cm 5] [--speed 0.02] [--min_occ_s 0.30]
        [--field_frac 0.30] [--min_field_cm 15] [--min_peak_hz 0.5]
        [--min_spikes 100] [--si_match_n 300] [--si_match_repeats 20] [--seed 0]
"""

import argparse
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages              # noqa: E402
from pynwb import NWBHDF5IO                                       # noqa: E402

from .. import behaviour, maze, nwb as nwbio, spike_metrics as SM, stats   # noqa: E402
from .. import place_fields as PF                                 # noqa: E402
from ..place_fields import place_field_metrics                    # noqa: E402

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:                    # graceful fallback if tqdm is absent
    _HAS_TQDM = False

    def tqdm(it, **k):
        return it


def _log(msg):
    (tqdm.write if _HAS_TQDM else print)(msg)


# Place-coding metrics, then the EXPOSURE variables they are all confounded with.
# The exposure columns are not decoration: without a spike count, an epoch
# duration and a covered-area count per (unit, epoch) nobody downstream can match
# spike counts, covary out duration, or even check that later sessions had less
# data — which is what would make every sampling confound below unfixable after
# the fact rather than merely present.
_PF_METRICS = ("spatial_info", "spatial_info_matched", "selectivity", "peak",
               "n_fields", "field_goal_m", "field_goal_null_m", "field_goal_over_null",
               "field_goal_largest_m", "field_goal_2ndlargest_m",
               "field_goal_smallest_m", "field_size_largest_m2")
_PF_EXPOSURE = ("n_spikes_epoch", "epoch_dur_s", "occ_total_s", "n_valid_bins")
_PF_KEYS = _PF_METRICS + _PF_EXPOSURE
_PF_PLOT = [("spatial_info", "pyramidal spatial information", "bits/spike"),
            ("spatial_info_matched", "spatial information, spike-count matched", "bits/spike"),
            ("selectivity", "pyramidal selectivity", "peak/mean"),
            ("peak", "pyramidal peak rate", "Hz"),
            ("n_fields", "pyramidal # place fields", "n fields"),
            ("field_goal_m", "field-to-goal (mean of all fields)", "metres"),
            ("field_goal_over_null", "field-to-goal / occupancy-matched null", "ratio"),
            ("field_goal_largest_m", "field-to-goal (largest field)", "metres"),
            ("field_goal_2ndlargest_m", "field-to-goal (2nd-largest field)", "metres"),
            ("field_goal_smallest_m", "field-to-goal (smallest field)", "metres"),
            ("n_valid_bins", "maze coverage (valid rate-map bins)", "bins")]
# Short on-panel warnings. n_fields and every field-to-goal distance are bounded by
# the bins the animal actually covered, and coverage collapses onto the direct
# start->goal route as the animal learns; spatial_info/selectivity/peak move with
# spike count. Reading any of these as a learning effect without the coverage panel
# and the matched-SI panel next to it is a mistake, so the figure says so.
_PANEL_NOTE = {"spatial_info": "biased by spike count — read the matched panel",
               "selectivity": "peak & mean move in opposite directions with sampling",
               "peak": "biased upward when data are sparse; sets the field threshold",
               "n_fields": "bounded by coverage (see coverage panel)",
               "field_goal_m": "bounded by coverage",
               "field_goal_largest_m": "bounded by coverage",
               "field_goal_2ndlargest_m": "only cells with >=2 fields contribute",
               "field_goal_smallest_m": "smallest field is the sampling speck"}
# 3-way subtype colours for the cell-type scatter
SUBTYPE_COLORS = {"pyramidal": "#2166ac",            # blue
                  "narrow interneuron": "#d62728",   # red
                  "wide interneuron": "#2ca02c"}     # green


def _subtype(cell_type, t2p_s):
    """pyramidal / narrow interneuron (t2p<=0.425ms) / wide interneuron."""
    if cell_type == "pyramidal":
        return "pyramidal"
    if np.isfinite(t2p_s) and t2p_s <= SM.TROUGH_PEAK_THRESH_S:
        return "narrow interneuron"
    return "wide interneuron"


def _unit_metrics(nwb, udf, fs=30000.0, windows=None):
    """Per-unit DataFrame with firing_rate_hz, trough_to_peak_s, acg_tau_rise_ms,
    cell_type and subtype — read from the columns step u stored, else recomputed.

    When trial `windows` are known the firing rate is computed over the SAME
    in-trial exposure the place-field metrics use. Over the whole recording a long
    off-task tail deflates every unit's rate, and classify_cell_type calls anything
    above 10 Hz an interneuron — so recording length would decide WHICH cells enter
    the place-field panels.
    """
    n = len(udf)
    has = lambda c: c in udf.columns
    if windows:
        dur = float(sum(b - a for a, b in windows)) or 1.0
        fr = pd.Series([int(_in_windows(np.asarray(udf.iloc[i]["spike_times"], float),
                                        windows).sum()) / dur for i in range(n)],
                       index=udf.index)
    else:
        dur = max((np.asarray(udf.iloc[i]["spike_times"]).max()
                   for i in range(n) if len(udf.iloc[i]["spike_times"])), default=1.0) or 1.0
        fr = udf["firing_rate_hz"].astype(float) if has("firing_rate_hz") else pd.Series(
            [len(np.asarray(udf.iloc[i]["spike_times"])) / dur for i in range(n)], index=udf.index)
    if has("cell_type") and has("trough_to_peak_s") and has("acg_tau_rise_ms"):
        t2p = udf["trough_to_peak_s"].astype(float)
        tau = udf["acg_tau_rise_ms"].astype(float)
        ct = udf["cell_type"].astype(str)
    else:                                   # recompute (NWB predates step u)
        has_wf = has("waveform_mean")
        ql = udf["quality_label"].astype(str) if has("quality_label") else pd.Series("good", index=udf.index)
        t2p_l, tau_l, ct_l = [], [], []
        for i in range(n):
            r = udf.iloc[i]; st = np.asarray(r["spike_times"], dtype=float)
            v = SM.waveform_metrics(r["waveform_mean"], fs).get("peak_to_trough_s", np.nan) if has_wf else np.nan
            g = SM.acg_tau_rise(st) if ql.iloc[i] == "good" else np.nan
            t2p_l.append(v); tau_l.append(g); ct_l.append(SM.classify_cell_type(fr.iloc[i], v, g))
        t2p = pd.Series(t2p_l, index=udf.index); tau = pd.Series(tau_l, index=udf.index)
        ct = pd.Series(ct_l, index=udf.index)
    out = pd.DataFrame({"firing_rate_hz": fr, "trough_to_peak_s": t2p,
                        "acg_tau_rise_ms": tau, "cell_type": ct}, index=udf.index)
    out["subtype"] = [_subtype(c, v) for c, v in zip(ct, t2p)]
    return out


_DEC_QUALS = ("good", "good_mua")           # file tags
_DEC_LEADS = (0.0, 1.0, 3.0)                 # prediction leads (s) shown in summary


def _dec_key(qtag, lead):
    return f"dec_err_{qtag}_lead{lead:g}"


def _decode_accuracy(session_dir):
    """Median decoding error (m) per unit set AND per prediction lead, read from the
    decoder's (step b) leads_summary_*.npz (falls back to decoded_*.npz for lead 0).
    Returns {} when a session has no decoding output."""
    d = Path(session_dir) / "decoding"
    out = {}
    for qtag in _DEC_QUALS:
        f = d / f"leads_summary_{qtag}.npz"
        if f.exists():
            try:
                z = np.load(f, allow_pickle=True)
                for L, m in zip(np.asarray(z["leads"], float), np.asarray(z["median_err"], float)):
                    out[_dec_key(qtag, float(L))] = float(m)
                continue
            except Exception:
                pass
        g = d / f"decoded_{qtag}.npz"          # fallback: lead-0 only
        if g.exists():
            try:
                z = np.load(g, allow_pickle=True)
                if "err" in z and len(z["err"]):
                    out[_dec_key(qtag, 0.0)] = float(np.median(z["err"]))
            except Exception:
                pass
    return out


def _session_goal(nwb, udf):
    """Dominant goal node id for the session: most common in the Trials_Data table,
    else parsed from session_description ('Goal_Node G'). None if unavailable."""
    try:
        tr = nwb.processing["Behavior"]["Trials_Data"].to_dataframe()
        if "Goal_node" in tr.columns:
            vals = [int(v.decode() if isinstance(v, bytes) else v) for v in tr["Goal_node"]]
            if vals:
                return Counter(vals).most_common(1)[0][0]
    except Exception:
        pass
    m = re.search(r"Goal_Node\s+(\d+)", str(nwb.session_description))
    return int(m.group(1)) if m else None


# ------------------------------------------------------------
#   trial windows + position preprocessing (see the module docstring)
# ------------------------------------------------------------
#: Position smoothing window before speed is computed. 1 s was tested on this
#: dataset and is worse (it blurs real slowing at nodes).
BOXCAR_S = 0.4
#: A single-frame displacement above this is not a rat, it is a discontinuity —
#: the carry-out jump to the sentinel pixel is tens of m/s. Smoothing ACROSS one of
#: those is what turns a single off-maze sample into a dozen fictitious positions
#: strung between the maze and the sentinel, so segments are cut here first.
JUMP_M = 0.25


def _merge_windows(wins):
    """Sorted, non-overlapping (t0, t1) — searchsorted lookups assume both."""
    out = []
    for a, b in sorted(wins):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _in_windows(times, windows):
    """Boolean mask: which `times` fall inside any of the merged `windows`."""
    times = np.asarray(times, float)
    if not windows or times.size == 0:
        return np.ones(times.shape, dtype=bool)
    starts = np.array([a for a, _ in windows]); ends = np.array([b for _, b in windows])
    i = np.searchsorted(starts, times, side="right") - 1
    ok = i >= 0
    ok[ok] &= times[ok] <= ends[i[ok]]
    return ok


def _trial_windows(session_dir, nwb, t):
    """(trials, windows, source) from :func:`nwb.build_trials`.

    build_trials silently falls back to align_trials(RecordingMeta), which derives
    its offset from session_start_time — a 1970 epoch in several sessions — and
    from behavioural-clock trial times that drift against the video clock. Trials
    from that path must not be allowed to define the windows that gate ALL
    occupancy, so we check the same inputs build_trials checks and report the
    provenance instead of trusting whatever comes back.
    """
    coords = (nwbio._pick_file(session_dir, "*Coordinates_Full_with_frames.csv")
              or nwbio._pick_file(session_dir, "*Coordinates_Full.csv"))
    have_files = coords is not None and nwbio.frame_to_seconds(session_dir) is not None
    try:
        trials = nwbio.build_trials(session_dir, nwb.session_start_time,
                                    float(t.min()), float(t.max()))
    except Exception as e:
        _log(f"    build_trials failed ({e}); place-field metrics use the WHOLE series.")
        return [], [], "unavailable"
    if not have_files:
        return trials, [], "recordingmeta_fallback"
    wins = _merge_windows([(float(a), float(b)) for (_tt, _g, _sn, a, b) in trials
                           if np.isfinite(a) and np.isfinite(b) and b > a])
    return trials, wins, ("build_trials" if wins else "empty")


def _prep_positions(x, y, t, windows, dt):
    """Trial-restricted positions, boxcar-smoothed and speed-tagged PER TRIAL.

    Order matters and is not interchangeable: smoothing first and trimming after
    turns every carry-out teleport into ~12 fictitious off-maze samples that sail
    through any speed gate (measured: 1 -> 21 valid bins off the run corridor).
    Computing the speed per trial also stops np.diff from manufacturing one
    tens-of-m/s sample at each trial boundary once the intervals are removed.
    """
    k = max(1, int(round(BOXCAR_S / dt)))
    xs, ys, ts, sp = [], [], [], []
    for a, b in (windows or [(float(t.min()), float(t.max()))]):
        m = (t >= a) & (t <= b)
        if m.sum() < 2:
            continue
        xw, yw, tw = x[m], y[m], t[m]
        # Cut each window again at teleports. Inside a trial this is a no-op; when
        # there are no trial windows at all it is what stops the boxcar from
        # inventing a trail of off-maze samples between the maze and the sentinel.
        cuts = np.flatnonzero(np.hypot(np.diff(xw), np.diff(yw)) > JUMP_M) + 1
        for xi, yi, ti in zip(np.split(xw, cuts), np.split(yw, cuts), np.split(tw, cuts)):
            if xi.size < 2:
                continue          # a lone sample has no defensible speed
            xi = behaviour.moving_average(xi, k)
            yi = behaviour.moving_average(yi, k)
            xs.append(xi); ys.append(yi); ts.append(ti)
            sp.append(behaviour.compute_speed_from_xy(xi, yi, 1.0 / dt))
    if not xs:
        return None
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(ts), np.concatenate(sp))


def collect_session(nwb_path, bin_cm=2.5, smooth_cm=5.0, speed=0.02,
                    field_frac=PF.DEFAULT_FIELD_FRAC, min_field_cm=PF.DEFAULT_MIN_FIELD_CM,
                    min_occ_s=PF.DEFAULT_MIN_OCC_S, min_peak_hz=PF.DEFAULT_MIN_PEAK_HZ,
                    min_spikes=100, si_match_n=300, si_match_repeats=20, seed=0):
    """Return a dict of per-session summary stats, or None on failure."""
    # load_namespaces=True re-opens the file to read its cached spec, and on an SMB
    # mount that second handle fails to close ("Bad file descriptor", errno 9),
    # killing the whole session. Nothing here reads a custom neurodata type - units,
    # position and Trials_Data are all standard - so skip the namespace load. For the
    # same reason the close below is allowed to fail: the data is already in memory.
    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=False)
    try:
        nwb = io.read()
        subj = nwb.subject.subject_id if nwb.subject is not None else "?"
        desc = str(nwb.session_description)
        # session date: prefer the parent folder name (session folders are named
        # YYYYMMDD), else the NWB session_id, else an 8-digit token in the filename.
        folder = nwb_path.parent.name
        sid = str(nwb.session_id) if nwb.session_id else ""
        if len(folder) == 8 and folder.isdigit():
            date = folder
        elif len(sid) == 8 and sid.isdigit():
            date = sid
        else:
            m = re.search(r"(\d{8})", nwb_path.stem)
            date = m.group(1) if m else (sid or nwb_path.stem)
        rep = re.search(r"Repeat\s+(\d+)", desc)
        ses = re.search(r"Session\s+(\d+)", desc)
        repeat = int(rep.group(1)) if rep else None
        session = int(ses.group(1)) if ses else None
        out = {"animal": f"Rat{int(subj)}" if str(subj).isdigit() else str(subj),
               "date": date, "repeat": repeat, "session": session, "split": False,
               "n_good": 0, "n_mua": 0, "n_pyr": 0, "n_int": 0,
               # units that actually contributed a place-field metric; every panel
               # of the figure has this denominator, and it moves with the data
               "n_pf_units": 0, "n_pf_units_post": 0,
               "trial_windows": "no_position", "n_trials": 0, "trial_time_s": np.nan,
               "bin_cm": float(bin_cm), "smooth_cm": float(smooth_cm),
               "sigma_bins": np.nan, "min_occ_s": float(min_occ_s),
               "field_frac": float(field_frac), "min_field_cm": float(min_field_cm),
               "speed_thresh_ms": float(speed), "min_spikes": int(min_spikes)}
        for qtag in _DEC_QUALS:                        # decoding accuracy per lead
            for L in _DEC_LEADS:
                out[_dec_key(qtag, L)] = np.nan
        for k in _PF_KEYS:
            out[k] = np.nan; out[k + "_post"] = np.nan
        out.update(_decode_accuracy(nwb_path.parent))   # decoding accuracy (step b)
        # pooled per-(trial,unit) metrics table written by step b (if present)
        out["trial_unit_metrics"] = _load_trial_unit_metrics(
            nwb, {"animal": out["animal"], "date": out["date"],
                  "repeat": out["repeat"], "session": out["session"]})
        if nwb.units is None or len(nwb.units.id) == 0:
            return out
        udf = nwb.units.to_dataframe()
        ql = udf["quality_label"].astype(str) if "quality_label" in udf else pd.Series("good", index=udf.index)
        out["n_good"] = int((ql == "good").sum())
        out["n_mua"] = int((ql == "mua").sum())

        # Trial windows come first: they define the exposure for the firing rates
        # (hence the cell-type labels) as well as for the rate maps, and both must
        # see the same behaviour or the population entering the place-field panels
        # is set by recording length rather than by physiology.
        pos = nwbio.load_position(nwb)
        trials, windows, tsrc = [], [], "no_position"
        if pos is not None:
            trials, windows, tsrc = _trial_windows(nwb_path.parent, nwb, pos[2])
        if tsrc != "build_trials":
            _log(f"    {nwb_path.name}: no trustworthy trial windows ({tsrc}) — "
                 f"metrics fall back to the WHOLE series, inter-trial intervals "
                 f"included; treat them as not comparable to windowed sessions.")
        out["trial_windows"] = tsrc
        out["n_trials"] = len(windows)
        out["trial_time_s"] = float(sum(b - a for a, b in windows)) if windows else np.nan

        um = _unit_metrics(nwb, udf, windows=windows)
        udf["cell_type"] = um["cell_type"]
        good = udf[ql == "good"]
        out["n_pyr"] = int((good["cell_type"] == "pyramidal").sum())
        out["n_int"] = int((good["cell_type"] == "interneuron").sum())
        # per-unit metrics (good units) for the cross-session cell-type scatter
        gm = um.loc[good.index].copy()
        gm.insert(0, "date", out["date"]); gm.insert(0, "animal", out["animal"])
        out["units"] = gm

        pyr = good[good["cell_type"] == "pyramidal"]
        if pos is None or not len(pyr):
            return out
        t_raw = pos[2]
        dt = float(np.median(np.diff(t_raw))) if t_raw.size > 1 else 1.0 / 30
        prep = _prep_positions(pos[0] / maze.SCALE_X, pos[1] / maze.SCALE_Y, t_raw,
                               windows, dt)
        if prep is None:
            return out
        x, y, t, spd = prep
        ext = maze.MAZE_EXTENT
        bm = bin_cm / 100.0
        bins = (max(5, int(round((ext[1] - ext[0]) / bm))),
                max(5, int(round((ext[3] - ext[2]) / bm))))
        # The grid is an integer count per axis, so the bin the maps actually use is
        # the extent divided by that count — every physical threshold below is
        # converted from the REALISED size, never from the requested bin_cm.
        bx_cm, by_cm = PF.bin_size_cm(ext, bins)
        if abs(bx_cm - by_cm) > 1e-3 * max(bx_cm, by_cm):
            raise ValueError(f"--bin_cm {bin_cm} gives non-square bins "
                             f"({bx_cm:.4f} x {by_cm:.4f} cm) on a "
                             f"{ext[1] - ext[0]:g} x {ext[3] - ext[2]:g} m maze")
        sigma = float(smooth_cm) / bx_cm         # scipy wants bins; we think in cm
        if not 0.75 <= sigma <= 6.0:
            raise ValueError(f"smoothing sigma = {sigma:.2f} bins "
                             f"({smooth_cm} cm / {bx_cm:.3f} cm): below ~0.75 the map "
                             "is aliased, above ~6 it is over-smoothed — pick a "
                             "bin_cm/smooth_cm pair in between")
        out["bin_cm"] = bx_cm; out["smooth_cm"] = float(smooth_cm)
        out["sigma_bins"] = sigma; out["min_occ_s"] = float(min_occ_s)
        out["field_frac"] = float(field_frac); out["min_field_cm"] = float(min_field_cm)
        out["speed_thresh_ms"] = float(speed); out["min_spikes"] = int(min_spikes)
        nodes = maze.node_positions_m()

        # Per-unit datapoints (for statistics) + the session-level means (for the
        # plots). A base row carries the unit's identity + waveform/ACG metrics.
        pyr_idx = list(good.index[good["cell_type"] == "pyramidal"])
        unit_rows = []

        def _base(idx):
            r = um.loc[idx]
            uid = int(udf.loc[idx, "phy_cluster_id"]) if "phy_cluster_id" in udf.columns else int(idx)
            return {"animal": out["animal"], "date": date, "repeat": repeat,
                    "session": session, "unit_id": uid, "quality_label": "good",
                    "cell_type": r["cell_type"], "subtype": r["subtype"],
                    "firing_rate_hz": r["firing_rate_hz"],
                    "trough_to_peak_s": r["trough_to_peak_s"],
                    "acg_tau_rise_ms": r["acg_tau_rise_ms"]}

        def _pyr_epoch(epoch, goal_node, t0, t1):
            """Compute per-pyramidal-unit place-field metrics for one epoch, append
            each unit's datapoint row, and return the session summary per metric.

            The summary is a MEDIAN over the contributing units, not a mean: the
            low-spike-count tail of the SI/selectivity distribution is heavy and
            grows every session as trials shorten, so a mean is dragged upward by
            exactly the cells that have the least data. Units below `min_spikes`
            contribute nothing at all (place_field_metrics returns all-NaN for
            them) and the surviving n is reported next to the panel.
            """
            gxy = nodes.get(goal_node)
            vals = {k: [] for k in _PF_KEYS}
            for idx in pyr_idx:
                st = np.asarray(udf.loc[idx, "spike_times"], dtype=float)
                # Spikes are restricted to the trial windows too. Dropping the
                # inter-trial samples leaves interior gaps in t, and np.interp
                # fills a gap with a straight line: an off-task spike would
                # otherwise be planted on a fabricated chord across the maze.
                st = st[_in_windows(st, windows)]
                m, _, _ = place_field_metrics(
                    x, y, t, st, ext, bins, dt, sigma, speed, goal_xy=gxy,
                    t0=t0, t1=t1, field_frac=field_frac, min_peak_hz=min_peak_hz,
                    min_field_cm=min_field_cm, min_occ_s=min_occ_s, speed=spd,
                    min_spikes=min_spikes, si_match_n=si_match_n,
                    si_match_repeats=si_match_repeats, seed=seed)
                d = _base(idx); d["epoch"] = epoch; d["goal_node"] = goal_node
                for k in _PF_KEYS:
                    d[k] = m[k]; vals[k].append(m[k])
                unit_rows.append(d)
            summ = {k: (float(np.nanmedian(v)) if np.any(np.isfinite(v)) else np.nan)
                    for k, v in vals.items()}
            summ["n_pf_units"] = int(np.isfinite(vals["spatial_info"]).sum())
            return summ

        # Split into before/after the type-5 (goal-switch) trial ONLY for RxS1
        # sessions with repeat > 1 (R1S1 and all S>1 stay whole-session). Trials
        # from the RecordingMeta fallback would misplace the boundary, so the split
        # is only attempted when build_trials used the coordinate/seconds files.
        type5 = None
        if session == 1 and repeat and repeat > 1 and tsrc == "build_trials":
            type5 = next((tr for tr in trials if tr[0] == 5), None)
            if type5 is not None:
                _tt, g5, _sn, t50, t51 = type5
                gb = [g for (tt, g, sn, a, b) in trials if b <= t50 and g is not None]
                goal_before = Counter(gb).most_common(1)[0][0] if gb else None
                pre = _pyr_epoch("before", goal_before, float(t.min()), t50)
                post = _pyr_epoch("after", g5, t51, float(t.max()))
                out["split"] = True
                for k in _PF_KEYS:
                    out[k] = pre[k]; out[k + "_post"] = post[k]
                # The two epochs are NOT matched in length or in spikes per cell, so
                # whichever is shorter is expected to show higher SI/peak/selectivity
                # and fewer fields even if the goal switch changed nothing. Their
                # exposure is carried in the table (epoch_dur_s / n_spikes_epoch,
                # per unit) so the offset can be checked before it is believed.
                out["n_pf_units"] = pre["n_pf_units"]
                out["n_pf_units_post"] = post["n_pf_units"]
        if type5 is None:
            whole = _pyr_epoch("whole", _session_goal(nwb, udf), None, None)
            for k in _PF_KEYS:
                out[k] = whole[k]
            out["n_pf_units"] = whole["n_pf_units"]

        # interneuron good units: one datapoint row each (no place fields)
        for idx in good.index[good["cell_type"] != "pyramidal"]:
            d = _base(idx); d["epoch"] = "whole"; d["goal_node"] = None
            for k in _PF_KEYS:
                d[k] = np.nan
            unit_rows.append(d)
        out["unit_rows"] = pd.DataFrame(unit_rows)
        return out
    finally:
        try:
            io.close()
        except Exception:
            pass


def _param_stamp(sessions):
    """One-line record of the parameters a figure/table was produced with. Every
    plotted metric except the unit counts moves with bin size and smoothing, so a
    figure that does not say which it used cannot be compared with another one."""
    s = next((s for s in sessions if np.isfinite(s.get("sigma_bins", np.nan))), None)
    if s is None:
        return "parameters unrecorded (no session produced place-field metrics)"
    return (f"bin {s['bin_cm']:.2f} cm · smooth {s['smooth_cm']:.1f} cm "
            f"(sigma {s['sigma_bins']:.2f} bins) · min pooled occ {s['min_occ_s']:.2f} s · "
            f"field >= {s['field_frac']:.2f}·peak & {s['min_field_cm']:.0f} cm long · "
            f"speed > {s['speed_thresh_ms']:.2f} m/s · min {s['min_spikes']:.0f} spikes/epoch")


def _plot_animal(pdf, animal, sessions, units_df=None):
    sessions = sorted(sessions, key=lambda s: s["date"])
    dates = [s["date"] for s in sessions]
    x = np.arange(len(sessions))
    labels = [f"{s['date']}\nR{s['repeat']}·S{s['session']}" for s in sessions]

    def col(key):
        return np.array([s.get(key) if s.get(key) is not None else np.nan
                         for s in sessions], dtype=float)

    # one panel per plotted metric, plus the two unit-count bars and decoding
    nrow = int(np.ceil((len(_PF_PLOT) + 3) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 3.6 * nrow))
    flat = list(axes.ravel())
    # units: good vs mua
    ax = axes[0, 0]; w = 0.4
    ax.bar(x - w / 2, col("n_good"), w, label="good", color="#2166ac")
    ax.bar(x + w / 2, col("n_mua"), w, label="mua", color="#b2182b")
    ax.set_title("units: good vs mua"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    # good composition: pyr vs int
    ax = axes[0, 1]
    ax.bar(x - w / 2, col("n_pyr"), w, label="pyramidal", color="#2166ac")
    ax.bar(x + w / 2, col("n_int"), w, label="interneuron", color="#f4a582")
    ax.set_title("good units: pyramidal vs interneuron"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    # pyramidal metrics: "pre / whole" line + "after type5" markers (RxS1 splits)
    any_split = any(s.get("split") for s in sessions)
    metric_axes = flat[2:2 + len(_PF_PLOT)]
    # decoding accuracy across sessions (median error, step b) at every lead:
    # colour = unit set (good/good+mua), line style = prediction lead (0/1/3 s).
    axd = flat[2 + len(_PF_PLOT)]
    for extra in flat[3 + len(_PF_PLOT):]:
        extra.axis("off")
    _qcol = {"good": "#2166ac", "good_mua": "#b2182b"}
    _qlab = {"good": "good", "good_mua": "good+mua"}
    _lsty = {0.0: "-", 1.0: "--", 3.0: ":"}
    any_dec = False
    for qtag in _DEC_QUALS:
        for L in _DEC_LEADS:
            vals = col(_dec_key(qtag, L))
            if np.isfinite(vals).any():
                any_dec = True
                axd.plot(x, vals, marker="o", ms=4, color=_qcol[qtag],
                         ls=_lsty.get(L, "-"), label=f"{_qlab[qtag]} +{L:g}s")
    if any_dec:
        axd.set_title("decoding accuracy (median error)"); axd.set_ylabel("error (m)")
        axd.legend(fontsize=6, ncol=2)
        axd.set_xticks(x); axd.set_xticklabels(labels, fontsize=6)
        axd.spines["top"].set_visible(False); axd.spines["right"].set_visible(False)
        axd.set_ylim(bottom=0)
    else:
        axd.axis("off")
    n_units = col("n_pf_units")
    for ax, (key, title, ylab) in zip(metric_axes, _PF_PLOT):
        ax.plot(x, col(key), "o-", color="#2166ac",
                label="whole / before type5" if any_split else None)
        if any_split:
            ax.plot(x, col(key + "_post"), "s", color="#d62728", label="after type5")
            ax.legend(fontsize=6)
        ax.set_title(title); ax.set_ylabel(ylab)
        # The number of units behind each point moves across sessions for its own
        # sampling reasons, so it is printed rather than left to be assumed equal.
        for xi, (vi, ni) in enumerate(zip(col(key), n_units)):
            if np.isfinite(vi) and np.isfinite(ni):
                ax.annotate(f"{int(ni)}", (xi, vi), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=5, color="0.45")
        note = _PANEL_NOTE.get(key)
        if note:
            ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=6, color="#b2182b")
        if units_df is not None:
            # Per-unit ANOVA across sessions. Units within a session share one
            # occupancy map, one duration and one spike-count regime, so they are
            # NOT independent replicates and this p-value is anticonservative by
            # roughly the number of units per session. It is kept as a descriptive
            # summary of the per-unit spread and labelled as such; the session-level
            # test lives on the pooled page, where there is more than one session
            # per group.
            _F, p, k, N = stats.oneway_anova(stats.groups_by(units_df, "date", key, order=dates))
            if np.isfinite(p):
                ax.text(0.98, 0.03, f"per-unit ANOVA p={p:.3g} (k={k}, pseudo-replicated)",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                        color="#b2182b" if p < 0.05 else "0.3")
    for ax in [axes[0, 0], axes[0, 1]] + metric_axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)          # all summary axes start from 0
    windowed = sum(1 for s in sessions if s.get("trial_windows") == "build_trials")
    fig.suptitle(f"{animal} — cross-session summary ({len(sessions)} sessions, "
                 f"{windowed} trial-windowed)\n{_param_stamp(sessions)}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


def _plot_scatter(pdf, animal, units):
    """Scatter of ALL good units across the animal's sessions, coloured by subtype
    (pyramidal blue, narrow interneuron red, wide interneuron green): trough-to-peak
    vs ACG tau_rise, and trough-to-peak vs firing rate."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 5))
    for sub, c in SUBTYPE_COLORS.items():
        d = units[units["subtype"] == sub]
        axa.scatter(d["trough_to_peak_s"] * 1e3, d["acg_tau_rise_ms"], s=14, alpha=0.6,
                    c=c, label=f"{sub} (n={len(d)})")
        axb.scatter(d["trough_to_peak_s"] * 1e3, d["firing_rate_hz"], s=14, alpha=0.6, c=c)
    axa.axvline(SM.TROUGH_PEAK_THRESH_S * 1e3, ls="--", c="grey", lw=1)
    axa.axhline(SM.ACG_TAU_RISE_THRESH_MS, ls="--", c="grey", lw=1)
    axa.set_xlabel("trough-to-peak (ms)"); axa.set_ylabel("ACG tau_rise (ms)")
    axa.legend(fontsize=7); axa.set_xlim(left=0); axa.set_ylim(bottom=0)
    axb.axvline(SM.TROUGH_PEAK_THRESH_S * 1e3, ls="--", c="grey", lw=1)
    axb.axhline(SM.RATE_THRESH_HZ, ls="--", c="grey", lw=1)
    axb.set_xlabel("trough-to-peak (ms)"); axb.set_ylabel("firing rate (Hz)")
    axb.set_yscale("log"); axb.set_xlim(left=0)
    for ax in (axa, axb):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"{animal} — all good units ({len(units)}) across sessions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def _label_order(labels):
    def key(s):
        r = re.search(r"R(\d+)", s); ss = re.search(r"S(\d+)", s)
        return (int(r.group(1)) if r else 0, int(ss.group(1)) if ss else 0)
    return sorted(labels, key=key)


def _session_level(units, group_col, value_col):
    """One value per (animal, session) — the unit of analysis for any test across
    sessions. Every unit in a session shares one occupancy map, one duration and
    one spike-count regime, which is the dominant source of the between-session
    differences, so treating units as replicates inflates F by roughly the number
    of units per session."""
    d = units[units["epoch"].isin(["whole", "before"])]
    d = d[["animal", "date", group_col, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    agg = d.groupby(["animal", "date", group_col])[value_col].median().reset_index()
    out = {}
    for g, sub in agg.groupby(group_col):
        v = sub[value_col].to_numpy(dtype=float)
        out[str(g)] = v[np.isfinite(v)]
    return out


def _plot_combined(pdf, units_all):
    """All animals pooled: pyramidal place-field metrics per session (grouped by
    R{repeat}S{session} label), box over units + the session-level medians, with a
    SESSION-level one-way ANOVA (one value per animal-session, not per unit)."""
    d = units_all[units_all["epoch"].isin(["whole", "before"])].copy()
    nrow = int(np.ceil(len(_PF_PLOT) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 4.2 * nrow))
    for extra in list(axes.ravel())[len(_PF_PLOT):]:
        extra.axis("off")
    for ax, (key, title, ylab) in zip(axes.ravel(), _PF_PLOT):
        groups = stats.groups_by(d, "session_label", key)          # per-unit, display
        sess = _session_level(d, "session_label", key)             # per-session, test
        labs = [l for l in _label_order(groups) if len(groups[l])]
        data = [groups[l] for l in labs]
        xx = np.arange(len(labs))
        if data:
            ax.boxplot(data, positions=xx, widths=0.6, showfliers=False)
            ax.plot(xx, [np.median(sess.get(l, [np.nan])) if len(sess.get(l, []))
                         else np.nan for l in labs], "o-", color="#2166ac",
                    label="session medians")
        _F, p, k, N = stats.oneway_anova(sess)
        if np.isfinite(p):
            ax.text(0.98, 0.97, f"session-level ANOVA p={p:.3g} (k={k}, N={N} sessions)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    color="#b2182b" if p < 0.05 else "0.3")
        else:
            ax.text(0.98, 0.97, "session-level ANOVA undefined (needs >=2 sessions "
                                "per group)", transform=ax.transAxes, ha="right",
                    va="top", fontsize=7, color="0.45")
        note = _PANEL_NOTE.get(key)
        if note:
            ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=6, color="#b2182b")
        ax.set_title(title); ax.set_ylabel(ylab)
        ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=7)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    n_an = units_all["animal"].nunique()
    fig.suptitle(f"All animals combined ({n_an}) — pyramidal metrics by session\n"
                 f"boxes are units (display only); tests are session-level",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def _constant_within_groups(g, tol=1e-9):
    """True when a metric has no variance inside any group with >1 member.

    Some columns are session-level quantities carried on every unit row -
    field_goal_null_m is the epoch's own occupancy null, identical for all units in
    the session. Running a units-as-replicates ANOVA on one gives within-group
    variance 0, hence F = inf and p = 0, which reads as overwhelming significance
    when it is really a degenerate test. Detect and refuse rather than print it.
    """
    seen = False
    for v in g.values():
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        if v.size > 1:
            seen = True
            if v.std() > tol * max(1.0, abs(float(v.mean()))):
                return False
    return seen


def _stats_tables(animal_units, units_all):
    """Per-animal (grouped by date) + combined (grouped by session label) one-way
    ANOVA and post-hoc pairwise tables for each pyramidal metric.

    Every row carries `unit_of_analysis` and the metric's sampling note: the
    per-animal rows can only be computed per UNIT (one date is one session, so a
    session-level test has n=1 per group) and are pseudo-replicated by
    construction; the pooled rows are computed per SESSION and are the ones to
    quote. The exposure columns are tested too — a significant trend in
    n_spikes_epoch or n_valid_bins is the direct evidence that a trend in the
    metrics above it may be sampling.
    """
    anova, posthoc = [], []
    for animal in sorted(animal_units):
        u = animal_units[animal]
        dates = sorted(u["date"].unique())
        for key in _PF_KEYS:
            g = stats.groups_by(u, "date", key, order=dates)
            degenerate = _constant_within_groups(g)
            F, p, k, N = stats.oneway_anova(g)
            if degenerate:
                F, p = np.nan, np.nan
            anova.append({"scope": animal, "metric": key, "F": F, "p": p,
                          "k_groups": k, "N": N, "unit_of_analysis": "unit",
                          "caveat": ("session-level quantity: constant within each "
                                     "session, so a units-as-replicates ANOVA is "
                                     "degenerate (F=inf) and is not reported"
                                     if degenerate else
                                     "pseudo-replicated: units within a session share "
                                     "one occupancy map and one spike-count regime"),
                          "note": PF.METRIC_NOTES.get(key, "")})
            posthoc += [dict(r, unit_of_analysis="unit") for r in stats.posthoc(g, animal, key)]
    for key in _PF_KEYS:
        g = _session_level(units_all, "session_label", key)
        g = {l: g[l] for l in _label_order(g)}
        F, p, k, N = stats.oneway_anova(g)
        anova.append({"scope": "ALL", "metric": key, "F": F, "p": p, "k_groups": k,
                      "N": N, "unit_of_analysis": "session", "caveat": "",
                      "note": PF.METRIC_NOTES.get(key, "")})
        posthoc += [dict(r, unit_of_analysis="session") for r in stats.posthoc(g, "ALL", key)]
    return pd.DataFrame(anova), pd.DataFrame(posthoc)


# ------------------------------------------------------------
#   pooled per-(trial, unit) metrics from step b (all sessions)
# ------------------------------------------------------------
# metrics correlated against trial performance (column, axis label)
_TU_MEASURES = [
    ("spatial_info", "Spatial info (bits/spk)"),
    ("field_size_m2", "Field size (m$^2$)"),
    ("selectivity", "Selectivity (peak/mean)"),
    ("firing_rate_hz", "Firing rate (Hz)"),
    ("decoding_error_m", "Decoding error (m)"),
    ("between_node_speed", "Between-node speed (m/s)"),
]
# Performance is log10(shortest_hops/actual_hops), and actual hops is monotone in
# path length, hence in trial duration, hence in how many spikes went into every
# per-trial metric. Correlating a sample-size-sensitive metric against it therefore
# has a guaranteed non-zero answer: an IDENTICAL simulated neuron, with trials
# varying only in length over the real 9 s - 210 s range, gives spatial_info
# r=+0.74 and field_size r=-0.52. Whatever exposure column step b wrote is used to
# partial that out and is plotted alongside, so the reader can see the confound
# rather than infer it.
_TU_EXPOSURE_COLS = ("n_spikes", "spike_count", "n_spikes_trial", "trial_dur_s",
                     "duration_s")
_TU_CONFOUND_NOTE = ("performance is monotone in trial length -> in spikes/trial; "
                     "these r values are confounded unless spike counts are matched")


def _tu_exposure(data):
    """(column, label) of the exposure variable step b happened to write, or None."""
    for c in _TU_EXPOSURE_COLS:
        if c in data.columns and pd.to_numeric(data[c], errors="coerce").notna().any():
            return c, f"{c} (exposure)"
    return None


def _partial_r(a, b, z):
    """Pearson r(a, b) with z partialled out — the correlation that survives once
    the amount of data per trial is held constant."""
    from scipy.stats import pearsonr
    if len(a) < 4:
        return np.nan
    rab = pearsonr(a, b)[0]; raz = pearsonr(a, z)[0]; rbz = pearsonr(b, z)[0]
    den = np.sqrt(max(0.0, (1 - raz ** 2) * (1 - rbz ** 2)))
    return (rab - raz * rbz) / den if den > 0 else np.nan


def _load_trial_unit_metrics(nwb, base):
    """Pooled per-(trial,unit) metrics tables written by step b into the NWB
    scratch ('trial_unit_metrics_<quality>'), tagged with this session's identity
    (base = animal/date/repeat/session). Concatenated over quality sets, or None."""
    sc = getattr(nwb, "scratch", None)
    if not sc:
        return None
    frames = []
    for name in list(sc.keys()):
        if not name.startswith("trial_unit_metrics_"):
            continue
        try:
            df = sc[name].to_dataframe()
        except Exception:
            continue
        if df is None or not len(df):
            continue
        df = df.reset_index(drop=True).copy()
        df["quality"] = name.replace("trial_unit_metrics_", "")
        for k, v in base.items():
            df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def _tu_trial_agg(data):
    """Collapse units within each (animal, date, quality, trial): mean of the
    per-unit metrics, performance/trial_type carried through (trial-level).

    `n_units` comes back too: a trial with one unit is a far noisier point than a
    trial with forty, and the panels say so rather than weighting them equally.
    """
    keys = [k for k in ("animal", "date", "quality", "trial") if k in data.columns]
    agg = {c: "mean" for c, _ in _TU_MEASURES if c in data.columns}
    exp = _tu_exposure(data)
    if exp:
        agg[exp[0]] = "sum" if exp[0] != "trial_dur_s" else "first"
    if "performance" in data.columns:
        agg["performance"] = "first"
    if "trial_type" in data.columns:
        agg["trial_type"] = "first"
    out = data.groupby(keys).agg(agg)
    out["n_units"] = data.groupby(keys).size()
    return out.reset_index()


def _tu_corr_grid(pdf, data, title, color_col="animal"):
    """2x3 grid: performance vs each metric, with a least-squares line + Pearson
    r/p/n. Points optionally coloured by `color_col` (e.g. animal).

    Each panel also reports n at the TRIAL level and, when step b wrote an exposure
    column, the partial r with that exposure held constant — the r that is left
    after the trial-length confound is removed.
    """
    from scipy.stats import pearsonr
    cats = (list(pd.unique(data[color_col].dropna()))
            if color_col and color_col in data.columns else None)
    exp = _tu_exposure(data)
    n_trials = (data[[c for c in ("animal", "date", "trial") if c in data.columns]]
                .drop_duplicates().shape[0]
                if "trial" in data.columns else None)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    for ax, (col, lab) in zip(axes.ravel(), _TU_MEASURES):
        if col not in data.columns or "performance" not in data.columns:
            ax.axis("off"); continue
        keep = list(dict.fromkeys(          # exp may BE the plotted column
            [col, "performance"] + ([color_col] if cats else []) + ([exp[0]] if exp else [])))
        d = data[keep].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) >= 3 and d[col].std() > 0 and d["performance"].std() > 0:
            if cats:
                for i, cval in enumerate(cats):
                    dd = d[d[color_col] == cval]
                    ax.scatter(dd[col], dd["performance"], s=12, alpha=0.45,
                               edgecolor="none", color=cmap(i % 10), label=str(cval))
            else:
                ax.scatter(d[col], d["performance"], s=12, alpha=0.45, edgecolor="none")
            r, p = pearsonr(d[col], d["performance"])
            b, a = np.polyfit(d[col], d["performance"], 1)
            xs = np.linspace(d[col].min(), d[col].max(), 50)
            ax.plot(xs, b * xs + a, color="crimson", lw=1.3)
            extra = ""
            if exp and d[exp[0]].std() > 0:
                rp = _partial_r(d[col].to_numpy(float), d["performance"].to_numpy(float),
                                d[exp[0]].to_numpy(float))
                extra = f"; r|{exp[0]}={rp:.2f}"
            ntxt = f"n={len(d)}" + (f" pts / {n_trials} trials" if n_trials else " pts")
            ax.set_title(f"{lab}\nr={r:.2f}, p={p:.2g}, {ntxt}{extra}", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{lab}\n(insufficient data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel("Performance  log10(short/actual)", fontsize=8)
        ax.grid(alpha=0.3)
    if cats and len(cats) <= 10:
        axes.ravel()[0].legend(fontsize=6, markerscale=1.6, loc="best")
    fig.suptitle(f"{title}\n{_TU_CONFOUND_NOTE}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def _tu_compare_page(pdf, tu_all):
    """Comparison page: (A) Pearson r of performance vs each metric, per quality
    set (trial-level) — which measures track performance; (B) key metrics split by
    trial type (goal-directed type-1 vs free-roaming type-4/5)."""
    from scipy.stats import pearsonr
    quals = list(pd.unique(tu_all["quality"]))
    trial_lvl = _tu_trial_agg(tu_all)
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.45)

    # (A) correlation-r bars, one group of bars per metric, one bar per quality set.
    # The exposure variable is charted as one more "measure": a ranked list of what
    # correlates with performance is largely a ranked list of what is sample-size
    # sensitive, and the exposure bar is the yardstick for reading the rest. Where
    # an exposure column exists the partial r (hollow bar) is drawn on top of it.
    exp = _tu_exposure(trial_lvl)
    measures = list(_TU_MEASURES) + ([exp] if exp else [])
    axA = fig.add_subplot(gs[0])
    xpos = np.arange(len(measures))
    width = 0.8 / max(1, len(quals))
    for qi, q in enumerate(quals):
        dq = trial_lvl[trial_lvl["quality"] == q]
        rs, rps = [], []
        for col, _ in measures:
            want = list(dict.fromkeys([col, "performance"] + ([exp[0]] if exp else [])))
            d = (dq[[c for c in want if c in dq.columns]]
                 .replace([np.inf, -np.inf], np.nan).dropna()
                 if col in dq.columns else pd.DataFrame())
            ok = len(d) >= 3 and col in d and d[col].std() > 0 and d["performance"].std() > 0
            rs.append(pearsonr(d[col], d["performance"])[0] if ok else np.nan)
            rps.append(_partial_r(d[col].to_numpy(float), d["performance"].to_numpy(float),
                                  d[exp[0]].to_numpy(float))
                       if ok and exp and col != exp[0] and d[exp[0]].std() > 0 else np.nan)
        axA.bar(xpos + qi * width, rs, width, label=q.replace("_", "+"))
        if exp and np.isfinite(rps).any():
            axA.bar(xpos + qi * width, rps, width, facecolor="none", edgecolor="k",
                    lw=0.8, label=f"{q.replace('_', '+')} | {exp[0]}")
    axA.axhline(0, color="k", lw=0.6)
    axA.set_xticks(xpos + width * (len(quals) - 1) / 2)
    axA.set_xticklabels([l for _, l in measures], rotation=18, ha="right", fontsize=7)
    axA.set_ylabel("Pearson r vs performance\n(unit-averaged per trial)", fontsize=8)
    axA.set_title("Correlation with trial performance — NOT a ranking of neural "
                  "measures: " + _TU_CONFOUND_NOTE, fontsize=8)
    axA.legend(fontsize=6, title="units"); axA.grid(axis="y", alpha=0.3)

    # (B) key metrics by trial-type group (richest quality set)
    qsel = "good_mua" if "good_mua" in quals else quals[0]
    sub = tu_all[tu_all["quality"] == qsel].copy()
    sub["grp"] = np.where(sub["trial_type"].isin([4, 5]), "free-roam (4/5)", "goal (1)")
    keys_b = [("spatial_info", "Spatial info"), ("firing_rate_hz", "Firing rate (Hz)"),
              ("decoding_error_m", "Decoding err (m)"), ("performance", "Performance")]
    gsB = gs[1].subgridspec(1, len(keys_b), wspace=0.5)
    order = ["goal (1)", "free-roam (4/5)"]
    for j, (col, lab) in enumerate(keys_b):
        axb = fig.add_subplot(gsB[0, j])
        data = [sub.loc[sub["grp"] == g, col].replace([np.inf, -np.inf], np.nan).dropna().values
                for g in order]
        if any(len(dd) for dd in data):
            axb.boxplot(data, tick_labels=["goal", "free"], showfliers=False, widths=0.6)
            for xi, dd in enumerate(data, 1):
                if len(dd):
                    axb.scatter(np.random.default_rng(0).normal(xi, 0.05, len(dd)), dd,
                                s=6, alpha=0.35, color="#2166ac", edgecolor="none")
        axb.set_title(lab, fontsize=8); axb.grid(axis="y", alpha=0.3)
        axb.tick_params(labelsize=7)
    fig.suptitle(f"Trial-type comparison (units: {qsel.replace('_', '+')})",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


def _plot_trial_unit_pool(pdf, tu_all):
    """Pooled step-b trial/unit pages: performance-vs-metric scatter at
    (trial,unit) granularity and unit-averaged trial granularity, plus a
    comparison page. Uses the richest quality set (good+mua) for the scatters."""
    quals = list(pd.unique(tu_all["quality"]))
    qsel = "good_mua" if "good_mua" in quals else quals[0]
    sub = tu_all[tu_all["quality"] == qsel]
    n_sess = sub[["animal", "date"]].drop_duplicates().shape[0] if len(sub) else 0
    _tu_corr_grid(pdf, sub,
                  f"Performance vs metrics — every (trial, unit) pooled over "
                  f"{n_sess} session(s) — units {qsel.replace('_', '+')}")
    _tu_corr_grid(pdf, _tu_trial_agg(sub),
                  f"Performance vs unit-averaged metrics — one point per trial, "
                  f"pooled over {n_sess} session(s) — units {qsel.replace('_', '+')}")
    _tu_compare_page(pdf, tu_all)


def _out_stem(bin_cm, smooth_cm, min_occ_s):
    """Output basename carrying the parameters that change every plotted metric —
    a 2.5 cm run must not overwrite a 5 cm one in place, because the two files
    would have identical names and identical columns while being incomparable."""
    return f"session_summary_bin{bin_cm:g}cm_sm{smooth_cm:g}cm_occ{min_occ_s:.2f}s"


def run(root, bin_cm=2.5, smooth_cm=5.0, speed=0.02,
        field_frac=PF.DEFAULT_FIELD_FRAC, min_field_cm=PF.DEFAULT_MIN_FIELD_CM,
        min_occ_s=PF.DEFAULT_MIN_OCC_S, min_peak_hz=PF.DEFAULT_MIN_PEAK_HZ,
        min_spikes=100, si_match_n=300, si_match_repeats=20, seed=0):
    root = Path(root)
    params = {"bin_cm": bin_cm, "smooth_cm": smooth_cm, "speed": speed,
              "field_frac": field_frac, "min_field_cm": min_field_cm,
              "min_occ_s": min_occ_s, "min_peak_hz": min_peak_hz,
              "min_spikes": min_spikes, "si_match_n": si_match_n,
              "si_match_repeats": si_match_repeats, "seed": seed}
    # Cross-session summary uses only the session NWBs, which sit at most a few
    # levels down (root[/animal]/session/RatX_date.nwb). Use BOUNDED-DEPTH globs
    # instead of '**' so we never descend into the huge *_sorting_output/phy_export
    # folders (thousands of files + recording.dat) — that recursion is what makes
    # this crawl for minutes on the SMB mount. ip* folders never hold NWBs anyway.
    patterns = ["*.nwb", "*/*.nwb", "*/*/*.nwb", "*/*/*/*.nwb"]
    nwbs = sorted({p for pat in patterns for p in root.glob(pat)
                   if not p.name.endswith(".tmp.nwb") and not p.name.startswith("._")
                   and not any(re.fullmatch(r"ip\d+", part, re.I) for part in p.parts)})
    if not nwbs:
        print(f"No .nwb files found under {root}.")
        return
    print(f"Found {len(nwbs)} NWB file(s).")
    # Resolve the physical parameters ONCE, up front, and say what they mean in
    # bins — running at a different bin size silently reinterprets every threshold
    # that is not stated in cm, and this line is the record that it did not.
    _bx, _by = PF.bin_size_cm(maze.MAZE_EXTENT,
                              (max(5, int(round((maze.MAZE_EXTENT[1] - maze.MAZE_EXTENT[0]) / (bin_cm / 100.0)))),
                               max(5, int(round((maze.MAZE_EXTENT[3] - maze.MAZE_EXTENT[2]) / (bin_cm / 100.0))))))
    print(f"Parameters: bin {_bx:.3f} x {_by:.3f} cm (requested {bin_cm} cm) · "
          f"smooth {smooth_cm} cm = sigma {smooth_cm / _bx:.2f} bins · "
          f"min pooled occupancy {min_occ_s} s · field >= {field_frac:g}·peak and "
          f"field >= {min_field_cm:g} cm long (bin-size invariant; area is not), "
          f"in-field peak >= {min_peak_hz} Hz · speed > {speed} m/s (after a "
          f"{BOXCAR_S:g} s per-trial boxcar) · units need >= {min_spikes} spikes · "
          f"SI matched at {si_match_n} spikes x {si_match_repeats} draws (seed {seed})")
    by_animal = defaultdict(list)
    for p in tqdm(nwbs, desc="sessions", unit="nwb"):
        try:
            s = collect_session(p, **params)
            if s is not None:
                by_animal[s["animal"]].append(s)
                _log(f"  {p.parent.name}/{p.name}: {s['animal']} {s['date']} "
                     f"good={s['n_good']} mua={s['n_mua']} pyr={s['n_pyr']} int={s['n_int']}")
        except Exception as e:
            print(f"  Failed on {p}: {e}")
            traceback.print_exc()

    # per-animal per-neuron datapoints (with a session_label for pooled grouping)
    animal_units = {}
    for animal in sorted(by_animal):
        urows = [s["unit_rows"] for s in by_animal[animal]
                 if isinstance(s.get("unit_rows"), pd.DataFrame) and not s["unit_rows"].empty]
        if urows:
            u = pd.concat(urows, ignore_index=True)
            u["session_label"] = ("R" + u["repeat"].astype("Int64").astype(str)
                                  + "S" + u["session"].astype("Int64").astype(str))
            animal_units[animal] = u
    units_all = pd.concat(animal_units.values(), ignore_index=True) if animal_units else pd.DataFrame()

    # pooled per-(trial,unit) metrics table (step b) across every session
    tu_frames = [s["trial_unit_metrics"] for animal in by_animal for s in by_animal[animal]
                 if isinstance(s.get("trial_unit_metrics"), pd.DataFrame)
                 and not s["trial_unit_metrics"].empty]
    tu_all = pd.concat(tu_frames, ignore_index=True) if tu_frames else pd.DataFrame()

    stem = _out_stem(bin_cm, smooth_cm, min_occ_s)
    out = root / f"{stem}.pdf"
    with PdfPages(str(out)) as pdf:
        for animal in sorted(by_animal):
            _plot_animal(pdf, animal, by_animal[animal], animal_units.get(animal))
            units = [s["units"] for s in by_animal[animal] if s.get("units") is not None]
            if units:
                _plot_scatter(pdf, animal, pd.concat(units, ignore_index=True))
        if not units_all.empty and units_all["animal"].nunique() > 1:
            _plot_combined(pdf, units_all)       # all animals pooled
        if not tu_all.empty:                     # step-b performance-vs-metrics pages
            _plot_trial_unit_pool(pdf, tu_all)
    print(f"\nWrote {out} ({len(by_animal)} animal(s)).")

    # Per-session summary table (before/after type5 columns for splits). The
    # analysis parameters and the trial-window provenance are constant columns:
    # without them two spreadsheets with identical column sets can differ in every
    # value and nothing in either says why.
    cols = (["animal", "date", "repeat", "session", "split", "trial_windows",
             "n_trials", "trial_time_s", "n_good", "n_mua", "n_pyr", "n_int",
             "n_pf_units", "n_pf_units_post",
             "bin_cm", "smooth_cm", "sigma_bins", "min_occ_s", "field_frac",
             "min_field_cm", "speed_thresh_ms", "min_spikes"]
            + [_dec_key(q, L) for q in _DEC_QUALS for L in _DEC_LEADS]
            + [k for key in _PF_KEYS for k in (key, key + "_post")])
    df = pd.DataFrame([{c: s.get(c) for c in cols}
                       for animal in sorted(by_animal)
                       for s in sorted(by_animal[animal], key=lambda z: z["date"])], columns=cols)
    ucols = ["animal", "date", "repeat", "session", "session_label", "epoch", "unit_id",
             "quality_label", "cell_type", "subtype", "firing_rate_hz", "trough_to_peak_s",
             "acg_tau_rise_ms", "goal_node"] + list(_PF_KEYS)
    units_dp = units_all.reindex(columns=ucols) if not units_all.empty else pd.DataFrame()
    anova_df, posthoc_df = _stats_tables(animal_units, units_all) if animal_units else (pd.DataFrame(), pd.DataFrame())
    # A reader of the spreadsheet alone gets the same warnings the figure carries.
    notes_df = pd.DataFrame(
        [{"metric": k, "note": PF.METRIC_NOTES.get(k, ""),
          "panel_warning": _PANEL_NOTE.get(k, "")} for k in _PF_KEYS]
        + [{"metric": f"param:{k}", "note": str(v), "panel_warning": ""}
           for k, v in params.items()])

    xlsx = root / f"{stem}.xlsx"
    try:
        with pd.ExcelWriter(xlsx) as xw:
            df.to_excel(xw, index=False, sheet_name="sessions")
            if not units_dp.empty:
                units_dp.to_excel(xw, index=False, sheet_name="units")
            if not tu_all.empty:
                tu_all.to_excel(xw, index=False, sheet_name="trial_unit_metrics")
            if not anova_df.empty:
                anova_df.to_excel(xw, index=False, sheet_name="anova")
            if not posthoc_df.empty:
                posthoc_df.to_excel(xw, index=False, sheet_name="posthoc")
            notes_df.to_excel(xw, index=False, sheet_name="notes")
        print(f"Wrote {xlsx} (sessions: {len(df)}, unit datapoints: {len(units_dp)}, "
              f"anova: {len(anova_df)}, posthoc: {len(posthoc_df)} rows)")
    except Exception as e:
        for name, d in [("", df), ("_units", units_dp), ("_anova", anova_df),
                        ("_posthoc", posthoc_df), ("_notes", notes_df)]:
            if not d.empty:
                d.to_csv(root / f"{stem}{name}.csv", index=False)
        print(f"Could not write xlsx ({e}); wrote CSVs instead.")
    return out, xlsx


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-session per-animal summary plots from NWBs.")
    ap.add_argument("--root", required=True, help="folder to scan recursively for session NWBs.")
    ap.add_argument("--config", required=False, default=None,
                    help="Unused; still accepted so old HM_Tracker_2025 step-[s] invocations keep working.")
    ap.add_argument("--bin_cm", type=float, default=2.5,
                    help="rate-map bin size in cm (default: 2.5).")
    ap.add_argument("--smooth_cm", type=float, default=5.0,
                    help="rate-map Gaussian smoothing sigma in CM (default: 5). "
                         "Converted to bins as smooth_cm/bin_cm, so the physical "
                         "kernel does not change when --bin_cm does.")
    ap.add_argument("--speed", type=float, default=0.02,
                    help="speed threshold in m/s, applied to the 400 ms-smoothed "
                         "per-trial trace (default: 0.02). On the raw integer-pixel "
                         "trace anything below ~0.20 m/s is a no-op.")
    ap.add_argument("--min_occ_s", type=float, default=PF.DEFAULT_MIN_OCC_S,
                    help="pooled seconds of real time a bin needs to be valid "
                         "(default: 0.30).")
    ap.add_argument("--field_frac", type=float, default=PF.DEFAULT_FIELD_FRAC,
                    help="a field is a connected region above this fraction of the "
                         "peak (default: 0.30, matching the stability analysis).")
    ap.add_argument("--min_field_cm", type=float, default=PF.DEFAULT_MIN_FIELD_CM,
                    help="minimum field LENGTH along the track in cm (default: 15). "
                         "Length, not area: the corridor is narrower than one bin, so "
                         "area scales with bin size (measured 1.8-2.2x between 5 and "
                         "2.5 cm bins) while length does not (within 6%%).")
    ap.add_argument("--min_peak_hz", type=float, default=PF.DEFAULT_MIN_PEAK_HZ,
                    help="minimum in-field peak rate in Hz (default: 0.5).")
    ap.add_argument("--min_spikes", type=int, default=100,
                    help="a unit needs this many in-epoch spikes to contribute "
                         "(default: 100); below it every metric is NaN.")
    ap.add_argument("--si_match_n", type=int, default=300,
                    help="spike count for the count-matched spatial information "
                         "(default: 300); 0 disables it.")
    ap.add_argument("--si_match_repeats", type=int, default=20,
                    help="draws averaged for the count-matched SI (default: 20).")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the count-matched SI subsampling (default: 0).")
    ap.add_argument("--smooth", type=float, default=None,
                    help="DEPRECATED sigma in BINS; converted to cm as "
                         "smooth*bin_cm so old invocations keep their physical "
                         "smoothing. Use --smooth_cm.")
    args = ap.parse_args(argv)
    smooth_cm = args.smooth_cm
    if args.smooth is not None:
        smooth_cm = args.smooth * args.bin_cm
        print(f"[session-summary] --smooth is a sigma in BINS and changes meaning "
              f"with --bin_cm; using --smooth_cm {smooth_cm:g} instead.")
    try:
        run(args.root, bin_cm=args.bin_cm, smooth_cm=smooth_cm, speed=args.speed,
            field_frac=args.field_frac, min_field_cm=args.min_field_cm,
            min_occ_s=args.min_occ_s, min_peak_hz=args.min_peak_hz,
            min_spikes=args.min_spikes, si_match_n=args.si_match_n or None,
            si_match_repeats=args.si_match_repeats, seed=args.seed)
    except Exception as e:
        print(f"[session-summary] Failed: {e}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
