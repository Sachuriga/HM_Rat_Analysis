"""
NWB reading + place-field metrics for the cross-session summary.

Extracted from HM_Tracker_2025 ``src/nwb/visualize_nwb.py`` — only the parts
``session_summary.py`` actually needs (position/trial loading, rate maps, place
fields). The tracker keeps the full ``visualize_nwb.py`` for its per-session
step [v] PDFs; this module is the analysis-side subset, so the summary no longer
needs the tracker repo on ``sys.path``.

The maths is unchanged, so metrics computed here match the tracker's. If a
rate-map or place-field convention is ever changed in one repo, change it in the
other too.
"""

from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label

# --- spatial frame (plot_trials.py convention) ---
# Pixel -> metre scaling and the fixed maze frame, copied from plot_trials.py so
# rate maps share its coordinate system. Positions divided by these give metres;
# the maze then lives in the fixed MAZE_EXTENT box (metres), identical for every
# session/trial so all spatial panels are directly comparable.
SCALE_X = 2352 / 2 / 9
SCALE_Y = 1424 / 2 / 5
MAZE_EXTENT = (0.0, 9.0, 0.0, 5.0)


# ------------------------------------------------------------
#                 locating inputs
# ------------------------------------------------------------
def find_nwb_file(output_folder):
    op = Path(output_folder)
    # skip *.tmp.nwb and macOS AppleDouble sidecars ("._*.nwb", not valid HDF5)
    _ok = lambda p: not p.name.endswith(".tmp.nwb") and not p.name.startswith("._")
    cands = [p for p in sorted(op.glob("*.nwb")) if _ok(p)]
    if cands:
        return cands[0]
    cands = [p for p in sorted(op.glob("**/*.nwb")) if _ok(p)]
    return cands[0] if cands else None


# ------------------------------------------------------------
#                 loading NWB content
# ------------------------------------------------------------
def load_position(nwb):
    """Return (x, y, t) for the animal in session-relative seconds, or None."""
    try:
        pos = nwb.processing["Behavior"]["Position"]
    except Exception:
        return None
    ss = pos.spatial_series
    # Prefer a series literally named 'Rat'; else the first one.
    key = "Rat" if "Rat" in ss else next(iter(ss), None)
    if key is None:
        return None
    s = ss[key]
    xy = np.asarray(s.data[:], dtype=float)
    t = np.asarray(s.timestamps[:], dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    order = np.argsort(t)
    return x[order], y[order], t[order]


def _pick_file(op, pat):
    """First op-folder file matching `pat`, skipping macOS AppleDouble sidecars
    ("._*") that glob would otherwise return first."""
    return next((p for p in sorted(Path(op).glob(pat))
                 if not p.name.startswith("._")), None)


def _read_csv_tol(p):
    """read_csv tolerant of the non-UTF-8 bytes (e.g. a degree sign) that some
    framewise CSVs carry in their headers."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return pd.read_csv(p, encoding="utf-8", encoding_errors="ignore")


def _first_int(v):
    """First integer in a scalar or comma-list (Start_Nodes may be '224' or
    '224,315'); None if unparseable."""
    try:
        return int(str(v).split(",")[0])
    except (TypeError, ValueError):
        return None


def read_trial_meta(op_folder):
    """Per-trial (type, goal_node, start_node) from RecordingMeta.xlsx, keyed by
    1-based trial number (row order == Trial_Num in the coordinate file). Rows are
    NOT skipped, so numbering stays aligned with the position data's Trial_Num."""
    meta = _pick_file(op_folder, "RecordingMeta.xlsx") or _pick_file(op_folder, "*RecordingMeta.xlsx")
    if meta is None:
        return {}
    try:
        df = pd.read_excel(meta, sheet_name=0)
    except Exception as e:
        print(f"  Could not read RecordingMeta.xlsx ({e}).")
        return {}
    out = {}
    for i, r in enumerate(df.itertuples(index=False), start=1):
        d = r._asdict()
        try:
            ttype = int(d.get("Trial_Type"))
        except (TypeError, ValueError):
            ttype = -1
        goal = int(d["Goal_Node"]) if pd.notna(d.get("Goal_Node")) else None
        start = _first_int(d["Start_Nodes"]) if pd.notna(d.get("Start_Nodes")) else None
        out[i] = (ttype, goal, start)
    return out


def read_trials_raw(op_folder):
    """Per-trial (type, goal_node, start_node, start_unix, end_unix) from
    RecordingMeta.xlsx, in the raw behavioural-sync unix clock. Fallback only —
    that clock is offset (and drifts) relative to the video/position clock, so it
    is not used when the coordinate Trial_Num blocks are available. [] if absent."""
    meta = _pick_file(op_folder, "RecordingMeta.xlsx") or _pick_file(op_folder, "*RecordingMeta.xlsx")
    if meta is None:
        return []
    try:
        df = pd.read_excel(meta, sheet_name=0)
    except Exception as e:
        print(f"  Could not read RecordingMeta.xlsx ({e}).")
        return []
    need = {"Trial_Type", "trial_start_time", "trial_end_time"}
    if not need <= set(df.columns):
        print("  RecordingMeta.xlsx missing Trial_Type/trial_start_time/trial_end_time.")
        return []
    trials = []
    for _, r in df.iterrows():
        if pd.isna(r["trial_start_time"]) or pd.isna(r["trial_end_time"]):
            continue
        try:
            ttype = int(r["Trial_Type"])
        except (TypeError, ValueError):
            ttype = -1
        goal = int(r["Goal_Node"]) if "Goal_Node" in df.columns and pd.notna(r["Goal_Node"]) else None
        start = _first_int(r["Start_Nodes"]) if "Start_Nodes" in df.columns and pd.notna(r["Start_Nodes"]) else None
        trials.append((ttype, goal, start,
                       float(r["trial_start_time"]), float(r["trial_end_time"])))
    return trials


def frame_to_seconds(op_folder):
    """{frame_number: seconds} from stitched_framewise_seconds.csv
    ('Seconds From Creation'), which IS the clock the NWB position/spike
    timestamps use. None if unavailable."""
    st = _pick_file(op_folder, "*stitched_framewise_seconds.csv")
    if st is None:
        return None
    try:
        df = _read_csv_tol(st)
    except Exception as e:
        print(f"  Could not read stitched_framewise_seconds.csv ({e}).")
        return None
    fcol = next((c for c in df.columns if "frame" in c.lower()), None)
    scol = next((c for c in df.columns if "second" in c.lower()), None)
    if fcol is None or scol is None:
        return None
    frames = pd.to_numeric(df[fcol], errors="coerce")
    secs = pd.to_numeric(df[scol], errors="coerce")
    ok = frames.notna() & secs.notna()
    return dict(zip(frames[ok].astype(int), secs[ok].astype(float)))


def build_trials(op_folder, nwb_start, t_min, t_max):
    """Per-trial (type, goal_node, start_node, t0, t1) on the NWB seconds clock.

    Trial windows come from the coordinate file's Trial_Num blocks mapped to
    seconds via stitched_framewise_seconds.csv — the SAME source plot_trials.py
    uses — so the summary's per-trial values align with the positions/spikes and
    with plot_trials. (RecordingMeta's trial_start/end_time live on the
    behavioural sync clock, which is offset from and drifts against the video
    clock, so they misplace every trial; they are only a last-resort fallback.)
    Metadata (type/goal/start) is joined from RecordingMeta by Trial_Num.
    Falls back to align_trials(read_trials_raw(...)) if the coordinate/seconds
    files are missing."""
    coords = (_pick_file(op_folder, "*Coordinates_Full_with_frames.csv")
              or _pick_file(op_folder, "*Coordinates_Full.csv"))
    f2s = frame_to_seconds(op_folder)
    meta = read_trial_meta(op_folder)
    if coords is not None and f2s is not None:
        try:
            cf = _read_csv_tol(coords)
        except Exception as e:
            print(f"  Could not read {coords.name} ({e}); falling back to RecordingMeta times.")
            cf = None
        if cf is not None and {"Trial_Num", "Frame_Index"} <= set(cf.columns):
            sec = pd.to_numeric(cf["Frame_Index"], errors="coerce").map(f2s)
            tnum = pd.to_numeric(cf["Trial_Num"], errors="coerce")
            trials = []
            for k, grp in sec.groupby(tnum):
                g = grp.dropna()
                if g.empty:
                    continue
                tt, goal, start = meta.get(int(k), (-1, None, None))
                trials.append((tt, goal, start, float(g.min()), float(g.max())))
            if trials:
                trials.sort(key=lambda r: r[3])
                return trials
            print("  No Trial_Num blocks resolved to seconds; falling back to RecordingMeta times.")
    return align_trials(read_trials_raw(op_folder), nwb_start, t_min, t_max)


def align_trials(raw_trials, nwb_start, t_min, t_max):
    """Fallback: convert raw-unix RecordingMeta trials to session-relative seconds
    when the coordinate Trial_Num blocks are unavailable. Recovers the relative
    zero from session_start_time, trying both the aware and tz-replaced readings
    and keeping whichever lands the most trials inside [t_min, t_max].
    Returns [(type, goal_node, start_node, t0_rel, t1_rel), ...]."""
    if not raw_trials:
        return []
    cands = []
    try:
        cands.append(nwb_start.timestamp())                                  # fixed NWBs
    except Exception:
        pass
    try:
        cands.append(nwb_start.replace(tzinfo=timezone.utc).timestamp())     # legacy NWBs
    except Exception:
        pass
    if not cands:
        return []
    in_range = lambda off: sum(1 for (_tt, _g, _s, s, e) in raw_trials if t_min <= s - off <= t_max)
    off = max(cands, key=in_range)
    return [(tt, g, sn, s - off, e - off) for (tt, g, sn, s, e) in raw_trials]


def load_nodes():
    """{node_id: (x_m, y_m)} maze-node coordinates (metres, plot_trials frame),
    from node_list_new.csv next to this file. Used to locate the goal node."""
    p = Path(__file__).resolve().parent / "node_list_new.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, header=None, names=["id", "x", "y"])
        return {int(r.id): (r.x / SCALE_X, r.y / SCALE_Y) for r in df.itertuples()}
    except Exception:
        return {}


# ------------------------------------------------------------
#                 rate maps and place fields
# ------------------------------------------------------------
def make_rate_map(x, y, t, spike_times, extent, bins, dt, sigma,
                  t0=None, t1=None, speed_thresh=0.0, return_occ=False):
    """Firing-rate (place-field) map, Hz: binned spike counts / occupancy, EXACTLY
    the plot_trials.py occupancy convention (count * dt seconds per bin), and
    masked to bins the animal actually visited (occupancy > 0) so nothing is drawn
    off the maze path. Optional speed gating (only samples/spikes with speed >
    speed_thresh, in position-units/s) mirrors plot_trials' "Speed > N" maps.

    x,y,t   : animal position (session-relative seconds), t ascending
    extent  : (xmin, xmax, ymin, ymax) in position (pixel) units
    bins    : (nx, ny)
    returns : (rate 2D [ny, nx] masked to visited bins, extent) or (None, extent)
    """
    xmin, xmax, ymin, ymax = extent
    nx, ny = bins
    rng = [[xmin, xmax], [ymin, ymax]]

    if t0 is not None:
        m = (t >= t0) & (t <= t1)
        x, y, t = x[m], y[m], t[m]
        spike_times = spike_times[(spike_times >= t0) & (spike_times <= t1)]
    good = np.isfinite(x) & np.isfinite(y)
    x, y, t = x[good], y[good], t[good]
    if x.size < 2:
        return (None, None, extent) if return_occ else (None, extent)

    # speed (position units / s), aligned to each position sample
    speed = np.zeros_like(x)
    if x.size > 1:
        d = np.hypot(np.diff(x), np.diff(y))
        dts = np.diff(t)
        speed[1:] = d / np.where(dts > 0, dts, np.inf)
    move = speed > speed_thresh if speed_thresh > 0 else np.ones_like(x, dtype=bool)

    # occupancy (seconds per bin) from moving samples — plot_trials convention
    occ, _, _ = np.histogram2d(x[move], y[move], bins=[nx, ny], range=rng)
    occ = occ.T * dt
    occ_raw = occ.copy()   # unsmoothed seconds/bin, for spatial-information p(bin)

    # spike positions (interpolated onto the trajectory), speed-gated the same way
    if spike_times.size:
        sx = np.interp(spike_times, t, x, left=np.nan, right=np.nan)
        sy = np.interp(spike_times, t, y, left=np.nan, right=np.nan)
        sv = np.interp(spike_times, t, speed, left=0.0, right=0.0)
        ok = np.isfinite(sx) & np.isfinite(sy)
        if speed_thresh > 0:
            ok &= sv > speed_thresh
        spk, _, _ = np.histogram2d(sx[ok], sy[ok], bins=[nx, ny], range=rng)
        spk = spk.T
    else:
        spk = np.zeros_like(occ)

    visited = occ > 0            # bins the animal actually entered
    if sigma and sigma > 0:      # optional light smoothing, kept ON the path only
        occ = gaussian_filter(occ, sigma)
        spk = gaussian_filter(spk, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(occ > 0, spk / occ, 0.0)
    rate = np.ma.masked_where(~visited, rate)   # draw ONLY on visited bins
    if return_occ:
        return rate, np.ma.masked_where(~visited, occ_raw), extent
    return rate, extent


def place_field_mask(rate, frac=0.5):
    """Boolean mask of a cell's main place field: the connected region of bins
    >= frac*peak that contains the peak bin. None if no field."""
    if rate is None or not rate.count():
        return None
    peak = float(np.ma.max(rate))
    if peak <= 0:
        return None
    binary = rate.filled(0) >= frac * peak
    lab, n = label(binary)
    if n == 0:
        return None
    iy, ix = np.unravel_index(np.ma.argmax(rate), rate.shape)
    pk = lab[iy, ix]
    if pk == 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        pk = int(np.argmax(sizes))
    return lab == pk


def place_fields(rate, field_frac=0.5, min_peak_hz=0.5, min_field_bins=6):
    """List of boolean masks, one per place field: connected regions >= field_frac
    * peak with >= min_field_bins bins and an in-field peak >= min_peak_hz. A cell
    can have several fields (large maze)."""
    if rate is None or not rate.count():
        return []
    lam = rate.filled(0.0)
    peak = float(lam.max())
    if peak <= 0:
        return []
    labmap, ncc = label((lam >= field_frac * peak) & ~np.ma.getmaskarray(rate))
    fields = []
    for c in range(1, ncc + 1):
        comp = labmap == c
        if comp.sum() >= min_field_bins and lam[comp].max() >= min_peak_hz:
            fields.append(comp)
    return fields


def place_field_metrics(x, y, t, spike_times, extent, bins, dt, sigma, speed_thresh,
                        goal_xy=None, t0=None, t1=None,
                        field_frac=0.5, min_peak_hz=0.5, min_field_bins=6):
    """Place-coding metrics for one cell over a window (defaults: a place field is
    a connected region >= 50% of the peak, >= 6 bins ~ a 15 cm field at 5 cm bins,
    with an in-field peak >= 0.5 Hz; a cell CAN have several fields on this large
    maze):
      n_fields      : # place fields = connected regions >= field_frac*peak with
                      >= min_field_bins bins and an in-field peak >= min_peak_hz
      spatial_info  : Skaggs spatial information (bits/spike)
      selectivity   : peak rate / mean rate
      field_goal_m  : mean distance (m) from each field's centroid to the goal node
    Returns (metrics_dict, rate, extent)."""
    rate, occ, ext = make_rate_map(x, y, t, spike_times, extent, bins, dt, sigma,
                                   t0=t0, t1=t1, speed_thresh=speed_thresh, return_occ=True)
    nan = {"n_fields": 0, "spatial_info": np.nan, "selectivity": np.nan,
           "field_goal_m": np.nan, "field_goal_largest_m": np.nan,
           "field_goal_2ndlargest_m": np.nan, "field_goal_smallest_m": np.nan,
           "peak": 0.0}
    if rate is None or not rate.count():
        return nan, rate, ext
    lam = rate.filled(0.0)
    p_occ = occ.filled(0.0)
    tot = p_occ.sum()
    if tot <= 0:
        return nan, rate, ext
    p = p_occ / tot
    lam_mean = float((p * lam).sum())
    peak = float(lam.max())
    if lam_mean <= 0:
        return {**nan, "peak": peak}, rate, ext
    # Skaggs spatial information (bits/spike)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lam / lam_mean
        terms = np.where(lam > 0, p * ratio * np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
    spatial_info = float(np.nansum(terms))
    selectivity = peak / lam_mean
    # all place fields + centroids (a cell can have several on this large maze)
    fields = place_fields(rate, field_frac, min_peak_hz, min_field_bins)
    ny, nx = rate.shape
    dists, sizes = [], []
    n_fields = len(fields)
    for comp in fields:
        iy, ix = np.where(comp)
        cx = ext[0] + (ix.mean() + 0.5) * (ext[1] - ext[0]) / nx
        cy = ext[2] + (iy.mean() + 0.5) * (ext[3] - ext[2]) / ny
        sizes.append(int(comp.sum()))
        dists.append(float(np.hypot(cx - goal_xy[0], cy - goal_xy[1]))
                     if goal_xy is not None else np.nan)
    # per-field distances ranked by field size (bins): largest / 2nd-largest / smallest
    largest = second = smallest = np.nan
    if dists and goal_xy is not None:
        order = np.argsort(sizes)[::-1]          # descending by size
        largest = dists[order[0]]
        second = dists[order[1]] if len(order) >= 2 else np.nan
        smallest = dists[order[-1]]              # smallest field
    return ({"n_fields": n_fields, "spatial_info": spatial_info,
             "selectivity": selectivity,
             "field_goal_m": float(np.nanmean(dists)) if np.any(np.isfinite(dists)) else np.nan,
             "field_goal_largest_m": largest,
             "field_goal_2ndlargest_m": second,
             "field_goal_smallest_m": smallest,
             "peak": peak}, rate, ext)
