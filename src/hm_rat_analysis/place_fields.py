"""Rate maps and place-field metrics.

Extracted from HM_Tracker_2025 ``src/nwb/visualize_nwb.py``. The maths is
unchanged, so metrics computed here match the tracker's per-session step [v]
PDFs — if a convention changes in one repo it must change in the other.

Coordinates are metres throughout (see :mod:`hm_rat_analysis.maze`).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label


def make_rate_map(x, y, t, spike_times, extent, bins, dt, sigma,
                  t0=None, t1=None, speed_thresh=0.0, return_occ=False):
    """Firing-rate (place-field) map, Hz: binned spike counts / occupancy, EXACTLY
    the plot_trials.py occupancy convention (count * dt seconds per bin), and
    masked to bins the animal actually visited (occupancy > 0) so nothing is drawn
    off the maze path. Optional speed gating (only samples/spikes with speed >
    speed_thresh, in position-units/s) mirrors plot_trials' "Speed > N" maps.

    x,y,t   : animal position (session-relative seconds), t ascending
    extent  : (xmin, xmax, ymin, ymax) in position units
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

    `goal_xy` must be a real maze node position — when it is None (or the goal node
    is not in the node table) every ``field_goal_*`` value comes back NaN rather
    than raising.

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
