"""Rate maps and place-field metrics.

Extracted from HM_Tracker_2025 ``src/nwb/visualize_nwb.py``. The maths is
unchanged, so metrics computed here match the tracker's per-session step [v]
PDFs — if a convention changes in one repo it must change in the other.

Coordinates are metres throughout (see :mod:`hm_rat_analysis.maze`).

Every threshold in this module is PHYSICAL — cm, cm^2, seconds, m/s — because the
maze is run at two bin sizes (5 cm and 2.5 cm) and a threshold stated in bins
silently changes meaning between them. The one exception is ``sigma``, which
scipy needs in bins; callers convert at the boundary (``sigma = smooth_cm /
bin_cm``) and the report never lets a default live in bin units.

Two properties of this dataset drive the defaults here:

* the rat runs a near-1-D line — the sampled surface is ~1.4 m^2 spread over 53 m
  of maze graph, an effective corridor width of only 2.6-2.9 cm — so a classic
  20-50 cm CA1 field is 60-150 cm^2 here, not the several-hundred cm^2 of an
  open-field disc, and a 2.5 cm corridor is barely one bin wide;
* Skaggs information in bits/spike is strongly biased upward at low spike counts
  (measured on this dataset: +161% going from 1200 to 300 spikes), and spikes per
  cell fall across learning as trials shorten. Anything derived from the spike
  count therefore carries a cross-session trend of its own; see
  :data:`METRIC_NOTES` and ``spatial_info_matched``.
"""

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, label

#: 8-connectivity. The default scipy structuring element is the 4-neighbour cross,
#: which severs a diagonal corridor: on this maze a bin-wide diagonal run is a
#: staircase whose bins touch only at their corners, so with 4-connectivity one
#: field shatters into a component per step (measured: 1 component at 5 cm becomes
#: 10 at 2.5 cm, and a straight 5 m diagonal run splits into 29). Connectivity is a
#: rule in bins, so it must be the permissive one or geometry decides n_fields.
_S8 = np.ones((3, 3), dtype=int)

DEFAULT_FIELD_FRAC = 0.30      #: field = connected region >= 30% of peak
DEFAULT_MIN_PEAK_HZ = 0.5      #: in-field peak floor
DEFAULT_MIN_FIELD_CM = 15.0    #: minimum field extent along the track, in cm
DEFAULT_MIN_FIELD_CM2 = 60.0   #: deprecated area spelling; kept for old call sites
DEFAULT_MIN_OCC_S = 0.30       #: pooled seconds of real time behind a valid bin

#: gaussian_filter returns a weighted AVERAGE (it is DC-preserving: the impulse
#: response peaks at 1/(2*pi*sigma^2)), so the amount of REAL time pooled into a
#: smoothed occupancy value is that average times the kernel's effective bin count,
#: 4*pi*sigma^2. Expressed in cm the pooled area is 4*pi*sigma_cm^2, which is why a
#: seconds threshold on it is bin-size independent only once sigma is fixed in cm.
POOLED_BINS_FACTOR = 4.0 * np.pi

#: Machine-readable notes on what each metric is confounded with. The plotting
#: layer reads these to annotate panels; the report writes them into the xlsx so a
#: reader of the spreadsheet alone can see which columns are not comparable across
#: sessions. Empty string = no known sampling confound.
METRIC_NOTES = {
    "spatial_info": "confounded_with_spike_count",
    "spatial_info_matched": "",
    "selectivity": "confounded_with_spike_count_and_coverage",
    "peak": "confounded_with_spike_count",
    "stability": "confounded_with_spike_count;each half holds ~half the spikes, "
                 "so a short session is penalised twice",
    "stability_n_bins": "exposure: bins valid in BOTH halves",
    "n_fields": "confounded_with_coverage",
    "field_goal_m": "confounded_with_coverage;compare against field_goal_null_m",
    "field_goal_null_m": "coverage_null: mean goal distance of this epoch's occupancy",
    "field_goal_over_null": "",
    "field_goal_largest_m": "confounded_with_coverage",
    "field_goal_2ndlargest_m": "confounded_with_coverage;defined_only_when_n_fields>=2",
    "field_goal_smallest_m": "confounded_with_coverage;smallest_field_is_noise",
    "field_size_largest_m2": "confounded_with_coverage;area is not bin-size "
                             "invariant on this maze — quote field_size_mean_cm",
    "field_size_mean_cm": "confounded_with_coverage",
    "field_size_largest_cm": "confounded_with_coverage",
    "n_spikes_epoch": "exposure",
    "epoch_dur_s": "exposure",
    "occ_total_s": "exposure",
    "n_valid_bins": "exposure",
}

#: Metrics that must not be read as a cross-session trend without matching spike
#: counts first (fact: SI inflates +161% from 1200 -> 300 spikes on this dataset).
SPIKE_COUNT_CONFOUNDED = ("spatial_info", "selectivity", "peak", "stability")
#: Metrics bounded by how much of the maze the animal covered that session.
COVERAGE_CONFOUNDED = ("n_fields", "field_goal_m", "field_goal_largest_m",
                       "field_goal_2ndlargest_m", "field_goal_smallest_m",
                       "field_size_largest_m2", "field_size_mean_cm",
                       "field_size_largest_cm")

#: Bins a split-half correlation needs before it is reported. Below this the
#: correlation is dominated by which few bins happened to survive in both halves.
DEFAULT_MIN_STABILITY_BINS = 20


def bin_size_cm(extent, bins):
    """Realised (bin_cm_x, bin_cm_y) for a grid — NOT the requested bin size.

    ``bins`` is an integer count per axis, so the bin the code actually uses is
    the extent divided by that count; at --bin_cm 3 over a 9 x 5 m maze that is
    3.000 cm in x and 2.994 cm in y. Every cm/cm^2 conversion must start here,
    never from the requested value.
    """
    nx, ny = bins
    return ((extent[1] - extent[0]) * 100.0 / nx,
            (extent[3] - extent[2]) * 100.0 / ny)


def _min_bins(min_field_cm2, bin_area_cm2, min_field_bins=None):
    """Minimum field size in BINS from a physical area (the bin count is derived,
    never configured). ``min_field_bins`` is the deprecated bin-count spelling and
    still wins when a caller passes it, so old call sites keep working."""
    if min_field_bins is not None:
        warnings.warn("min_field_bins is a bin count and changes meaning with bin "
                      "size; pass min_field_cm (with bin_cm) instead.",
                      DeprecationWarning, stacklevel=3)
        return max(1, int(min_field_bins))
    if not min_field_cm2 or bin_area_cm2 is None or bin_area_cm2 <= 0:
        return 1
    return max(1, int(np.ceil(float(min_field_cm2) / float(bin_area_cm2))))


def field_length_cm(comp, bin_cm):
    """Longest straight-line extent of a field, in cm: the greatest distance between
    any two of its bin centres.

    This is the size measure to threshold on, because AREA is not bin-size invariant
    on this maze. The rat runs a corridor narrower than one bin at either resolution
    (measured: 0.99-1.11 bins wide at both 2.5 cm and 5 cm), so a field is a bin-wide
    line and its area is length x bin width — it scales with the bin. Measured on
    three real sessions, the median field area is 1.8-2.2x larger at 5 cm than at
    2.5 cm for the same cells, while the median length agrees to within 6%.
    """
    iy, ix = np.where(comp)
    if len(iy) < 2:
        return 0.0
    bx, by = bin_cm
    pts = np.column_stack([ix * float(bx), iy * float(by)])
    if len(pts) > 2000:              # diameter is O(n^2); huge blobs pass anyway
        pts = pts[:: len(pts) // 2000 + 1]
    d = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def occupancy_parts(x, y, t, extent, bins, dt, sigma, t0=None, t1=None,
                    speed_thresh=0.0, speed=None, max_gap_s=None):
    """Everything about a window that does NOT depend on which cell you look at:
    the gated trajectory, the raw and smoothed occupancy, and the histogram
    geometry. None when the window holds no usable data.

    Split out from :func:`_rate_map_parts` so a caller scoring MANY cells over the
    SAME trials computes each trial's occupancy once instead of once per cell —
    with 33 trials and 84 cells that is 2772 redundant passes. The spike half is
    :func:`spike_parts`; composing the two reproduces the original exactly, which
    is the point: there is still one implementation of these conventions.
    """
    xmin, xmax, ymin, ymax = extent
    nx, ny = bins
    rng = [[xmin, xmax], [ymin, ymax]]
    x = np.asarray(x, float); y = np.asarray(y, float); t = np.asarray(t, float)
    sp = None if speed is None else np.asarray(speed, float)

    if t0 is not None:
        m = (t >= t0) & (t <= t1)
        x, y, t = x[m], y[m], t[m]
        if sp is not None:
            sp = sp[m]
    good = np.isfinite(x) & np.isfinite(y)
    if sp is not None:
        good &= np.isfinite(sp)
    x, y, t = x[good], y[good], t[good]
    if sp is not None:
        sp = sp[good]
    if x.size < 2:
        return None

    # Speed, aligned to each position sample. The caller should hand in a speed
    # computed from a 400 ms-smoothed, PER-TRIAL trace: positions are integer
    # pixels (7.65 mm in x, 7.02 mm in y), so a difference of the raw trace can
    # only be 0 or >= ~0.20 m/s and any threshold below that is a duplicate-frame
    # filter, not a speed gate. The internal fallback keeps old callers working.
    if sp is None:
        sp = np.zeros_like(x)
        d = np.hypot(np.diff(x), np.diff(y))
        dts = np.diff(t)
        sp[1:] = d / np.where(dts > 0, dts, np.inf)
        sp[0] = sp[1]      # else the first sample of every window is always dropped
    move = sp > speed_thresh if speed_thresh > 0 else np.ones_like(x, dtype=bool)

    # occupancy (seconds per bin) from moving samples — plot_trials convention
    occ_raw, _, _ = np.histogram2d(x[move], y[move], bins=[nx, ny], range=rng)
    occ_raw = occ_raw.T * dt
    if max_gap_s is None:
        max_gap_s = max(0.5, 10.0 * float(dt))
    occ_s = gaussian_filter(occ_raw, sigma) if (sigma and sigma > 0) else occ_raw
    return {"x": x, "y": y, "t": t, "move": move, "occ_raw": occ_raw, "occ_s": occ_s,
            "sigma": sigma, "dt": float(dt), "max_gap_s": float(max_gap_s),
            "t0": t0, "t1": t1,
            "dur_s": float(move.sum()) * float(dt),
            "hist_kw": {"bins": [nx, ny], "range": rng}}


def spike_parts(occ, spike_times):
    """One cell's smoothed spike map over a window, given that window's
    :func:`occupancy_parts`. Returns (smoothed spike map, gated spike x, y)."""
    x, y, t, move = occ["x"], occ["y"], occ["t"], occ["move"]
    sigma, max_gap_s = occ["sigma"], occ["max_gap_s"]
    spike_times = np.asarray(spike_times, float)
    if occ["t0"] is not None:
        spike_times = spike_times[(spike_times >= occ["t0"]) & (spike_times <= occ["t1"])]

    # Spike positions. np.interp only returns NaN OUTSIDE [t[0], t[-1]]: an
    # interior gap — which is exactly what removing the inter-trial intervals
    # leaves behind — is filled with a straight line across the maze, so a spike
    # emitted while the rat is off the maze would be planted on a trajectory it
    # never ran. Drop any spike whose bracketing samples are further apart than
    # max_gap_s, and gate on the move flag of the NEAREST REAL FRAME rather than
    # on an interpolated speed (interpolating the speed keeps a different fraction
    # of the spikes than of the samples: measured 68.5% of spikes vs 43.5% of
    # frames at a 0.05 m/s gate, i.e. rate inflated 1.6x in the slowest bins).
    if spike_times.size:
        sx = np.interp(spike_times, t, x, left=np.nan, right=np.nan)
        sy = np.interp(spike_times, t, y, left=np.nan, right=np.nan)
        j = np.clip(np.searchsorted(t, spike_times), 1, t.size - 1)
        near = np.where(np.abs(t[j] - spike_times) <= np.abs(spike_times - t[j - 1]),
                        j, j - 1)
        ok = (np.isfinite(sx) & np.isfinite(sy)
              & ((t[j] - t[j - 1]) <= max_gap_s) & move[near])
        sx, sy = sx[ok], sy[ok]
        spk, _, _ = np.histogram2d(sx, sy, **occ["hist_kw"])
        spk = spk.T
    else:
        sx = sy = np.array([])
        spk = np.zeros_like(occ["occ_raw"])
    spk_s = gaussian_filter(spk, sigma) if (sigma and sigma > 0) else spk
    return spk_s, sx, sy


def valid_bins(occ_raw, occ_s, sigma, min_occ_s):
    """Which bins carry a usable rate estimate.
    A bin is valid iff the animal actually entered it AND the smoothed estimate
    pools enough real time. Judging validity on raw occupancy alone (one 33 ms
    sample) while reading the rate off a smoothed map is inconsistent: at 2.5 cm
    it punches single-bin holes inside corridors that shatter a field into
    specks, and it lets a bin whose smoothed denominator is essentially kernel
    tail become the map peak — which then sets the field threshold and the
    selectivity numerator. A raw seconds-per-bin floor is NOT a substitute: on a
    1-D track it scales with bin_cm (one pass deposits 0.167 s in a 5 cm bin but
    0.083 s in a 2.5 cm bin), so it would change meaning between the two runs.
    """
    pooled = (occ_s * POOLED_BINS_FACTOR * float(sigma) ** 2
              if (sigma and sigma > 0) else occ_raw)
    return (occ_raw > 0) & (pooled >= float(min_occ_s))


def rate_from_maps(spk_s, occ_s, visited):
    """Rate map (Hz) masked to the valid bins, from a smoothed spike and occupancy
    map. Additive in the spike and occupancy maps, which is what lets a caller
    build a leave-one-out reference by subtraction instead of a rebuild."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(occ_s > 0, spk_s / occ_s, 0.0)
    return np.ma.masked_where(~visited, rate)


def _rate_map_parts(x, y, t, spike_times, extent, bins, dt, sigma,
                    t0=None, t1=None, speed_thresh=0.0, speed=None,
                    min_occ_s=0.0, max_gap_s=None):
    """Everything both public entry points need from one pass over the data:
    the masked rate map, the raw and smoothed occupancies, the gated spike
    positions (so spikes can be re-histogrammed for the count-matched estimate)
    and the exposure counters. Returns None when the window holds no usable data.
    """
    occ = occupancy_parts(x, y, t, extent, bins, dt, sigma, t0=t0, t1=t1,
                          speed_thresh=speed_thresh, speed=speed,
                          max_gap_s=max_gap_s)
    if occ is None:
        return None
    spk_s, sx, sy = spike_parts(occ, spike_times)
    visited = valid_bins(occ["occ_raw"], occ["occ_s"], sigma, min_occ_s)
    return {"rate": rate_from_maps(spk_s, occ["occ_s"], visited),
            "occ_raw": occ["occ_raw"], "occ_s": occ["occ_s"], "visited": visited,
            "spike_xy": (sx, sy), "n_spikes": int(sx.size),
            "dur_s": occ["dur_s"], "hist_kw": occ["hist_kw"]}


def make_rate_map(x, y, t, spike_times, extent, bins, dt, sigma,
                  t0=None, t1=None, speed_thresh=0.0, return_occ=False,
                  speed=None, min_occ_s=0.0, max_gap_s=None):
    """Firing-rate (place-field) map, Hz: binned spike counts / occupancy, EXACTLY
    the plot_trials.py occupancy convention (count * dt seconds per bin), masked to
    the bins that are VALID: entered by the animal (raw occupancy > 0) and backed
    by at least ``min_occ_s`` seconds of pooled time (see :data:`POOLED_BINS_FACTOR`).

    x,y,t        : animal position (session-relative seconds), t ascending
    extent       : (xmin, xmax, ymin, ymax) in position units
    bins         : (nx, ny)
    sigma        : Gaussian smoothing in BINS — callers convert from cm
                   (``sigma = smooth_cm / bin_cm``) so the physical kernel is
                   fixed; sigma in bins halves the kernel when the bins halve.
    speed        : optional precomputed speed per sample (m/s). Pass one computed
                   from a 400 ms-smoothed, per-trial trace; the raw integer-pixel
                   trace cannot express a speed below ~0.20 m/s, so a gate below
                   that selects exactly the same samples as no gate at all.
    speed_thresh : m/s; 0 disables gating entirely (keeps stationary samples).
    min_occ_s    : pooled seconds of real time required behind a valid bin. 0
                   reproduces the historical "one position sample is enough" rule.
    returns      : (rate 2D [ny, nx] masked to valid bins, extent) or
                   (rate, masked raw occupancy, extent) when return_occ.
    """
    parts = _rate_map_parts(x, y, t, spike_times, extent, bins, dt, sigma,
                            t0=t0, t1=t1, speed_thresh=speed_thresh, speed=speed,
                            min_occ_s=min_occ_s, max_gap_s=max_gap_s)
    if parts is None:
        return (None, None, extent) if return_occ else (None, extent)
    if return_occ:
        # the occupancy carries the SAME mask as the rate, so the Skaggs prior and
        # the rate are always defined on one bin set
        return (parts["rate"],
                np.ma.masked_where(~parts["visited"], parts["occ_raw"]), extent)
    return parts["rate"], extent


def place_field_mask(rate, frac=DEFAULT_FIELD_FRAC, min_field_cm=0.0, bin_cm=None,
                     min_field_cm2=None, bin_area_cm2=None):
    """Boolean mask of a cell's main place field: the connected region of valid
    bins >= frac*peak that contains the peak bin. None if no field.

    Uses the same connectivity and the same physical size floor as
    :func:`place_fields`, so there is exactly one field definition in the package.
    """
    if rate is None or not rate.count():
        return None
    peak = float(np.ma.max(rate))
    if peak <= 0:
        return None
    fields = place_fields(rate, field_frac=frac, min_peak_hz=0.0,
                          min_field_cm=min_field_cm, bin_cm=bin_cm,
                          min_field_cm2=min_field_cm2, bin_area_cm2=bin_area_cm2)
    if not fields:
        return None
    iy, ix = np.unravel_index(np.ma.argmax(rate), rate.shape)
    for comp in fields:
        if comp[iy, ix]:
            return comp
    return max(fields, key=lambda c: c.sum())     # peak fell below the size floor


def place_fields(rate, field_frac=DEFAULT_FIELD_FRAC, min_peak_hz=DEFAULT_MIN_PEAK_HZ,
                 min_field_cm=DEFAULT_MIN_FIELD_CM, bin_cm=None,
                 min_field_cm2=None, bin_area_cm2=None, min_field_bins=None):
    """List of boolean masks, one per place field: 8-connected regions of VALID
    bins >= ``field_frac`` * peak, spanning at least ``min_field_cm`` of track
    (default 15 cm — below the 20-50 cm of a classic CA1 field, so the floor
    excludes specks without cutting into the distribution), with an in-field peak
    >= ``min_peak_hz``. A cell can have several fields on this large maze.

    The size floor is a LENGTH, not an area — see :func:`field_length_cm` for the
    measurement showing area doubles between 5 cm and 2.5 cm bins while length does
    not. ``bin_cm`` is the (x, y) bin size from :func:`bin_size_cm`. The deprecated
    ``min_field_cm2`` (area) and ``min_field_bins`` (bin count) still win when
    passed, so unmigrated call sites keep their old behaviour.

    ``field_frac`` is 0.30, matching the stability analysis and figures/msca_fig1a:
    at 2.5 cm bins with a 5 cm kernel the half-max contour on a bin-wide corridor
    is only a few bins long, so 0.5 splits one field in two and the connectivity,
    not the cell, decides n_fields. The cost is that 0.30 merges adjacent fields on
    the same corridor, so n_fields is a LOWER bound.
    """
    if rate is None or not rate.count():
        return []
    lam = rate.filled(0.0)
    peak = float(lam.max())
    if peak <= 0:
        return []
    legacy = min_field_bins is not None or min_field_cm2 is not None
    min_bins = _min_bins(min_field_cm2, bin_area_cm2, min_field_bins) if legacy else 1
    if min_field_cm2 is not None and min_field_bins is None:
        warnings.warn("min_field_cm2 is an area and is not bin-size invariant on a "
                      "corridor narrower than one bin; pass min_field_cm instead.",
                      DeprecationWarning, stacklevel=2)
    labmap, ncc = label((lam >= field_frac * peak) & ~np.ma.getmaskarray(rate),
                        structure=_S8)
    fields = []
    for c in range(1, ncc + 1):
        comp = labmap == c
        if lam[comp].max() < min_peak_hz:
            continue
        if legacy:
            if comp.sum() >= min_bins:
                fields.append(comp)
        elif bin_cm is None:
            fields.append(comp)          # no geometry supplied: size cannot be judged
        elif field_length_cm(comp, bin_cm) >= float(min_field_cm):
            fields.append(comp)
    return fields


def _half_masks(t, windows=None, t0=None, t1=None):
    """(A, B) boolean masks splitting position samples into two halves.

    With trial `windows` in hand the split is ALTERNATE TRIALS — trial 1, 3, 5 ...
    against 2, 4, 6 ... — which is the split a place-field stability number is
    normally quoted from. A first-half/second-half cut is not equivalent on this
    task: the animal's route collapses onto the direct start->goal path within a
    session, so the two halves would differ in coverage as well as in the cells,
    and a drop in r could not be attributed to either. Without windows there is
    nothing to alternate over and the epoch is cut at its midpoint in time, which
    is why the fallback is reported (`stability_split`) rather than silently used.

    Returns (mask_a, mask_b, split_name).
    """
    t = np.asarray(t, float)
    inside = np.ones(t.shape, dtype=bool)
    if t0 is not None:
        inside &= t >= t0
    if t1 is not None:
        inside &= t <= t1
    if windows:
        a = np.zeros(t.shape, dtype=bool)
        b = np.zeros(t.shape, dtype=bool)
        for i, (w0, w1) in enumerate(windows):
            m = (t >= w0) & (t <= w1)
            (a if i % 2 == 0 else b)[m] = True
        if a.any() and b.any():
            return a & inside, b & inside, "alternate_trials"
    ti = t[inside]
    if ti.size < 4:
        return np.zeros(t.shape, bool), np.zeros(t.shape, bool), "none"
    mid = float(np.median(ti))
    return inside & (t <= mid), inside & (t > mid), "time_halves"


def split_half_stability(x, y, t, spike_times, extent, bins, dt, sigma,
                         windows=None, t0=None, t1=None, speed=None,
                         speed_thresh=0.0, min_occ_s=0.0, max_gap_s=None,
                         min_bins=DEFAULT_MIN_STABILITY_BINS):
    """Split-half spatial stability: Pearson r between two half-session rate maps.

    The halves are alternate trials when `windows` is given (see :func:`_half_masks`),
    and r is taken over the bins that are VALID IN BOTH — a bin the animal only
    entered in one half carries no information about whether the map repeated.

    Two properties make the raw number hard to compare across sessions and both
    are reported alongside it rather than hidden: each half holds about half the
    spikes, so r inherits the same low-count bias as every other rate-derived
    measure (twice over); and the co-valid bin count falls as coverage narrows,
    which both thins the estimate and shrinks the range of positions r is computed
    over. Below `min_bins` co-valid bins the value is NaN, not a noisy number.

    Returns ``(r, n_covalid_bins, split_name)``.
    """
    a, b, split = _half_masks(t, windows=windows, t0=t0, t1=t1)
    if split == "none" or a.sum() < 2 or b.sum() < 2:
        return np.nan, 0, split

    x = np.asarray(x, float); y = np.asarray(y, float); t = np.asarray(t, float)
    st = np.asarray(spike_times, float)
    sp = None if speed is None else np.asarray(speed, float)

    maps = []
    for m in (a, b):
        tm = t[m]
        # Spikes follow their half. Masking out the other half's samples leaves
        # interior gaps in t, and _rate_map_parts already refuses to interpolate a
        # spike across one — so a spike from the OTHER half cannot be planted on a
        # fabricated chord here; it is simply dropped.
        parts = _rate_map_parts(x[m], y[m], tm, st[(st >= tm[0]) & (st <= tm[-1])],
                                extent, bins, dt, sigma,
                                speed_thresh=speed_thresh,
                                speed=None if sp is None else sp[m],
                                min_occ_s=min_occ_s, max_gap_s=max_gap_s)
        if parts is None:
            return np.nan, 0, split
        maps.append(parts)

    both = maps[0]["visited"] & maps[1]["visited"]
    n = int(both.sum())
    if n < int(min_bins):
        return np.nan, n, split
    va = maps[0]["rate"].filled(0.0)[both]
    vb = maps[1]["rate"].filled(0.0)[both]
    if va.std() <= 0 or vb.std() <= 0:
        return np.nan, n, split          # a silent half has no map to correlate
    return float(np.corrcoef(va, vb)[0, 1]), n, split


#: Co-visited bins a trial-to-reference divergence needs before it is reported.
DEFAULT_MIN_KL_BINS = 20
#: Trial types that are FREE ROAMING — no goal is set, so the animal's own
#: exploration decides the coverage. Used to build the goal-independent reference.
FREE_ROAM_TRIAL_TYPES = (4,)


def poisson_kl_bits(lam, eta, min_bins=DEFAULT_MIN_KL_BINS, symmetric=True,
                    normalise=False):
    """Divergence between two rate maps of the same cell, in BITS.

    Spike counts in a bin over a time step are Poisson, so the divergence between
    two rate maps is the Poisson KL per unit time,

        d(lam || eta) = sum_bins [ lam*ln(lam/eta) - (lam - eta) ] / ln 2

    which is the estimator used by Quattrocolo et al. for trial-to-trial map
    stability. Note the conversion: the bracket is derived in NATS and the whole
    expression is divided by ln 2 — taking log2 of the ratio while leaving the
    (lam - eta) term alone would mix units.

    Only bins VALID IN BOTH maps contribute, so a bin the animal entered in one
    map and not the other cannot register as a change in firing; that is an
    occupancy difference, not a map difference.

    Where the reference is exactly zero and the other map is not, the divergence
    diverges. That happens for a smoothed map only if the cell went completely
    silent there, which at these spike counts is usually a sampling fluctuation
    rather than true silence, so the reference is floored at its own smallest
    nonzero rate (the paper's regularisation).

    KL is asymmetric; `symmetric` averages d(lam||eta) and d(eta||lam), which is
    what makes the number a distance between two maps rather than a statement
    about which one is the reference.

    Returns ``(total_bits, bits_per_bin, n_bins)`` — the total to compare with the
    paper, and the per-bin mean because on this maze the co-visited support varies
    by an order of magnitude between trials and a sum would mostly track how far
    the animal ran.
    """
    if lam is None or eta is None:
        return np.nan, np.nan, 0
    both = ~(np.ma.getmaskarray(lam) | np.ma.getmaskarray(eta))
    n = int(both.sum())
    if n < int(min_bins):
        return np.nan, np.nan, n
    a = np.asarray(np.ma.getdata(lam))[both].astype(float)
    b = np.asarray(np.ma.getdata(eta))[both].astype(float)

    # The Poisson KL is NOT scale-invariant: multiply both maps by 2 and it
    # doubles. That is correct when the two maps are comparable stretches of
    # behaviour, as in the paper, but on this task a goal run is ~30 s and a
    # free-roaming bout ~600 s, so their mean RATES differ and the divergence
    # would rank them by how fast the cell was firing rather than by whether the
    # map moved. Normalising each map to unit mean first compares map SHAPE only.
    if normalise:
        ma, mb = a.mean(), b.mean()
        if not (ma > 0 and mb > 0):
            return np.nan, np.nan, n
        a, b = a / ma, b / mb

    def _d(p, q):
        if not np.any(q > 0):
            return np.nan            # a silent reference is no reference at all
        # The floor goes ONLY where it is needed — the reference is zero and the
        # other map is not. Flooring every zero bin would raise q above p in bins
        # where both are silent, and the -(p - q) term would then score agreement
        # as divergence: a map against itself came out at +0.0067 bits.
        need = (q <= 0) & (p > 0)
        q = np.where(need, q[q > 0].min(), q)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p / q), 0.0) - (p - q)
        return float(np.nansum(terms) / np.log(2.0))

    total = _d(a, b)
    if symmetric:
        other = _d(b, a)
        total = np.nan if not np.isfinite([total, other]).all() else (total + other) / 2
    return total, (total / n if np.isfinite(total) else np.nan), n


def trial_divergence(occs, spike_times, sigma, min_occ_s=DEFAULT_MIN_OCC_S,
                     reference="template", free_roam=None,
                     min_bins=DEFAULT_MIN_KL_BINS, symmetric=True,
                     normalise=False):
    """One cell's map divergence on every trial, against a reference map.

    `occs` is that session's per-trial :func:`occupancy_parts`, in trial order —
    computed once by the caller and shared across cells.

    The reference decides what "unstable" means, and on this maze the choice is
    not free:

    ``template``  the cell's map from all the animal's OTHER trials that day.
                  Leave-one-out matters: a trial that helped build its own
                  reference correlates with itself, which would flatter exactly
                  the short sessions where the estimate is weakest.
    ``freeroam``  the map from the free-roaming trials only (`free_roam` holds
                  their indices). Goal-independent — it does not move as the
                  animal learns the goal — and on this dataset those trials are
                  ~600 s each and cover 71-79% of the ground the whole session
                  covers, so the support is there.
    ``prev``      the previous trial, as in the paper. Kept for comparison, but
                  see the caveat: two consecutive HexMaze runs start from
                  different nodes and share a median of 0-5 valid bins, so most
                  pairs come back NaN.

    Returns one dict per trial: ``kl_bits`` (the sum, comparable to the paper),
    ``kl_bits_per_bin`` (the mean, comparable ACROSS trials of different length),
    ``kl_n_bins`` and ``n_spikes_trial``.
    """
    maps, spk, occ_s, occ_raw, counts = [], [], [], [], []
    for occ in occs:
        s, sx, _sy = spike_parts(occ, spike_times)
        spk.append(s); occ_s.append(occ["occ_s"]); occ_raw.append(occ["occ_raw"])
        counts.append(int(sx.size))
        vis = valid_bins(occ["occ_raw"], occ["occ_s"], sigma, min_occ_s)
        maps.append(rate_from_maps(s, occ["occ_s"], vis))

    # Pool once, then subtract the scored trial. Smoothing is linear, so the sum
    # of smoothed maps IS the smoothed sum, and the leave-one-out reference is a
    # subtraction rather than a rebuild — the difference between one pass over the
    # trials and one pass per trial.
    pool_idx = (list(free_roam or []) if reference == "freeroam"
                else list(range(len(occs))))
    if reference == "freeroam" and not pool_idx:
        return [{"kl_bits": np.nan, "kl_bits_per_bin": np.nan, "kl_n_bins": 0,
                 "n_spikes_trial": c} for c in counts]
    SPK = sum(spk[i] for i in pool_idx) if pool_idx else None
    OCC_S = sum(occ_s[i] for i in pool_idx) if pool_idx else None
    OCC_R = sum(occ_raw[i] for i in pool_idx) if pool_idx else None

    out = []
    for k in range(len(occs)):
        if reference == "prev":
            ref = maps[k - 1] if k else None
        else:
            inc = k in pool_idx          # only subtract what this trial put in
            s = SPK - spk[k] if inc else SPK
            o_s = OCC_S - occ_s[k] if inc else OCC_S
            o_r = OCC_R - occ_raw[k] if inc else OCC_R
            ref = (rate_from_maps(s, o_s, valid_bins(o_r, o_s, sigma, min_occ_s))
                   if np.any(o_r > 0) else None)
        tot, per, n = poisson_kl_bits(maps[k], ref, min_bins=min_bins,
                                      symmetric=symmetric, normalise=normalise)
        out.append({"kl_bits": tot, "kl_bits_per_bin": per, "kl_n_bins": n,
                    "n_spikes_trial": counts[k]})
    return out


def cumulative_kl_slope(values):
    """Slope of the CUMULATIVE divergence against trial number — the paper's
    per-cell statistic for how fast a map is changing.

    The cumulative curve of a non-negative quantity always rises, so the slope is
    a rate of change, not a direction: a flat map gives a shallow slope and a
    drifting one a steep slope. NaN below two finite trials, where a line through
    the points is not a summary of anything.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 2:
        return np.nan
    return float(np.polyfit(np.arange(1, v.size + 1), np.cumsum(v), 1)[0])


def _skaggs(lam, p, lam_mean):
    """Skaggs spatial information (bits/spike) over the bins p is defined on."""
    if lam_mean <= 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lam / lam_mean
        terms = np.where(lam > 0, p * ratio * np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
    return float(np.nansum(terms))


def place_field_metrics(x, y, t, spike_times, extent, bins, dt, sigma, speed_thresh,
                        goal_xy=None, t0=None, t1=None,
                        field_frac=DEFAULT_FIELD_FRAC, min_peak_hz=DEFAULT_MIN_PEAK_HZ,
                        min_field_cm=DEFAULT_MIN_FIELD_CM,
                        min_field_cm2=None, min_field_bins=None,
                        min_occ_s=DEFAULT_MIN_OCC_S, speed=None, min_spikes=0,
                        si_match_n=None, si_match_repeats=20, seed=0,
                        max_gap_s=None, stability_windows=None,
                        min_stability_bins=DEFAULT_MIN_STABILITY_BINS):
    """Place-coding metrics for one cell over a window.

    A place field is an 8-connected region of valid bins >= ``field_frac`` of the
    peak covering at least ``min_field_cm2`` with an in-field peak >=
    ``min_peak_hz``; a cell CAN have several on this large maze. Every threshold is
    physical, so the same arguments mean the same thing at 5 cm and at 2.5 cm bins.

      n_fields            : number of place fields (NaN, not 0, when the cell has
                            too little data for the metric to be defined, so every
                            panel of the report has the same denominator)
      spatial_info        : Skaggs spatial information (bits/spike) — RAW, and
                            biased upward at low spike counts; see METRIC_NOTES
      spatial_info_matched: Skaggs SI re-estimated from a fixed number of spikes
                            (``si_match_n``, averaged over ``si_match_repeats``
                            seeded draws), NaN when the cell has fewer. This is the
                            only spatial-tuning number here that is comparable
                            across sessions of different length.
      stability           : split-half spatial correlation (Pearson r) between the
                            maps built from alternate trials — pass
                            ``stability_windows`` (the epoch's trial windows) or
                            the epoch is cut at its midpoint instead; NaN below
                            ``min_stability_bins`` bins valid in both halves
      selectivity         : peak rate / mean rate
      peak                : peak rate (Hz) of the smoothed map — reported because
                            it sets the field threshold and the selectivity
                            numerator, and it is the most sample-size sensitive
                            quantity in the pipeline
      field_goal_m        : mean distance (m) from each field to the goal node
      field_size_mean_cm  : mean field LENGTH along the track (cm) — the size
                            measure to quote, because it survives a change of bin
                            size and the area does not (see field_length_cm)
      field_size_*_m2     : field areas in physical units, so the field
                            decomposition is auditable
      n_spikes_epoch, epoch_dur_s, occ_total_s, n_valid_bins : the exposure
                            variables every one of the above is confounded with.
                            Without them the confound cannot be checked post hoc.

    Field position is the field's PEAK bin, not the arithmetic centroid of its
    bins: on a maze the centroid of a component that follows a bend or a junction
    lands off the trajectory, in a bin the animal never entered.

    `goal_xy` must be a real maze node position — when it is None (or the goal node
    is not in the node table) every ``field_goal_*`` value comes back NaN rather
    than raising.

    Returns (metrics_dict, rate, extent)."""
    bx_cm, by_cm = bin_size_cm(extent, bins)
    if abs(bx_cm - by_cm) > 1e-3 * max(bx_cm, by_cm):
        raise ValueError(f"non-square bins ({bx_cm:.4f} x {by_cm:.4f} cm): a scalar "
                         "sigma would be an anisotropic kernel and areas would be "
                         "wrong; pick a bin size that divides the maze extent.")
    bin_area_cm2 = bx_cm * by_cm

    nan = {"n_fields": np.nan, "spatial_info": np.nan, "spatial_info_matched": np.nan,
           "selectivity": np.nan, "stability": np.nan, "stability_n_bins": 0,
           "stability_split": "none", "field_goal_m": np.nan,
           "field_goal_null_m": np.nan, "field_goal_over_null": np.nan,
           "field_goal_largest_m": np.nan, "field_goal_2ndlargest_m": np.nan,
           "field_goal_smallest_m": np.nan, "field_size_largest_m2": np.nan,
           "field_size_smallest_m2": np.nan, "field_size_mean_cm": np.nan,
           "field_size_largest_cm": np.nan, "peak": np.nan,
           "n_spikes_epoch": 0, "epoch_dur_s": np.nan, "occ_total_s": np.nan,
           "n_valid_bins": 0}

    parts = _rate_map_parts(x, y, t, spike_times, extent, bins, dt, sigma,
                            t0=t0, t1=t1, speed_thresh=speed_thresh, speed=speed,
                            min_occ_s=min_occ_s, max_gap_s=max_gap_s)
    if parts is None:
        return nan, None, extent
    rate, visited = parts["rate"], parts["visited"]
    exposure = {"n_spikes_epoch": parts["n_spikes"], "epoch_dur_s": parts["dur_s"],
                "occ_total_s": float(parts["occ_raw"][visited].sum()),
                "n_valid_bins": int(visited.sum())}
    nan = {**nan, **exposure}
    if not rate.count() or parts["n_spikes"] < int(min_spikes):
        # Too little data for ANY of these metrics to mean anything. Everything is
        # NaN — including n_fields, which used to come back as a finite 0 and so
        # silently gave the "# place fields" panel a larger denominator than every
        # other panel, falling as sessions got shorter even with unchanged cells.
        return nan, rate, extent

    lam = rate.filled(0.0)
    # Skaggs prior: RAW occupancy, restricted to the same valid bins the rate is
    # defined on. Raw is the right pairing (it preserves sum(p*lam) = N/T; taking p
    # from the smoothed map loses ~11% to edge leakage), but the support must match
    # the rate's or the sum runs over bins that have no rate estimate.
    p_occ = np.where(visited, parts["occ_raw"], 0.0)
    tot = p_occ.sum()
    if tot <= 0:
        return nan, rate, extent
    p = p_occ / tot
    lam_mean = float((p * lam).sum())
    peak = float(lam.max())
    if lam_mean <= 0:
        return {**nan, "peak": peak}, rate, extent
    spatial_info = _skaggs(lam, p, lam_mean)
    selectivity = peak / lam_mean

    # Spike-count-matched spatial information. Skaggs bits/spike is invariant to a
    # global scaling of the rate, so thinning the spikes and rebuilding the map on
    # the SAME occupancy isolates the count bias: with counts matched, this
    # dataset's SI-vs-performance correlation drops from rho +0.128 to +0.061.
    si_matched = np.nan
    if si_match_n and parts["n_spikes"] >= int(si_match_n):
        sx, sy = parts["spike_xy"]
        gen = np.random.default_rng(seed)
        vals = []
        for _ in range(int(si_match_repeats)):
            sel = gen.choice(sx.size, int(si_match_n), replace=False)
            spk_m, _, _ = np.histogram2d(sx[sel], sy[sel], **parts["hist_kw"])
            spk_m = spk_m.T
            if sigma and sigma > 0:
                spk_m = gaussian_filter(spk_m, sigma)
            with np.errstate(divide="ignore", invalid="ignore"):
                lam_m = np.where(parts["occ_s"] > 0, spk_m / parts["occ_s"], 0.0)
            lam_m = np.where(visited, lam_m, 0.0)
            vals.append(_skaggs(lam_m, p, float((p * lam_m).sum())))
        si_matched = float(np.nanmean(vals)) if vals else np.nan

    # Split-half stability. It is built from the SAME positions, speed gate and
    # bin/kernel geometry as the map above — a stability number computed on a
    # different grid than the fields it is quoted next to is not about those fields.
    stability, stab_bins, stab_split = split_half_stability(
        x, y, t, spike_times, extent, bins, dt, sigma, windows=stability_windows,
        t0=t0, t1=t1, speed=speed, speed_thresh=speed_thresh, min_occ_s=min_occ_s,
        max_gap_s=max_gap_s, min_bins=min_stability_bins)

    fields = place_fields(rate, field_frac=field_frac, min_peak_hz=min_peak_hz,
                          min_field_cm=min_field_cm, bin_cm=(bx_cm, by_cm),
                          min_field_cm2=min_field_cm2, bin_area_cm2=bin_area_cm2,
                          min_field_bins=min_field_bins)
    ny, nx = rate.shape

    # Occupancy-matched null for the field-to-goal distances. Every centroid is
    # bounded by the region the animal covered, and coverage collapses onto the
    # direct start->goal route as it learns (measured: 2304 -> 203 valid bins), so
    # the maximum attainable distance shrinks with learning all on its own. This is
    # the mean distance-to-goal of the animal's OWN occupancy that session; the
    # ratio to it is what a downward trend has to beat to mean anything.
    goal_null = np.nan
    if goal_xy is not None:
        cx_grid = extent[0] + (np.arange(nx) + 0.5) * (extent[1] - extent[0]) / nx
        cy_grid = extent[2] + (np.arange(ny) + 0.5) * (extent[3] - extent[2]) / ny
        dgrid = np.hypot(cx_grid[None, :] - goal_xy[0], cy_grid[:, None] - goal_xy[1])
        goal_null = float((p * dgrid).sum())
    dists, areas, lengths = [], [], []
    for comp in fields:
        iy, ix = np.unravel_index(np.argmax(np.where(comp, lam, -np.inf)), lam.shape)
        cx = extent[0] + (ix + 0.5) * (extent[1] - extent[0]) / nx
        cy = extent[2] + (iy + 0.5) * (extent[3] - extent[2]) / ny
        areas.append(comp.sum() * bin_area_cm2 / 1e4)      # m^2
        lengths.append(field_length_cm(comp, (bx_cm, by_cm)))
        dists.append(float(np.hypot(cx - goal_xy[0], cy - goal_xy[1]))
                     if goal_xy is not None else np.nan)
    # per-field distances ranked by field AREA: largest / 2nd-largest / smallest
    largest = second = smallest = np.nan
    a_large = a_small = np.nan
    len_mean = len_large = np.nan
    if areas:
        order = np.argsort(areas)[::-1]          # descending by area
        a_large, a_small = areas[order[0]], areas[order[-1]]
        # Size in track cm, ranked the same way, so "largest field" means the same
        # field in both spellings even though only the cm one survives a bin change.
        len_mean = float(np.mean(lengths))
        len_large = float(lengths[order[0]])
        if goal_xy is not None:
            largest = dists[order[0]]
            # 2nd-largest exists only for multi-field cells, so its N is itself
            # sampling-driven; NaN (not a silent copy of the only field) keeps that
            # visible in the panel's n.
            second = dists[order[1]] if len(order) >= 2 else np.nan
            smallest = dists[order[-1]] if len(order) >= 2 else np.nan
    mean_dist = float(np.nanmean(dists)) if np.any(np.isfinite(dists)) else np.nan
    return ({"n_fields": len(fields), "spatial_info": spatial_info,
             "spatial_info_matched": si_matched,
             "selectivity": selectivity,
             "stability": stability, "stability_n_bins": stab_bins,
             "stability_split": stab_split,
             "field_goal_m": mean_dist,
             "field_goal_null_m": goal_null,
             "field_goal_over_null": (mean_dist / goal_null
                                      if goal_null and np.isfinite(goal_null) and goal_null > 0
                                      else np.nan),
             "field_goal_largest_m": largest,
             "field_goal_2ndlargest_m": second,
             "field_goal_smallest_m": smallest,
             "field_size_largest_m2": a_large,
             "field_size_smallest_m2": a_small,
             "field_size_mean_cm": len_mean,
             "field_size_largest_cm": len_large,
             "peak": peak, **exposure}, rate, extent)
