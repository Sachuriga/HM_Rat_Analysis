"""LFP power spectra per tetrode, split by locomotion state.

Two questions, one session:

    1. what does each tetrode's spectrum look like while the animal runs
       (> `SPEED_TH` cm/s) versus while it is still, and
    2. how much theta power the running state adds over the still state.

Speed comes from the tracked position through the same 400 ms-smoothed
differentiation the MSCA figures use (``msca_fig1a.trace_speed``), so the
2.5 cm/s gate here means the same thing as the spike-display gate in figure 1.

Untracked stretches are dropped rather than counted as "still": this session
carries two long tracking gaps, and a gap is not a behavioural state.

Usage::

    python figures/lfp_speed_spectra.py <session_dir> [-o OUT_DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import palette as P
from hm_rat_analysis import maze
from hm_rat_analysis import nwb as N
from msca_fig1a import trace_speed

# ---- analysis constants -------------------------------------------------
SPEED_TH = 2.5          # cm/s, the moving/still boundary (matches figure 1)
MIN_EPOCH_S = 1.4       # s, an epoch must hold at least one Welch window
NPERSEG = 2048          # samples, 1.365 s at 1500 Hz -> 0.73 Hz resolution
FMAX = 200.0            # Hz, plotted range
MAX_GAP_S = 0.20        # s, how far a position sample may be from an LFP sample

# shaded on every spectrum, in the order they are drawn
BANDS = {"delta": (2.0, 4.0), "theta": (6.0, 10.0),
         "gamma": (30.0, 60.0), "high gamma": (60.0, 120.0)}
BAND_COLOR = {"delta": P.MUTED, "theta": P.AMBER,
              "gamma": P.GREEN, "high gamma": P.RED}
THETA = BANDS["theta"]

C_MOVE, C_STILL = P.ORANGE, P.BLUE


# ---- data ---------------------------------------------------------------
def load_lfp(session_dir):
    """(data mmap, timestamps, ntrode numbers, emg index) from LFP_Output."""
    out = Path(session_dir) / "LFP_Output"

    def pick(pat, *, without=("emg",)):
        """First non-sidecar match, skipping the EMG derivatives.

        ``*_lfp_timestamps.npy`` also matches ``*_emg_from_lfp_timestamps.npy``
        and sorts after it, which silently yields a 5 Hz time base.
        """
        cands = [p for p in sorted(out.glob(pat))
                 if not p.name.startswith("._")
                 and not any(w in p.name for w in without)]
        if not cands:
            raise FileNotFoundError(f"{pat} in {out}")
        return cands[0]
    data = np.load(pick("*_lfp_data.npy"), mmap_mode="r")
    ts = np.load(pick("*_lfp_timestamps.npy"))
    cmap = np.load(pick("*_channel_map.npy"), allow_pickle=True)
    ntrodes = [int(c["ntrode"]) for c in cmap]
    try:
        emg = int(np.load(pick("*_emg_channel_index.npy", without=()))[0])
    except FileNotFoundError:
        emg = None
    return data, ts, ntrodes, emg


def speed_on_lfp(session_dir, lfp_t):
    """Speed in cm/s sampled at `lfp_t`, plus a validity mask.

    Invalid means the nearest tracked sample is either not finite or further
    away than `MAX_GAP_S`; those LFP samples belong to no state at all.
    """
    from pynwb import NWBHDF5IO

    nwb_path = N.find_nwb_file(session_dir)
    with NWBHDF5IO(str(nwb_path), "r") as io:
        x, y, t = N.load_position(io.read())

    good = np.isfinite(x) & np.isfinite(y)
    # speed is differentiated on the tracked trace only, so a gap does not
    # manufacture a huge jump on either side of itself
    sp = np.full(len(t), np.nan)
    sp[good] = trace_speed(x[good] / maze.SCALE_X,
                           y[good] / maze.SCALE_Y, t[good]) * 100.0

    idx = np.searchsorted(t, lfp_t).clip(1, len(t) - 1)
    left = np.abs(lfp_t - t[idx - 1]) <= np.abs(lfp_t - t[idx])
    near = np.where(left, idx - 1, idx)
    gap = np.abs(lfp_t - t[near])

    out = sp[near]
    valid = np.isfinite(out) & (gap <= MAX_GAP_S)
    return out, valid


def epochs(mask, fs, min_s=MIN_EPOCH_S):
    """[(start, stop), ...] index pairs for runs of True at least `min_s` long."""
    m = np.asarray(mask).astype(np.int8)
    d = np.diff(m)
    starts = np.flatnonzero(d == 1) + 1
    stops = np.flatnonzero(d == -1) + 1
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        stops = np.r_[stops, len(m)]
    need = int(round(min_s * fs))
    return [(a, b) for a, b in zip(starts, stops) if b - a >= need]


def psd(chan, eps, fs):
    """Welch PSD averaged over `eps`, each epoch weighted by its length.

    `NPERSEG` is fixed so every epoch lands on the same frequency grid; epochs
    too short to hold one window are skipped (`epochs` already filters them).
    """
    acc = None
    total = 0
    freqs = None
    for a, b in eps:
        if b - a < NPERSEG:
            continue
        seg = np.asarray(chan[a:b], dtype=np.float64)
        f, p = signal.welch(seg, fs=fs, nperseg=NPERSEG, noverlap=NPERSEG // 2,
                            window="hann", detrend="constant")
        w = len(seg)
        acc = p * w if acc is None else acc + p * w
        total += w
        freqs = f
    if acc is None:
        raise RuntimeError("no epoch long enough for one Welch window")
    return freqs, acc / total


def band_power(f, p, lo, hi):
    """Integrated power in [lo, hi) Hz."""
    m = (f >= lo) & (f < hi)
    return float(np.trapz(p[m], f[m]))


# ---- figures ------------------------------------------------------------
def fig_grid(f, pm, ps, ntrodes, emg, n_move_s, n_still_s):
    """One PSD panel per tetrode, both states overlaid."""
    ncol, nrow = 8, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 9), sharex=True, sharey=True)
    band = (f >= 1) & (f <= FMAX)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(ntrodes):
            ax.axis("off")
            continue
        for name, (lo, hi) in BANDS.items():
            ax.axvspan(lo, hi, color=BAND_COLOR[name], alpha=0.13, lw=0, zorder=0)
        ax.loglog(f[band], ps[i][band], color=C_STILL, lw=1.0, zorder=2)
        ax.loglog(f[band], pm[i][band], color=C_MOVE, lw=1.0, zorder=3)
        tag = f"nt{ntrodes[i]}" + ("  (EMG)" if emg is not None and i == emg else "")
        ax.set_title(tag, fontsize=8, pad=2,
                     color=P.RED if (emg is not None and i == emg) else P.INK)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD (µV²/Hz)", fontsize=8)
    handles = [Line2D([], [], color=C_MOVE, lw=1.6,
                      label=f"moving > {SPEED_TH:g} cm/s  ({n_move_s:.0f} s)"),
               Line2D([], [], color=C_STILL, lw=1.6,
                      label=f"still ≤ {SPEED_TH:g} cm/s  ({n_still_s:.0f} s)")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR[n], alpha=0.30,
                              label=f"{n} {lo:g}-{hi:g} Hz")
                for n, (lo, hi) in BANDS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.962))
    fig.suptitle("LFP power spectrum per tetrode, by locomotion state", y=0.992,
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    return fig


def fig_theta(f, pm, ps, ntrodes, emg):
    """Theta power moving vs still, per tetrode, three views."""
    tm = np.array([band_power(f, p, *THETA) for p in pm])
    tsl = np.array([band_power(f, p, *THETA) for p in ps])
    dm = np.array([band_power(f, p, *BANDS["delta"]) for p in pm])
    dsl = np.array([band_power(f, p, *BANDS["delta"]) for p in ps])
    keep = np.ones(len(ntrodes), bool)
    if emg is not None:
        keep[emg] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    ax = axes[0]
    for i in range(len(ntrodes)):
        col = P.MUTED if not keep[i] else P.INK2
        ax.plot([0, 1], [tsl[i], tm[i]], color=col, lw=0.8, alpha=0.55, zorder=1)
    ax.scatter(np.zeros(keep.sum()), tsl[keep], s=26, color=C_STILL, zorder=3)
    ax.scatter(np.ones(keep.sum()), tm[keep], s=26, color=C_MOVE, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["still", f"moving\n> {SPEED_TH:g} cm/s"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_yscale("log")
    ax.set_ylabel(f"theta power {THETA[0]:g}-{THETA[1]:g} Hz (µV²)")
    ax.set_title("paired by tetrode", fontsize=11)

    ax = axes[1]
    ratio = tm / tsl
    order = np.argsort(-ratio)
    cols = [C_MOVE if keep[i] else P.MUTED for i in order]
    ax.bar(range(len(order)), ratio[order], color=cols, width=0.78)
    ax.axhline(1.0, color=P.INK2, lw=0.9, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{ntrodes[i]}" for i in order], fontsize=7, rotation=90)
    ax.set_xlabel("tetrode")
    ax.set_ylabel("theta power, moving / still")
    ax.set_title("theta gain when running", fontsize=11)

    ax = axes[2]
    rm, rs = tm / dm, tsl / dsl
    for i in range(len(ntrodes)):
        ax.plot([0, 1], [rs[i], rm[i]], color=P.INK2 if keep[i] else P.MUTED,
                lw=0.8, alpha=0.55, zorder=1)
    ax.scatter(np.zeros(keep.sum()), rs[keep], s=26, color=C_STILL, zorder=3)
    ax.scatter(np.ones(keep.sum()), rm[keep], s=26, color=C_MOVE, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["still", f"moving\n> {SPEED_TH:g} cm/s"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("theta / delta ratio")
    ax.set_title("theta relative to delta", fontsize=11)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    from scipy.stats import wilcoxon
    k = np.flatnonzero(keep)
    stat, pval = wilcoxon(tm[k], tsl[k])
    gain = (tm / tsl)[k]
    fig.suptitle("Theta power, moving versus still", y=0.995, fontsize=13,
                 weight="bold")
    fig.text(0.5, 0.935,
             f"{len(k)} tetrodes (EMG channel excluded)   "
             f"median gain {np.median(gain):.2f}×   "
             f"range {gain.min():.2f}-{gain.max():.2f}×   "
             f"Wilcoxon signed-rank W={stat:.0f}, p={pval:.1e}",
             ha="center", fontsize=9.5, color=P.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    return fig, tm, tsl, dm, dsl


# ---- main ---------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir")
    ap.add_argument("-o", "--out-dir", default=None)
    a = ap.parse_args(argv)

    sess = Path(a.session_dir)
    out = Path(a.out_dir) if a.out_dir else (
        Path.home() / "Desktop" / f"LFP_speed_{sess.parent.name}_{sess.name}")
    out.mkdir(parents=True, exist_ok=True)

    data, ts, ntrodes, emg = load_lfp(sess)
    fs = 1.0 / float(np.median(np.diff(ts[:5000])))
    print(f"LFP  {data.shape[0]} samples x {data.shape[1]} tetrodes, "
          f"{fs:.1f} Hz, {(ts[-1] - ts[0]) / 60:.1f} min")

    sp, valid = speed_on_lfp(sess, ts)
    move = valid & (sp > SPEED_TH)
    still = valid & (sp <= SPEED_TH)
    print(f"tracked {100 * valid.mean():5.1f}%   "
          f"moving {move.sum() / fs:7.1f} s   still {still.sum() / fs:7.1f} s   "
          f"untracked {(~valid).sum() / fs:.1f} s (dropped)")

    em, es = epochs(move, fs), epochs(still, fs)
    print(f"epochs >= {MIN_EPOCH_S}s:  moving {len(em)} "
          f"({sum(b - a for a, b in em) / fs:.1f} s), "
          f"still {len(es)} ({sum(b - a for a, b in es) / fs:.1f} s)")

    pm, ps = [], []
    for i in range(data.shape[1]):
        chan = np.asarray(data[:, i], dtype=np.float32)
        f, p1 = psd(chan, em, fs)
        f, p2 = psd(chan, es, fs)
        pm.append(p1)
        ps.append(p2)
        print(f"  nt{ntrodes[i]:<3d} done", end="\r", flush=True)
    print(" " * 30, end="\r")

    n_move_s = sum(b - a for a, b in em) / fs
    n_still_s = sum(b - a for a, b in es) / fs
    g = fig_grid(f, pm, ps, ntrodes, emg, n_move_s, n_still_s)
    g.savefig(out / "01_spectra_per_tetrode.png", dpi=200)
    g.savefig(out / "01_spectra_per_tetrode.pdf")

    t, tm, tsl, dm, dsl = fig_theta(f, pm, ps, ntrodes, emg)
    t.savefig(out / "02_theta_moving_vs_still.png", dpi=200)
    t.savefig(out / "02_theta_moving_vs_still.pdf")

    import csv
    with open(out / "band_power.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ntrode", "is_emg_channel"]
                   + [f"{b}_{s}" for b in BANDS for s in ("moving", "still")]
                   + ["theta_gain", "theta_delta_moving", "theta_delta_still"])
        for i, nt in enumerate(ntrodes):
            row = [nt, int(emg is not None and i == emg)]
            for lo, hi in BANDS.values():
                row += [band_power(f, pm[i], lo, hi), band_power(f, ps[i], lo, hi)]
            row += [tm[i] / tsl[i], tm[i] / dm[i], tsl[i] / dsl[i]]
            w.writerow(row)

    keep = [i for i in range(len(ntrodes)) if emg is None or i != emg]
    gain = (tm / tsl)[keep]
    from scipy.stats import wilcoxon
    stat, pval = wilcoxon(tm[keep], tsl[keep])
    print(f"\ntheta gain (moving/still) over {len(keep)} tetrodes: "
          f"median {np.median(gain):.2f}x, range {gain.min():.2f}-{gain.max():.2f}")
    print(f"Wilcoxon signed-rank, theta moving vs still: W={stat:.0f}, p={pval:.2e}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
