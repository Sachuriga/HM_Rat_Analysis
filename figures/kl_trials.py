"""Per-trial map divergence across the whole experiment, both references.

One point per trial: the median over that session's pyramidal cells of the
Poisson-KL divergence between the trial's rate map and a reference map, in bits
per co-visited bin (Quattrocolo et al.'s stability measure — see
``place_fields.poisson_kl_bits``). Trials run left to right inside their session,
sessions are grouped by goal block, and colour is the animal.

    top     reference = the day's own map, leave-one-out. Internal consistency:
            does this trial's map match the rest of the day.
    bottom  reference = the FREE-ROAMING trials only. Goal-independent: the
            baseline does not move as the animal learns the goal.

Free-roaming trials are drawn as open rings rather than dropped. They are the
control — a normal free-roam trial is not expected to move the map, so where its
rings sit relative to the filled goal points is the measure's own sanity check.

Per-bin, not per-trial-total: a trial that covered more ground shares more bins
with the reference, and an unnormalised sum would rank the long trials as the
unstable ones.

Usage:
    python figures/kl_trials.py --summary <session_summary_*.xlsx> [--out kl_trials]
"""

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import msca_fig1 as F1                                            # noqa: E402

from hm_rat_analysis.reports import session_summary as SS         # noqa: E402

INK, INK2, MUTED, SURFACE = F1.INK, F1.INK2, F1.MUTED, F1.SURFACE

PANELS = [("template", "a", "Reference: the day's own map (leave-one-out)"),
          ("freeroam", "b", "Reference: the free-roaming trials"),
          ("spikes", "c", "Exposure: spikes per cell per trial")]


def trial_points(div, animals):
    """``{animal: {slot label: (trial numbers, values)}}`` — one value per trial,
    the median over the session's cells."""
    d = div.copy()
    d["label"], _r, _s = F1._label(d)
    out = {}
    for a in animals:
        sub = d[d["animal"] == a]
        out[a] = {k: g for k, g in sub.groupby("label")}
    return out


def draw(ax, div, ref, keys, pos, meta, animals, colors, scale=1.0, session_stat="mean"):
    """Trials spread across their session's slot, in order, one line per session,
    with that session's value for the whole session on top.

    The session value is over GOAL trials only — a free-roaming bout is a different
    behaviour and averaging it in would move the session number for a reason that
    has nothing to do with the goal-directed map.
    """
    col = f"kl_{ref}_per_bin" if ref != "spikes" else "n_spikes_trial"
    by = trial_points(div, animals)
    lo, hi = np.inf, -np.inf
    for a, c in zip(animals, colors):
        sess_x, sess_y = [], []
        for k, g in by[a].items():
            if k not in keys:
                continue
            g = g.sort_values("trial")
            v = g.groupby("trial")[col].median()
            tt = g.groupby("trial")["trial_type"].first()
            v = v[np.isfinite(v)]
            if v.empty:
                continue
            # trials fill the slot left to right; a single-trial session sits mid-slot
            n = len(v)
            span = np.linspace(-0.40, 0.40, n) if n > 1 else np.zeros(1)
            x0 = pos[keys.index(k)]
            xs = x0 + span
            ax.plot(xs, v.to_numpy(), "-", color=c, lw=0.8 * scale, alpha=0.45,
                    zorder=2)
            free = np.array([tt.get(i, 1) in (4,) for i in v.index])
            ax.plot(xs[~free], v.to_numpy()[~free], "o", ms=2.6, color=c,
                    mec="none", alpha=0.65, zorder=3)
            if free.any():          # the control, marked rather than hidden
                ax.plot(xs[free], v.to_numpy()[free], "o", ms=4.2,
                        mfc="none", mec=c, mew=1.0 * scale, alpha=0.8, zorder=4)
            goal = pd.to_numeric(g.loc[g.trial_type == 1, col], errors="coerce").dropna()
            if len(goal):
                sess_x.append(x0)
                sess_y.append(float(goal.mean() if session_stat == "mean"
                                    else goal.median()))
            lo, hi = min(lo, v.min()), max(hi, v.max())
        if sess_x:
            o = np.argsort(sess_x)
            sx, sy = np.asarray(sess_x)[o], np.asarray(sess_y)[o]
            ax.plot(sx, sy, "-", color=c, lw=1.9 * scale, zorder=5)
            ax.plot(sx, sy, "o", ms=max(2.6, 5.6 * scale), color=c, mec=SURFACE,
                    mew=1.3 * scale, zorder=6)
    return lo, hi


def build(summary_path, out_stem, width=F1.A4_TEXT_WIDTH_IN, transparent=True,
          animal=None):
    summary = F1.load_summary(summary_path)
    div = summary.get("trial_divergence")
    if div is None or div.empty:
        raise ValueError("this summary has no 'trial_divergence' sheet — re-run "
                         "hm-session-summary (it is skipped with --no-kl, and on "
                         "sessions without trustworthy trial windows)")
    sess, keys, meta, animals = F1.session_axis(summary, animal)
    div = div[div["animal"].isin(animals)]
    pos = F1.slot_positions(meta)
    colors = F1.ANIMAL_COLORS[:len(animals)]

    scale = width / F1.REF_WIDTH_IN
    row_h, row_gap = 2.05 * scale, max(0.86, 1.2 * scale)
    top_pad, bottom_pad = max(0.34, 0.6 * scale), max(0.80, 1.1 * scale)
    n_row = len(PANELS)
    height = n_row * row_h + (n_row - 1) * row_gap + top_pad + bottom_pad + 0.2
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    left = max(0.62, 0.055 * F1.REF_WIDTH_IN * scale) / width
    w = (1 - 0.06 / width) - left

    for i, (ref, letter, title) in enumerate(PANELS):
        y0 = (bottom_pad + (n_row - 1 - i) * (row_h + row_gap)) / height
        ax = fig.add_axes([left, y0, w, row_h / height])
        F1._frame(ax, letter, title,
                  "spikes per trial" if ref == "spikes" else "bits per co-visited bin",
                  meta, pos, scale=scale)
        if ref == "spikes":
            # 4 to 2400 spikes per trial: on a linear axis the late sessions —
            # the ones the divergence rises in — are a flat line on the floor.
            # The scale has to be set BEFORE the limits, and a log axis has no 0.
            ax.set_yscale("log")
        lo, hi = draw(ax, div, ref, keys, pos, meta, animals, colors, scale=scale)
        if np.isfinite([lo, hi]).all():
            if ref == "spikes":
                ax.set_ylim(max(0.5, lo * 0.6), hi * 1.8)
            else:
                pad = 0.08 * (hi - lo) if hi > lo else 0.5
                ax.set_ylim(max(0.0, lo - pad), hi + pad)
        if i == 0:
            h = [plt.Line2D([], [], color=c, marker="o", ms=3.5, lw=1.0, label=a)
                 for a, c in zip(animals, colors)]
            h.append(plt.Line2D([], [], color=INK2, marker="o", ms=4.2, mfc="none",
                                ls="none", label="free-roaming trial"))
            h.append(plt.Line2D([], [], color=INK2, lw=1.9, marker="o", ms=5.0,
                                mec=SURFACE, mew=1.3, label="session mean"))
            ax.legend(handles=h, fontsize=F1.FONT["legend"], frameon=False,
                      labelcolor=INK2, handlelength=1.4, borderpad=0.15,
                      ncol=len(h), loc="upper left")

    stamp = SS._param_stamp(sess.to_dict("records"))
    chars = int((width - left * width - 0.06) * 72 / (0.56 * F1.FONT["stamp"]))
    fig.text(left, 0.07 / height, "\n".join(textwrap.wrap(stamp, max(30, chars))),
             fontsize=F1.FONT["stamp"], color=MUTED, va="bottom", linespacing=1.4)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(transparent=True) if transparent else dict(facecolor=SURFACE)
    written = []
    for suffix, extra in ((".pdf", {}), (".svg", {}), (".png", dict(dpi=220))):
        p = out_stem.with_suffix(suffix)
        fig.savefig(p, **kw, **extra)
        written.append(p)
    plt.close(fig)
    for p in written:
        print(f"wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="kl_trials")
    ap.add_argument("--animal", default=None, nargs="*")
    ap.add_argument("--width", type=float, default=F1.A4_TEXT_WIDTH_IN)
    ap.add_argument("--opaque", action="store_true")
    a = ap.parse_args(argv)
    build(a.summary, a.out, width=a.width, transparent=not a.opaque,
          animal=a.animal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
