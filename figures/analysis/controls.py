"""Five controls, side by side: which of them moves the goal-island effect.

The apparent result is that the island carrying the goal has more cells coding
position. The animal also spends far longer there, so the question is whether
anything is left once that is controlled. Five controls were tried, and the
figure puts them in one row each so they can be compared rather than argued
about.

Top row  how strongly the pass rate still follows exposure, measured where no
         goal effect can contaminate it: non-goal islands only.

         Exposure here is ``occ_s``, and it is RUNNING time on that island, not
         time spent there. Three filters build it: trial windows only, so the
         gaps between trials are out; speed above 0.05 m/s, which on this
         tracking removes about two thirds of the frames, so an animal sitting
         on an island contributes nothing; and only that island's bins that were
         visited and passed the occupancy floor. On one session the chain runs
         3994 s of video, 2397 s inside trials, 817 s of running. Calling it
         "dwell" overstates it by a factor of three, which is why the axis says
         what it is.
Bottom   the goal-island comparison itself, paired within island.

A control that works should flatten the top row. Only one does, and it is the
one that pays for it by shrinking the map the rate is computed on.

The columns are NOT drawn from the same cell-blocks: each control drops the
cell-blocks it cannot apply to (bins below 500, dwell below 100 s), so the n
under each column differs and is printed. Comparing a column with itself across
the two rows is safe; reading an absolute level across columns is not.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from hm_rat_analysis import maze

from analysis import data, style
from analysis.style import ISLAND_COLOURS, lw as _lw

#: Each control: the column it writes, a short name, and what it holds fixed.
#: Each control: the column it writes, what it HOLDS FIXED, and at what value.
#: The names say what was fixed, which is the one thing that differs between
#: columns. Note that the x axis of the top row is the OPPOSITE: the running
#: time as recorded, varying from 12 s to 355 s. "Fixed" belongs on the column,
#: never on that axis.
CONTROLS = (
    ("sig95", "nothing fixed", "every cell as recorded"),
    ("sig95_matched", "spikes fixed", "40 spikes"),
    ("sig95_bm", "bins fixed", "500 bins"),
    ("sig95_bm_m", "bins + spikes fixed", "500 bins, 40 spikes"),
    ("sig95_occ_matched", "run time + spikes fixed", "100 s running, 40 spikes"),
)


def table():
    """One frame carrying every control's verdict for every cell-block."""
    from analysis import plots
    si = plots._blocks(data.island_si())
    bm = data.island_si_bin()
    bm["phase"] = bm["phase"].fillna("")
    bm = plots._blocks(bm)
    oc = data.island_si_occ()
    oc["phase"] = oc["phase"].fillna("")
    oc = plots._blocks(oc)
    key = ["animal", "slot", "island", "unit_id"]
    m = (si.merge(bm[key + ["sig95_bm", "sig95_bm_m"]], on=key, how="left")
           .merge(oc[key + ["sig95_occ", "sig95_occ_matched"]], on=key,
                  how="left"))
    m = m[m["island"].between(1, 4)].copy()
    m["gi"] = [maze.island_of_node(str(g)) if str(g) not in ("", "nan")
               else np.nan for g in m["goal_node"]]
    m = m.dropna(subset=["gi"])
    m["goal_island"] = m["island"] == m["gi"].astype(int)
    return m


def _blocks_for(m, col):
    """Per block and island, the pass rate under `col`, where it is defined."""
    d = m[m[col].notna()]
    return (d.groupby(["animal", "slot", "island", "goal_island"])
              .agg(occ=("occ_s", "first"), rate=(col, "mean"),
                   n=(col, "size")).reset_index())


def controls_figure(name="fig_controls", out_dir=None):
    m = table()
    W = 180.0
    H = 104.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM), facecolor="none")
    n = len(CONTROLS)
    left, right, gap = 14.0, 3.0, 5.0
    w = (W - left - right - gap * (n - 1)) / n

    def axes(i, y, h):
        a = fig.add_axes([(left + i * (w + gap)) / W, 1 - (y + h) / H,
                          w / W, h / H])
        a.set_facecolor("none")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(style.P.MUTED)
            a.spines[s].set_linewidth(style.P.lw(0.8))
        a.tick_params(length=2.5, pad=2, colors=style.P.MUTED,
                      labelcolor=style.P.INK, labelsize=style.FS["tick"] - 1)
        a.grid(axis="y", color=style.P.MUTED, lw=style.P.lw(0.4), alpha=0.3,
               zorder=0)
        a.set_axisbelow(True)
        return a

    report = []
    for i, (col, short, held) in enumerate(CONTROLS):
        b = _blocks_for(m, col)
        off = b[~b["goal_island"]]
        # --- top: does the pass rate still follow dwell? ---
        ax = axes(i, 10, 34)
        rho = spearmanr(off["occ"], off["rate"]).correlation
        for _, r in off.iterrows():
            ax.plot(r["occ"], 100 * r["rate"], "o", ms=2.6,
                    color=ISLAND_COLOURS[int(r["island"])], alpha=0.65,
                    mec="none", zorder=3)
        ax.set_xlim(0, 380)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 150, 300])
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("cells above shuffle (%)\nnon-goal islands",
                          fontsize=style.FS["tick"], color=style.P.INK,
                          labelpad=3)
        # As RECORDED, not fixed: this is the variable the controls are meant
        # to disconnect the pass rate from, so it has to be the raw one in
        # every column, including the one that fixed it.
        ax.set_xlabel("running time on island, as recorded (s)",
                      fontsize=style.FS["tick"] - 1.5, color=style.P.MUTED,
                      labelpad=1)
        ax.set_title(f"{short}\n{held}", fontsize=style.FS["tick"] - 1,
                     color=style.P.INK, pad=3)
        ax.text(0.04, 0.96, f"$\\rho$ = {rho:+.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=style.FS["tick"],
                color=style.P.RED if abs(rho) > 0.3 else style.P.GREEN_DARK)

        # --- bottom: the goal-island comparison, paired within island ---
        ax2 = axes(i, 62, 34)
        pairs = []
        for (a, k), d in b.groupby(["animal", "island"]):
            on, of = d[d["goal_island"]], d[~d["goal_island"]]
            if on.empty or of.empty:
                continue
            pairs.append((a, k, of["rate"].mean(), on["rate"].mean()))
        up = sum(1 for _a, _k, o, g in pairs if g > o)
        diff = np.median([g - o for _a, _k, o, g in pairs])
        p = wilcoxon([g for *_x, g in pairs], [o for *_x, o, _g in pairs]).pvalue
        for a, k, o, g in pairs:
            ls = "-" if a == "Rat5" else (0, (3, 2))
            ax2.plot([0, 1], [100 * o, 100 * g], color=ISLAND_COLOURS[int(k)],
                     lw=_lw(1.2), ls=ls, zorder=3)
            ax2.plot([0, 1], [100 * o, 100 * g], "o", ms=3.2,
                     color=ISLAND_COLOURS[int(k)], mec="white", mew=_lw(0.8),
                     zorder=4)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["other", "goal"], fontsize=style.FS["tick"] - 1,
                            color=style.P.INK)
        ax2.set_xlim(-0.35, 1.35)
        ax2.set_ylim(0, 100)
        if i:
            ax2.set_yticklabels([])
        else:
            ax2.set_ylabel("cells above shuffle (%)",
                           fontsize=style.FS["tick"], color=style.P.INK,
                           labelpad=3)
        ax2.text(0.04, 0.96, f"{up}/{len(pairs)}  $p$={p:.3f}\n"
                 f"$\\Delta$ {100 * diff:+.0f}%", transform=ax2.transAxes,
                 ha="left", va="top", fontsize=style.FS["tick"] - 0.5,
                 color=style.P.INK)
        ax2.text(0.5, -0.30, f"n = {len(b)} block-islands",
                 transform=ax2.transAxes, ha="center", va="top",
                 fontsize=style.FS["tick"] - 1.5, color=style.P.MUTED)
        report.append((short, rho, up, len(pairs), 100 * diff, p))

    for lab, y in (("a", 10), ("b", 62)):
        fig.text(left / W - 0.055, 1 - (y - 4) / H, lab,
                 fontsize=style.FS["title"], fontweight="bold",
                 color=style.P.INK, va="top")

    out = Path(out_dir) if out_dir is not None else data.out_dir()
    written = []
    for suffix, extra in ((".pdf", {}), (".png", dict(dpi=600))):
        f = (out / name).with_suffix(suffix)
        fig.savefig(f, transparent=True, **extra)
        written.append(f)
    plt.close(fig)
    print(f"  {name}:")
    for short, rho, up, npair, diff, p in report:
        print(f"    {short:24} rho(running time as recorded)={rho:+.2f}   "
              f"paired {up}/{npair}  median {diff:+5.1f}%  p={p:.3f}")
    for f in written:
        print(f"  wrote {f}  ({f.stat().st_size / 1024:.0f} KB)")
    return written
