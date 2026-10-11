"""The goal-island figure: an effect that turns out to be dwell time.

Six panels, in the order the argument runs.

a  The design. Each animal's goal moved to a different island each time, and
   between them the two animals put the goal on all four. So every island can be
   compared with ITSELF, as the goal island once and as a plain island three
   times, which removes every difference between islands by construction.
b  The example. One session's occupancy. The animal spends three times longer on
   the island carrying the goal than on any other. That is the confound, in one
   picture.
c  The confound, for all of it: dwell time per island, paired.
d  The apparent result: the fraction of cells that beat their own shuffle is
   higher on the goal island, in 8 of 8 pairs.
e  The same comparison with dwell time forced equal at the source. The effect is
   gone: 4 of 8, and the median difference is +0.01.
f  The positive control. The same matching leaves the across-session rise in
   place and in fact strengthens it, so panel e is not an artefact of a
   procedure that flattens everything.

Panels d and e are the point. Panel f is what keeps panel e honest.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from hm_rat_analysis import maze, nwb as nwbio, place_fields as PF
from hm_rat_analysis.reports import session_summary as SS
from hm_rat_analysis import islands as I

from analysis import data, style
from analysis.schematic import draw_maze, _maze_axes, _island_numbers
from analysis.style import ISLAND_COLOURS, ISLAND_DASH, lw as _lw

#: The session drawn in panel b: the clearest case, and a typical one. Rat6's
#: last session on goal 421, which sits on island 4: 407 s there against a
#: median of 123 s on the other three.
EXAMPLE = {"animal": "Rat6", "date": "20260717", "slot": "GL4S5",
           "goal": 421, "island": 4}

#: Which significance column each panel uses. Each cell is tested at its OWN
#: spike count, on both sides, so panels d and e differ in dwell time and in
#: nothing else.
#:
#: Count matched on top of the dwell, and it has to be. The shuffle equalises
#: the BIAS — the null carries whatever Skaggs adds at a low count, so it is
#: subtracted off — but it does not equalise the POWER. A cell with few spikes
#: has a wide null and cannot beat it whatever it is doing; a cell with many has
#: a narrow one. On this dataset the pass rate runs 10%, 35%, 48%, 47%, 26%
#: across spike-count quintiles, so the test plainly still depends on count.
#:
#: That matters here because the goal island keeps MORE spikes even in the same
#: 100 s (median 212 against 158): the animal fires faster there. Comparing pass
#: rates without matching the count therefore favours the goal island by
#: construction, and part of panel d's 8 of 8 is exactly that.
SIG_RAW = "sig95_matched"
SIG_OCC = "sig95_occ_matched"


def _paired(si, col, occ_col="occ_s"):
    """One row per (animal, island): the region as goal island, and as not.

    The unit of the test. Pairing inside an island is what makes the comparison
    free of island size, shape and position, none of which the animals' goals
    were balanced against.
    """
    g = (si.groupby(["animal", "slot", "repeat", "island", "goal_node"])
           .agg(val=(col, "mean"), occ=(occ_col, "first")).reset_index())
    g["gi"] = [maze.island_of_node(str(q)) if str(q) not in ("", "nan") else np.nan
               for q in g["goal_node"]]
    g = g[g["island"].between(1, 4)].copy()
    g["goal_island"] = g["island"] == g["gi"]
    rows = []
    for (a, k), d in g.groupby(["animal", "island"]):
        on, off = d[d["goal_island"]], d[~d["goal_island"]]
        if on.empty or off.empty:
            continue
        rows.append({"animal": a, "island": k,
                     "on": on["val"].mean(), "off": off["val"].mean(),
                     "occ_on": on["occ"].median(), "occ_off": off["occ"].median(),
                     "n_on": len(on), "n_off": len(off)})
    return pd.DataFrame(rows)


def _paired_panel(ax, pr, ycol_on="on", ycol_off="off", ylabel="", title="",
                  pct=True, note_xy=(0.03, 0.97)):
    """Eight paired lines, one per (animal, island), coloured by island."""
    scale = 100.0 if pct else 1.0
    for _, r in pr.iterrows():
        c = ISLAND_COLOURS[int(r["island"])]
        ls = "-" if r["animal"] == sorted(pr["animal"].unique())[0] else (0, (3, 2))
        ax.plot([0, 1], [r[ycol_off] * scale, r[ycol_on] * scale], color=c,
                lw=_lw(1.3), ls=ls, zorder=3, alpha=0.9)
        ax.plot([0, 1], [r[ycol_off] * scale, r[ycol_on] * scale], "o", ms=4.0,
                color=c, mec="white", mew=_lw(0.9), zorder=4)
    d = (pr[ycol_on] - pr[ycol_off])
    up = int((d > 0).sum())
    p = wilcoxon(pr[ycol_on], pr[ycol_off]).pvalue if len(pr) > 2 else np.nan
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other\nislands", "goal\nisland"],
                       fontsize=style.FS["tick"], color=style.P.INK)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(ylabel, fontsize=style.FS["label"], color=style.P.INK,
                  labelpad=3)
    # Headroom first, then the note: at the first size it sat on the topmost
    # point of the steepest pair.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.26 * (hi - lo))
    ax.text(*note_xy, f"{up}/{len(pr)} up,  median $\\Delta$ = "
            f"{d.median() * scale:+.1f}{'%' if pct else ''},  $p$ = {p:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=style.FS["tick"] - 0.5, color=style.P.INK)
    if title:
        ax.set_title(title, fontsize=style.FS["label"], color=style.P.INK,
                     loc="left", pad=4)
    return up, len(pr), d.median() * scale, p


def example_occupancy(animal=None, date=None, speed=None, bin_cm=2.5,
                      smooth_cm=5.0, cache=True):
    """Occupancy map and per-island dwell seconds for the panel-b session.

    Cached as an .npz beside the tables: it costs an NWB read, and the figure is
    redrawn far more often than the session changes.
    """
    animal = EXAMPLE["animal"] if animal is None else animal
    date = EXAMPLE["date"] if date is None else date
    speed = I.SPEED_THRESH if speed is None else speed
    f = data.out_dir() / f"example_occupancy_{animal}_{date}.npz"
    if cache and f.exists():
        z = np.load(f, allow_pickle=True)
        return z["occ"], {int(k): float(v) for k, v in z["dwell"]}

    from pynwb import NWBHDF5IO
    root = data.ROOT
    hits = [p for p in root.glob(f"*/{date}")
            if p.parent.name.lower().startswith(animal.lower())]
    if not hits:
        raise FileNotFoundError(f"no session {animal} {date} under {root}")
    d = hits[0]
    nwbf = next(p for p in sorted(d.glob("*.nwb"))
                if not p.name.startswith("._") and not p.name.endswith(".tmp.nwb"))
    io = NWBHDF5IO(str(nwbf), "r", load_namespaces=True)
    try:
        nwb = io.read()
        pos = nwbio.load_position(nwb)
        t = pos[2]
        dt = float(np.median(np.diff(t)))
        _tr, wins, _s = SS._trial_windows(d, nwb, t)
        x, y, tt, spd = SS._prep_positions(pos[0] / maze.SCALE_X,
                                           pos[1] / maze.SCALE_Y, t, wins, dt)
    finally:
        io.close()
    ext = maze.MAZE_EXTENT
    bm = bin_cm / 100.0
    bins = (max(5, int(round((ext[1] - ext[0]) / bm))),
            max(5, int(round((ext[3] - ext[2]) / bm))))
    bx_cm, _ = PF.bin_size_cm(ext, bins)
    sigma = float(smooth_cm) / bx_cm
    occ = PF.occupancy_parts(x, y, tt, ext, bins, dt, sigma,
                             speed_thresh=speed, speed=spd)
    isl = maze.bin_islands(ext, bins, radius_m=maze.ISLAND_BIN_RADIUS_M)
    visited = PF.valid_bins(occ["occ_raw"], occ["occ_s"], sigma,
                            PF.DEFAULT_MIN_OCC_S)
    dwell = {int(k): float(occ["occ_raw"][visited & (isl == k)].sum())
             for k in maze.ISLANDS}
    occ_raw = np.where(visited, occ["occ_raw"], np.nan)
    if cache:
        np.savez_compressed(f, occ=occ_raw,
                            dwell=np.array(list(dwell.items()), dtype=float))
    return occ_raw, dwell


def _panel_design(ax):
    """a: the crossover. Each island carries the goal once per animal."""
    pos, G = draw_maze(ax, lw=1.3)
    _maze_axes(ax, pos, pad=0.80)
    _island_numbers(ax, pos, G, dy=0.26, fs=style.FS["tick"])
    from analysis.schematic import GOAL_NODES
    for animal, nodes in GOAL_NODES.items():
        colour = style.ANIMAL_COLOURS.get(animal, style.P.INK)
        xs = [pos[str(n)][0] for n in nodes if str(n) in pos]
        ys = [pos[str(n)][1] for n in nodes if str(n) in pos]
        ax.scatter(xs, ys, s=7.5 ** 2, facecolors="none", edgecolors=colour,
                   linewidths=_lw(1.5), zorder=6, label=animal)
    style.legend(ax, loc="upper left", ncol=1)


def _panel_example(ax, cax=None):
    """b: one session's occupancy, with the dwell time written on each island."""
    occ, dwell = example_occupancy()
    ext = maze.MAZE_EXTENT
    im = ax.imshow(occ, origin="lower", extent=ext, aspect="equal",
                   cmap="magma_r", zorder=2,
                   vmax=float(np.nanpercentile(occ, 99)))
    G = maze.build_graph()
    pos = maze.node_positions_m()
    pos = {str(k): v for k, v in pos.items()}
    _maze_axes(ax, pos, pad=0.42)
    for k in maze.ISLANDS:
        pts = [pos[n] for n in pos if maze.island_of_node(n) == k]
        cx = sum(q[0] for q in pts) / len(pts)
        cy = sum(q[1] for q in pts) / len(pts)
        goal = k == EXAMPLE["island"]
        ax.text(cx, cy, f"{dwell[k]:.0f} s", ha="center", va="center",
                fontsize=style.FS["tick"], zorder=7,
                color=style.P.RED if goal else style.P.INK,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=style.P.RED if goal else "none",
                          lw=_lw(1.2) if goal else 0, alpha=0.88))
    gx, gy = pos[str(EXAMPLE["goal"])]
    ax.plot(gx, gy, "*", ms=11, color=style.P.RED, mec="white",
            mew=_lw(0.9), zorder=8)
    # Inside the axes, not as a title: a title here runs into the panel letter.
    ax.text(0.02, 0.02, f"{EXAMPLE['animal']} {EXAMPLE['slot']}, goal "
            f"{EXAMPLE['goal']} on island {EXAMPLE['island']}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=style.FS["tick"], color=style.P.INK)
    if cax is not None:
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("occupancy (s/bin)", fontsize=style.FS["tick"],
                     color=style.P.INK)
        cb.ax.tick_params(labelsize=style.FS["tick"] - 1, colors=style.P.MUTED)
        cb.outline.set_visible(False)


def _panel_trend(ax, si, occ):
    """f: the across-session rise, before and after the dwell matching.

    The positive control. A procedure that removed the goal-island difference by
    destroying signal would remove this too; it does the opposite.
    """
    slots, meta = data.slot_order(si)
    idx = {s: i for i, s in enumerate(slots)}
    pos = style.slot_positions(meta)
    out = []
    for tag, d, col, colour, dash in (
            ("as recorded", si, SIG_RAW, style.P.MUTED, None),
            ("dwell matched", occ, SIG_OCC, style.P.GREEN_DARK, None)):
        d = d[d["island"].between(1, 4)]
        g = d.groupby("slot")[col].mean()
        y = np.array([100 * g.get(s, np.nan) for s in slots], float)
        good = ~np.isnan(y)
        rho = spearmanr([idx[s] for s in g.index], g.values).correlation
        kw = {"dashes": dash} if dash else {}
        for j, run in enumerate(_runs(meta)):
            k = dict(kw)
            if j == 0:
                k["label"] = f"{tag}  ($\\rho$ = {rho:+.2f})"
            ax.plot(pos[run], y[run], "-", color=colour, lw=_lw(1.5), zorder=3,
                    **k)
        ax.plot(pos[good], y[good], "o", ms=3.2, color=colour, mec="white",
                mew=_lw(0.8), zorder=4)
        out.append((tag, rho))
    style.slot_axis(ax, slots, pos, goals=None)
    ax.set_ylabel("cells above shuffle (%)\nislands 1-4",
                  fontsize=style.FS["label"], color=style.P.INK, labelpad=3)
    ax.set_ylim(bottom=0)
    style.legend(ax, loc="upper left", ncol=1)
    return out


def _runs(meta):
    runs = []
    for i, (rep, _s) in enumerate(meta):
        if runs and runs[-1][0] == rep:
            runs[-1][1].append(i)
        else:
            runs.append((rep, [i]))
    return [ix for _r, ix in runs]


def goal_island_figure(name="fig_goal_island", out_dir=None, si=None, occ=None):
    """The whole argument in one figure. Returns the written paths."""
    from analysis import plots
    si = plots._blocks(data.island_si() if si is None else si)
    occ = data.island_si_occ() if occ is None else occ
    occ["phase"] = occ["phase"].fillna("")
    occ = plots._blocks(occ)

    # H leaves 30 mm under the bottom row: panel f carries rotated slot labels,
    # and at the first size they were clipped to their last three characters,
    # so GL1S1 read as "1S1".
    W, H = 180.0, 196.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM), facecolor="none")

    def axes(x, y, w, h):
        a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        a.set_facecolor("none")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(style.P.MUTED)
            a.spines[s].set_linewidth(style.P.lw(0.8))
        a.tick_params(length=2.5, pad=2, colors=style.P.MUTED,
                      labelcolor=style.P.INK, labelsize=style.FS["tick"])
        a.grid(axis="y", color=style.P.MUTED, lw=style.P.lw(0.4), alpha=0.3,
               zorder=0)
        a.set_axisbelow(True)
        return a

    ax_a = axes(3, 6, 82, 46)
    ax_b = axes(92, 6, 74, 46)
    cax = fig.add_axes([(92 + 75.5) / W, 1 - (6 + 40) / H, 2.6 / W, 28 / H])
    ax_c = axes(16, 68, 40, 44)
    ax_d = axes(74, 68, 40, 44)
    ax_e = axes(132, 68, 40, 44)
    ax_f = axes(16, 128, 156, 38)

    for a, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                   (ax_e, "e"), (ax_f, "f")):
        a.text(-0.08 if a in (ax_c, ax_d, ax_e) else -0.03, 1.10, lab,
               transform=a.transAxes, fontsize=style.FS["title"],
               fontweight="bold", color=style.P.INK, va="top", ha="left")

    for a in (ax_a, ax_b):
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)

    _panel_design(ax_a)
    _panel_example(ax_b, cax)

    pr_occ = _paired(si, SIG_RAW)
    _paired_panel(ax_c, pr_occ, "occ_on", "occ_off",
                  ylabel="dwell time (s)\nmedian per block", pct=False)
    ax_c.set_title("the confound", fontsize=style.FS["label"],
                   color=style.P.INK, loc="left", pad=4)
    d_raw = _paired_panel(ax_d, pr_occ,
                          ylabel="cells above shuffle (%)",
                          title="as recorded")
    pr_m = _paired(si.merge(occ[["animal", "slot", "island", "unit_id",
                                 SIG_OCC]],
                            on=["animal", "slot", "island", "unit_id"],
                            how="inner"), SIG_OCC)
    d_occ = _paired_panel(ax_e, pr_m, ylabel="cells above shuffle (%)",
                          title="dwell matched")
    # d and e are the comparison, so they share an axis.
    top = max(ax_d.get_ylim()[1], ax_e.get_ylim()[1])
    for a in (ax_d, ax_e):
        a.set_ylim(0, top)

    trend = _panel_trend(ax_f, si, occ)

    out = Path(out_dir) if out_dir is not None else data.out_dir()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, extra in ((".pdf", {}), (".png", dict(dpi=600))):
        f = (out / name).with_suffix(suffix)
        fig.savefig(f, transparent=True, **extra)
        written.append(f)
    plt.close(fig)
    print(f"  {name}: d (as recorded) {d_raw[0]}/{d_raw[1]} up, median "
          f"{d_raw[2]:+.1f}%, p={d_raw[3]:.3f};  e (dwell matched) "
          f"{d_occ[0]}/{d_occ[1]} up, median {d_occ[2]:+.1f}%, p={d_occ[3]:.3f}")
    print("    f: " + ";  ".join(f"{t} rho={r:+.2f}" for t, r in trend))
    for f in written:
        print(f"  wrote {f}  ({f.stat().st_size / 1024:.0f} KB)")
    return written
