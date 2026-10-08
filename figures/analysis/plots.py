"""One function per figure. Each writes ONE plot to ONE file.

Adding a figure means adding a function here and naming it in :data:`REGISTRY`;
``python -m analysis`` then draws it.

Path inefficiency is ``log10(actual hops / optimal hops)``: 0 is an optimal run,
0.3 is a route twice as long as it needed to be, and the log makes a two-fold
detour the same distance from optimal wherever on the maze it starts.
"""

import numpy as np

from analysis import data, style

#: Axis label used wherever the detour is plotted, so the figures agree.
DETOUR_LABEL = "path inefficiency\n$\\log_{10}$(actual / optimal hops)"


def _lw(w):
    return style.P.lw(w)


def _goal_runs(df, slots):
    """``[(goal, [slot indices]), ...]`` — runs of consecutive slots sharing a goal.

    The MODAL goal of each slot, not the per-trial value: one mistyped Goal_Node
    (Rat6 GL2S4 has a single 110 among seventeen 109s) would otherwise cut the run
    in three and draw two gaps that mean nothing.

    The runs are what the lines follow, so a line never joins two different goals:
    the route to GL1's goal and the route to GL2's are different tasks, and a
    continuous curve across the switch claims they are one. The "a" block of a
    split day belongs to the PREVIOUS goal's run, which is what it is: the last
    retrievals of the goal the rat already knew.
    """
    mode = df.groupby("slot")["goal_node"].agg(
        lambda v: v.mode().iat[0] if not v.mode().empty else np.nan)
    runs = []
    for i, s in enumerate(slots):
        if s not in mode.index or not np.isfinite(mode[s]):
            continue
        g = int(mode[s])
        if runs and runs[-1][0] == g:
            runs[-1][1].append(i)
        else:
            runs.append((g, [i]))
    return runs


def _mean_sem(df, slots, value="detour"):
    """``(mean, sem, n)`` per slot, in `slots` order, NaN where a slot is empty."""
    g = df.groupby("slot")[value]
    m = np.array([g.get_group(s).mean() if s in g.groups else np.nan for s in slots])
    sd = np.array([g.get_group(s).std(ddof=1) if s in g.groups else np.nan
                   for s in slots])
    n = np.array([len(g.get_group(s)) if s in g.groups else 0 for s in slots])
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.where(n > 1, sd / np.sqrt(n), 0.0)
    return m, sem, n


def _draw(kept, slots, meta, name, ylabel=DETOUR_LABEL, missing=None):
    """The shared panel: one line per goal, mean +/- SEM per animal, GL axis.

    `slots` and `meta` come from the FULL goal-trial set rather than from `kept`,
    so the three figures in this folder share one x axis and can be stacked or
    flipped between without the slots moving.

    `missing` is ``{animal: boolean array over slots}`` for trials that were RUN
    but could not be scored. They are drawn as crosses along the top rather than
    left out: a trial the rat failed to finish is the worst outcome there is, and
    a figure that drops it shows a performance better than the one it measured.
    """
    pos = style.slot_positions(meta)
    fig, ax = style.new_figure(height_mm=72.0)
    for animal, colour in style.ANIMAL_COLOURS.items():
        d = kept[kept["animal"] == animal]
        if d.empty:
            continue
        m, sem, n = _mean_sem(d, slots)
        if np.any(sem > 0):
            ax.errorbar(pos, m, yerr=sem, fmt="none", ecolor=colour,
                        elinewidth=_lw(0.9), capsize=2.2, capthick=_lw(0.9),
                        zorder=2)
        for j, (_goal, idx) in enumerate(_goal_runs(d, slots)):
            ii = np.asarray(idx, int)
            ax.plot(pos[ii], m[ii], "-", color=colour, lw=_lw(1.7), zorder=3,
                    label=animal if j == 0 else None)
        # HOLLOW where the trials ran to the goal the rat already knew: on a split
        # day the "a" block is the tail of the previous goal's curve, and a filled
        # marker there reads as one more point on the new goal's.
        old = np.array([s.endswith("a") for s in slots])
        ax.plot(pos[~old], m[~old], "o", ms=4.4, color=colour, mec="white",
                mew=_lw(1.3), zorder=4)
        ax.plot(pos[old], m[old], "o", ms=4.4, mfc=style.P.SURFACE, mec=colour,
                mew=_lw(1.4), zorder=4)
    ax.axhline(0, color=style.P.MUTED, lw=_lw(0.8), ls=(0, (4, 3)), zorder=1)
    if missing:
        top = ax.get_ylim()[1]
        ax.set_ylim(top=top * 1.12)
        y = ax.get_ylim()[1] * 0.955
        for animal, mask in missing.items():
            mask = np.asarray(mask, bool)
            if mask.any():
                ax.plot(pos[mask], np.full(mask.sum(), y), "x", ms=4.6,
                        mew=_lw(1.5), color=style.ANIMAL_COLOURS[animal],
                        ls="none", zorder=5)
        ax.plot([], [], "x", ms=4.6, mew=_lw(1.5), color=style.P.MUTED,
                ls="none", label="goal not reached")
    style.slot_axis(ax, slots, pos)
    ax.plot([], [], "o", ms=4.4, mfc=style.P.SURFACE, mec=style.P.MUTED,
            mew=_lw(1.4), ls="none", label="to the previous goal")
    ax.set_ylabel(ylabel, fontsize=style.FS["label"], color=style.P.INK, labelpad=3)
    ax.set_ylim(bottom=min(-0.02, ax.get_ylim()[0]))
    style.legend(ax, loc="upper right")
    return style.save(fig, name)


def _report(name, notes, kept, slots, extra="", warn_small=True):
    print(f"  {name}: {notes['scored']}/{notes['goal_trials']} goal trials scored; "
          f"{notes['goal_never_reached']} dropped (goal never reached), "
          f"{notes['clamped_below_zero']} clamped to 0. Habituation and the "
          f"free-roaming periods are out. a/b = goal trials before/after the goal "
          f"switched that day.{extra}")
    if not warn_small:
        return
    short = []
    for animal in style.ANIMAL_COLOURS:
        d = kept[kept["animal"] == animal]
        per = d.groupby("slot").size()
        short += [f"{s} {animal} n={int(per.get(s, 0))}" for s in slots
                  if int(per.get(s, 0)) < 4]
    if short:
        print("    slots with fewer than 4 trials behind the point: "
              + "; ".join(short))


def detour_by_session(trials, name="detour_by_session"):
    """Path inefficiency per session, every goal trial, mean +/- SEM per animal."""
    kept, notes = data.scorable(trials)
    slots, meta = data.slot_order(kept)
    _report(name, notes, kept, slots)
    return _draw(kept, slots, meta, name)


def detour_first_trial(trials, name="detour_first_trial"):
    """The FIRST goal trial of every block: what the rat did before relearning.

    One trial per block per animal, so there is nothing to average and no error
    bar is drawn. That is the point of the figure: the first run after a break is
    a single measurement, and showing it with a spurious interval would invite the
    reader to treat it as a mean.
    """
    kept_all, notes = data.scorable(trials)
    slots, meta = data.slot_order(kept_all)
    first = data.goal_trials(trials)
    first = first[first["trial_rank"] == 1]
    kept = first[first["detour"].notna()]
    # Which first trials were RUN but never reached the goal. 20.5% of them, against
    # 2.6% of every later trial: the first run of a block is the one the rat fails,
    # so these are the points a gap would hide.
    missing = {}
    for animal in style.ANIMAL_COLOURS:
        d = first[(first["animal"] == animal) & (first["detour"].isna())]
        bad = set(d["slot"])
        missing[animal] = np.array([s in bad for s in slots])
    n_missing = int(sum(m.sum() for m in missing.values()))
    _report(name, notes, kept, slots, warn_small=False,
            extra=f" Trial 1 of each block only: n=1 per point, so no SEM. "
                  f"{len(kept)} first trials scored, {n_missing} run but never "
                  f"reached the goal (crosses).")
    return _draw(kept, slots, meta, name, missing=missing,
                 ylabel="first trial of the block\n"
                        "$\\log_{10}$(actual / optimal hops)")


def detour_trials_2_5(trials, name="detour_trials_2_5"):
    """Goal trials 2 to 5 of every block: the first few runs after trial 1.

    Paired with :func:`detour_first_trial`, this is the within-session relearning
    the first trial does not show. Blocks shorter than five goal trials, which is
    every "b" block, contribute what they have and say so.
    """
    kept_all, notes = data.scorable(trials)
    slots, meta = data.slot_order(kept_all)
    kept = kept_all[kept_all["trial_rank"].between(2, 5)]
    _report(name, notes, kept, slots, extra=" Goal trials 2-5 of each block only.")
    return _draw(kept, slots, meta, name,
                 ylabel="trials 2-5 of the block\n"
                        "$\\log_{10}$(actual / optimal hops)")


#: name -> function. ``python -m analysis`` draws these.
REGISTRY = {
    "detour_by_session": detour_by_session,
    "detour_first_trial": detour_first_trial,
    "detour_trials_2_5": detour_trials_2_5,
}
