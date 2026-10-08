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


def _substituted(kept, slots, nominal):
    """Which slots drew on a trial later than the nominal window, and by how much.

    ``(mask_by_animal, lines)``. A point is a stand-in when the LAST trial behind
    it sits beyond `nominal` in block order: the window counts successes, so a
    block whose early trials failed reaches further into the session to fill it.
    """
    masks, lines = {}, []
    for animal in style.ANIMAL_COLOURS:
        d = kept[kept["animal"] == animal]
        worst = d.groupby("slot")["trial_rank"].max()
        flag = {s: bool(worst.get(s, 0) > nominal) for s in slots}
        masks[animal] = np.array([flag[s] for s in slots])
        for s in slots:
            if flag[s]:
                ranks = sorted(d[d["slot"] == s]["trial_rank"].astype(int))
                lines.append(f"{animal} {s}: trial(s) "
                             f"{', '.join(map(str, ranks))} of the block")
    return masks, lines


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


def _draw(kept, slots, meta, name, ylabel=DETOUR_LABEL, missing=None, ref=None,
          substituted=None):
    """The shared panel: one line per goal, mean +/- SEM per animal, GL axis.

    `slots` and `meta` come from the FULL goal-trial set rather than from `kept`,
    so the three figures in this folder share one x axis and can be stacked or
    flipped between without the slots moving.

    `ref` is the frame the goal runs are read from, and defaults to `kept`. Pass
    the FULL goal-trial table when `kept` is a subset: runs taken from a subset
    omit the slots the subset happens to be missing, and the line then bridges a
    gap instead of breaking at it, which draws a continuity the data does not have.

    `substituted` is ``{animal: boolean array over slots}`` for points that did
    not come from the nominal trial because that trial failed. They get a ring,
    so a reader can see at a glance which points are stand-ins: without it the
    figure would claim a first-trial route the rat never ran.

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
        r = (ref if ref is not None else kept)
        r = r[r["animal"] == animal]
        for j, (_goal, idx) in enumerate(_goal_runs(r, slots)):
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
        if substituted is not None:
            sub = np.asarray(substituted.get(animal, []), bool)
            if sub.size and sub.any():
                ax.plot(pos[sub], m[sub], "o", ms=8.2, mfc="none", mec=colour,
                        mew=_lw(0.9), alpha=0.75, zorder=5)
    if substituted is not None:
        ax.plot([], [], "o", ms=8.2, mfc="none", mec=style.P.MUTED, mew=_lw(0.9),
                ls="none", label="earlier trials failed")
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
    return _draw(kept, slots, meta, name, ref=data.goal_trials(trials))


def detour_first_trial(trials, name="detour_first_trial"):
    """The FIRST goal trial of every block: what the rat did before relearning.

    One trial per block per animal, so there is nothing to average and no error
    bar is drawn. That is the point of the figure: the first run after a break is
    a single measurement, and showing it with a spurious interval would invite the
    reader to treat it as a mean.
    """
    kept_all, notes = data.scorable(trials)
    slots, meta = data.slot_order(kept_all)
    kept = kept_all[kept_all["success_rank"] == 1]
    sub, where = _substituted(kept, slots, nominal=1)
    _report(name, notes, kept, slots, warn_small=False,
            extra=f" The FIRST SUCCESSFUL trial of each block: n=1 per point, so "
                  f"no SEM. {len(kept)} points, {sum(int(v.sum()) for v in sub.values())} "
                  f"of them from a later trial because the earlier ones failed "
                  f"(ringed). How late is in trials_to_first_success.")
    for line in where:
        print(f"    {line}")
    return _draw(kept, slots, meta, name, ref=data.goal_trials(trials),
                 substituted=sub,
                 ylabel="first successful trial\n"
                        "$\\log_{10}$(actual / optimal hops)")


def detour_trials_2_5(trials, name="detour_trials_2_5"):
    """Goal trials 2 to 5 of every block: the first few runs after trial 1.

    Paired with :func:`detour_first_trial`, this is the within-session relearning
    the first trial does not show. Blocks shorter than five goal trials, which is
    every "b" block, contribute what they have and say so.
    """
    kept_all, notes = data.scorable(trials)
    slots, meta = data.slot_order(kept_all)
    kept = kept_all[kept_all["success_rank"].between(2, 5)]
    sub, where = _substituted(kept, slots, nominal=5)
    ref = data.goal_trials(trials)
    _report(name, notes, kept, slots,
            extra=" Successful trials 2-5 of each block. Where earlier trials "
                  "failed these come from later in the block (ringed).")
    for line in where:
        print(f"    {line}")
    return _draw(kept, slots, meta, name, ref=ref, substituted=sub,
                 ylabel="successful trials 2-5\n"
                        "$\\log_{10}$(actual / optimal hops)")




def _wilson(k, n, z=1.96):
    """Wilson 95% interval for k successes in n, as ``(lo, hi)`` arrays.

    Not k/n +/- the normal interval: at n=4 and k=4 that interval has zero width
    and claims certainty the four trials cannot carry. Wilson keeps the interval
    inside [0, 1] and stays honest at the ends, which is where every interesting
    point in this figure sits.
    """
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / d
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def _block_series(comp, slots, animal, column):
    """`column` per slot for one animal, NaN where that block is absent."""
    d = comp[comp["animal"] == animal].set_index("slot")
    return np.array([d[column].get(s, np.nan) for s in slots], float)


def _draw_blocks(comp, trials, slots, meta, name, ylabel, column, err=None,
                 ref_line=None, integer_y=False, legend_loc="upper right"):
    """One value per block per animal on the shared GL axis.

    These are block-level counts, not means of trials, so there is no SEM; where
    an interval is meaningful it is passed in as `err`.
    """
    pos = style.slot_positions(meta)
    fig, ax = style.new_figure(height_mm=66.0)
    ref = data.goal_trials(trials)
    for animal, colour in style.ANIMAL_COLOURS.items():
        y = _block_series(comp, slots, animal, column)
        if np.all(np.isnan(y)):
            continue
        if err is not None:
            lo, hi = err(comp, slots, animal)
            good = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
            # The interval is drawn as a SEGMENT, not as arms either side of the
            # point: a Wilson interval is not centred on k/n, and at k=n its whole
            # width lies below the point. Arms would have to be clipped to stay
            # positive, which silently redraws the interval as something else.
            ax.vlines(pos[good], lo[good], hi[good], color=colour,
                      lw=_lw(0.9), zorder=2)
            for yy in (lo, hi):
                ax.plot(pos[good], yy[good], "_", ms=3.6, mew=_lw(0.9),
                        color=colour, ls="none", zorder=2)
        r = ref[ref["animal"] == animal]
        for j, (_goal, idx) in enumerate(_goal_runs(r, slots)):
            ii = np.asarray(idx, int)
            ax.plot(pos[ii], y[ii], "-", color=colour, lw=_lw(1.7), zorder=3,
                    label=animal if j == 0 else None)
        old = np.array([s.endswith("a") for s in slots])
        ax.plot(pos[~old], y[~old], "o", ms=4.4, color=colour, mec="white",
                mew=_lw(1.3), zorder=4)
        ax.plot(pos[old], y[old], "o", ms=4.4, mfc=style.P.SURFACE, mec=colour,
                mew=_lw(1.4), zorder=4)
    if ref_line is not None:
        ax.axhline(ref_line, color=style.P.MUTED, lw=_lw(0.8), ls=(0, (4, 3)),
                   zorder=1)
    style.slot_axis(ax, slots, pos)
    ax.plot([], [], "o", ms=4.4, mfc=style.P.SURFACE, mec=style.P.MUTED,
            mew=_lw(1.4), ls="none", label="to the previous goal")
    ax.set_ylabel(ylabel, fontsize=style.FS["label"], color=style.P.INK, labelpad=3)
    if integer_y:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # The legend goes where the DATA is not: these series sit against the top of
    # the panel, so the default upper-right box lands on the points it explains.
    style.legend(ax, loc=legend_loc)
    return style.save(fig, name)


def success_rate(trials, name="success_rate"):
    """How often the goal was reached in each block, with Wilson 95% intervals.

    The companion to the detour figures, which can only draw the trials that
    SUCCEEDED: a block where the rat failed two thirds of its runs scores no worse
    there than one where it never failed, because the failures have no route
    efficiency to plot. This is where they live.
    """
    comp, notes = data.completion(trials)
    kept, _ = data.scorable(trials)
    slots, meta = data.slot_order(kept)

    def err(c, sl, animal):
        n = _block_series(c, sl, animal, "n")
        p = _block_series(c, sl, animal, "success_rate")
        return _wilson(p * n, n)

    print(f"  {name}: {notes['scored']}/{notes['goal_trials']} goal trials usable "
          f"({notes['tracking_dropouts']} recordings stopped early and are out of "
          f"both numerator and denominator). "
          f"{notes['blocks_with_a_failure']}/44 blocks contain a failure. "
          f"Bars are Wilson 95% intervals; the b blocks are short, so theirs are wide.")
    worst = comp.nsmallest(4, "success_rate")[["animal", "slot", "n", "success_rate"]]
    for r in worst.itertuples(index=False):
        print(f"    lowest: {r.animal} {r.slot} {r.success_rate:.2f} (n={int(r.n)})")
    return _draw_blocks(comp, trials, slots, meta, name,
                        "goal reached\n(fraction of trials in the block)",
                        "success_rate", err=err, ref_line=1.0,
                        legend_loc="lower left")


def trials_to_first_success(trials, name="trials_to_first_success"):
    """Which trial of the block first reached the goal. 1 means straight away.

    A single count per block, so no interval: the quantity is "how many runs did
    this take", and an error bar on one observation would be an invention.
    """
    comp, notes = data.completion(trials)
    kept, _ = data.scorable(trials)
    slots, meta = data.slot_order(kept)
    print(f"  {name}: {notes['blocks_needing_more_than_one']}/44 blocks took more "
          f"than one trial; {notes['blocks_never_solved']} were never solved.")
    slow = comp[comp["first_success"] > 1][["animal", "slot", "first_success", "n"]]
    for r in slow.sort_values("first_success", ascending=False).itertuples(index=False):
        print(f"    {r.animal} {r.slot}: first success on trial "
              f"{int(r.first_success)} of {int(r.n)}")
    return _draw_blocks(comp, trials, slots, meta, name,
                        "first trial that reached the goal",
                        "first_success", ref_line=1.0, integer_y=True,
                        legend_loc="upper left")


#: name -> function. ``python -m analysis`` draws these.
REGISTRY = {
    "detour_by_session": detour_by_session,
    "detour_first_trial": detour_first_trial,
    "detour_trials_2_5": detour_trials_2_5,
    "success_rate": success_rate,
    "trials_to_first_success": trials_to_first_success,
}
