"""One function per figure. Each writes ONE plot to ONE file.

Adding a figure means adding a function here and naming it in :data:`REGISTRY`;
``python -m analysis`` then draws it.

Path inefficiency is ``log10(actual hops / optimal hops)``: 0 is an optimal run,
0.3 is a route twice as long as it needed to be, and the log makes a two-fold
detour the same distance from optimal wherever on the maze it starts.
"""

import numpy as np

from hm_rat_analysis import islands

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
          substituted=None, goals=None):
    """The shared panel: one line per goal, mean +/- SEM per animal, GL axis.

    `slots` and `meta` come from the FULL goal-trial set rather than from `kept`,
    so the three figures in this folder share one x axis and can be stacked or
    flipped between without the slots moving.

    `goals` is ``{slot: node}``; it defaults to the goal nodes of `ref`, so a
    figure drawn from a subset of the trials still labels every slot on the axis.

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
    fig, ax = style.new_figure(height_mm=78.0, bottom_mm=26.0)
    if goals is None:
        goals, goal_animals = data.slot_goals(ref if ref is not None else kept)
    else:
        goals, goal_animals = goals
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
    style.slot_axis(ax, slots, pos, goals=goals, animals=goal_animals)
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
    # Only the animals this figure actually draws: on a per-animal figure the
    # other one is absent by construction, and reporting its 22 empty slots as
    # thin data would bury the real warnings.
    present = set(kept["animal"].dropna().unique())
    for animal in [a for a in style.ANIMAL_COLOURS if a in present]:
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
    fig, ax = style.new_figure(height_mm=72.0, bottom_mm=26.0)
    ref = data.goal_trials(trials)
    goals, goal_animals = data.slot_goals(ref)
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
    style.slot_axis(ax, slots, pos, goals=goals, animals=goal_animals)
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




#: The regions in the order they sit along the maze, then the bridges. Node ids
#: run right to left, so the legend says WHERE each island is rather than leaving
#: "1" to be guessed from the number.
ISLAND_ORDER = (4, 3, 2, 1, 5)
ISLAND_WHERE = {4: "island 4 (left)", 3: "island 3 (centre-left)",
                2: "island 2 (centre-right)", 1: "island 1 (right)",
                5: "bridges (5 pooled)"}
ISLAND_COLOURS = {4: style.P.BLUE, 3: style.P.GREEN_DARK,
                  2: style.P.AMBER_INK, 1: style.P.RED, 5: style.P.INK}
#: The bridges are drawn dashed: they are corridors between the islands, not one
#: of them, and a fifth solid line would read as a fifth island.
ISLAND_DASH = {5: (3.5, 2.0)}


def _blocks(si):
    """The island table reduced to ONE segment per session-block.

    islands.py returns each day the goal changed three times: whole, and each
    side of the switch. A day that was split must be read as its two halves (the
    "a" trials still run to the previous goal) and every other day as itself, so
    the axis here is the same 22 blocks the behavioural figures use rather than
    19 sessions with three of them straddling two goals.
    """
    si = si[si["repeat"] > data.HABITUATION_REPEAT].copy()
    si["phase"] = si["phase"].fillna("")
    split = set(map(tuple, si.loc[si["phase"].isin(("a", "b")),
                                  ["animal", "date"]].drop_duplicates().values))
    key = list(zip(si["animal"], si["date"]))
    want_split = np.array([k in split for k in key])
    keep = np.where(want_split, si["phase"].isin(("a", "b")), si["phase"] == "")
    return si[keep].copy()


def _si_by_slot(df, slots, island, column="spatial_info"):
    """Median `column` per slot for one island, with the IQR and the unit count."""
    d = df[df["island"] == island]
    g = d.groupby("slot")[column]

    def per_slot(fn):
        return np.array([fn(g.get_group(s)) if s in g.groups else np.nan
                         for s in slots], float)

    # Median, not mean: Skaggs bits/spike has a long right tail made of the cells
    # with the fewest spikes, and a mean follows that tail rather than the session.
    med = per_slot(lambda v: v.median())
    q1 = per_slot(lambda v: v.quantile(0.25))
    q3 = per_slot(lambda v: v.quantile(0.75))
    n = per_slot(lambda v: v.notna().sum())
    return med, q1, q3, np.nan_to_num(n).astype(int)


def _repeat_runs(meta):
    """``[[slot indices], ...]`` — one run per repeat, in axis order.

    The lines follow these, so a line never joins two repeats. Note this is NOT
    :func:`_goal_runs`, which the behavioural figures use: that groups by the
    GOAL NODE, which puts the "a" block of a split day in the PREVIOUS repeat's
    run because those trials still run to the old goal. Here the run is the
    repeat, so GL2S1a opens GL2 rather than closing GL1. The rate maps are the
    reason: a block's spatial information is a property of the recording session
    and its position in the training schedule, not of which goal the trials in it
    happened to target, and the question this figure answers is how coding
    changed over the schedule.
    """
    runs = []
    for i, (rep, _ses) in enumerate(meta):
        if runs and runs[-1][0] == rep:
            runs[-1][1].append(i)
        else:
            runs.append((rep, [i]))
    return [idx for _rep, idx in runs]


def _si_lines(ax, si, slots, pos, runs, columns, dash_regions=None, label=None):
    """Draw the whole-maze line and one line per region, broken between repeats.

    `columns` maps region -> the column to read, because the matched figure reads
    a DIFFERENT column per region (each is thinned to its own count). `label`
    maps region -> legend text, and the label goes on the FIRST run only, or the
    legend would carry one entry per repeat per region.
    """
    dash_regions = ISLAND_DASH if dash_regions is None else dash_regions
    label = label or {}
    for isl in (0,) + ISLAND_ORDER:
        colour = style.P.MUTED if isl == 0 else ISLAND_COLOURS[isl]
        med, q1, q3, _n = _si_by_slot(si, slots, isl, columns[isl])
        good = ~np.isnan(med)
        if isl:
            ax.vlines(pos[good], q1[good], q3[good], color=colour, lw=_lw(0.7),
                      alpha=0.55, zorder=3)
        # `dashes` cannot be passed as None for a solid line, so the kwarg is
        # only present for the regions that want one.
        dash = {"dashes": dash_regions[isl]} if isl in dash_regions else {}
        for j, run in enumerate(runs):
            kw = dict(dash)
            if j == 0 and isl in label:
                kw["label"] = label[isl]
            ax.plot(pos[run], med[run], "-", color=colour,
                    lw=_lw(1.4), zorder=2 if isl == 0 else 4, **kw)
        if isl:
            ax.plot(pos[good], med[good], "o", ms=3.4, color=colour, mec="white",
                    mew=_lw(0.9), zorder=5)


def _significant(si):
    """Rows whose cell beat its OWN shuffle in THAT region, at 95%.

    Per region, not per cell: a cell can carry position on one island and not on
    the next, and that is the thing these figures are about. Filtering on a
    whole-maze test instead would let a cell into the island 3 line on the
    strength of a field on island 1.

    The threshold is a one-sided 95th percentile of 500 circular shifts, so about
    5% of the rows that pass are expected to be cells with no spatial coding at
    all. With thousands of cell-blocks that is tens of false positives per line,
    which is why these figures report the pass RATE: a line whose pass rate is
    near 5% is a line of noise, whatever its median says.
    """
    if "sig95" not in si.columns:
        raise KeyError("no sig95 column: rebuild the island table with "
                       "`python -m analysis island_spatial_info --rebuild`")
    return si[si["sig95"].astype(bool)].copy()


def _pass_rates(si, name):
    """Print the fraction of cell-blocks that beat their own shuffle, per region."""
    n_null = int(si["n_null"].max()) if "n_null" in si.columns else 0
    print(f"  {name}: cell-blocks above their own 95th percentile "
          f"({n_null} circular shifts):")
    g = si.groupby("island")
    for k in (0,) + ISLAND_ORDER:
        if k not in g.groups:
            continue
        d = g.get_group(k)
        ok = d["sig95"].astype(bool)
        where = "whole maze" if k == 0 else ISLAND_WHERE[k]
        print(f"    {where:24} {int(ok.sum()):5d}/{len(d):5d} = "
              f"{100 * ok.mean():4.0f}%   median p = {d['si_null_p'].median():.3f}")
    print("    5% is what no spatial coding at all would give.")


def _si_axis(si):
    """``(si, slots, meta, pos, goals, animals)`` shared by the SI figures."""
    si = _blocks(si)
    slots, meta = data.slot_order(si)
    goals, animals = data.slot_goals(si)
    return si, slots, meta, style.slot_positions(meta), goals, animals


def _si_exposure(si, name, column, header=True):
    """Print what each value in a figure rests on, per region."""
    g = si.groupby("island")
    if header:
        print(f"  {name}: good pyramidal only, no spike floor, positions gated at "
              f"{islands.SPEED_THRESH:.2f} m/s; values from {column}.")
    for k in (0,) + ISLAND_ORDER:
        if k not in g.groups:
            continue
        d = g.get_group(k)
        got = int(d[column].notna().sum())
        print(f"    {('whole maze' if k == 0 else ISLAND_WHERE[k]):24} "
              f"n={got:5d}/{len(d):5d} cell-blocks   "
              f"median spikes {int(d['n_spikes'].median()):5d}   "
              f"median occupancy {d['occ_s'].median():6.0f} s")


ISLAND_ORDER = (4, 3, 2, 1, 5)
ISLAND_WHERE = {4: "island 4 (left)", 3: "island 3 (centre-left)",
                2: "island 2 (centre-right)", 1: "island 1 (right)",
                5: "bridges (5 pooled)"}
ISLAND_SHORT = {0: "whole", 4: "isl 4", 3: "isl 3", 2: "isl 2", 1: "isl 1",
                5: "bridges"}
ISLAND_COLOURS = {4: style.P.BLUE, 3: style.P.GREEN_DARK,
                  2: style.P.AMBER_INK, 1: style.P.RED, 5: style.P.INK}
ISLAND_DASH = {5: (3.5, 2.0)}

#: Spikes each region's count-matched estimate is thinned to. Skaggs bits/spike
#: is biased upward at low counts and the counts here FALL as trials shorten with
#: learning: corr(session order, log10 spikes) = -0.67 to -0.71 and
#: corr(log10 spikes, raw SI) = -0.93, which is enough to manufacture the whole
#: cross-session rise on its own.
#:
#: It does not, as it turns out. Matching at every available level leaves the
#: rise in place in every region (Spearman rho of the per-block median against
#: session order: +0.33 to +0.68, against +0.52 to +0.83 raw). What matching
#: changes is the SIZE: pooled over the four islands the medians go 0.56 -> 1.86
#: across the four goals raw, 1.05 -> 1.60 matched at 40 and 0.29 -> 0.79 at 200.
#: So the direction is a result and the amplitude is not.
#:
#: Each level is chosen for COVERAGE, not for being the highest possible: a
#: cell-block below the level is dropped, and since the cells with the most
#: spikes in a region are the cells with fields there, a high level selects for
#: exactly the population the figure is about. Under the 0.05 m/s gate and with
#: no spike floor the island medians are 106-149 spikes, so 40 keeps 67-72% of
#: island cell-blocks where 100 keeps 51-59% and 200 keeps 35-43%.
#:
#: 40 is also the only level the BRIDGES can reach at all (40% of their
#: cell-blocks; their median is 20 spikes), and putting the islands there too
#: buys something the earlier version could not offer: all five regions are
#: matched at the SAME count, so they may be read against each other. Only the
#: whole-maze line stands apart, at 200.
ISLAND_MATCH = {0: 200, 1: 40, 2: 40, 3: 40, 4: 40, 5: 40}


def _matched_column(island):
    return f"si_m{ISLAND_MATCH[island]}"


def island_spatial_info(trials_unused=None, name="island_spatial_info", si=None,
                        sig_only=False):
    """Spatial information per block, computed inside each island separately.

    Four solid lines, one per island, plus a dashed line for the five
    inter-island corridors pooled; each is the median over good pyramidal cells
    with the interquartile range. The whole-maze value is the grey line: it sits
    ABOVE every region because it also carries WHICH region a cell fired on,
    which is information about the maze rather than about position within a block
    of it.

    These are the RAW values, and the rise across sessions in them is not safe to
    read: spike counts fall by a factor of three or four over the same axis and
    Skaggs is biased upward at low counts. ``island_spatial_info_matched`` is the
    same figure with the counts held fixed, and it is the one a claim about
    learning has to rest on. This one is kept because it is what the unmatched
    statistic actually looks like, and because the two together show how much of
    the trend was exposure.

    The bridges are pooled because one short bridge is about fourteen bins and
    under 1% of a session; the five together are ~200 bins and ~7%, which is
    enough for an estimate, and they are the only part of the maze where position
    is one-dimensional.
    """
    si = data.island_si() if si is None else si
    if sig_only:
        _pass_rates(_blocks(si), name)
        si = _significant(si)
    si, slots, meta, pos, goals, goal_animals = _si_axis(si)
    fig, ax = style.new_figure(height_mm=80.0, bottom_mm=26.0)

    label = {0: "whole maze", **ISLAND_WHERE}
    _si_lines(ax, si, slots, pos, _repeat_runs(meta),
              {k: "spatial_info" for k in (0,) + ISLAND_ORDER}, label=label)
    style.slot_axis(ax, slots, pos, goals=goals, animals=goal_animals)
    ax.set_ylabel("spatial information\n(bits/spike, median over cells)",
                  fontsize=style.FS["label"], color=style.P.INK, labelpad=3)
    ax.set_ylim(bottom=0)
    style.legend(ax, loc="upper left", ncol=2)
    _si_exposure(si, name, "spatial_info")
    print("    RAW bits/spike, no spike floor: Skaggs is biased upward at low "
          "counts, so the cells with the fewest spikes sit highest here, and "
          "spike counts also fall with learning. The TREND survives matching "
          "(rho +0.46 to +0.69 against +0.58 to +0.67 raw) but the amplitude "
          "does not, so quote island_spatial_info_matched for a size.")
    return style.save(fig, name)


def island_spatial_info_matched(trials_unused=None,
                                name="island_spatial_info_matched", si=None,
                                sig_only=False):
    """The same figure with every estimate thinned to a fixed number of spikes.

    Every region is thinned to 40 spikes and the whole maze to 200 (see
    ``ISLAND_MATCH`` for why those), averaged over eight draws. Within a line the
    count is constant, so a change along x is a change in how sharply the cells
    coded position and not a change in how much data there was. The four islands
    and the bridges share a count, so they may also be read against EACH OTHER,
    which the raw figure does not allow. The whole-maze line is matched at 200
    and carries less upward bias than the five below it, so the gap between it
    and them is a floor on the real gap rather than a measure of it.

    A gap in a line is a block where too few cells reached the match count.

    With the spike floor gone this figure and the raw one can disagree about more
    than scale: the raw medians now include cells with a handful of spikes, which
    the matching drops, and those cells sit at the top of the raw distribution.
    """
    si = data.island_si() if si is None else si
    if sig_only:
        _pass_rates(_blocks(si), name)
        si = _significant(si)
    si, slots, meta, pos, goals, goal_animals = _si_axis(si)
    fig, ax = style.new_figure(height_mm=80.0, bottom_mm=26.0)

    label = {0: f"whole maze ({ISLAND_MATCH[0]} spikes)",
             **{k: f"{v} ({ISLAND_MATCH[k]})" for k, v in ISLAND_WHERE.items()}}
    _si_lines(ax, si, slots, pos, _repeat_runs(meta),
              {k: _matched_column(k) for k in (0,) + ISLAND_ORDER}, label=label)
    style.slot_axis(ax, slots, pos, goals=goals, animals=goal_animals)
    ax.set_ylabel("spatial information\n(bits/spike at matched count)",
                  fontsize=style.FS["label"], color=style.P.INK, labelpad=3)
    ax.set_ylim(bottom=0)
    style.legend(ax, loc="upper left", ncol=2)
    print(f"  {name}: good pyramidal only, no spike floor, positions gated at "
          f"{islands.SPEED_THRESH:.2f} m/s; each region thinned to its own count.")
    for k in (0,) + ISLAND_ORDER:
        _si_exposure(si[si["island"] == k], name, _matched_column(k), header=False)
    print("    The four islands and the bridges are matched at the same count, "
          "so they are comparable to each other as well as along x; the "
          "whole-maze line is matched higher and is not comparable to them.")
    return style.save(fig, name)


def island_spatial_info_bars(trials_unused=None, name="island_spatial_info_bars",
                             si=None, column=None, sig_only=False):
    """Count-matched spatial information per region, one bar per goal.

    Nineteen slots times six lines is too much to read a trend off, so this
    collapses the sessions of each goal into one bar: five groups of four, which
    is the cross-goal comparison at a glance. What it gives up is the
    within-goal time course, which the line figures keep.

    Bars are the median over cell-blocks of the matched estimate, whiskers the
    interquartile range, and the number inside each bar is how many cell-blocks
    it rests on. The islands and the bridges share a match count, so their bars
    are comparable across groups as well as within one; the whole-maze group is
    matched higher and stands on its own.
    """
    si = data.island_si() if si is None else si
    if sig_only:
        _pass_rates(_blocks(si), name)
        si = _significant(si)
    si = _blocks(si)
    goals = sorted(si["repeat"].unique())
    width = 0.80 / max(1, len(goals))
    fig, ax = style.new_figure(width_mm=180.0, height_mm=82.0, bottom_mm=24.0)
    shades = [style.P.BLUE, style.P.GREEN_DARK, style.P.AMBER_INK, style.P.RED,
              style.P.INK]

    regions = (0,) + ISLAND_ORDER
    for gi, rep in enumerate(goals):
        d = si[si["repeat"] == rep]
        xs, med, lo, hi, ns = [], [], [], [], []
        for ri, k in enumerate(regions):
            col = _matched_column(k)
            v = d.loc[d["island"] == k, col].dropna()
            xs.append(ri - 0.40 + width * (gi + 0.5))
            med.append(v.median() if len(v) else np.nan)
            lo.append(v.quantile(0.25) if len(v) else np.nan)
            hi.append(v.quantile(0.75) if len(v) else np.nan)
            ns.append(len(v))
        colour = shades[gi % len(shades)]
        ax.bar(xs, med, width=width * 0.92, color=colour, alpha=0.80, lw=0,
               zorder=3, label=f"goal {int(rep)}")
        ax.vlines(xs, lo, hi, color=style.P.INK, lw=_lw(0.8), alpha=0.75, zorder=4)
        # Inside the bar, not under it: under the axis they collide with the
        # region label, and a count belongs to its own bar rather than to the
        # group.
        for x, n in zip(xs, ns):
            ax.text(x, 0.015, str(n), ha="center", va="bottom", rotation=90,
                    fontsize=style.FS["tick"] - 1, color="white",
                    transform=ax.get_xaxis_transform(), zorder=6)

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([f"{ISLAND_SHORT[k]}\n({ISLAND_MATCH[k]} spk)"
                        for k in regions], fontsize=style.FS["tick"],
                       color=style.P.INK)
    ax.set_xlim(-0.6, len(regions) - 0.4)
    ax.set_ylabel("spatial information\n(bits/spike at matched count)",
                  fontsize=style.FS["label"], color=style.P.INK, labelpad=3)
    ax.set_ylim(bottom=0)
    style.legend(ax, loc="upper right", ncol=2)
    print(f"  {name}: bars are medians of the count-matched estimate over "
          f"cell-blocks, whiskers the IQR, numbers inside the bars the "
          f"cell-block count. Islands and bridges share a match count; the "
          f"whole-maze group does not.")
    nodes = (si.groupby("repeat")["goal_node"].agg(
        lambda v: "/".join(sorted(set(str(x) for x in v.dropna() if str(x))))))
    print("    goal nodes: "
          + ", ".join(f"goal {int(k)} -> {v}" for k, v in nodes.items()))
    return style.save(fig, name)


def island_sig_fraction(trials_unused=None, name="island_sig_fraction", si=None):
    """The FRACTION of cells that beat their own shuffle, per block and region.

    This is where the effect in this dataset actually lives. Restricted to the
    cells that do code position, the information each one carries barely changes
    across the schedule (per-island Spearman rho -0.08 to +0.35 against +0.52 to
    +0.83 over all cells); what changes is how MANY cells code position at all,
    from about 35-46% in GL1 to 49-55% by GL4S2-S4. The rise in the earlier
    figures was largely a mixture moving, not a population sharpening.

    Two things make the pattern readable rather than an artefact of exposure.
    Spike counts FALL across these sessions (rho -0.45) while the pass rate
    RISES (+0.34), and fewer spikes means less power, so the rise is understated
    here rather than manufactured. And the drop at every a/b block (21-27%
    against 46-55% either side) survives restricting to cell-blocks of 80-200
    spikes, where the two groups have matched medians of 125 and 133: 27%
    against 49%. A half-session has less power, but not that much less.

    The dashed line is 5%, which is what a population with no spatial coding
    would give at a one-sided 95% threshold. Nothing here is near it.
    """
    si = data.island_si() if si is None else si
    si, slots, meta, pos, goals, goal_animals = _si_axis(si)
    fig, ax = style.new_figure(height_mm=80.0, bottom_mm=26.0)
    runs = _repeat_runs(meta)
    for isl in (0,) + ISLAND_ORDER:
        colour = style.P.MUTED if isl == 0 else ISLAND_COLOURS[isl]
        d = si[si["island"] == isl]
        g = d.groupby("slot")["sig95"]
        frac = np.array([100 * g.get_group(x).astype(bool).mean()
                         if x in g.groups else np.nan for x in slots], float)
        dash = {"dashes": ISLAND_DASH[isl]} if isl in ISLAND_DASH else {}
        label = "whole maze" if isl == 0 else ISLAND_WHERE[isl]
        for j, run in enumerate(runs):
            kw = dict(dash)
            if j == 0:
                kw["label"] = label
            ax.plot(pos[run], frac[run], "-", color=colour, lw=_lw(1.4),
                    zorder=2 if isl == 0 else 4, **kw)
        good = ~np.isnan(frac)
        if isl:
            ax.plot(pos[good], frac[good], "o", ms=3.4, color=colour, mec="white",
                    mew=_lw(0.9), zorder=5)
    ax.axhline(5, color=style.P.INK, lw=_lw(0.8), ls=(0, (4, 3)), zorder=1)
    ax.text(pos[0] - 0.6, 5, " chance (5%)", fontsize=style.FS["tick"],
            color=style.P.MUTED, va="bottom", ha="left")
    style.slot_axis(ax, slots, pos, goals=goals, animals=goal_animals)
    ax.set_ylabel("cells coding position\n(% above their own shuffle, 95%)",
                  fontsize=style.FS["label"], color=style.P.INK, labelpad=3)
    ax.set_ylim(0, 100)
    style.legend(ax, loc="upper left", ncol=2)
    _pass_rates(si, name)
    return style.save(fig, name)


def island_spatial_info_sig(trials=None, name="island_spatial_info_sig", si=None):
    """:func:`island_spatial_info`, restricted to cells that beat their own shuffle."""
    return island_spatial_info(trials, name=name, si=si, sig_only=True)


def island_spatial_info_matched_sig(trials=None,
                                    name="island_spatial_info_matched_sig", si=None):
    """:func:`island_spatial_info_matched`, significant cell-blocks only.

    Belt and braces: the shuffle already controls the count bias, because the
    null is built at the cell's own spike count. Matching on top of it answers a
    different question, which is whether the SIZE of the coding changed once both
    the bias and the non-coding cells are out.
    """
    return island_spatial_info_matched(trials, name=name, si=si, sig_only=True)


def island_spatial_info_bars_sig(trials=None, name="island_spatial_info_bars_sig",
                                 si=None):
    """:func:`island_spatial_info_bars`, significant cell-blocks only."""
    return island_spatial_info_bars(trials, name=name, si=si, sig_only=True)


#: Figures whose data is the island table rather than the trial table. They take
#: `si` and ignore `trials`, so the dispatcher has to know which is which.
SI_FIGURES = ("island_spatial_info", "island_spatial_info_matched",
              "island_spatial_info_bars", "island_spatial_info_sig",
              "island_spatial_info_matched_sig", "island_spatial_info_bars_sig",
              "island_sig_fraction")


def draw(name, trials, si=None, animals=None, combined=True):
    """Draw figure `name` once per animal, and once with the animals together.

    The animals are separated because they do not share goals: Rat5 runs its
    goals to nodes 410, 314, 107, 219 and Rat6 to 204, 109, 318, 421, so a single
    axis can label the goal number but not the place, and the two lines in one
    panel are two different experiments drawn on top of each other. One file per
    animal says where each goal was.

    The combined file is still written, because comparing the two animals is
    worth one figure and the behavioural ones read well that way.
    """
    fn = REGISTRY[name]
    uses_si = name in SI_FIGURES
    if uses_si and si is None:
        si = data.island_si()
    out = []
    if combined:
        out += list(fn(trials, name=name, **({"si": si} if uses_si else {})))
    src = si if uses_si else trials
    for a in (animals if animals is not None
              else sorted(src["animal"].dropna().unique())):
        kw = {"si": si[si["animal"] == a]} if uses_si else {}
        out += list(fn(trials[trials["animal"] == a], name=f"{name}_{a}", **kw))
    return out


#: name -> function. ``python -m analysis`` draws these.
REGISTRY = {
    "island_spatial_info": island_spatial_info,
    "island_spatial_info_matched": island_spatial_info_matched,
    "island_spatial_info_bars": island_spatial_info_bars,
    "island_spatial_info_sig": island_spatial_info_sig,
    "island_spatial_info_matched_sig": island_spatial_info_matched_sig,
    "island_spatial_info_bars_sig": island_spatial_info_bars_sig,
    "island_sig_fraction": island_sig_fraction,
    "detour_by_session": detour_by_session,
    "detour_first_trial": detour_first_trial,
    "detour_trials_2_5": detour_trials_2_5,
    "success_rate": success_rate,
    "trials_to_first_success": trials_to_first_success,
}
