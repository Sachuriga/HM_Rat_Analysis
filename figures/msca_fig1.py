"""MSCA proposal, Figure 1: what one HexMaze session looks like, and what every
session gives.

    a   simultaneously recorded CA1 place cells on the idealised maze (one
        session — the whole of figures/msca_fig1a.py)
    b   unit yield: good and MUA units per session
    c   behavioural performance: log10(shortest hops / actual hops) per trial
    d   spatial information (Skaggs, bits/spike)
    e   number of place fields per cell
    f   place-field size, in cm of track
    g   map stability: alternate-trial split-half correlation

Panels b-g read the cross-session tables written by ``hm-session-summary`` rather
than recomputing anything, so the figure and the summary report cannot disagree:
one xlsx, three sheets (``sessions`` for the unit counts and the behavioural
median, ``units`` for the per-cell distributions, ``trials`` for the per-trial
behaviour). Run the summary first; a summary produced before stability and field
size existed as columns will need re-running for panels f and g.

Panels b-g share one x axis: a slot per session, ordered by
repeat, with a GAP wherever the repeat changes so the blocks are visible and the
tick only has to carry the session number. Session numbers restart inside every
repeat, so a flat 0..N axis puts R1S4 beside R2S1 with nothing to mark the
boundary. A repeat IS a goal location, so the blocks are labelled 'goal 1',
'goal 2' ... and repeat 0 — before any goal was set — is 'habituation'.

The page is transparent and fully vector by default (no rasterised layers, text
embedded as TrueType/real <text>), written as PDF, SVG and PNG, so the figure can
be composited and edited rather than re-made. --rasterize-spikes trades panel a's
editability for a much smaller file; --opaque paints the background back in.

It is authored at the size it will be PLACED: 170 mm wide by default, which is A4
less 20 mm margins, and every type size is fixed in the 8-11 pt band. Points are
absolute, so a smaller page gets smaller artwork around the same-sized text rather
than shrunken labels — but only if the figure is inserted at 100%. Rescaling it in
Word scales the type with it and undoes the whole arrangement; set the picture
width to 17 cm instead. --width-mm re-authors it for a different column: the type
stays 8-11 pt and the LAYOUT gives instead — at 110 mm the strip becomes 2 x 3
rather than 3 x 2, because a third of 110 mm cannot hold a dozen session ticks.

Every animal is drawn on that shared axis, one hue each, as separate series — the
animals were not recorded on the same days, so repeat/session is the only slot
they can share, and pooling them would let one line hide a disagreement between
two rats. Each of panels c-g shows the individual measurements as well as the
per-animal session median, because every one of these quantities has a heavy tail
and a median alone cannot show whether a session was uniform or split.

The metrics in panels d-g are all confounded with how much data a session holds —
this script prints the confounds to stdout when it runs, for the caption.

Usage:
    python figures/msca_fig1.py --nwb <session.nwb> --units 25 149 221 294 437 538 \
        --summary <session_summary_*.xlsx> [--out fig1] [--animal Rat5 Rat6]
    python figures/msca_fig1.py --summary <...xlsx> --summary-only [--out fig1bg]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.transforms import offset_copy                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import msca_fig1a as F1A                                          # noqa: E402

from hm_rat_analysis import place_fields as PF                    # noqa: E402
from hm_rat_analysis.reports import session_summary as SS         # noqa: E402

INK, INK2, MUTED, SURFACE = F1A.INK, F1A.INK2, F1A.MUTED, F1A.SURFACE

# Hue is the ANIMAL, in every panel — the entity the reader tracks across the
# strip. Validated against this surface: worst pair CVD dE 24.7, normal-vision
# dE 33.6, all above 3:1 contrast on #fcfcfb. Beyond four animals hue alone stops
# separating under CVD, so the strip refuses rather than inventing a fifth.
ANIMAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE = ANIMAL_COLORS[0]
#: Within panel b the SECOND dimension is unit quality, carried by lightness of the
#: animal's own hue (good solid, MUA tinted) rather than by a fifth hue.
MUA_ALPHA = 0.40

#: Panels b-g, in order: (key, letter, title, y label). The per-unit panels name
#: the column in the summary's `units` sheet; b and c are session/trial level.
STRIP = [("units", "B", "Unit yield", "units per session"),
         ("performance", "C", "Behaviour", "log10(shortest/actual)"),
         ("spatial_info", "D", "Spatial information", "bits/spike"),
         ("n_fields", "E", "Place fields", "fields per cell"),
         ("field_size_mean_cm", "F", "Field size", "cm of track"),
         ("stability", "G", "Map stability", "split-half r")]

#: Panels whose quantity can legitimately be negative, so they get a zero line and
#: no positive clamp.
SIGNED = {"performance", "stability"}

#: Fixed upper limits, where the distribution has a tail long enough to flatten the
#: part of the axis that carries the result. How many measurements each cap puts
#: outside its panel goes to stdout for the caption — the panels stay unannotated,
#: so a figure reproduced without its caption does not say what it cut.
YMAX = {"spatial_info": 5.0, "field_size_mean_cm": 50.0, "performance": 0.0}

#: A block of sessions is one GOAL location: repeat N is the Nth goal the animal
#: was trained to, and repeat 0 is the habituation day before any goal was set.
HABITUATION_REPEAT = 0


def grid_columns(width, left_in, right_in, hgap_in, n=len(STRIP)):
    """``(ncol, nrow)`` for the strip at this page width.

    The widest grid whose panels still clear :data:`MIN_PANEL_IN`. This is the one
    thing that gives when the page narrows, because the alternative — shrinking the
    type with the page — is what the whole fixed-point-size scheme exists to
    prevent. At 170 mm the panels are 1.7 in and the strip is 3 x 2; at 110 mm a
    third of the page is 0.9 in, so it becomes 2 x 3 instead.
    """
    for ncol in GRID_CHOICES:
        if (width - left_in - right_in - hgap_in * (ncol - 1)) / ncol >= MIN_PANEL_IN:
            break
    return ncol, -(-n // ncol)


def group_label(repeat):
    return "habituation" if repeat == HABITUATION_REPEAT else f"goal {repeat}"


def group_label_short(repeat):
    """The fallback when the full block name cannot fit its block at 8 pt — which
    happens to 'habituation' as soon as the page is narrow and that block is one
    session long. Shrinking the type below 8 pt instead is not an option: the whole
    point of the fixed band is that every label stays legible in print."""
    return "hab" if repeat == HABITUATION_REPEAT else f"g{repeat}"

#: How wide a panel has to be, in INCHES, before it can carry what it must hold: a
#: dozen session slots with an 8 pt number under each, a 10 pt title, and a legend.
#: Below this the ticks run together into '1234' and the title is clipped — so the
#: grid drops a column instead (see :func:`grid_columns`). Type size is never the
#: thing that gives: points are absolute, and 8 pt is the floor for print.
MIN_PANEL_IN = 1.5

#: The column counts tried, widest grid first. Six panels: 3x2, then 2x3, then 6x1.
GRID_CHOICES = (3, 2, 1)

#: Default page width, INCHES: A4 (210 mm) less 20 mm margins each side. Authoring
#: at the width the figure will actually occupy is what makes the type sizes below
#: mean something — placed at 100% in a document, 9 pt here IS 9 pt on the page.
#: Scaling the picture in Word afterwards scales the text with it and breaks that.
A4_TEXT_WIDTH_IN = 170 / 25.4

#: The width this layout's artwork sizes were tuned at. Everything geometric is
#: scaled by width / REF_WIDTH_IN; the type is NOT (see FONT).
REF_WIDTH_IN = 15.0

#: Type sizes in POINTS, all inside the 8-11 pt band that survives print. Absolute
#: by design: they do not change with page width, so a smaller figure gets smaller
#: artwork around the same-sized text rather than unreadable 4 pt labels.
FONT = {"letter": 11, "title": 10, "ylabel": 9, "tick": 8, "group": 8,
        "legend": 8, "note": 8}


# ------------------------------------------------------------
#                     the summary tables
# ------------------------------------------------------------
def load_summary(path):
    """``{sheet: DataFrame}`` from an ``hm-session-summary`` output.

    Accepts the xlsx, one of the CSVs it falls back to when openpyxl fails, or the
    folder either was written into (newest xlsx wins). Missing sheets simply come
    back absent — panels whose data is missing say so rather than failing.
    """
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("session_summary_*.xlsx"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"no session_summary_*.xlsx under {p}")
        p = cands[0]
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        return {k: v for k, v in pd.read_excel(p, sheet_name=None).items()}
    # CSV fallback: <stem>.csv is the sessions table, siblings carry the rest
    stem = p.with_suffix("")
    if stem.name.endswith(("_units", "_trials")):
        stem = Path(str(stem).rsplit("_", 1)[0])
    out = {}
    for sheet, suffix in (("sessions", ""), ("units", "_units"), ("trials", "_trials")):
        f = Path(f"{stem}{suffix}.csv")
        if f.exists():
            out[sheet] = pd.read_csv(f)
    if not out:
        raise FileNotFoundError(f"no session summary tables at {p}")
    return out


def _norm_key(s):
    """Session keys as strings, whatever the sheet stored them as.

    A YYYYMMDD date is all digits, so a reader is free to hand it back as int64 —
    or as float64 the moment one row is blank, at which point ``astype(str)`` gives
    '20260622.0' and every join against the sessions table silently matches
    nothing. Normalising both sides through the same function is what stops a
    figure from coming back empty with no error.
    """
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().all():
        return v.astype("int64").astype(str)
    return s.astype(str)


def _si_column(units, mode="auto"):
    """Which spatial-information column panel d should show.

    Raw Skaggs SI is biased upward at low spike counts (+161% from 1200 to 300
    spikes on this dataset) and spikes per cell fall as trials shorten with
    learning, so a raw cross-session trend is partly a trend in how much data each
    session held. The count-matched column is the comparable one and is preferred
    whenever the summary computed it; `mode` forces either.
    """
    matched = "spatial_info_matched"
    if mode == "raw" or units is None or matched not in units.columns:
        return "spatial_info", "bits/spike"
    finite = pd.to_numeric(units[matched], errors="coerce").notna().sum()
    if mode == "matched" or finite >= 0.5 * len(units):
        # The label stays "bits/spike": rotated, it has only the panel's height to
        # live in, and "count-matched" doubles its length. Which column was used is
        # printed for the caption instead.
        return matched, "bits/spike"
    return "spatial_info", "bits/spike"


def _label(df):
    """The R{repeat}S{session} slot label, as its (repeat, session) parts too."""
    rep = pd.to_numeric(df["repeat"], errors="coerce").astype("Int64")
    ses = pd.to_numeric(df["session"], errors="coerce").astype("Int64")
    return "R" + rep.astype(str) + "S" + ses.astype(str), rep, ses


def session_axis(summary, animals=None):
    """The x axis shared by panels b-g.

    One slot per R{repeat}S{session}, ordered by repeat then session, with EVERY
    animal on the same slot — animals were not recorded on the same days, so the
    date cannot be the axis, and the repeat/session label is the only thing two
    rats' sessions genuinely share.

    Returns ``(sessions, keys, meta, animals)`` where `meta` is the (repeat,
    session) pair per slot, which is what lets the slots be grouped by repeat.
    """
    sess = summary.get("sessions")
    if sess is None or sess.empty:
        raise ValueError("the summary has no 'sessions' table")
    sess = sess.copy()
    sess["date"] = _norm_key(sess["date"])
    sess["label"], rep, ses = _label(sess)

    have = list(pd.unique(sess["animal"].dropna()))
    if animals is not None and str(animals).upper() != "ALL":
        want = [animals] if isinstance(animals, str) else list(animals)
        missing = [a for a in want if a not in have]
        if missing:
            raise ValueError(f"animal(s) {missing} not in the summary (have: "
                             f"{', '.join(map(str, have))})")
        sess = sess[sess["animal"].isin(want)]
        have = want
    if len(have) > len(ANIMAL_COLORS):
        raise ValueError(f"{len(have)} animals but only {len(ANIMAL_COLORS)} hues "
                         "separate under colour-vision deficiency — plot them in "
                         "separate figures rather than adding a fifth colour")

    order = (sess[["label"]].assign(rep=rep, ses=ses).dropna()
             .drop_duplicates("label").sort_values(["rep", "ses"]))
    keys = list(order["label"])
    meta = list(zip(order["rep"].astype(int), order["ses"].astype(int)))
    return sess, keys, meta, sorted(have)


def slot_positions(meta, gap=0.85):
    """x position per slot, with a gap inserted wherever the repeat changes.

    Sessions run 1..N inside a repeat and then start again, so a plain 0..N-1 axis
    puts R1S4 next to R2S1 with nothing to say a new block of training began. The
    gap is what makes the blocks readable at a glance; the tick then only has to
    carry the session number.
    """
    pos, x = [], 0.0
    for i, (rep, _ses) in enumerate(meta):
        if i and rep != meta[i - 1][0]:
            x += gap
        pos.append(x)
        x += 1.0
    return np.array(pos, float)


def repeat_groups(meta, pos):
    """``[(repeat, first_x, last_x), ...]`` — one entry per run of equal repeat."""
    groups = []
    for rep, x in zip([m[0] for m in meta], pos):
        if groups and groups[-1][0] == rep:
            groups[-1][2] = x
        else:
            groups.append([rep, x, x])
    return [tuple(g) for g in groups]


# ------------------------------------------------------------
#                       panel drawing
# ------------------------------------------------------------
def _offset_axes(ax, dy_points):
    """A transform of (data x, axis bottom + `dy_points`) — for annotations that
    must sit a fixed distance under the axis whatever the panel's height."""
    return offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=dy_points,
                       units="points")


def _units_to_inches(ax, pos):
    """Inches per x-axis unit, from the axes' own place on the page.

    Needed because a block label ('habituation') can be wider than the block it
    names when that block is one session long, and the only way to know is to
    compare the two in the same units.
    """
    ax_w = ax.figure.get_size_inches()[0] * ax.get_position().width
    span = (pos[-1] - pos[0]) + 1.5 if len(pos) > 1 else 1.5
    return ax_w / span


def _fits(text, avail_in, fontsize, char_w=0.56):
    """Whether `text` fits `avail_in` inches at `fontsize` points."""
    return len(text) * char_w * fontsize / 72.0 <= avail_in


def _frame(ax, letter, title, ylab, meta, pos, highlight=None, show_repeats=True,
           scale=1.0):
    ax.set_title(title, fontsize=FONT["title"], color=INK, loc="left", pad=6)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-26 * max(scale, 0.6), 9), textcoords="offset points",
                fontsize=FONT["letter"], fontweight="bold", color=INK,
                annotation_clip=False)
    ax.set_ylabel(ylab, fontsize=FONT["ylabel"], color=INK2, labelpad=2)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(s) for _r, s in meta], fontsize=FONT["tick"], color=INK2)
    ax.set_xlim(pos[0] - 0.75, pos[-1] + 0.75)
    ax.tick_params(length=2.5, pad=2, colors=MUTED, labelcolor=INK2,
                   labelsize=FONT["tick"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8 * scale)
    ax.grid(axis="y", color=MUTED, lw=0.4 * scale, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    if show_repeats:
        # The goal each block of sessions belongs to, named once under its block.
        # The tick then carries only the session number, which is what actually
        # varies inside a block — repeating "goal 3" on every tick is the thing
        # that made this axis hard to read.
        #
        # Both the rule and its label hang a fixed number of POINTS below the axis
        # rather than a fraction of it, so they keep clear of the tick labels
        # whatever height the panel ends up with.
        in_per_unit = _units_to_inches(ax, pos)
        groups = repeat_groups(meta, pos)
        # Only the labels that do not fit get shortened. Shortening all of them the
        # moment one is too wide is what turned a readable axis into 'g1 g2 g3'.
        full = []
        for rep, x0, x1 in groups:
            text = group_label(rep)
            # A label may use its own block plus HALF the gap on either side — the
            # allowance that lets two neighbours grow without ever meeting.
            if not _fits(text, (x1 - x0 + 1.85) * in_per_unit, FONT["group"]):
                text = group_label_short(rep)
            full.append(text)
        rule = _offset_axes(ax, -14 - FONT["tick"])
        text_y = _offset_axes(ax, -17 - FONT["tick"])
        for text, (rep, x0, x1) in zip(full, groups):
            ax.plot([x0 - 0.3, x1 + 0.3], [0, 0], transform=rule, color=MUTED,
                    lw=0.9 * scale, clip_on=False, solid_capstyle="butt")
            ax.text((x0 + x1) / 2, 0, text, transform=text_y, ha="center", va="top",
                    fontsize=FONT["group"], color=INK2, clip_on=False)

    if highlight is not None and highlight in [f"R{r}S{s}" for r, s in meta]:
        # The session panel a is drawn from, banded in every panel so the example is
        # placed among the rest. A band rather than a mark under the axis: the space
        # below the ticks now belongs to the repeat labels.
        i = [f"R{r}S{s}" for r, s in meta].index(highlight)
        ax.axvspan(pos[i] - 0.46, pos[i] + 0.46, color=BLUE, alpha=0.08, lw=0,
                   zorder=0)


def _offsets(n, span=0.34):
    """Sub-slot x offsets for `n` animals sharing one session slot."""
    if n <= 1:
        return np.zeros(1)
    return np.linspace(-span, span, n)


def _fit_legend(ax, ncol, **kw):
    """The legend, with `ncol` reduced until it actually fits its panel.

    The legend holds text at a fixed 8 pt while the panel shrinks with the page, so
    the arrangement that fits at 170 mm runs off the edge at 110 mm and lands on the
    next panel. Stacking it instead costs height, which the panel has more of.

    Measured against the drawn legend rather than estimated from character counts:
    the estimate is what decides whether a figure is readable at a given width, and
    a wrong guess either overflows or stacks a legend that would have fitted.
    """
    leg = ax.legend(ncol=ncol, **kw)
    r = ax.figure.canvas.get_renderer()
    while ncol > 1 and (leg.get_window_extent(r).width
                        > ax.get_window_extent(r).width * 0.98):
        ncol -= 1
        leg.remove()
        leg = ax.legend(ncol=ncol, **kw)
    return leg, ncol


def units_panel(ax, sess, keys, pos, animals, colors, scale=1.0):
    """Panel b — units per session: one bar per animal, MUA stacked on good.

    Counts that sum to a whole (everything the session yielded), so a stack; hue is
    the animal, as everywhere else in the strip, and quality is the lightness step
    within it rather than a fifth colour.
    """
    off = _offsets(len(animals), span=0.24)
    w = min(0.42, (0.62 if len(animals) == 1 else 0.46 / len(animals)))
    top = 0.0
    for a, dx, c in zip(animals, off, colors):
        d = sess[sess["animal"] == a].drop_duplicates("label").set_index("label")
        good = np.array([d["n_good"].get(k, np.nan) for k in keys], float)
        mua = np.array([d["n_mua"].get(k, np.nan) for k in keys], float)
        ax.bar(pos + dx, good, w, color=c, zorder=2, label=f"{a} good")
        # a surface gap between the two segments, so the boundary is a line the
        # reader sees rather than a colour change they have to infer
        ax.bar(pos + dx, mua, w, bottom=good + 1.2, color=c, alpha=MUA_ALPHA,
               linewidth=0, zorder=2, label=f"{a} MUA")
        if np.isfinite(good + mua).any():
            top = max(top, float(np.nanmax(good + mua)))
    _leg, ncol = _fit_legend(ax, len(animals), fontsize=FONT["legend"], frameon=False,
                             labelcolor=INK2, handlelength=0.9, handletextpad=0.5,
                             borderpad=0.15, labelspacing=0.25, columnspacing=0.9,
                             loc="upper left")
    # headroom for the legend's own rows, so it never sits on a bar — counted from
    # the rows it actually ended up with, which is not len(animals) once the legend
    # has had to stack itself to fit a narrow panel
    rows = -(-2 * len(animals) // ncol)
    ax.set_ylim(0, top * (1.15 + 0.10 * rows) if top > 0 else 1)
    return top


def dist_panel(ax, per_animal, keys, pos, animals, colors, signed=False, seed=0,
               ymax=None, rasterize=False, scale=1.0):
    """Panels c-g — every measurement as a dot in its animal's hue, that animal's
    session median as the line through them.

    The median, not the mean: each of these distributions has a long tail made of
    the cells or trials with the least data behind them, and a mean follows that
    tail rather than the session. The animals are drawn as separate series rather
    than pooled — with two rats, one line hiding a disagreement between them is a
    worse failure than a slightly busier panel.
    """
    rng = np.random.default_rng(seed)
    off = _offsets(len(animals))
    allv, lines, n_over = [], [], 0
    for a, dx, c in zip(animals, off, colors):
        vals = per_animal.get(a, {})
        meds = []
        for x, k in zip(pos, keys):
            v = np.asarray(vals.get(k, []), float)
            v = v[np.isfinite(v)]
            meds.append(float(np.median(v)) if v.size else np.nan)
            if v.size:
                j = rng.normal(0, 0.05, v.size) if v.size > 1 else np.zeros(1)
                # A dot big enough to read as a measurement rather than as grain,
                # with a white rim so a cluster of them stays countable instead of
                # merging into one blob at print size.
                ax.scatter(x + dx + np.clip(j, -0.13, 0.13), v,
                           s=max(3.0, 28 * scale ** 2), color=c, alpha=0.50,
                           edgecolors="white", linewidths=0.5 * scale,
                           zorder=2, rasterized=rasterize)
                allv.append(v)
        meds = np.array(meds, float)
        ax.plot(pos + dx, meds, "-", color=c, lw=1.7 * scale, zorder=3,
                label=str(a))
        # a white ring, so a median never disappears into the dots underneath it —
        # the same rim the measurements carry, one step wider
        ax.plot(pos + dx, meds, "o", ms=max(2.6, 6.4 * scale), color=c,
                mec="white", mew=1.3 * scale, zorder=4)
        lines.append(meds)

    # Limits from the data rather than autoscale: a dot sitting exactly on the top
    # spine reads as a clipped distribution.
    pool = [v for v in allv] + [m[np.isfinite(m)] for m in lines]
    pool = [v for v in pool if len(v)]
    if pool:
        v = np.concatenate(pool)
        lo, hi = float(v.min()), float(v.max())
        pad = 0.10 * (hi - lo) if hi > lo else max(0.1, abs(hi) * 0.1)
        top = hi + pad
        if ymax is not None:
            # The cap trims the TAIL, never the result: a median above it would
            # vanish from the panel it is the subject of, so the axis grows to hold
            # the medians even when that means overshooting the requested cap.
            med_max = max((float(np.nanmax(m)) for m in lines
                           if np.isfinite(m).any()), default=-np.inf)
            top = max(ymax, med_max * 1.06 if np.isfinite(med_max) else -np.inf)
        ax.set_ylim(min(lo - pad, 0.0) if signed else 0.0, top)
        # The count of points above the cap is returned rather than drawn on the
        # panel: it belongs in the caption now. It still has to be said somewhere —
        # a capped axis that says nothing at all about what it cut shows a tight
        # distribution where there is a long tail.
        if ymax is not None and allv:
            n_over = int((np.concatenate(allv) > top).sum())
    if signed:
        ax.axhline(0, color=INK2, lw=0.8 * scale, ls=(0, (4, 3)), zorder=1)
    return lines, n_over


def _by_animal_key(df, value_col, keys, animals):
    """``{animal: {slot label: values}}`` for one metric column."""
    out = {a: {} for a in animals}
    if df is None or df.empty or value_col not in df.columns:
        return out
    d = df.copy()
    d["label"], _r, _s = _label(d)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    for a, sub in d.groupby("animal"):
        if a not in out:
            continue
        by = sub.groupby("label")[value_col]
        out[a] = {k: by.get_group(k).dropna().to_numpy() for k in keys
                  if k in by.groups}
    return out


def _prepared_units(summary, animals):
    """The `units` rows for the plotted animals, one epoch per session.

    Sessions split at a goal switch carry two epochs; the 'before' epoch is the one
    the whole-session sessions are comparable to, so mixing both would put two
    populations in one slot.
    """
    u = summary.get("units")
    if u is None or u.empty:
        return None
    u = u.copy()
    if "epoch" in u.columns:
        u = u[u["epoch"].isin(["whole", "before"])]
    return u[u["animal"].isin(animals)]


def summary_strip(axes, summary, animal=None, si_mode="auto", highlight=None,
                  rasterize=False, scale=1.0):
    """Draw panels b-g into `axes` (six of them). Returns the parameter stamp."""
    sess, keys, meta, animals = session_axis(summary, animal)
    pos = slot_positions(meta)
    colors = ANIMAL_COLORS[:len(animals)]
    units = _prepared_units(summary, animals)
    trials = summary.get("trials")
    if trials is not None and not trials.empty:
        trials = trials[trials["animal"].isin(animals)].copy()

    si_col, si_unit = _si_column(units, si_mode)
    # ONE legend for the animal hues, in panel c. The hue means the same thing in
    # all six panels, so a legend per panel is five repetitions competing with the
    # data — and at A4 width a legend is a sizeable fraction of a panel. Panel b
    # carries its own, because it also has to name the quality stack.
    legend_on = {1}
    capped = {}
    for i, (ax, (key, letter, title, ylab)) in enumerate(zip(axes, STRIP)):
        ymax = YMAX.get(key)
        if key == "spatial_info":
            key, ylab = si_col, si_unit
        _frame(ax, letter, title, ylab, meta, pos, highlight=highlight,
               scale=scale)
        if key == "units":
            units_panel(ax, sess, keys, pos, animals, colors, scale=scale)
            continue
        if key == "performance":
            vals = _by_animal_key(trials, "performance", keys, animals)
            if not any(len(v) for per in vals.values() for v in per.values()):
                # no per-trial sheet: the session medians alone still draw the lines
                vals = _by_animal_key(sess.assign(performance=sess.get(
                    "performance_med")), "performance", keys, animals)
            _, n_over = dist_panel(ax, vals, keys, pos, animals, colors,
                                   signed=True, ymax=ymax, rasterize=rasterize,
                                   scale=scale)
        else:
            _, n_over = dist_panel(ax, _by_animal_key(units, key, keys, animals),
                                   keys, pos, animals, colors,
                                   signed=key in SIGNED, ymax=ymax,
                                   rasterize=rasterize, scale=scale)
            if units is None or key not in getattr(units, "columns", []):
                ax.text(0.5, 0.5, f"no '{key}' column\nin this summary\n"
                                  "— re-run hm-session-summary",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=FONT["note"], color=MUTED)
        if n_over:
            capped[title] = (n_over, ymax)
        if i in legend_on and len(animals) > 1:
            ax.legend(fontsize=FONT["legend"], frameon=False, labelcolor=INK2,
                      handlelength=1.2, borderpad=0.15, loc="best")

    stamp = SS._param_stamp(sess.to_dict("records"))
    if highlight is not None and highlight in keys:
        stamp += f"  ·  panel a: {highlight}"
    return stamp, si_col, capped


def confound_notes(si_col, capped=None):
    """The caption's worth of warnings, for stdout rather than for the figure.

    The capped panels are in here rather than printed on the axes: the panel says
    nothing about the tail it cut, so the caption has to.
    """
    keys = [si_col, "n_fields", "field_size_mean_cm", "stability"]
    out = []
    for k in keys:
        note = PF.METRIC_NOTES.get(k, "")
        if note:
            out.append(f"  {k}: {note}")
    for title, (n, ymax) in sorted((capped or {}).items()):
        out.append(f"  {title}: axis capped at {ymax:g}; {n} points above it "
                   f"are outside the panel")
    return out


# ------------------------------------------------------------
def _panel_a_slot(summary, data):
    """The R{repeat}S{session} slot panel a's session occupies, or None.

    Panel a is identified by animal + date, but the strip's axis is repeat/session
    — the only label two animals can share — so the highlight has to be looked up
    rather than carried across.
    """
    if data is None or summary is None or "sessions" not in summary:
        return None
    s = summary["sessions"].copy()
    s["date"] = _norm_key(s["date"])
    subj = str(data["animal"])
    animal = f"Rat{int(subj)}" if subj.isdigit() else subj
    hit = s[(s["animal"] == animal) & (s["date"] == str(data["date"]))]
    if hit.empty:
        return None
    lab, _r, _ses = _label(hit)
    return str(lab.iloc[0])


def build(out_stem, nwb_path=None, units=None, summary_path=None, animal=None,
          si_mode="auto", strip_only=False, transparent=True, rasterize=False,
          goal_label_offset=F1A.GOAL_LABEL_OFFSET, width=A4_TEXT_WIDTH_IN,
          **panel_a_kw):
    summary = load_summary(summary_path) if summary_path else None
    if summary is None and strip_only:
        raise ValueError("--summary-only needs --summary")

    data = None
    if not strip_only:
        if not nwb_path or not units:
            raise ValueError("panel a needs --nwb and --units (or pass --summary-only)")
        # the caption offset goes in too: the unit labels are laid out against it,
        # so they can be pushed clear of the "goal N" text before anything is drawn
        data = F1A.load_session(nwb_path, units,
                                goal_label_offset=goal_label_offset, **panel_a_kw)

    scale = width / REF_WIDTH_IN
    # Margins in INCHES, not fractions: the left one has to hold a 9 pt y label plus
    # its tick numbers, which do not shrink with the page, so a fractional margin
    # would swallow them on a narrow figure. They fix the panel width, which is what
    # decides how many columns the strip can be.
    left_in = max(0.58, 0.052 * REF_WIDTH_IN * scale)
    right_in = 0.06
    hgap_in = max(0.46, 0.075 * REF_WIDTH_IN * scale)
    ncol, nrow = grid_columns(width, left_in, right_in, hgap_in)

    highlight = _panel_a_slot(summary, data)

    # The plotting areas scale with the page; the paddings around them do NOT, or
    # not fully — they hold TEXT, whose size is fixed in points, so a padding that
    # scaled linearly would stop fitting its own labels on a small page. Each is
    # therefore a scaled value with a floor derived from what it has to contain:
    # bottom_pad holds the tick row and the goal brackets under it; row_gap holds
    # the same brackets plus the next row's title.
    row_h = max(0.85, 2.25 * scale)
    row_gap = max(0.80, 1.15 * scale)
    top_pad = max(0.30, 0.55 * scale)
    # what hangs below the bottom row: ticks, then the goal rule and its label, both
    # drawn a fixed number of points down (see _frame)
    labels_below = (17 + FONT["tick"] + 2 * FONT["group"]) / 72.0
    bottom_pad = max(1.05 * scale, labels_below + 0.10)
    strip_block = nrow * row_h + (nrow - 1) * row_gap + top_pad + bottom_pad
    # A band above the maze for the panel letter. Fixed in INCHES: it holds text, and
    # the correlogram window captions sit just above the maze axes, so a band that
    # shrank with the page would run into them.
    head_pad = 0.30
    if strip_only:
        height = strip_block + 0.20
        maze_frac = None
    else:
        bbox = data["bbox"]
        maze_h = (width * 0.99) / ((bbox[1] - bbox[0]) / (bbox[3] - bbox[2]))
        height = maze_h + head_pad + strip_block + 0.20
        maze_frac = maze_h / height

    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    if not strip_only:
        ax_a = fig.add_axes([0.005, 1 - maze_frac - head_pad / height, 0.99,
                             maze_frac])
        F1A.draw_panel_a(ax_a, data, rasterize=rasterize,
                         goal_label_offset=goal_label_offset, scale=scale,
                         fonts=F1A.PRINT_FONT)
        # Only the letter. Which animal and which day this panel is belongs in the
        # caption, not on the page — it is printed to stdout when this runs.
        fig.text(0.006, 1 - 0.055 / height, "A", fontsize=FONT["letter"],
                 fontweight="bold", color=INK, va="top")

    stamp = si_col = None
    capped = {}
    if summary is not None:
        left = left_in / width
        right = 1 - right_in / width
        hgap = hgap_in / width
        w = (right - left - hgap * (ncol - 1)) / ncol
        axes = []
        for i in range(len(STRIP)):
            r, c = divmod(i, ncol)
            y0 = (bottom_pad + (nrow - 1 - r) * (row_h + row_gap)) / height
            axes.append(fig.add_axes([left + c * (w + hgap), y0, w, row_h / height]))
        # The parameter stamp is NOT drawn on the page — it goes to stdout below, for
        # the caption. On the figure it was three lines of 8 pt competing with the
        # panels for the reader's attention and for the page's bottom margin.
        stamp, si_col, capped = summary_strip(axes, summary, animal=animal,
                                              si_mode=si_mode, highlight=highlight,
                                              rasterize=rasterize, scale=scale)

    out_stem = F1A.out_path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    # PDF and SVG both, because vector editors disagree about which they open
    # cleanly: Illustrator prefers the PDF, Inkscape and the browser the SVG. Text
    # is TrueType in the PDF and real <text> in the SVG (see msca_fig1a's rcParams),
    # so labels stay editable in either rather than arriving as outlines.
    kw = dict(transparent=True) if transparent else dict(facecolor=SURFACE)
    written = []
    for suffix, extra in ((".pdf", {}), (".svg", {}), (".png", dict(dpi=1200))):
        p = out_stem.with_suffix(suffix)
        fig.savefig(p, **kw, **extra)
        written.append(p)
    plt.close(fig)
    for p in written:
        print(f"wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    if data is not None:
        # off the page now, so it has to arrive somewhere the caption can use it
        print(f"\npanel a: Rat {data['animal']}  ·  {data['date']}")
    if si_col:
        print("\nFor the caption — every panel b-g quantity and what it is "
              "confounded with:")
        for line in confound_notes(si_col, capped):
            print(line)
        print(f"  parameters: {stamp}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nwb", help="session NWB drawn in panel a")
    ap.add_argument("--units", type=int, nargs="+", help="phy cluster ids for panel a")
    ap.add_argument("--summary", help="session_summary_*.xlsx (or the folder holding it)")
    ap.add_argument("--summary-only", action="store_true",
                    help="draw only panels b-g, without opening an NWB")
    ap.add_argument("--animal", default=None, nargs="*",
                    help="restrict panels b-g to these animals (default: every "
                         "animal in the summary, one colour each, sharing the "
                         "R{repeat}S{session} axis)")
    ap.add_argument("--si", default="auto", choices=["auto", "matched", "raw"],
                    help="spatial information: the spike-count-matched column when "
                         "the summary has it (default), or force either")
    ap.add_argument("--out", default="fig1",
                    help=f"output stem (default %(default)s). A bare name lands in "
                         f"{F1A.OUT_DIR}; set MSCA_FIG_DIR to move that, or give a "
                         "path of your own to bypass it")
    ap.add_argument("--width", type=float, default=A4_TEXT_WIDTH_IN,
                    help="page width in INCHES (default %(default).2f = A4 less "
                         "20 mm margins). Place the figure at this width and do "
                         "not rescale it, or the 8-11 pt type rescales with it")
    ap.add_argument("--width-mm", type=float, default=None,
                    help="the same width in MILLIMETRES, which is how a journal "
                         "states its column (e.g. 110). Overrides --width. The type "
                         "stays 8-11 pt at any width; the grid drops a column "
                         "instead once a panel would be under "
                         f"{MIN_PANEL_IN * 25.4:.0f} mm")
    ap.add_argument("--opaque", action="store_true",
                    help="paint the page background instead of leaving it "
                         "transparent (default: transparent, for compositing)")
    ap.add_argument("--rasterize-spikes", action="store_true",
                    help="embed the trajectory and spike clouds as an image inside "
                         "the vector page — a much smaller file, but those two "
                         "layers stop being editable (default: everything vector)")
    ap.add_argument("--bin-cm", type=float, default=5.0)
    ap.add_argument("--min-occ", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=0.025)
    ap.add_argument("--field-frac", type=float, default=0.30)
    ap.add_argument("--min-field-bins", type=int, default=3)
    ap.add_argument("--palette", default=F1A.DEFAULT_PALETTE,
                    choices=["turbo"] + sorted(F1A.FIXED_PALETTES))
    ap.add_argument("--max-jitter", type=float, default=0.10)
    ap.add_argument("--goal-label", type=float, nargs=2, metavar=("DX", "DY"),
                    default=list(F1A.GOAL_LABEL_OFFSET),
                    help="where panel a's 'goal N' caption sits relative to the "
                         "star, in metres: +DX right, +DY DOWN the page (default "
                         f"{F1A.GOAL_LABEL_OFFSET[0]:g} {F1A.GOAL_LABEL_OFFSET[1]:g})")
    ap.add_argument("--gamma", type=float, default=2.0)
    a = ap.parse_args(argv)
    width = a.width_mm / 25.4 if a.width_mm else a.width
    build(a.out, nwb_path=a.nwb, units=a.units, summary_path=a.summary,
          animal=a.animal, si_mode=a.si, strip_only=a.summary_only,
          transparent=not a.opaque, rasterize=a.rasterize_spikes,
          goal_label_offset=tuple(a.goal_label), width=width,
          bin_cm=a.bin_cm, min_occ=a.min_occ, sigma=a.sigma, speed_thresh=a.speed,
          gamma=a.gamma, field_frac=a.field_frac, min_field_bins=a.min_field_bins,
          max_jitter=a.max_jitter, palette=a.palette)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
