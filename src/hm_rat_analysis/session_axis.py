"""The session axis every cross-session figure shares.

One slot per R{repeat}S{session}, ordered by repeat then session, with a gap
wherever the repeat changes. The tick carries the session number alone and the
goal each block of sessions belongs to is named ONCE under its block, with a rule;
the first session at every goal is banded, because that is the day the goal moved.

This is the axis of the MSCA proposal figure (``figures/msca_fig1.py``,
``_frame``): sessions run 1..N inside a repeat and then start again, so a plain
0..N-1 axis puts R1S4 next to R2S1 with nothing to say a new block of training
began, and repeating "goal 3" on every tick is what made the old axis hard to
read. The cross-session report draws the same axis so the two can be read side
by side.
"""

import re

import numpy as np
from matplotlib.transforms import offset_copy

#: The repeat that predates any goal.
HABITUATION_REPEAT = 0
#: Extra x distance opened wherever the repeat changes (slot units).
GAP = 0.85

# The report's own furniture colours (the figure scripts have their palette).
_INK2 = "#52514e"
_MUTED = "#8a8985"
_BAND = "#2a78d6"


def group_label(repeat):
    return "habituation" if repeat == HABITUATION_REPEAT else f"goal {repeat}"


def group_label_short(repeat):
    """The fallback when the full block name cannot fit its block — which happens
    to 'habituation' as soon as that block is one session long on a narrow panel."""
    return "hab" if repeat == HABITUATION_REPEAT else f"g{repeat}"


def slot_label(repeat, session):
    return f"R{repeat}S{session}"


def parse_slot_label(label):
    """``(repeat, session)`` from an ``R{repeat}S{session}`` label, else None."""
    m = re.fullmatch(r"R(\d+)S(\d+)", str(label))
    return (int(m.group(1)), int(m.group(2))) if m else None


def slot_positions(meta, gap=GAP):
    """x position per slot, with a gap inserted wherever the repeat changes."""
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


def _units_to_inches(ax, pos):
    """Inches per x-axis unit, from the axes' own place on the page."""
    ax_w = ax.figure.get_size_inches()[0] * ax.get_position().width
    span = (pos[-1] - pos[0]) + 1.5 if len(pos) > 1 else 1.5
    return ax_w / span


def _fits(text, avail_in, fontsize, char_w=0.56):
    return len(text) * char_w * fontsize / 72.0 <= avail_in


def decorate(ax, meta, pos, tick_fontsize=7, group_fontsize=7, band=True,
             ink=_INK2, muted=_MUTED, band_color=_BAND):
    """Dress `ax` as the shared session axis.

    `meta` is the (repeat, session) pair per slot and `pos` the x of each slot
    (:func:`slot_positions`). Ticks show the session number; a rule and the goal
    name hang a fixed number of POINTS under the axis so they clear the tick labels
    whatever height the panel has; the first session of every goal is banded.
    """
    pos = np.asarray(pos, float)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(s) for _r, s in meta], fontsize=tick_fontsize, color=ink)
    ax.set_xlim(pos[0] - 0.75, pos[-1] + 0.75)
    ax.tick_params(axis="x", length=2.5, pad=2, colors=muted, labelcolor=ink,
                   labelsize=tick_fontsize)

    in_per_unit = _units_to_inches(ax, pos)
    groups = repeat_groups(meta, pos)
    # Only HABITUATION is ever shortened: 'goal N' over a one-session block still
    # fits its block plus half a gap either side, and 'g1' next to 'goal 2' read
    # as two different things.
    names = []
    for rep, x0, x1 in groups:
        text = group_label(rep)
        if rep == HABITUATION_REPEAT and not _fits(
                text, (x1 - x0 + 1.85) * in_per_unit, group_fontsize):
            text = group_label_short(rep)
        names.append(text)
    rule = offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                       y=-14 - tick_fontsize, units="points")
    text_y = offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                         y=-17 - tick_fontsize, units="points")
    for text, (rep, x0, x1) in zip(names, groups):
        ax.plot([x0 - 0.3, x1 + 0.3], [0, 0], transform=rule, color=muted, lw=0.9,
                clip_on=False, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, 0, text, transform=text_y, ha="center", va="top",
                fontsize=group_fontsize, color=ink, clip_on=False)

    if band:
        for x, (rep, ses) in zip(pos, meta):
            if rep != HABITUATION_REPEAT and ses == 1:
                ax.axvspan(x - 0.46, x + 0.46, color=band_color, alpha=0.10, lw=0,
                           zorder=0)
