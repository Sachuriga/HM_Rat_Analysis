"""The one colour system and type scale the MSCA proposal figures share.

Figures 1, 2 and 3 each used to carry their own copy of these hexes. They agreed,
until one of them was edited — so they live here instead, and a change to the
proposal's look is a change to one file rather than three that quietly drift.

The four HUES are the ones validated for figure 1's panel b-g strip: worst pair
dE 24.7 under colour-vision deficiency, 33.6 with normal vision, all above 3:1
contrast on SURFACE. What a hue MEANS is each figure's business — figure 1 spends
them on animals, figure 2 on activities, figure 3 on killing points — but no
figure invents a fifth.

The NEUTRALS are deliberately warm. Matplotlib's default greys are blue-tinted,
and a warm figure beside a cool one is what makes two pages look like two
different documents even when every other choice matches.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# --------------------------------------------------------------- neutrals
INK = "#0b0b0b"        # body text and anything that must read first
INK2 = "#52514e"       # secondary text, axis labels, annotations
MUTED = "#8a8985"      # spines, ticks, rules, "this is furniture" marks
SURFACE = "#fcfcfb"    # the page
MAZE_LINE = "#c9c8c2"  # maze corridors, and the rule around a filled box
FILL = "#f2f1ed"       # a panel or callout fill
RULE = MAZE_LINE

# ------------------------------------------------------------------ hues
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
AMBER = "#eda100"
RED = "#d64545"        # the only red in the proposal: barrier, and KP3

#: GREEN one lightness step down. Pure GREEN is 2.4:1 against SURFACE, so small
#: white type on it, and thin marks in it, fall below the 3:1 floor; this clears
#: it at 3.7:1 with the hue and saturation untouched.
GREEN_DARK = "#17976a"

#: AMBER is 1.9:1 on SURFACE — fine as a fill, unreadable as text. Type that
#: wants to read as "amber" uses this instead.
AMBER_INK = "#8a6100"

#: A pale wash of each hue, for the fill a block of content sits on while the
#: saturated hue stays for its marker. Lightness carries the second dimension,
#: which is the same rule figure 1 uses for good vs MUA units inside one panel.
TINT = {BLUE: "#e5eefb", AMBER: "#fbf2dc", RED: "#fbe9e9", GREEN: "#e4f6ef",
        ORANGE: "#fdeae2"}

#: Type that must read as a hue, where the hue itself is too pale to set text in.
TEXT_ON_SURFACE = {BLUE: BLUE, ORANGE: ORANGE, GREEN: GREEN_DARK,
                   AMBER: AMBER_INK, RED: RED}

# ------------------------------------------------------------------ type
#: The floor, in POINTS. Nothing in any of the three figures may be set smaller:
#: below this, print and a 110 mm column stop being legible. Points are absolute,
#: so this holds at every page width — what gives instead is the LAYOUT.
MIN_PT = 8.0


#: The thinnest stroke, in POINTS, that survives print. Below about 0.25 pt
#: (0.09 mm) a line can drop out on press or render as a single device pixel; a
#: stroke asked for at 0 pt asks the device for its thinnest possible line, which
#: on a 2400 dpi imagesetter is invisible. Line widths here are multiplied by a
#: page-scale factor, so a width that was fine on a 15-inch poster can fall
#: through this floor on a 180 mm column without anyone editing it.
MIN_LW = 0.3


def lw(width_pt):
    """`width_pt` in points, never below :data:`MIN_LW`."""
    return max(MIN_LW, float(width_pt))


def pt(size):
    """`size` in points, never below :data:`MIN_PT`."""
    return max(MIN_PT, float(size))


def scale(sizes):
    """A dict of point sizes with the floor applied to every entry."""
    return {k: pt(v) for k, v in sizes.items()}


def text_width_mm(s, pt_size, family=None, weight="normal"):
    """Width of `s` at `pt_size`, in mm, from the FONT'S OWN METRICS.

    Not an estimate. A characters-times-half-an-em rule is out by about a fifth
    between Times and DejaVu Sans, which is the difference between a caption that
    fits its panel and one that hangs off the page — and the error goes the wrong
    way for the wider font, so the estimate is optimistic exactly where it matters.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family=family or plt.rcParams["font.family"],
                        size=pt_size, weight=weight)
    return max(TextPath((0, 0), line or " ", prop=fp).get_extents().width
               for line in str(s).split("\n")) * 25.4 / 72


def fits_mm(s, avail_mm, pt_size, family=None, weight="normal"):
    return text_width_mm(s, pt_size, family, weight) <= avail_mm


def wrap_mm(s, avail_mm, pt_size, family=None, weight="normal"):
    """`s` wrapped, greedily, to lines that measure no wider than `avail_mm`.

    The text always comes back WHOLE: dropping the tail of a caption to make it
    fit is a silent edit to what the figure says, so it is the panel's height that
    has to give, not the sentence.
    """
    lines, cur = [], ""
    for word in str(s).split():
        trial = f"{cur} {word}".strip()
        if cur and text_width_mm(trial, pt_size, family, weight) > avail_mm:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return "\n".join(lines) if lines else str(s)


# --------------------------------------------------------------- helpers
def luminance(colour):
    r, g, b = mcolors.to_rgb(colour)
    f = lambda u: u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b):
    """WCAG contrast ratio between two colours. 3:1 is the floor for a small mark."""
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def ink_on(colour):
    """White or ink, whichever a label on `colour` can actually be read in.

    The palette spans pale tints to deep blues, so a fixed white label is
    invisible on half of it.
    """
    return "white" if luminance(colour) < 0.34 else INK


def tint(colour):
    """The pale wash of `colour`, mixed on the fly for hues not in TINT."""
    if colour in TINT:
        return TINT[colour]
    r, g, b = mcolors.to_rgb(colour)
    s, t = mcolors.to_rgb(SURFACE), 0.86
    return mcolors.to_hex((r + (s[0] - r) * t, g + (s[1] - g) * t,
                           b + (s[2] - b) * t))


#: rcParams every figure sets: text stays TEXT in the PDF and SVG (TrueType, real
#: <text>), so a figure can be opened in Illustrator and its labels retyped rather
#: than arriving as traced outlines.
VECTOR_TEXT = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}

#: Times New Roman, with fallbacks so a machine without it still renders.
SERIF_STACK = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
