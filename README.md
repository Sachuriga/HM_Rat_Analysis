# HM_Rat_Analysis

Analysis and figures for the HexMaze rat experiments. It consumes what the
[HM_Tracker_2025](https://github.com/Sachuriga/HM_Tracker_2025) pipeline produces
— session `.nwb` files and the tracker's per-session log/CSV outputs — and turns
them into reports. It does not track, sort, or package anything itself.

## Install

```bash
pip install -e ".[dev]"     # drop [dev] if you don't want pytest
```

Installing puts two commands on your PATH; both are also runnable as
`python -m hm_rat_analysis.reports.<module>`.

## The two reports

### `hm-session-summary` — cross-session, per animal

Scans a folder tree for session NWBs, groups them by animal (NWB `subject_id`),
and plots each metric against session date (labelled with Repeat & Session).

```bash
hm-session-summary --root <folder> [--bin_cm 5] [--speed 0.05] [--smooth 2]
```

Searches up to four levels deep and skips `ip*` folders, so `root/`,
`root/animal/session/` and similar layouts all work without descending into the
tracker's huge `phy_export` directories. Writes into `--root`:

| Output | Contents |
| --- | --- |
| `session_summary.pdf` | 2 pages per animal (metrics over sessions; cell-type scatter), a pooled page when there is more than one animal, and decoder pages when step `b` was run |
| `session_summary.xlsx` | sheets `sessions`, `units`, `trials`, `trial_unit_metrics`, `anova`, `posthoc` — falls back to `session_summary*.csv` if the workbook cannot be written |

Metrics per session: GOOD/MUA unit counts, pyramidal vs interneuron counts,
behavioural performance, pyramidal spatial information (Skaggs, bits/spike),
selectivity (peak/mean), map stability, place-field count, field size, and
place-field distance to the goal node.

Behavioural performance is `log10(shortest hops / actual hops)` per trial, read
from the session's `RecordingMeta.xlsx` (`Start_Nodes`, `Goal_Node`, `paths`) and
scored against the maze graph — the same definition the tracker's decoder uses, so
the two agree. Map stability is the Pearson correlation between rate maps built
from **alternate trials**, over the bins valid in both halves.

### `hm-trial-report` — one behavioural session

Parses one session folder's tracker logs and writes a per-trial report.

```bash
hm-trial-report -o <session folder> [--out-dir <where to write>]
```

| Output | Contents |
| --- | --- |
| `<stem>_analysis_final.pdf` | metadata page, one page per trial (speed track, time evolution, shortest routes, occupancy, speed traces), aggregate speed distribution |
| `<stem>_trial_metrics.csv` | tidy per-trial metrics — one row per trial |
| `<stem>_all_plot_data.pkl` | every plotted series, for further analysis |

Each trial is scored against the maze graph as `log(optimal / actual)`, so 0 is a
perfect route and more negative is more wandering. It is computed over physical
distance (`dist_log_score`) and over hop count (`hops_log_score`), because a rat
can take a topologically direct route that is physically long, or vice versa.

## The MSCA figures

`figures/` holds proposal figures rather than reports; they are run directly, not
installed as commands.

```bash
python figures/msca_fig1.py --nwb <session.nwb> --units 25 149 221 294 437 538 \
    --summary <session_summary_*.xlsx> --out MSCA_figures/Fig1
```

| Panel | Shows |
| --- | --- |
| a | every listed unit's spikes on one idealised maze, fields outlined, correlograms inset (this is all of `msca_fig1a.py`, which still runs standalone) |
| b–g | unit yield, behavioural performance, spatial information, place-field count, field size and map stability, as a 2 × 3 grid |

Panels b–g share one x axis: a slot per session, ordered by repeat, with a gap
wherever the repeat changes, so the tick only carries the session number and the
blocks are visible. A repeat **is** a goal location, so blocks are labelled
`goal 1`, `goal 2` …, and repeat 0 — before any goal was set — is `habituation`.
Every animal in the summary is drawn on that axis in its own hue, as a separate
series: the animals were not recorded on the same days, so repeat/session is the
only slot they can share, and pooling them would let one line hide a disagreement
between two rats. The session panel a comes from is banded in every panel.

Output is **transparent and fully vector** by default — PDF, SVG and PNG, with no
rasterised layers and text embedded as TrueType (PDF) / real `<text>` (SVG), so
labels stay editable in Illustrator or Inkscape. `--rasterize-spikes` embeds panel
a's trajectory and spike clouds as an image for a much smaller file (those two
layers stop being editable); `--opaque` paints the page background back in.

**Sized for the page it goes on.** The default is 170 mm wide (A4 less 20 mm
margins) and every type size is fixed in the 8–11 pt band — points are absolute, so
a smaller page gets smaller *artwork* around the same-sized text, not shrunken
labels. That only holds if the figure is placed at 100%: in Word, insert it and set
the picture width to **17 cm**, never drag-resize, or the type rescales with the
picture. `--width <inches>` re-authors it for a different column.
Panels d and f cap the y axis (`YMAX` in the script) because both have a tail long
enough to flatten the rest; each says how many measurements are above the cap, and
the cap is raised if it would ever hide a session median.

Panels b–g **read the summary xlsx** rather than recomputing, so the figure and
the report cannot disagree — run `hm-session-summary` first. A summary produced
before stability and field size existed as columns has to be re-run, and the
panels say so on the figure instead of failing. `--summary-only` draws the grid
without opening an NWB; `--animal Rat6` restricts it to named animals.

## Gotchas worth knowing

**Goal nodes are real node ids, 101–502.** They are not 1..N. If a goal node is
not in the node table, every `field_goal_*` / route score comes back NaN and
nothing raises — so empty distance columns usually mean a wrong goal node, not
broken place-field maths.

**Short hops score badly.** A goal counts as reached within
`GOAL_RADIUS_PX = 50`, but neighbouring maze nodes are only ~61 px apart, so on a
single-hop trial the goal registers almost immediately and `dist_log_score` is
strongly positive. Treat one-hop trials with suspicion.

**Session layout matters for grouping.** Session folders should be named
`YYYYMMDD` (otherwise the date falls back to the NWB `session_id`, then to an
8-digit token in the filename), and the NWB `session_description` needs
`Repeat N Session M` for the repeat/session grouping to work.

## Layout

```text
src/hm_rat_analysis/
├── maze.py            # geometry constants, node table, connectivity graph, shortest paths
├── behaviour.py       # reading tracker logs; speed, path length, clock alignment
├── nwb.py             # reading position and trial windows out of session NWBs
├── place_fields.py    # rate maps, place fields, place-field metrics
├── spike_metrics.py   # waveform metrics, ACG rise time, cell-type classifier
├── stats.py           # one-way ANOVA and Holm-corrected pairwise tests
├── data/              # node_list_new.csv
└── reports/           # one module per PDF, each with a main()
figures/               # proposal figures, run directly (msca_fig1.py composes msca_fig1a.py)
tests/                 # pytest suite; fixtures build synthetic sessions
notebooks/             # exploratory notebooks
```

Two coordinate systems are in play and mixing them is the easiest mistake to
make: **pixels** (what the tracker writes, and what the maze graph is built in)
and **metres** (pixels / `SCALE_X`, `SCALE_Y` — what every plot and metric uses).
Columns are suffixed accordingly: `x`/`y` are pixels, `x_m`/`y_m` are metres.

## Tests

```bash
pytest
```

The fixtures synthesise a behaviour session and a tree of NWBs, so the suite
needs no real recordings and runs in a few seconds.

## Relationship to HM_Tracker_2025

`reports/session_summary.py`, `nwb.py`, `place_fields.py` and `spike_metrics.py`
were split out of HM_Tracker_2025, where the summary used to be runner step `[s]`.

`nwb.py` and `place_fields.py` are the subset of the tracker's
`src/nwb/visualize_nwb.py` that the summary needs; the tracker keeps the full
module for its per-session step `[v]` PDFs, so **the rate-map and place-field code
now exists in both repos and must be kept in step if either changes.**
`spike_metrics.py` is an unmodified copy of the tracker's file — the same
implementation that writes the metrics into the NWB, so stored and recomputed
values cannot drift.
