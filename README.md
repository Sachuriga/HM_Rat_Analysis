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
| `session_summary.xlsx` | sheets `sessions`, `units`, `trial_unit_metrics`, `anova`, `posthoc` — falls back to `session_summary*.csv` if the workbook cannot be written |

Metrics per session: GOOD/MUA unit counts, pyramidal vs interneuron counts,
pyramidal spatial information (Skaggs, bits/spike), selectivity (peak/mean),
place-field count, and place-field distance to the goal node.

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
