# HM_Rat_Analysis

Analysis and figures for the HexMaze rat experiments. This repo takes the files
the [HM_Tracker_2025](https://github.com/Sachuriga/HM_Tracker_2025) pipeline
produces (session `.nwb` files, tracker `.log`/`.txt`/`.csv` outputs) and turns
them into plots — it does not track, sort, or package anything itself.

## Install

```bash
pip install -r requirements.txt
```

## Entry points

### `session_summary.py` — cross-session per-animal summary

Scans a folder tree for session NWBs, groups them by animal (NWB `subject_id`),
and plots each metric against session date (labelled with Repeat & Session):

- number of GOOD and MUA units
- among GOOD units, pyramidal vs interneuron counts
- pyramidal spatial information (Skaggs, bits/spike), selectivity (peak/mean),
  number of place fields
- pyramidal place-field distance to the goal node (mean / largest / 2nd-largest /
  smallest field)
- Bayesian decoding accuracy, and trial-level unit metrics where present

Cell type and waveform metrics are read from the NWB when the tracker's step `u`
stored them, and recomputed (`spike_metrics.py`) otherwise. Place-field metrics
are computed here from the NWB position + units, using each session's dominant
goal node.

```bash
python session_summary.py --root <folder> [--bin_cm 5] [--speed 0.05] [--smooth 2]
```

Writes `session_summary.pdf` and `session_summary.xlsx` (sheets: `sessions`,
`units`, `trial_unit_metrics`, `anova`, `posthoc`) into `--root`, falling back to
`session_summary*.csv` if the workbook cannot be written. `--root` is searched up
to four levels deep, so `root/`, `root/animal/session/` and similar layouts all
work.

### `hexmaze_analyzer.py` — per-session behavioural report

Parses the tracker's `.log` files (plus node sequences, `RecordingMeta.xlsx`, and
the stitched-time CSVs) for one session folder and writes a per-trial PDF with
trajectories, speed, occupancy, and shortest-path comparisons.

```bash
python hexmaze_analyzer.py -o <folder containing the .log files>
```

## Supporting modules

| File | Role |
| --- | --- |
| `nwb_io.py` | Reads position and trial windows out of an NWB; rate maps, place fields, and place-field metrics. |
| `spike_metrics.py` | Waveform metrics, ACG rise time, and the CellExplorer putative cell-type classifier. |
| `node_list_new.csv` | Maze node coordinates, used to locate the goal node. |

## Relationship to HM_Tracker_2025

`session_summary.py`, `nwb_io.py`, and `spike_metrics.py` were split out of
HM_Tracker_2025, where the summary used to be runner step `[s]`. `nwb_io.py` is
the subset of the tracker's `src/nwb/visualize_nwb.py` that the summary needs;
the tracker keeps the full module for its per-session step `[v]` PDFs, so the two
copies of the rate-map and place-field code must be kept in step if either is
changed. `spike_metrics.py` is an unmodified copy of the tracker's file — the same
implementation that writes the metrics into the NWB, so stored and recomputed
values cannot drift.
