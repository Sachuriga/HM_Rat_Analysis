"""Reading position and trial windows out of the session NWBs.

Extracted from HM_Tracker_2025 ``src/nwb/visualize_nwb.py`` — the subset the
cross-session summary needs. The tracker keeps the full module for its own
per-session PDFs.
"""

from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd


def find_nwb_file(output_folder):
    """First session NWB in `output_folder` (recursing if needed), or None."""
    op = Path(output_folder)
    # skip *.tmp.nwb and macOS AppleDouble sidecars ("._*.nwb", not valid HDF5)
    def ok(p):
        return not p.name.endswith(".tmp.nwb") and not p.name.startswith("._")
    cands = [p for p in sorted(op.glob("*.nwb")) if ok(p)]
    if cands:
        return cands[0]
    cands = [p for p in sorted(op.glob("**/*.nwb")) if ok(p)]
    return cands[0] if cands else None


def load_position(nwb):
    """Return (x, y, t) for the animal in session-relative seconds, or None.

    x and y are in the tracker's PIXEL units — divide by ``maze.SCALE_X`` /
    ``maze.SCALE_Y`` for metres.
    """
    try:
        pos = nwb.processing["Behavior"]["Position"]
    except Exception:
        return None
    ss = pos.spatial_series
    # Prefer a series literally named 'Rat'; else the first one.
    key = "Rat" if "Rat" in ss else next(iter(ss), None)
    if key is None:
        return None
    s = ss[key]
    xy = np.asarray(s.data[:], dtype=float)
    t = np.asarray(s.timestamps[:], dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    order = np.argsort(t)
    return x[order], y[order], t[order]


def _pick_file(op, pat):
    """First op-folder file matching `pat`, skipping macOS AppleDouble sidecars
    ("._*") that glob would otherwise return first."""
    return next((p for p in sorted(Path(op).glob(pat))
                 if not p.name.startswith("._")), None)


def _read_csv_tol(p):
    """read_csv tolerant of the non-UTF-8 bytes (e.g. a degree sign) that some
    framewise CSVs carry in their headers."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return pd.read_csv(p, encoding="utf-8", encoding_errors="ignore")


def _first_int(v):
    """First integer in a scalar or comma-list (Start_Nodes may be '224' or
    '224,315'); None if unparseable."""
    try:
        return int(str(v).split(",")[0])
    except (TypeError, ValueError):
        return None


def read_trial_meta(op_folder):
    """Per-trial (type, goal_node, start_node) from RecordingMeta.xlsx, keyed by
    1-based trial number (row order == Trial_Num in the coordinate file). Rows are
    NOT skipped, so numbering stays aligned with the position data's Trial_Num."""
    meta = _pick_file(op_folder, "RecordingMeta.xlsx") or _pick_file(op_folder, "*RecordingMeta.xlsx")
    if meta is None:
        return {}
    try:
        df = pd.read_excel(meta, sheet_name=0)
    except Exception as e:
        print(f"  Could not read RecordingMeta.xlsx ({e}).")
        return {}
    out = {}
    for i, r in enumerate(df.itertuples(index=False), start=1):
        d = r._asdict()
        try:
            ttype = int(d.get("Trial_Type"))
        except (TypeError, ValueError):
            ttype = -1
        goal = int(d["Goal_Node"]) if pd.notna(d.get("Goal_Node")) else None
        start = _first_int(d["Start_Nodes"]) if pd.notna(d.get("Start_Nodes")) else None
        out[i] = (ttype, goal, start)
    return out


def read_trials_raw(op_folder):
    """Per-trial (type, goal_node, start_node, start_unix, end_unix) from
    RecordingMeta.xlsx, in the raw behavioural-sync unix clock. Fallback only —
    that clock is offset (and drifts) relative to the video/position clock, so it
    is not used when the coordinate Trial_Num blocks are available. [] if absent."""
    meta = _pick_file(op_folder, "RecordingMeta.xlsx") or _pick_file(op_folder, "*RecordingMeta.xlsx")
    if meta is None:
        return []
    try:
        df = pd.read_excel(meta, sheet_name=0)
    except Exception as e:
        print(f"  Could not read RecordingMeta.xlsx ({e}).")
        return []
    need = {"Trial_Type", "trial_start_time", "trial_end_time"}
    if not need <= set(df.columns):
        print("  RecordingMeta.xlsx missing Trial_Type/trial_start_time/trial_end_time.")
        return []
    trials = []
    for _, r in df.iterrows():
        if pd.isna(r["trial_start_time"]) or pd.isna(r["trial_end_time"]):
            continue
        try:
            ttype = int(r["Trial_Type"])
        except (TypeError, ValueError):
            ttype = -1
        goal = int(r["Goal_Node"]) if "Goal_Node" in df.columns and pd.notna(r["Goal_Node"]) else None
        start = _first_int(r["Start_Nodes"]) if "Start_Nodes" in df.columns and pd.notna(r["Start_Nodes"]) else None
        trials.append((ttype, goal, start,
                       float(r["trial_start_time"]), float(r["trial_end_time"])))
    return trials


def frame_to_seconds(op_folder):
    """{frame_number: seconds} from stitched_framewise_seconds.csv
    ('Seconds From Creation'), which IS the clock the NWB position/spike
    timestamps use. None if unavailable."""
    st = _pick_file(op_folder, "*stitched_framewise_seconds.csv")
    if st is None:
        return None
    try:
        df = _read_csv_tol(st)
    except Exception as e:
        print(f"  Could not read stitched_framewise_seconds.csv ({e}).")
        return None
    fcol = next((c for c in df.columns if "frame" in c.lower()), None)
    scol = next((c for c in df.columns if "second" in c.lower()), None)
    if fcol is None or scol is None:
        return None
    frames = pd.to_numeric(df[fcol], errors="coerce")
    secs = pd.to_numeric(df[scol], errors="coerce")
    ok = frames.notna() & secs.notna()
    return dict(zip(frames[ok].astype(int), secs[ok].astype(float)))


def build_trials(op_folder, nwb_start, t_min, t_max):
    """Per-trial (type, goal_node, start_node, t0, t1) on the NWB seconds clock.

    Trial windows come from the coordinate file's Trial_Num blocks mapped to
    seconds via stitched_framewise_seconds.csv — the SAME source the tracker's
    plot_trials.py uses — so summary values align with the positions/spikes.
    (RecordingMeta's trial_start/end_time live on the behavioural sync clock,
    which is offset from and drifts against the video clock, so they misplace
    every trial; they are only a last-resort fallback.) Metadata
    (type/goal/start) is joined from RecordingMeta by Trial_Num.
    Falls back to align_trials(read_trials_raw(...)) if the coordinate/seconds
    files are missing."""
    coords = (_pick_file(op_folder, "*Coordinates_Full_with_frames.csv")
              or _pick_file(op_folder, "*Coordinates_Full.csv"))
    f2s = frame_to_seconds(op_folder)
    meta = read_trial_meta(op_folder)
    if coords is not None and f2s is not None:
        try:
            cf = _read_csv_tol(coords)
        except Exception as e:
            print(f"  Could not read {coords.name} ({e}); falling back to RecordingMeta times.")
            cf = None
        if cf is not None and {"Trial_Num", "Frame_Index"} <= set(cf.columns):
            sec = pd.to_numeric(cf["Frame_Index"], errors="coerce").map(f2s)
            tnum = pd.to_numeric(cf["Trial_Num"], errors="coerce")
            trials = []
            for k, grp in sec.groupby(tnum):
                g = grp.dropna()
                if g.empty:
                    continue
                tt, goal, start = meta.get(int(k), (-1, None, None))
                trials.append((tt, goal, start, float(g.min()), float(g.max())))
            if trials:
                trials.sort(key=lambda r: r[3])
                return trials
            print("  No Trial_Num blocks resolved to seconds; falling back to RecordingMeta times.")
    return align_trials(read_trials_raw(op_folder), nwb_start, t_min, t_max)


def align_trials(raw_trials, nwb_start, t_min, t_max):
    """Fallback: convert raw-unix RecordingMeta trials to session-relative seconds
    when the coordinate Trial_Num blocks are unavailable. Recovers the relative
    zero from session_start_time, trying both the aware and tz-replaced readings
    and keeping whichever lands the most trials inside [t_min, t_max].
    Returns [(type, goal_node, start_node, t0_rel, t1_rel), ...]."""
    if not raw_trials:
        return []
    cands = []
    try:
        cands.append(nwb_start.timestamp())                                  # fixed NWBs
    except Exception:
        pass
    try:
        cands.append(nwb_start.replace(tzinfo=timezone.utc).timestamp())     # legacy NWBs
    except Exception:
        pass
    if not cands:
        return []

    def in_range(off):
        return sum(1 for (_tt, _g, _s, s, e) in raw_trials if t_min <= s - off <= t_max)

    off = max(cands, key=in_range)
    return [(tt, g, sn, s - off, e - off) for (tt, g, sn, s, e) in raw_trials]
