"""Reading the tracker's behavioural session outputs, and kinematics on them.

A session folder holds, for one recording day:

===========================  ==================================================
``*.log``                    per-frame rat positions + trial markers
``*.txt``                    per-trial node sequences ("Summary Trial N" blocks)
``*Meta.xlsx``               session metadata, including ``Goal_Node``
``*framewise_ts.csv``        video frame -> unix timestamp
``*second.csv``              video frame -> stitched seconds (the ephys clock)
===========================  ==================================================

The two CSVs are what let a log line (unix clock) be placed on the stitched
seconds clock that the NWB position and spike timestamps use — see
:func:`load_time_reference` and :func:`nearest_stitched_times`.
"""

import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

#: Tracker camera sampling rate (Hz). Position samples are assumed evenly spaced.
FS = 30.0

# "[LEVEL: ][HH:MM:SS.mmm ][unix ][: ]message"
_TS_LINE = re.compile(
    r'^(?:(?P<level>[A-Z]+)\s*:\s*)?'
    r'(?:(?P<video>\d{1,2}:\d{1,2}:\d{1,2}\.\d{3})\s*)?'
    r'(?:(?P<sys>\d+(?:\.\d+)?)\s*)?(?::\s*)?(?P<msg>.*)$')
_POS_LINE = re.compile(
    r'The rat position is:\s*\(\s*(?P<x>-?[\d\.]+),\s*(?P<y>-?[\d\.]+)\s*\)'
    r'\s*@\s*(?P<frame>[\d\.]+)')
_TRIAL_LINE = re.compile(r"Recording\s*Trial\s*(\d+)\b", flags=re.I)


# ------------------------------------------------------------
#                        kinematics
# ------------------------------------------------------------
def parse_video_to_seconds(ts_str):
    """Parse an ``HH:MM:SS.mmm`` video timestamp into seconds. None if unparseable."""
    if not ts_str:
        return None
    try:
        h, m, s_ms = ts_str.split(":")
        s, ms = s_ms.split(".")
        td = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
        return td.total_seconds()
    except ValueError:
        return None


def moving_average(a: np.ndarray, k: int) -> np.ndarray:
    """Centred `k`-sample moving average, edge-padded so the length is preserved."""
    if k <= 1 or a.size == 0:
        return a.astype(float, copy=True)
    kernel = np.ones(k) / k
    pad = k // 2
    a_pad = np.pad(a, (pad, pad), mode="edge")
    out = np.convolve(a_pad, kernel, mode="valid")
    if out.size > a.size:
        out = out[:a.size]
    return out


def compute_speed_from_xy(x: np.ndarray, y: np.ndarray, fs: float) -> np.ndarray:
    """Instantaneous speed from a trajectory, in position-units per second."""
    dt = 1.0 / fs
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    spd = np.hypot(vx, vy)
    return np.nan_to_num(spd, nan=0.0, posinf=0.0, neginf=0.0)


def compute_path_length(x: np.ndarray, y: np.ndarray) -> float:
    """Total distance travelled along a trajectory, in position units."""
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


# ------------------------------------------------------------
#                     locating session inputs
# ------------------------------------------------------------
def find_session_files(work_dir):
    """``{kind: [paths]}`` for the inputs a session folder is expected to hold.

    Keys are ``log``, ``txt``, ``meta``, ``framewise_ts`` and ``stitched_seconds``;
    each maps to a sorted list, empty when that input is absent.
    """
    work_dir = Path(work_dir)
    stitched = sorted(work_dir.glob("*second.csv"))
    if not stitched:
        stitched = sorted(work_dir.glob("stitched_framewise_seconds.csv"))
    return {"log": sorted(work_dir.glob("*.log")),
            "txt": sorted(work_dir.glob("*.txt")),
            "meta": sorted(work_dir.glob("*Meta.xlsx")),
            "framewise_ts": sorted(work_dir.glob("*framewise_ts.csv")),
            "stitched_seconds": stitched}


def parse_node_sequences(txt_path):
    """``{trial_id: "node, node, ..."}`` from a tracker ``.txt``.

    Each trial's node list is the line immediately preceding its
    ``Summary Trial N`` header.
    """
    sequences = {}
    try:
        lines = [l.strip() for l in Path(txt_path).read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    except OSError as e:
        print(f"Error reading node sequence file {txt_path}: {e}")
        return sequences
    header_re = re.compile(r"Summary Trial\s+(\d+)", re.IGNORECASE)
    node_line_re = re.compile(r"^[\d, ]+$")
    for i, line in enumerate(lines):
        m = header_re.search(line)
        if m and i > 0:
            prev = lines[i - 1]
            if node_line_re.match(prev.replace(" ", "").rstrip(",")):
                sequences[int(m.group(1))] = prev.strip(", ")
    return sequences


def load_session_meta(work_dir):
    """``(metadata_dict, goal_node_id)`` from ``*Meta.xlsx``.

    `goal_node_id` is a STRING, because maze graph nodes are keyed by string.
    Returns ``(None, None)`` when there is no readable metadata file.
    """
    paths = find_session_files(work_dir)["meta"]
    if not paths:
        return None, None
    try:
        df = pd.read_excel(paths[0])
    except Exception as e:
        print(f"Error parsing metadata {paths[0]}: {e}")
        return None, None
    if df.empty:
        return None, None
    meta = df.iloc[0].to_dict()
    goal = None
    if "Goal_Node" in meta and pd.notna(meta["Goal_Node"]):
        try:
            goal = str(int(meta["Goal_Node"]))
        except (TypeError, ValueError):
            goal = None
    return meta, goal


def load_time_reference(work_dir):
    """``(unix_timestamps, stitched_seconds)`` aligning the log clock to the ephys
    clock, or ``(None, None)`` when the reference CSVs are missing or unusable.

    Both files are expected to carry a ``Corrected Time Stamp`` column; if that
    name is absent the second column is used, which is what the tracker wrote
    before the column was named.
    """
    files = find_session_files(work_dir)
    ts_paths, sec_paths = files["framewise_ts"], files["stitched_seconds"]
    if not ts_paths or not sec_paths:
        print("Warning: time-reference CSVs not found; stitched time unavailable.")
        return None, None

    def column(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "Corrected Time Stamp" in df.columns:
            return df["Corrected Time Stamp"].values
        print(f"Warning: 'Corrected Time Stamp' not in {path.name}; "
              f"found {df.columns.tolist()}. Falling back to the 2nd column.")
        return df.iloc[:, 1].values if len(df.columns) > 1 else None

    try:
        ref_ts, ref_sec = column(ts_paths[0]), column(sec_paths[0])
    except Exception as e:
        print(f"Error loading time reference CSVs: {e}")
        return None, None
    if ref_ts is None or ref_sec is None:
        return None, None

    ref_ts = pd.to_numeric(ref_ts, errors="coerce")
    ref_sec = pd.to_numeric(ref_sec, errors="coerce")
    ok = ~np.isnan(ref_ts) & ~np.isnan(ref_sec)
    ref_ts, ref_sec = ref_ts[ok], ref_sec[ok]
    if len(ref_ts) != len(ref_sec):
        print(f"Length mismatch in time reference: ts={len(ref_ts)}, sec={len(ref_sec)}.")
        return None, None
    print(f"Loaded {len(ref_ts)} aligned time points.")
    return ref_ts, ref_sec


def nearest_stitched_times(log_sys_times, ref_ts, ref_secs):
    """Stitched seconds for each log unix timestamp, by nearest neighbour in `ref_ts`.

    NaN-filled if the reference is empty. `ref_ts` must be ascending.
    """
    log_sys_times = np.asarray(log_sys_times, dtype=float)
    ref_ts = np.asarray(ref_ts, dtype=float)
    ref_secs = np.asarray(ref_secs, dtype=float)
    if ref_ts.size == 0 or ref_secs.size == 0:
        return np.full_like(log_sys_times, np.nan)

    idx = np.clip(np.searchsorted(ref_ts, log_sys_times, side="left"), 0, len(ref_ts) - 1)
    left = np.clip(idx - 1, 0, len(ref_ts) - 1)
    closer_left = np.abs(log_sys_times - ref_ts[left]) < np.abs(log_sys_times - ref_ts[idx])
    return ref_secs[np.where(closer_left, left, idx)]


# ------------------------------------------------------------
#                        parsing the logs
# ------------------------------------------------------------
def parse_logs(log_paths):
    """Every parseable log line as a DataFrame.

    Columns: ``video_seconds``, ``sys_time`` (unix), ``event``
    (``rat_position`` / ``recording_start`` / ``message``), ``x``, ``y`` (pixels),
    and ``raw`` (the message text). Empty DataFrame if nothing parsed.
    """
    frames = []
    for log_path in log_paths:
        rows = []
        with Path(log_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = _TS_LINE.match(line)
                if not m:
                    continue
                msg = m.group("msg")
                sys_time_str = m.group("sys")

                x = y = None
                event = "message"
                mpos = _POS_LINE.search(msg)
                if mpos:
                    try:
                        x = int(float(mpos.group("x")))
                        y = int(float(mpos.group("y")))
                        event = "rat_position"
                    except ValueError:
                        pass
                elif msg.startswith("Recording Trial"):
                    event = "recording_start"

                rows.append({"video_seconds": parse_video_to_seconds(m.group("video")),
                             "sys_time": float(sys_time_str) if sys_time_str else None,
                             "event": event, "x": x, "y": y, "raw": msg})
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["video_seconds", "sys_time", "event", "x", "y", "raw"])
    df = pd.concat(frames, ignore_index=True)
    for col in ("video_seconds", "x", "y", "sys_time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def assign_trial_ids(df):
    """Add a ``trial_id`` column, carrying each ``Recording Trial N`` marker forward
    over the rows that follow it. Rows before the first marker belong to trial 1."""
    trial_ids = []
    current = 1
    for event, raw in zip(df["event"].astype(str), df["raw"].astype(str)):
        if event.lower() == "recording_start":
            m = _TRIAL_LINE.search(raw)
            if m:
                current = int(m.group(1))
        trial_ids.append(current)
    df = df.copy()
    df["trial_id"] = trial_ids
    return df


def trial_trajectories(df, drop_tail_samples=5):
    """Per-trial position arrays from a parsed, trial-tagged log.

    Returns a DataFrame with one row per trial: ``trial_id``, ``xy`` (N x 2 pixel
    array) and ``stitched_time`` (N stitched seconds). The last
    `drop_tail_samples` samples of each trial are discarded — the tracker keeps
    emitting positions briefly after a trial ends.
    """
    pos = df[df["event"] == "rat_position"].copy()
    if pos.empty:
        return pd.DataFrame(columns=["trial_id", "xy", "stitched_time"])
    sort_cols = [c for c in ("trial_id", "sys_time", "video_seconds") if c in pos.columns]
    pos = pos.sort_values(sort_cols, na_position="last")

    records = []
    for tid, g in pos.groupby("trial_id", sort=False):
        g = g.dropna(subset=["x", "y"])
        if g.empty:
            continue
        # guard drop_tail_samples == 0: iloc[:-0] is iloc[:0], i.e. everything
        if 0 < drop_tail_samples < len(g):
            g = g.iloc[:-drop_tail_samples]
        stitched = (g["stitched_time"].values if "stitched_time" in g.columns
                    else np.full(len(g), np.nan))
        records.append({"trial_id": tid,
                        "xy": np.column_stack([g["x"].values, g["y"].values]),
                        "stitched_time": stitched})
    return pd.DataFrame.from_records(
        records, columns=["trial_id", "xy", "stitched_time"])


def load_session(work_dir):
    """Everything the trial report needs from a session folder.

    Returns a dict with ``work_dir``, ``stem`` (the first log's filename stem),
    ``trials`` (:func:`trial_trajectories` output), ``node_sequences``, ``meta``
    and ``goal_node``. Raises ``FileNotFoundError`` if the folder has no logs.
    """
    work_dir = Path(work_dir)
    files = find_session_files(work_dir)
    if not files["log"]:
        raise FileNotFoundError(f"No .log files found in {work_dir}")
    print(f"Found {len(files['log'])} log file(s).")

    df = parse_logs(files["log"])
    if df.empty:
        raise ValueError(f"No parseable log lines in {work_dir}")
    df = assign_trial_ids(df)

    ref_ts, ref_sec = load_time_reference(work_dir)
    if ref_ts is not None:
        df["stitched_time"] = nearest_stitched_times(df["sys_time"].values, ref_ts, ref_sec)
    else:
        df["stitched_time"] = np.nan

    meta, goal_node = load_session_meta(work_dir)
    return {"work_dir": work_dir,
            "stem": files["log"][0].stem,
            "trials": trial_trajectories(df),
            "node_sequences": parse_node_sequences(files["txt"][0]) if files["txt"] else {},
            "meta": meta,
            "goal_node": goal_node}
