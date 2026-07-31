"""Synthetic sessions, so the tests never need real recordings.

Both fixtures are session-scoped: building them costs a second or two (pynwb
writes real HDF5), and nothing mutates them.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from hm_rat_analysis import maze

#: A real node id, so route scoring has something to resolve against.
GOAL_NODE = 217
LOG_T0 = 1623484800.0


@pytest.fixture(scope="session")
def session_dir(tmp_path_factory):
    """A behaviour session folder: .log, .txt, Meta.xlsx and the two time CSVs."""
    out = tmp_path_factory.mktemp("behaviour_session")
    rng = np.random.default_rng(7)
    nodes = maze.node_table()
    goal = nodes[nodes["id"] == GOAL_NODE].iloc[0]

    lines, node_blocks = [], []
    frame, sys_t = 0, LOG_T0
    for trial in range(1, 5):
        lines.append(f"INFO: 00:00:00.000 {sys_t:.3f} : Recording Trial {trial}")
        n = 220 + 40 * trial
        start = nodes.iloc[(trial * 13) % len(nodes)]
        ramp = np.linspace(0, 1, n)
        x = start.x + (goal.x - start.x) * ramp + rng.normal(0, 25, n)
        y = start.y + (goal.y - start.y) * ramp + rng.normal(0, 25, n)
        if trial % 2 == 0:                       # even trials never reach the goal
            x = x - 400
        for i in range(n):
            sys_t += 1.0 / 30.0
            frame += 1
            secs = frame / 30.0
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            lines.append(
                f"INFO: {int(h):02d}:{int(m):02d}:{int(s):02d}.{int((s % 1) * 1000):03d} "
                f"{sys_t:.3f} : The rat position is: ({x[i]:.1f}, {y[i]:.1f}) @ {frame}")
        path = [str(int(nodes.iloc[(trial * 13 + k) % len(nodes)].id)) for k in range(6)]
        if trial % 2 == 1:
            path.append(str(GOAL_NODE))
        node_blocks += [", ".join(path), f"Summary Trial {trial}"]

    (out / "20210612_Rat5.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "20210612_Rat5.txt").write_text("\n".join(node_blocks) + "\n", encoding="utf-8")
    pd.DataFrame([{"Goal_Node": GOAL_NODE, "Rat": 5, "Repeat": 1, "Session": 1}]).to_excel(
        out / "20210612_Rat5_Meta.xlsx", index=False)

    n_ref = frame + 10
    pd.DataFrame({"Frame": np.arange(n_ref),
                  "Corrected Time Stamp": LOG_T0 + np.arange(n_ref) / 30.0}).to_csv(
        out / "20210612_Rat5_framewise_ts.csv", index=False)
    pd.DataFrame({"Frame": np.arange(n_ref),
                  "Corrected Time Stamp": np.arange(n_ref) / 30.0}).to_csv(
        out / "20210612_Rat5_stitched_second.csv", index=False)
    return out


@pytest.fixture(scope="session")
def nwb_root(tmp_path_factory):
    """A root/<animal>/<date>/*.nwb tree: 2 animals x 2 sessions, units + position."""
    pytest.importorskip("pynwb")
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.file import Subject

    root = tmp_path_factory.mktemp("nwb_root")
    rng = np.random.default_rng(0)

    for subj, dates in (("5", ["20210612", "20210613"]), ("6", ["20210612", "20210614"])):
        for si, date in enumerate(dates, start=1):
            sess = root / f"Rat{subj}" / date
            sess.mkdir(parents=True, exist_ok=True)

            nwbfile = NWBFile(
                session_description=f"HexMaze Repeat 1 Session {si} Goal_Node {GOAL_NODE}",
                identifier=f"Rat{subj}_{date}",
                session_start_time=datetime(2021, 6, 12, 10, 0, tzinfo=timezone.utc),
                session_id=date)
            nwbfile.subject = Subject(subject_id=subj, species="Rattus norvegicus")

            n, fs = 6000, 30.0
            t = np.arange(n) / fs
            walk = rng.normal(0, 6.0, size=(n, 2)).cumsum(axis=0)
            x = np.clip(walk[:, 0] % (9.0 * maze.SCALE_X), 0, 9.0 * maze.SCALE_X)
            y = np.clip(walk[:, 1] % (5.0 * maze.SCALE_Y), 0, 5.0 * maze.SCALE_Y)

            beh = nwbfile.create_processing_module("Behavior", "tracked position")
            beh.add(Position(spatial_series=SpatialSeries(
                name="Rat", data=np.column_stack([x, y]), timestamps=t,
                reference_frame="maze top-left", unit="pixels")))

            nwbfile.add_unit_column("quality_label", "curation label")
            for u in range(8):
                cx, cy = rng.uniform(0, 9.0 * maze.SCALE_X), rng.uniform(0, 5.0 * maze.SCALE_Y)
                d = np.hypot(x - cx, y - cy) / maze.SCALE_X
                spikes = t[rng.random(n) < np.exp(-(d ** 2) / (2 * 0.6 ** 2)) * 0.35]
                if spikes.size < 20:
                    spikes = np.sort(rng.uniform(0, t[-1], 60))
                nwbfile.add_unit(spike_times=spikes,
                                 quality_label="good" if u < 6 else "mua")

            with NWBHDF5IO(str(sess / f"Rat{subj}_{date}.nwb"), "w") as io:
                io.write(nwbfile)
    return root
