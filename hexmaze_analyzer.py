import sys
import argparse
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import timedelta
import textwrap

# --- Imports for Graph Theory ---
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy import stats 

# --- Helper Functions ---

def parse_video_to_seconds(ts_str):
    """Parses HH:MM:SS.mmm strings into total seconds."""
    if not ts_str:
        return None
    try:
        h, m, s_ms = ts_str.split(":")
        s, ms = s_ms.split(".")
        td = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
        return td.total_seconds()
    except Exception:
        return None

def moving_average(a: np.ndarray, k: int) -> np.ndarray:
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
    dt = 1.0 / fs
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    spd = np.hypot(vx, vy)
    spd = np.nan_to_num(spd, nan=0.0, posinf=0.0, neginf=0.0)
    return spd

def compute_path_length(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)
    return np.sum(dists)

def parse_node_sequences(txt_path):
    sequences = {}
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        trial_header_re = re.compile(r"Summary Trial\s+(\d+)", re.IGNORECASE)
        node_line_re = re.compile(r"^[\d, ]+$")
        for i, line in enumerate(lines):
            m = trial_header_re.search(line)
            if m:
                trial_id = int(m.group(1))
                if i > 0:
                    prev_line = lines[i-1]
                    if node_line_re.match(prev_line.replace(" ", "").rstrip(',')):
                         sequences[trial_id] = prev_line.strip(', ')
    except Exception as e:
        print(f"Error parsing node sequence file: {e}")
    return sequences

def find_nearest_stitched_times(log_sys_times, ref_ts, ref_secs):
    """
    Finds the stitched time for each log timestamp by finding the nearest neighbor in ref_ts.
    
    Args:
        log_sys_times (np.array): Array of Unix timestamps from the .log file.
        ref_ts (np.array): Array of Unix timestamps from framewise_ts.csv.
        ref_secs (np.array): Array of stitched seconds from stitched_framewise_seconds.csv.
    
    Returns:
        np.array: Array of stitched seconds corresponding to log_sys_times.
    """
    # Ensure inputs are numpy arrays
    log_sys_times = np.array(log_sys_times)
    ref_ts = np.array(ref_ts)
    ref_secs = np.array(ref_secs)
    
    if ref_ts.size == 0 or ref_secs.size == 0:
        return np.full_like(log_sys_times, np.nan)

    # Find insertion points
    # searchsorted returns the index where elements should be inserted to maintain order
    idx = np.searchsorted(ref_ts, log_sys_times, side="left")
    
    # Clip indices to valid range [0, len-1]
    # This initially maps everything to the right neighbor
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    
    # Now check if the left neighbor (idx - 1) is actually closer
    left_idx = np.clip(idx - 1, 0, len(ref_ts) - 1)
    right_idx = idx
    
    dist_left = np.abs(log_sys_times - ref_ts[left_idx])
    dist_right = np.abs(log_sys_times - ref_ts[right_idx])
    
    # Choose the index with smaller distance
    final_idx = np.where(dist_left < dist_right, left_idx, right_idx)
    
    # Return the corresponding values from the seconds file
    return ref_secs[final_idx]

def build_hexmaze_graph(nodes_df):
    G = nx.Graph()
    nodes_df['id_str'] = nodes_df['id'].astype(int).astype(str)
    pos_dict = {}
    for idx, row in nodes_df.iterrows():
        node_id = row['id_str']
        G.add_node(node_id, pos=(row['x'], row['y']))
        pos_dict[node_id] = np.array([row['x'], row['y']])

    coords = nodes_df[['x', 'y']].values
    distances = squareform(pdist(coords))
    threshold = 65
    node_ids = nodes_df['id_str'].tolist()
    
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            dist = distances[i, j]
            if dist < threshold:
                G.add_edge(node_ids[i], node_ids[j], weight=dist)

    for n in ['501', '502']:
        if n in G: G.remove_node(n)

    manual_edges = [
        ('121', '302'), ('324', '401'), ('305', '220'),
        ('404', '223'), ('201', '124'), ('224', '218'),
    ]
    for u, v in manual_edges:
        if u in G and v in G:
            p1 = pos_dict[u]
            p2 = pos_dict[v]
            w = np.linalg.norm(p1 - p2)
            G.add_edge(u, v, weight=w)
    return G

def get_all_shortest_paths_plot_data(G, start_node, end_node, weight_mode='weight'):
    all_paths_segments = []
    label = "No Path"
    metric_val = 0.0
    try:
        if start_node not in G or end_node not in G: return [], "Node not found", 0.0
        if not nx.has_path(G, start_node, end_node): return [], "No Path", 0.0
        paths_iter = nx.all_shortest_paths(G, source=start_node, target=end_node, weight=weight_mode)
        metric_val = nx.shortest_path_length(G, source=start_node, target=end_node, weight=weight_mode)
        path_count = 0
        pos = nx.get_node_attributes(G, 'pos')
        for path in paths_iter:
            path_count += 1
            current_segments = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                current_segments.append((pos[u], pos[v]))
            all_paths_segments.append(current_segments)
        label = f"{'Dist' if weight_mode=='weight' else 'Hops'}: {metric_val:.1f} (N={path_count})"
        return all_paths_segments, label, metric_val
    except nx.NetworkXNoPath: return [], "No Path", 0.0
    except Exception as e: return [], str(e), 0.0

# --- Main Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Log Files from Output Folder')
    parser.add_argument('-o', "--output", dest='output_folder', required=True, 
                        help='Folder path containing .log files')
    args = parser.parse_args()
    work_dir = Path(args.output_folder)
    if not work_dir.exists(): sys.exit(f"Error: {work_dir} does not exist.")

    # 1. Find logs
    LOG_GLOB = str(work_dir / "*.log")
    log_paths = sorted(glob.glob(LOG_GLOB, recursive=True))
    if not log_paths: sys.exit(f"No .log files found in {work_dir}")
    print(f"Found {len(log_paths)} log files.")
    log_file_stem = Path(log_paths[0]).stem

    # --- 1b. Node Sequences ---
    TXT_GLOB = str(work_dir / "*.txt")
    txt_paths = sorted(glob.glob(TXT_GLOB))
    trial_node_sequences = parse_node_sequences(txt_paths[0]) if txt_paths else {}

    # --- 1c. Metadata ---
    XLSX_GLOB = str(work_dir / "*Meta.xlsx")
    xlsx_paths = sorted(glob.glob(XLSX_GLOB))
    session_meta, target_goal_node_id = None, None
    if xlsx_paths:
        try:
            meta_df = pd.read_excel(xlsx_paths[0])
            if not meta_df.empty:
                session_meta = meta_df.iloc[0].to_dict()
                if 'Goal_Node' in session_meta:
                    target_goal_node_id = str(int(session_meta['Goal_Node']))
        except Exception as e: print(f"Error parsing metadata: {e}")

    # ... (前面的代码不变) ...

    # -------------------------------------------------------------
    # [FIXED] 1d. Find and Load Stitched Time CSVs by Column Name
    # -------------------------------------------------------------
    FRAMEWISE_TS_GLOB = str(work_dir / "*framewise_ts.csv")
    STITCHED_SEC_GLOB = str(work_dir / "*second.csv")  # 根据你的描述，可能是 *second.csv 或类似的
    # 如果你的文件名是 "stitched_framewise_seconds.csv"，请保留原来的GLOB，或者确保它能被找到

    # 尝试找到匹配的文件
    fw_ts_paths = sorted(glob.glob(FRAMEWISE_TS_GLOB))
    # 注意：这里稍微放宽搜索条件以匹配 "*second.csv"
    if not sorted(glob.glob(STITCHED_SEC_GLOB)):
        # 如果找不到 *second.csv，尝试找原来的名字
        STITCHED_SEC_GLOB = str(work_dir / "stitched_framewise_seconds.csv")
    stitched_sec_paths = sorted(glob.glob(STITCHED_SEC_GLOB))
    
    ref_ts_data = None
    ref_sec_data = None
    has_stitched_time = False

    if fw_ts_paths and stitched_sec_paths:
        print(f"Found Reference TS file: {fw_ts_paths[0]}")
        print(f"Found Stitched Seconds file: {stitched_sec_paths[0]}")
        try:
            # 关键修改：显式读取列名
            # 1. 读取 TS 文件
            df_ts = pd.read_csv(fw_ts_paths[0])
            # 清除列名的空格，防止 "Corrected Time Stamp " 这种情况
            df_ts.columns = df_ts.columns.str.strip()
            
            if "Corrected Time Stamp" in df_ts.columns:
                ref_ts_data = df_ts["Corrected Time Stamp"].values
            else:
                # 如果找不到名字，打印所有列名供调试，并尝试回退到第二列(假设第一列是framenumber)
                print(f"Warning: 'Corrected Time Stamp' column not found in TS file. Found: {df_ts.columns.tolist()}")
                if len(df_ts.columns) > 1:
                    print("Attempting to use the 2nd column as data...")
                    ref_ts_data = df_ts.iloc[:, 1].values
            
            # 2. 读取 Seconds 文件
            df_sec = pd.read_csv(stitched_sec_paths[0])
            df_sec.columns = df_sec.columns.str.strip()
            
            if "Corrected Time Stamp" in df_sec.columns:
                ref_sec_data = df_sec["Corrected Time Stamp"].values
            else:
                print(f"Warning: 'Corrected Time Stamp' column not found in Seconds file. Found: {df_sec.columns.tolist()}")
                if len(df_sec.columns) > 1:
                    print("Attempting to use the 2nd column as data...")
                    ref_sec_data = df_sec.iloc[:, 1].values

            # 3. 验证数据
            if ref_ts_data is not None and ref_sec_data is not None:
                # 确保是数值类型
                ref_ts_data = pd.to_numeric(ref_ts_data, errors='coerce')
                ref_sec_data = pd.to_numeric(ref_sec_data, errors='coerce')
                
                # 去除 NaN (如果有标题行被误读)
                valid_mask = ~np.isnan(ref_ts_data) & ~np.isnan(ref_sec_data)
                ref_ts_data = ref_ts_data[valid_mask]
                ref_sec_data = ref_sec_data[valid_mask]

                if len(ref_ts_data) == len(ref_sec_data):
                    has_stitched_time = True
                    print(f"Successfully loaded {len(ref_ts_data)} aligned time points.")
                else:
                    print(f"Length Mismatch! TS: {len(ref_ts_data)}, Sec: {len(ref_sec_data)}")
        except Exception as e:
            print(f"Error loading time reference CSVs: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: Reference CSV files not found.")
    
    # -------------------------------------------------------------

    # --- 2. Parse Logs ---
    ts_line_new = re.compile(r'^(?:(?P<level>[A-Z]+)\s*:\s*)?(?:(?P<video>\d{1,2}:\d{1,2}:\d{1,2}\.\d{3})\s*)?(?:(?P<sys>\d+(?:\.\d+)?)\s*)?(?::\s*)?(?P<msg>.*)$')
    pos_line = re.compile(r'The rat position is:\s*\(\s*(?P<x>-?[\d\.]+),\s*(?P<y>-?[\d\.]+)\s*\)\s*@\s*(?P<frame>[\d\.]+)')
    all_dfs = []

    for log_path in log_paths:
        rows_new = []
        with Path(log_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                m = ts_line_new.match(line)
                if not m: continue
                msg = m.group("msg")
                sys_time_str = m.group("sys")
                
                x = y = frame = None
                mpos = pos_line.search(msg)
                event = "message"
                if mpos:
                    try:
                        x = int(float(mpos.group("x")))
                        y = int(float(mpos.group("y")))
                        frame = int(float(mpos.group("frame")))
                        event = "rat_position"
                    except: pass
                else:
                    if msg.startswith("Recording Trial"): event = "recording_start"
                
                rows_new.append({
                    "video_seconds": parse_video_to_seconds(m.group("video")),
                    "sys_time": float(sys_time_str) if sys_time_str else None,
                    "event": event,
                    "x": x, "y": y, "raw": msg
                })
        if rows_new: all_dfs.append(pd.DataFrame(rows_new))

    if not all_dfs: sys.exit("No valid data.")
    df = pd.concat(all_dfs, ignore_index=True)
    
    for col in ["video_seconds", "x", "y", "sys_time"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 3. Assign Trial IDs ---
    trial_id_list = []
    current = 1
    trial_re = re.compile(r"Recording\s*Trial\s*(\d+)\b", flags=re.I)
    for _, row in df.iterrows():
        if str(row.get("event", "")).lower() == "recording_start":
            m = trial_re.search(str(row.get("raw", "")))
            if m: current = int(m.group(1))
        trial_id_list.append(current)
    df["trial_id"] = trial_id_list

    # -------------------------------------------------------------
    # [NEW] 3b. Perform Time Synchronization
    # -------------------------------------------------------------
    if has_stitched_time:
        print("Synchronizing Log Sys Time to Stitched Seconds...")
        # Get valid sys_times from log
        log_sys_vals = df['sys_time'].values
        # Perform Nearest Neighbor lookup
        stitched_vals = find_nearest_stitched_times(log_sys_vals, ref_ts_data, ref_sec_data)
        df['stitched_time'] = stitched_vals
    else:
        df['stitched_time'] = np.nan
    # -------------------------------------------------------------

    # --- 4. Process Position Data ---
    pos_df = df[df["event"] == "rat_position"].copy()
    if not pos_df.empty:
        sort_cols = [c for c in ["trial_id", "sys_time", "video_seconds"] if c in pos_df.columns]
        pos_df = pos_df.sort_values(sort_cols, na_position="last")

    # --- 5. Per-Trial Aggregation ---
    records = []
    grouped = pos_df.groupby("trial_id", sort=False)
    
    for tid, g in grouped:
        if g.empty: continue
        g_valid = g.dropna(subset=["x", "y"])
        if g_valid.empty: continue
        if len(g_valid) > 5: g_valid = g_valid.iloc[:-5]

        # Extract stitched time alongside XY
        stitched_t_seq = g_valid['stitched_time'].values if 'stitched_time' in g_valid.columns else np.full(len(g_valid), np.nan)

        xy_seq = list(zip(g_valid["x"], g_valid["y"]))
        records.append({
            "trial_id": tid,
            "xy": xy_seq,
            "stitched_time": stitched_t_seq # Store in intermediate record
        })

    per_trial_df = pd.DataFrame.from_records(records)
    if "xy" in per_trial_df.columns:
        per_trial_df["xy"] = per_trial_df["xy"].apply(lambda p: np.asarray(list(p)) if p else np.empty((0, 2)))

    # --- 6. Init Mega Data Storage ---
    mega_data_storage = {
        "trial_ids": [],
        "raw_x_scaled": [],
        "raw_y_scaled": [],
        "speed_raw_smoothed": [],
        "speed_0_5s": [], "speed_1_0s": [], "speed_2_0s": [], "speed_5_0s": [],
        "time_seconds": [], 
        "normalized_time": [],
        "stitched_time_seconds": [], # [NEW] Storage Column
        "physical_score_val": [], "hops_score_val": [],
        "path_physical_segments": [], "path_topological_segments": [],
        "node_sequence_str": []
    }

    FS = 30.0
    DT = 1.0 / FS
    X_SCALE_DEN, Y_SCALE_DEN = (2352 / 2 / 9), (1424 / 2 / 5)
    
    # Smooth kernels
    SMOOTH_SAMPLES_RAW = max(1, int(round((400.0 / 1000.0) * FS))) 
    kernels = {k: max(1, int(round(k * FS))) for k in [0.5, 1.0, 2.0, 5.0]}

    # Graph Setup
    node_file = Path("node_list_new.csv")
    nodes_data, maze_graph = None, None
    if node_file.exists():
        try:
            nodes_df = pd.read_csv(node_file, header=None, names=["id", "x", "y"])
            nodes_df["x_scaled"] = nodes_df["x"] / X_SCALE_DEN
            nodes_df["y_scaled"] = nodes_df["y"] / Y_SCALE_DEN
            nodes_data = nodes_df
            maze_graph = build_hexmaze_graph(nodes_df)
        except: pass

    pdf_path = work_dir / f"{log_file_stem}_analysis_final.pdf"
    
    agg_data = {'0.5s': [], '1.0s': [], '2.0s': [], '5.0s': []}
    all_trials_speed_raw_list = []
    global_x_scaled, global_y_scaled, global_speed_vals = [], [], []
    summary_metrics = []

    print(f"Generating PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        if session_meta:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111); ax.axis('off')
            txt = "SESSION METADATA\n\n" + "\n".join([f"{k:<20}: {str(v)[:80]}" for k,v in session_meta.items()])
            ax.text(0.1, 0.8, txt, family='monospace', va='top')
            pdf.savefig(fig); plt.close(fig)

        for i, row in per_trial_df.iterrows():
            trial_id = row.get("trial_id", "Unknown")
            xy_arr = row["xy"]
            stitched_t_arr = row["stitched_time"] # [NEW] Retrieve stitched time array

            if xy_arr.size == 0: continue
            x_raw, y_raw = xy_arr[:, 0], xy_arr[:, 1]
            x_calc, y_calc = x_raw / X_SCALE_DEN, y_raw / Y_SCALE_DEN
            speed = compute_speed_from_xy(x_calc, y_calc, FS)
            
            x_plot, y_plot = x_calc.copy(), y_calc.copy()
            
            # Goal Logic
            goal_reached_naturally = False
            first_goal_visit_idx = -1
            gx_scaled, gy_scaled, gx_raw, gy_raw = None, None, None, None
            
            if target_goal_node_id and nodes_data is not None:
                g_row = nodes_data[nodes_data['id_str'] == target_goal_node_id]
                if not g_row.empty:
                    gx_scaled, gy_scaled = g_row.iloc[0]['x_scaled'], g_row.iloc[0]['y_scaled']
                    gx_raw, gy_raw = g_row.iloc[0]['x'], g_row.iloc[0]['y']
                    d_sq = (x_raw - gx_raw)**2 + (y_raw - gy_raw)**2
                    arr_idx = np.where(d_sq < (50**2))[0]
                    if len(arr_idx) > 0:
                        goal_reached_naturally = True
                        first_goal_visit_idx = arr_idx[0]

            appended = False
            if target_goal_node_id and not goal_reached_naturally and gx_scaled is not None:
                if np.sqrt((x_plot[-1]-gx_scaled)**2 + (y_plot[-1]-gy_scaled)**2) > 0.5:
                    x_plot = np.append(x_plot, gx_scaled)
                    y_plot = np.append(y_plot, gy_scaled)
                    appended = True

            global_x_scaled.append(x_calc); global_y_scaled.append(y_calc); global_speed_vals.append(speed)
            
            spd_raw_sm = moving_average(speed, SMOOTH_SAMPLES_RAW)
            spd_smooths = {k: moving_average(speed, v) for k,v in kernels.items()}
            
            all_trials_speed_raw_list.append((trial_id, spd_raw_sm))
            
            if len(speed) > 1:
                norm_common = np.linspace(0, 1, 100)
                curr_norm = np.linspace(0, 1, len(speed))
                for k in agg_data: agg_data[k].append(np.interp(norm_common, curr_norm, spd_smooths[float(k[:-1])]))

            time_vec = np.arange(len(speed)) * DT
            norm_time_vec = np.linspace(0, 1, len(speed)) if len(speed) > 1 else np.array([0.0])
            speed_vis = spd_raw_sm.copy()
            speed_vis[speed_vis>1] = 1
            if appended: speed_vis = np.append(speed_vis, 0.0)

            # Histogram
            H, _, _ = np.histogram2d(x_calc, y_calc, bins=[50, 30], range=[[0, 9], [0, 5]])
            H_rel_mask = np.ma.masked_where(H.T == 0, H.T / (H.sum() or 1))
            H_sec_mask = np.ma.masked_where(H.T == 0, H.T * DT)

            # Scoring
            score_note = "(Full Path)"
            if goal_reached_naturally and first_goal_visit_idx > 0:
                act_dist = compute_path_length(x_raw[:first_goal_visit_idx+1], y_raw[:first_goal_visit_idx+1])
                score_note = "(Start->FirstGoal)"
            else:
                act_dist = compute_path_length(x_raw, y_raw)
                if appended: act_dist += np.sqrt((x_raw[-1]-gx_raw)**2 + (y_raw[-1]-gy_raw)**2)

            act_hops = 0
            seq_str = trial_node_sequences.get(trial_id, "")
            start_node = None
            if seq_str:
                p_nodes = [t.strip() for t in seq_str.split(',') if t.strip()]
                if p_nodes:
                    start_node = p_nodes[0]
                    if target_goal_node_id in p_nodes:
                        act_hops = p_nodes.index(target_goal_node_id)
                        if "FirstGoal" not in score_note: score_note = "(Start->GoalNode)"
                    else:
                        act_hops = max(0, len(p_nodes)-1)
            
            opt_dist, opt_hops = 0.0, 0
            d_msg, h_msg, d_val, h_val = "N/A", "N/A", np.nan, np.nan
            segs_dist, segs_hops = [], []
            
            if maze_graph and start_node and target_goal_node_id:
                try:
                    opt_dist = nx.shortest_path_length(maze_graph, start_node, target_goal_node_id, weight='weight')
                    d_val = np.log(opt_dist/act_dist) if act_dist > 0 else np.nan
                    d_msg = f"{d_val:.3f}" if not np.isnan(d_val) else "Err"
                    
                    opt_hops = nx.shortest_path_length(maze_graph, start_node, target_goal_node_id, weight=None)
                    h_val = np.log(opt_hops/act_hops) if act_hops > 0 else (0.0 if opt_hops==0 and act_hops==0 else np.nan)
                    h_msg = f"{h_val:.3f}" if not np.isnan(h_val) else "Err"
                    
                    segs_dist, _, _ = get_all_shortest_paths_plot_data(maze_graph, start_node, target_goal_node_id, 'weight')
                    segs_hops, _, _ = get_all_shortest_paths_plot_data(maze_graph, start_node, target_goal_node_id, None)
                except: d_msg, h_msg = "No Path", "No Path"

            summary_metrics.append({'trial_id': trial_id, 'avg_speed': np.mean(speed) if len(speed)>0 else 0, 'median_speed': np.median(speed) if len(speed)>0 else 0, 'dist_log_score': d_val, 'hops_log_score': h_val})

            # --- [NEW] Store Data ---
            mega_data_storage["trial_ids"].append(trial_id)
            mega_data_storage["raw_x_scaled"].append(x_plot.tolist())
            mega_data_storage["raw_y_scaled"].append(y_plot.tolist())
            mega_data_storage["speed_raw_smoothed"].append(spd_raw_sm.tolist())
            mega_data_storage["speed_0_5s"].append(spd_smooths[0.5].tolist())
            mega_data_storage["speed_1_0s"].append(spd_smooths[1.0].tolist())
            mega_data_storage["speed_2_0s"].append(spd_smooths[2.0].tolist())
            mega_data_storage["speed_5_0s"].append(spd_smooths[5.0].tolist())
            mega_data_storage["time_seconds"].append(time_vec.tolist())
            mega_data_storage["normalized_time"].append(norm_time_vec.tolist())
            mega_data_storage["stitched_time_seconds"].append(stitched_t_arr.tolist()) # [NEW]
            mega_data_storage["physical_score_val"].append(d_val)
            mega_data_storage["hops_score_val"].append(h_val)
            mega_data_storage["node_sequence_str"].append(seq_str)
            
            def serialize_path(paths): return [[[float(p[0]), float(p[1])] for p in s] for path in paths for s in path]
            mega_data_storage["path_physical_segments"].append(serialize_path(segs_dist))
            mega_data_storage["path_topological_segments"].append(serialize_path(segs_hops))

            # --- Plotting (Simplified for brevity as logic is identical) ---
            fig = plt.figure(figsize=(12, 23))
            gs = fig.add_gridspec(6, 2, height_ratios=[0.3, 1, 1, 1, 0.6, 0.6])
            
            # Text Summary
            ax_txt = fig.add_subplot(gs[0, :]); ax_txt.axis('off')
            txt_info = f"Trial {trial_id} | Goal: {target_goal_node_id}\nScore(Dist): {d_msg} | Score(Hops): {h_msg}"
            ax_txt.text(0.5, 0.5, txt_info, ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", fc="#f0f0f0"))

            # Speed Track
            ax0 = fig.add_subplot(gs[1, 0])
            sc = ax0.scatter(x_plot, y_plot, c=speed_vis, s=10, vmax=1, cmap='hot', rasterized=True)
            ax0.set_title("Speed Track")
            
            # Time Track
            ax1 = fig.add_subplot(gs[1, 1])
            if len(x_plot) > 1:
                pts = np.column_stack([x_plot, y_plot])
                lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="cool", norm=plt.Normalize(0,1), rasterized=True)
                lc.set_array(np.linspace(0,1,len(pts)-1)); ax1.add_collection(lc)
            ax1.set_title("Time Evolution")

            # Paths
            ax_pd, ax_ph = fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])
            for segs in segs_dist: 
                for p1, p2 in segs: ax_pd.plot([p1[0]/X_SCALE_DEN, p2[0]/X_SCALE_DEN], [p1[1]/Y_SCALE_DEN, p2[1]/Y_SCALE_DEN], 'b', alpha=0.4, lw=3)
            for segs in segs_hops:
                for p1, p2 in segs: ax_ph.plot([p1[0]/X_SCALE_DEN, p2[0]/X_SCALE_DEN], [p1[1]/Y_SCALE_DEN, p2[1]/Y_SCALE_DEN], 'purple', alpha=0.4, lw=3)
            
            # Occupancy
            ax2, ax3 = fig.add_subplot(gs[3,0]), fig.add_subplot(gs[3,1])
            ax2.imshow(H_rel_mask, extent=[0,9,5,0], cmap='jet', aspect='auto')
            ax3.imshow(H_sec_mask, extent=[0,9,5,0], cmap='jet', aspect='auto', vmax=5)

            # Line Plots
            ax4, ax5 = fig.add_subplot(gs[4,:]), fig.add_subplot(gs[5,:])
            ax4.plot(time_vec, spd_raw_sm, color='gray', alpha=0.3); ax4.plot(time_vec, spd_smooths[0.5], label='0.5s')
            ax5.plot(norm_time_vec, spd_raw_sm, color='gray', alpha=0.3); ax5.plot(norm_time_vec, spd_smooths[0.5], label='0.5s')
            
            for ax in [ax0, ax1, ax_pd, ax_ph, ax2, ax3]:
                ax.set_xlim(0,9); ax.set_ylim(5,0); ax.set_aspect('equal')
                if nodes_data is not None: ax.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], c='none', edgecolors='grey', alpha=0.3)
            
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # --- Aggregate Plots (Speed Hists, etc.) ---
        # (Included in simplified form to ensure PDF completeness)
        if all_trials_speed_raw_list:
             fig, ax = plt.subplots()
             all_s = np.concatenate([s for _, s in all_trials_speed_raw_list])
             ax.hist(all_s, bins=50, density=True, histtype='step', color='k')
             ax.set_title("Aggregate Speed Dist"); pdf.savefig(fig); plt.close(fig)
        
        # --- Save Mega Data ---
        mega_df = pd.DataFrame([mega_data_storage])
        pkl_path = work_dir / f"{log_file_stem}_all_plot_data.pkl"
        print(f"Saving compiled data (incl. stitched time) to: {pkl_path}")
        mega_df.to_pickle(pkl_path)

    print("Done.")