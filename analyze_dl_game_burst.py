"""Derive 10 Hz game-speed response after ball snap for rookie DL blitz plays.

This module reads only player-play keys in the already-filtered DL blitz cohort,
then scans the three game-tracking files in chunks. It retains movement features
before a play outcome window can bias the analysis: speed/displacement at fixed
times after snap plus a thresholded, explicitly approximate movement-onset delay.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from run_analysis import DEFAULT_ARCHIVE, open_data_source, parse_sensor_times

MEMBER = "nfl-big-data-bowl-2027/"
CHUNK_ROWS = 400_000
MOVE_THRESHOLD = 1.0


def read_csv(archive: zipfile.ZipFile, filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(archive.open(MEMBER + filename), **kwargs)


def interp(t: np.ndarray, values: np.ndarray, target: float) -> float:
    if len(t) < 2 or target < t[0] or target > t[-1]:
        return np.nan
    return float(np.interp(target, t, values))


def summarize_player_play(rows: list[tuple[str, float, float, float, str]]) -> dict:
    frame = pd.DataFrame(rows, columns=["time", "s", "x", "y", "event"])
    frame["time"] = parse_sensor_times(frame.time)
    frame[["s", "x", "y"]] = frame[["s", "x", "y"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["time", "s", "x", "y"]).sort_values("time")
    if len(frame) < 4:
        return {"game_tracking_status": "too_few_frames", "tracking_frames": int(len(frame))}
    frame = frame.drop_duplicates("time", keep="last")
    absolute_time_ns = frame.time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    # Subtract integers before converting to floating-point seconds. Some pandas
    # versions infer millisecond timestamps; assuming their integer unit is always
    # nanoseconds silently compresses a 0.1 s frame interval to 0.0001 s.
    t = (absolute_time_ns - absolute_time_ns[0]).astype(float) / 1e9
    speed_raw = frame.s.to_numpy(float)
    speed = pd.Series(speed_raw).rolling(3, center=True, min_periods=1).median().to_numpy()
    x, y = frame.x.to_numpy(float), frame.y.to_numpy(float)
    event = frame.event.fillna("").astype(str).to_numpy()
    snap_hits = np.flatnonzero(event == "ball_snap")
    if not len(snap_hits):
        return {"game_tracking_status": "ball_snap_missing", "tracking_frames": int(len(frame))}
    snap = int(snap_hits[0])
    rel_t = t - t[snap]
    pre = rel_t < 0
    post = rel_t >= 0
    result = {
        "game_tracking_status": "usable",
        "tracking_frames": int(len(frame)),
        "snap_time_resolution_s": 0.1,
        "speed_at_snap_ydps": float(speed[snap]),
        "pre_snap_max_speed_0_5s_ydps": float(np.max(speed[(rel_t >= -0.5) & pre])) if np.any((rel_t >= -0.5) & pre) else np.nan,
        "already_moving_before_snap": int(np.any((rel_t >= -0.5) & pre & (speed >= MOVE_THRESHOLD))),
        "movement_onset_speed_threshold_ydps": MOVE_THRESHOLD,
    }
    for horizon in (0.3, 0.5, 1.0):
        sh = interp(rel_t, speed, horizon)
        xh, yh = interp(rel_t, x, horizon), interp(rel_t, y, horizon)
        result[f"speed_{horizon:g}s_after_snap_ydps"] = sh
        result[f"speed_gain_0_{horizon:g}s_ydps"] = sh - speed[snap] if np.isfinite(sh) else np.nan
        result[f"net_displacement_0_{horizon:g}s_yd"] = float(np.hypot(xh - x[snap], yh - y[snap])) if np.isfinite(xh + yh) else np.nan
    t1 = interp(rel_t, speed, 1.0)
    result["early_speed_gain_0_1s_ydps"] = t1 - speed[snap] if np.isfinite(t1) else np.nan
    for horizon in (0.5, 1.0):
        section = post & (rel_t <= horizon)
        result[f"peak_speed_first_{horizon:g}s_after_snap_ydps"] = float(np.max(speed[section])) if section.any() else np.nan

    # Detect post-snap movement onset only for players who were below threshold
    # during the pre-snap observation window. At 10 Hz this is a coarse movement
    # onset estimate, not a physiological reaction-time measurement.
    onset = np.flatnonzero(post & (speed >= MOVE_THRESHOLD) & np.r_[speed[1:] >= MOVE_THRESHOLD, False])
    onset_time = float(rel_t[onset[0]]) if len(onset) else np.nan
    result["movement_onset_after_snap_s"] = onset_time
    result["movement_onset_within_1s_s"] = onset_time if np.isfinite(onset_time) and onset_time <= 1.0 else np.nan
    result["movement_onset_eligible"] = int(
        np.isfinite(onset_time) and onset_time <= 1.0 and not result["already_moving_before_snap"]
    )
    forward = np.flatnonzero(event == "pass_forward")
    result["time_to_pass_forward_after_snap_s"] = float(rel_t[forward[0]]) if len(forward) and rel_t[forward[0]] >= 0 else np.nan
    return result


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=root / "outputs")
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    play_path = args.output / "dl_rookie_blitz_opportunities.csv"
    plays = pd.read_csv(play_path)
    key_cols = ["game_id", "play_id", "nfl_id"]
    for col in key_cols:
        plays[col] = pd.to_numeric(plays[col], errors="coerce")
    plays = plays.dropna(subset=key_cols).copy()
    plays[key_cols] = plays[key_cols].astype(np.int64)
    expected = {(int(r.game_id), int(r.play_id), int(r.nfl_id)): r for r in plays.itertuples(index=False)}
    wanted_ids = set(plays.nfl_id.unique())
    frames: dict[tuple[int, int, int], list[tuple[str, float, float, float, str]]] = defaultdict(list)
    counts_by_file = {}

    with open_data_source(args.archive) as archive:
        for year in (2023, 2024, 2025):
            member = f"game_tracking_{year}.csv"
            read_rows = kept_rows = 0
            for chunk in read_csv(archive, member, usecols=key_cols + ["time", "s", "x", "y", "event"], chunksize=args.chunk_rows, low_memory=False):
                read_rows += len(chunk)
                for col in key_cols:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                chunk = chunk.loc[chunk.nfl_id.isin(wanted_ids)]
                if chunk.empty:
                    continue
                keys = list(zip(chunk.game_id.astype(np.int64), chunk.play_id.astype(np.int64), chunk.nfl_id.astype(np.int64)))
                mask = np.fromiter((key in expected for key in keys), dtype=bool, count=len(keys))
                chunk = chunk.loc[mask]
                keys = [key for key, keep in zip(keys, mask) if keep]
                if chunk.empty:
                    continue
                kept_rows += len(chunk)
                chunk = chunk.assign(_tracking_key=keys)
                for key, group in chunk.groupby("_tracking_key", sort=False):
                    records = list(group[["time", "s", "x", "y", "event"]].itertuples(index=False, name=None))
                    frames[key].extend((str(t), float(s), float(x), float(y), "" if pd.isna(ev) else str(ev)) for t, s, x, y, ev in records)
            counts_by_file[member] = {"rows_scanned": read_rows, "matching_player_play_frames": kept_rows}

    metrics = []
    for key, record in expected.items():
        base = {"game_id": key[0], "play_id": key[1], "nfl_id": key[2],
                "draft_year": int(record.draft_year), "nfl_position": record.nfl_position,
                "lined_up_position": record.lined_up_position,
                "down": record.down, "yards_to_go": record.yards_to_go,
                "pressure_event": int(record.pressure_event),
                "time_to_pressure": record.time_to_pressure,
                "player_get_off": record.player_get_off,
                "tracking_row_count": len(frames.get(key, []))}
        base.update(summarize_player_play(frames.get(key, [])))
        metrics.append(base)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.output / "dl_rookie_blitz_game_burst.csv", index=False)
    matched = metrics_df.game_tracking_status.eq("usable")
    snap_match = metrics_df.game_tracking_status.eq("usable")
    coverage = {
        "blitz_player_play_keys": int(len(metrics_df)),
        "keys_with_any_game_tracking_frames": int(metrics_df.tracking_row_count.gt(0).sum()),
        "keys_with_ball_snap_and_usable_tracking": int(snap_match.sum()),
        "keys_with_pressure_event": int(metrics_df.pressure_event.sum()),
        "keys_with_pressure_and_usable_tracking": int((metrics_df.pressure_event.eq(1) & matched).sum()),
        "player_count": int(metrics_df.nfl_id.nunique()),
        "plays_with_eligible_movement_onset_within_1s": int(metrics_df.movement_onset_eligible.eq(1).sum()),
        "players_with_eligible_movement_onset_within_1s": int(metrics_df.loc[metrics_df.movement_onset_eligible.eq(1), "nfl_id"].nunique()),
        "share_already_moving_pre_snap": float(metrics_df.loc[matched, "already_moving_before_snap"].mean()) if matched.any() else None,
        "coverage_by_file": counts_by_file,
        "interpretation": "Movement onset at 10 Hz is at most a coarse time-to-observed-movement measure. Exclude pre-snap movers from that outcome; do not call it pure reaction time. Fixed post-snap speed and displacement are reported separately.",
    }
    (args.output / "dl_game_burst_coverage.json").write_text(json.dumps(coverage, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
