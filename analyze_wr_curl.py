"""Extract event-aligned WR curl-route speed features and rookie separation outcomes.

This is the first focused analysis module. It deliberately separates movement-event
measurement from NFL outcome construction so both can be inspected before modeling.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from run_analysis import DEFAULT_ARCHIVE, open_data_source, parse_sensor_times

MEMBER = "nfl-big-data-bowl-2027/"


def read_csv(archive: zipfile.ZipFile, filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(archive.open(MEMBER + filename), **kwargs)


def circular_turn_degrees(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Absolute heading change over a centered four-frame (0.4 s) window."""
    turn = np.full(len(vx), np.nan)
    for idx in range(2, len(vx) - 2):
        before = np.array([vx[idx - 2], vy[idx - 2]])
        after = np.array([vx[idx + 2], vy[idx + 2]])
        if np.linalg.norm(before) < 1.5 or np.linalg.norm(after) < 1.5:
            continue
        cross = before[0] * after[1] - before[1] * after[0]
        dot = float(before @ after)
        turn[idx] = abs(np.degrees(np.arctan2(cross, dot)))
    return turn


def summarize_curl_attempt(group: pd.DataFrame, turn_threshold: float = 35.0, min_pre_speed: float = 1.5) -> dict:
    g = group.sort_values("time")
    t0 = parse_sensor_times(g.time)
    t = (t0 - t0.iloc[0]).dt.total_seconds().to_numpy(float)
    x = pd.to_numeric(g.x, errors="coerce").to_numpy(float)
    y = pd.to_numeric(g.y, errors="coerce").to_numpy(float)
    s = pd.to_numeric(g.s, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    t, x, y, s = t[valid], x[valid], y[valid], s[valid]
    base = {"frames": int(len(t)), "duration_s": float(t[-1] - t[0]) if len(t) else np.nan}
    if len(t) < 12 or np.any(np.diff(t) <= 0):
        return {**base, "event_status": "too_few_or_invalid_frames"}

    smooth_s = pd.Series(s).rolling(3, center=True, min_periods=1).median().to_numpy()
    smooth_x = pd.Series(x).rolling(3, center=True, min_periods=1).median().to_numpy()
    smooth_y = pd.Series(y).rolling(3, center=True, min_periods=1).median().to_numpy()
    vx = np.gradient(smooth_x, t)
    vy = np.gradient(smooth_y, t)
    vx = pd.Series(vx).rolling(3, center=True, min_periods=1).median().to_numpy()
    vy = pd.Series(vy).rolling(3, center=True, min_periods=1).median().to_numpy()
    turn = circular_turn_degrees(vx, vy)

    # The first substantial sustained direction change after a running approach is
    # the operational route-break event. The threshold is exposed for sensitivity work.
    pre_speed = np.r_[smooth_s[0], smooth_s[:-1]]
    # The curl break occurs after an approach. Constrain detection to the central
    # route phase, so a later return/pivot is not mistaken for the initial break.
    candidates = np.flatnonzero((t >= 1.5) & (t <= 3.3) & (turn >= turn_threshold) & (pre_speed >= min_pre_speed))
    event_idx = None
    for idx in candidates:
        future = turn[idx:min(idx + 3, len(turn))]
        if np.isfinite(future).sum() >= 2 and np.nanmax(future) >= turn_threshold:
            event_idx = int(idx)
            break
    if event_idx is None:
        return {**base, "event_status": f"no_{turn_threshold:g}deg_turn_after_approach",
                "max_detectable_turn_deg": float(np.nanmax(turn)) if np.isfinite(turn).any() else np.nan}

    event_t = float(t[event_idx])
    approach = (t >= event_t - 0.6) & (t <= event_t - 0.1)
    post_min = (t >= event_t) & (t <= event_t + 0.8)
    post_slope = (t >= event_t + 0.1) & (t <= event_t + 0.8)
    approach_speed = float(np.mean(smooth_s[approach])) if approach.any() else np.nan
    min_speed = float(np.min(smooth_s[post_min])) if post_min.any() else np.nan
    slope = np.polyfit(t[post_slope], smooth_s[post_slope], 1)[0] if post_slope.sum() >= 4 else np.nan
    features = {
        **base,
        "event_status": "detected",
        "turn_rule_deg": turn_threshold,
        "prebreak_speed_floor_ydps": min_pre_speed,
        "break_search_window_start_s": 1.5,
        "break_search_window_end_s": 3.3,
        "break_time_s": event_t,
        "break_turn_deg_0_4s": float(turn[event_idx]),
        "approach_speed_ydps": approach_speed,
        "minimum_speed_0_8s_after_break_ydps": min_speed,
        "speed_lost_at_break_ydps": approach_speed - min_speed if np.isfinite(approach_speed + min_speed) else np.nan,
        "speed_retained_at_min_ratio": min_speed / approach_speed if np.isfinite(approach_speed + min_speed) and approach_speed > 0 else np.nan,
        "reacceleration_0_1_to_0_8s_after_break_ydps2": float(slope),
    }
    for horizon in (0.5, 1.0, 1.5):
        j = int(np.argmin(np.abs(t - (event_t + horizon))))
        if abs(t[j] - (event_t + horizon)) <= 0.16:
            features[f"speed_{horizon:g}s_after_break_ydps"] = float(smooth_s[j])
        else:
            features[f"speed_{horizon:g}s_after_break_ydps"] = np.nan
    recovered_1s = features["speed_1s_after_break_ydps"] - min_speed
    total_loss = approach_speed - min_speed
    features["speed_recovered_from_postbreak_min_by_1s_ydps"] = recovered_1s
    features["fraction_of_speed_loss_recovered_by_1s"] = (
        recovered_1s / total_loss if np.isfinite(recovered_1s + total_loss) and total_loss >= 0.25 else np.nan
    )
    recovery = np.flatnonzero((t >= event_t) & (t <= event_t + 2.0) & (smooth_s >= 0.9 * approach_speed)) if np.isfinite(approach_speed) else []
    features["time_to_recover_90pct_approach_s"] = float(t[recovery[0]] - event_t) if len(recovery) else np.nan
    features["recovered_90pct_within_2s"] = int(len(recovery) > 0)
    return features


def extract_combine_features(archive: zipfile.ZipFile, out: Path, turn_threshold: float, min_pre_speed: float) -> pd.DataFrame:
    combine = read_csv(archive, "combine_tracking.csv")
    combine = combine.loc[(combine.entity_type == "PLAYER") & (combine.drill_name == "CURL_ROUTE_RIGHT")].copy()
    keys = ["draft_year", "event_id", "nfl_id", "drill_name", "attempt"]
    rows = []
    for key, group in combine.groupby(keys, sort=False, dropna=False):
        record = dict(zip(keys, key))
        record.update(summarize_curl_attempt(group, turn_threshold, min_pre_speed))
        rows.append(record)
    features = pd.DataFrame(rows)
    features["nfl_id"] = pd.to_numeric(features.nfl_id, errors="coerce")
    features.to_csv(out / "wr_curl_attempt_speed_features.csv", index=False)
    return features


def rookie_wr_separation(archive: zipfile.ZipFile, players: pd.DataFrame, out: Path) -> pd.DataFrame:
    games = read_csv(archive, "games.csv", usecols=["game_id", "season", "season_type"])
    games = games.loc[games.season_type.eq("REG"), ["game_id", "season"]]
    plays = read_csv(archive, "player_play.csv", usecols=[
        "game_id", "play_id", "nfl_id", "lined_up_position", "route_ran",
        "separation_at_pass_forward", "target", "down", "yards_to_go", "expected_points_added"
    ])
    plays = plays.merge(games, on="game_id", how="inner", validate="many_to_one")
    plays = plays.merge(players[["nfl_id", "draft_year", "nfl_position"]], on="nfl_id", how="inner", validate="many_to_one")
    plays = plays.loc[(plays.nfl_position == "WR") & (plays.season == plays.draft_year)].copy()
    plays["separation_at_pass_forward"] = pd.to_numeric(plays.separation_at_pass_forward, errors="coerce")
    plays = plays.loc[plays.separation_at_pass_forward.notna() & plays.route_ran.notna()]
    plays.to_csv(out / "wr_rookie_route_separation.csv", index=False)
    return plays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--turn-threshold", type=float, default=35.0, help="Heading-change threshold in degrees; alter for sensitivity analysis.")
    parser.add_argument("--pre-speed-threshold", type=float, default=1.5, help="Minimum speed before the candidate turn, in yd/s.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with open_data_source(args.archive) as archive:
        players = read_csv(archive, "players.csv")
        players["nfl_id"] = pd.to_numeric(players.nfl_id, errors="coerce")
        players["draft_year"] = pd.to_numeric(players.draft_year, errors="coerce")
        features = extract_combine_features(archive, args.output, args.turn_threshold, args.pre_speed_threshold)
        plays = rookie_wr_separation(archive, players, args.output)
    overlap = features.loc[features.event_status.eq("detected"), "nfl_id"].nunique()
    outcome_ids = set(plays.nfl_id.unique())
    matched = features.loc[features.event_status.eq("detected") & features.nfl_id.isin(outcome_ids), "nfl_id"].nunique()
    print({
        "curl_attempts": int(len(features)),
        "players_with_curl_tracking": int(features.nfl_id.nunique()),
        "attempts_with_detected_break": int(features.event_status.eq("detected").sum()),
        "players_with_detected_break": int(overlap),
        "rookie_WR_route_separation_rows": int(len(plays)),
        "rookie_WR_players_with_separation": int(plays.nfl_id.nunique()),
        "players_with_both_detected_curl_break_and_separation": int(matched),
    })


if __name__ == "__main__":
    main()
