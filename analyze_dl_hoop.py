"""Build event- and path-aware speed profiles for DL Combine hoop attempts.

The main arc metric weights speed by the amount of change in travel direction.
This avoids selecting a single turn frame with an arbitrary heading threshold.
Rookie-year pressure outcomes are joined only to regular-season blitz opportunities.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from run_analysis import DEFAULT_ARCHIVE, open_data_source, parse_sensor_times

MEMBER = "nfl-big-data-bowl-2027/"
DL_POSITIONS = {"DE", "DT", "NT", "OLB"}


def read_csv(archive: zipfile.ZipFile, filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(archive.open(MEMBER + filename), **kwargs)


def circular_smooth_direction(degrees: np.ndarray, window: int = 5) -> np.ndarray:
    rad = np.deg2rad(degrees)
    sin_s = pd.Series(np.sin(rad)).rolling(window, center=True, min_periods=1).mean().to_numpy()
    cos_s = pd.Series(np.cos(rad)).rolling(window, center=True, min_periods=1).mean().to_numpy()
    return np.unwrap(np.arctan2(sin_s, cos_s))


def at_cumulative_turn(turn: np.ndarray, speed: np.ndarray, target_deg: float) -> float:
    if len(turn) < 2 or turn[-1] < target_deg:
        return np.nan
    idx = int(np.searchsorted(turn, target_deg, side="left"))
    if idx == 0:
        return float(speed[0])
    lo, hi = turn[idx - 1], turn[idx]
    frac = (target_deg - lo) / (hi - lo) if hi > lo else 0.0
    return float(speed[idx - 1] + frac * (speed[idx] - speed[idx - 1]))


def summarize_hoop_attempt(group: pd.DataFrame) -> dict:
    g = group.sort_values("time")
    t0 = parse_sensor_times(g.time)
    t = (t0 - t0.iloc[0]).dt.total_seconds().to_numpy(float)
    speed_raw = pd.to_numeric(g.s, errors="coerce").to_numpy(float)
    direction = pd.to_numeric(g.dir, errors="coerce").to_numpy(float)
    x = pd.to_numeric(g.x, errors="coerce").to_numpy(float)
    y = pd.to_numeric(g.y, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(speed_raw) & np.isfinite(direction) & np.isfinite(x) & np.isfinite(y)
    t, speed_raw, direction, x, y = t[valid], speed_raw[valid], direction[valid], x[valid], y[valid]
    result = {"frames": int(len(t)), "duration_s": float(t[-1] - t[0]) if len(t) else np.nan}
    if len(t) < 20 or np.any(np.diff(t) <= 0):
        result["quality_status"] = "too_few_or_invalid_frames"
        return result

    speed = pd.Series(speed_raw).rolling(3, center=True, min_periods=1).median().to_numpy()
    heading = circular_smooth_direction(direction, window=5)
    # Ignore stationary pre-movement frames, then calculate accumulated absolute
    # change in path heading. This remains comparable if the player runs the hoop
    # clockwise or counterclockwise.
    onset_hits = np.flatnonzero((speed >= 1.0) & np.r_[speed[1:] >= 1.0, False])
    if not len(onset_hits):
        result["quality_status"] = "movement_onset_not_found"
        return result
    start = int(onset_hits[0])
    t, speed, heading, x, y = t[start:], speed[start:], heading[start:], x[start:], y[start:]
    dt = np.diff(t)
    dtheta = np.diff(heading)
    arc_increment = np.abs(np.rad2deg(dtheta))
    cumulative_turn = np.r_[0.0, np.cumsum(arc_increment)]
    if not len(dt) or cumulative_turn[-1] < 180:
        result["quality_status"] = "insufficient_direction_change"
        return {**result, "total_absolute_direction_change_deg": float(cumulative_turn[-1])}

    interval_speed = (speed[:-1] + speed[1:]) / 2
    turn_total = float(arc_increment.sum())
    arc_speed = float(np.average(interval_speed, weights=arc_increment)) if arc_increment.sum() > 0 else np.nan
    sustained = pd.Series(speed).rolling(5, center=True, min_periods=3).median().to_numpy()
    peak_sustained = float(np.nanmax(sustained)) if np.isfinite(sustained).any() else np.nan
    # Angular rate is a smoothed finite difference. Threshold features are retained
    # for sensitivity analysis; the primary arc speed itself is threshold-free.
    omega = np.rad2deg(np.gradient(heading, t))
    omega = pd.Series(omega).rolling(3, center=True, min_periods=1).median().to_numpy()
    result.update({
        "quality_status": "usable",
        "movement_onset_from_recording_s": float(t[0]),
        "moving_duration_s": float(t[-1] - t[0]),
        "total_absolute_direction_change_deg": turn_total,
        "mean_speed_ydps": float(np.mean(speed)),
        "peak_sustained_speed_ydps": peak_sustained,
        "direction_change_weighted_speed_ydps": arc_speed,
        "curve_speed_retention_ratio": arc_speed / peak_sustained if np.isfinite(arc_speed + peak_sustained) and peak_sustained > 0 else np.nan,
        "peak_to_curve_speed_deficit_ydps": peak_sustained - arc_speed if np.isfinite(arc_speed + peak_sustained) else np.nan,
        "mean_abs_turn_rate_deg_s": float(np.average(np.abs(omega), weights=np.r_[dt[0], (dt[:-1] + dt[1:]) / 2, dt[-1]])) if len(omega) == len(t) else np.nan,
        "max_abs_turn_rate_deg_s": float(np.max(np.abs(omega))),
        "path_length_yd": float(np.hypot(np.diff(x), np.diff(y)).sum()),
    })
    for threshold in (20.0, 40.0, 60.0):
        turning = np.abs(omega) >= threshold
        result[f"speed_during_turn_{int(threshold)}deg_s_ydps"] = float(np.mean(speed[turning])) if turning.any() else np.nan
        result[f"turning_time_fraction_{int(threshold)}deg_s"] = float(np.mean(turning))
    for target in (90.0, 180.0, 270.0, 360.0):
        result[f"speed_at_cumulative_turn_{int(target)}deg_ydps"] = at_cumulative_turn(cumulative_turn, speed, target)
    for target in (180.0, 270.0, 300.0, 360.0):
        idx = int(np.searchsorted(cumulative_turn, target, side="left"))
        result[f"time_to_cumulative_turn_{int(target)}deg_s"] = float(t[idx] - t[0]) if idx < len(t) and cumulative_turn[-1] >= target else np.nan
    return result


def combine_hoop_profiles(archive: zipfile.ZipFile, output: Path) -> pd.DataFrame:
    tracking = read_csv(archive, "combine_tracking.csv")
    tracking = tracking.loc[(tracking.entity_type == "PLAYER") & (tracking.drill_name == "RUN_THE_HOOP_DRILL")].copy()
    player_keys = ["draft_year", "event_id", "nfl_id", "drill_name", "attempt"]
    rows = []
    for key, group in tracking.groupby(player_keys, sort=False, dropna=False):
        row = dict(zip(player_keys, key))
        row.update(summarize_hoop_attempt(group))
        rows.append(row)
    attempts = pd.DataFrame(rows)
    attempts["nfl_id"] = pd.to_numeric(attempts.nfl_id, errors="coerce")
    attempts.to_csv(output / "dl_hoop_attempt_profiles.csv", index=False)
    numeric = [c for c in attempts.select_dtypes(include=["number"]).columns if c not in {"draft_year", "nfl_id", "attempt", "event_id"}]
    player = attempts.loc[attempts.quality_status.eq("usable")].groupby(["nfl_id", "draft_year"], as_index=False)[numeric].mean()
    variability = attempts.loc[attempts.quality_status.eq("usable")].groupby("nfl_id")["curve_speed_retention_ratio"].agg(["std", "count"]).rename(columns={"std": "curve_speed_retention_attempt_sd", "count": "usable_hoop_attempts"}).reset_index()
    player = player.merge(variability, on="nfl_id", how="left", validate="one_to_one")
    player.to_csv(output / "dl_hoop_player_profiles.csv", index=False)
    return player


def rookie_dl_blitz_profiles(archive: zipfile.ZipFile, players: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = read_csv(archive, "games.csv", usecols=["game_id", "season", "season_type"])
    games = games.loc[games.season_type.eq("REG"), ["game_id", "season"]]
    plays = read_csv(archive, "player_play.csv", usecols=[
        "game_id", "play_id", "nfl_id", "lined_up_position", "down", "yards_to_go",
        "blitzing", "time_to_pressure", "player_get_off"
    ])
    plays = plays.merge(games, on="game_id", how="inner", validate="many_to_one")
    players = players.loc[players.nfl_position.isin(DL_POSITIONS), ["nfl_id", "draft_year", "nfl_position"]]
    plays = plays.merge(players, on="nfl_id", how="inner", validate="many_to_one")
    plays = plays.loc[(plays.season == plays.draft_year) & plays.blitzing.eq(True)].copy()
    plays["pressure_event"] = plays.time_to_pressure.notna().astype(int)
    plays["time_to_pressure"] = pd.to_numeric(plays.time_to_pressure, errors="coerce")
    plays["player_get_off"] = pd.to_numeric(plays.player_get_off, errors="coerce")
    plays.to_csv(output / "dl_rookie_blitz_opportunities.csv", index=False)
    summary = (plays.groupby(["nfl_id", "draft_year", "nfl_position"], as_index=False)
               .agg(rookie_blitz_opportunities=("play_id", "size"),
                    rookie_pressures=("pressure_event", "sum"),
                    rookie_pressure_rate=("pressure_event", "mean"),
                    median_time_to_pressure_s=("time_to_pressure", "median"),
                    pressure_events_with_time=("time_to_pressure", "count"),
                    median_get_off_s=("player_get_off", "median"),
                    get_off_events=("player_get_off", "count")))
    summary.to_csv(output / "dl_rookie_blitz_outcomes.csv", index=False)
    return plays, summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=root / "outputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with open_data_source(args.archive) as archive:
        players = read_csv(archive, "players.csv")
        players["nfl_id"] = pd.to_numeric(players.nfl_id, errors="coerce")
        players["draft_year"] = pd.to_numeric(players.draft_year, errors="coerce")
        combine = combine_hoop_profiles(archive, args.output)
        blitz_plays, blitz = rookie_dl_blitz_profiles(archive, players, args.output)
    merged = combine.merge(blitz, on=["nfl_id", "draft_year"], how="left", validate="one_to_one")
    stats = {
        "hoop_attempts": int(pd.read_csv(args.output / "dl_hoop_attempt_profiles.csv").shape[0]),
        "players_with_hoop_tracking": int(combine.nfl_id.nunique()),
        "players_with_usable_hoop_profile": int(combine.nfl_id.nunique()),
        "rookie_blitz_players": int(blitz.nfl_id.nunique()),
        "rookie_blitz_opportunities": int(len(blitz_plays)),
        "hoop_blitz_overlap": int(merged.rookie_blitz_opportunities.notna().sum()),
        "hoop_blitz_overlap_min_50_plays": int(merged.rookie_blitz_opportunities.ge(50).sum()),
        "pressure_definition": "Among rookie-season REG player-play rows labeled blitzing=True, time_to_pressure present is counted as a pressure event, following the competition host's clarification. Missing labels outside that specified population are not generalized.",
    }
    (args.output / "dl_hoop_coverage.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
