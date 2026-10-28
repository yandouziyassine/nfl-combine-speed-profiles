"""Cohort-held-out screen of DL Combine speed traits against rookie NFL tracking.

The player is the independent unit. Outcomes are summarized over rookie regular-
season blitz plays, while the game-burst outcome uses 10 Hz frames after ball snap.
Each Combine feature is added singly to the same conventional-test baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight", "interior_line"]
OUTCOMES = {
    "game_early_speed_gain": ("median_speed_gain_0_1s_ydps", "n_stationary_snap_bursts", 20),
    "game_early_displacement": ("median_net_displacement_0_1s_yd", "n_stationary_snap_bursts", 20),
    "blitz_pressure_rate": ("rookie_pressure_rate", "rookie_blitz_opportunities", 50),
    "get_off_time": ("median_get_off_s", "get_off_events", 20),
}
CANDIDATES = {
    "front_start_early_acceleration": ("FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "mean_accel_first_1s_ydps2_mean"),
    "front_start_sustained_speed": ("FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "peak_sustained_speed_ydps_mean"),
    "front_start_peak_acceleration": ("FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "peak_accel_ydps2_mean"),
    "pass_rush_sustained_speed": ("PASS_RUSH_DRILL", "peak_sustained_speed_ydps_mean"),
}
HOOP_CANDIDATES = {
    "hoop_turn_weighted_speed": "direction_change_weighted_speed_ydps",
    "hoop_curve_speed_retention": "curve_speed_retention_ratio",
}


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, alpha: float) -> np.ndarray:
    mu = train[features].mean()
    sd = train[features].std().replace(0, 1)
    x = ((train[features] - mu) / sd).to_numpy(float)
    z = ((test[features] - mu) / sd).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(float))
    return z @ beta


def correlation_ci(x: np.ndarray, y: np.ndarray, seed: int = 249, n_boot: int = 2000) -> tuple[float, list[float]]:
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) and np.std(y) else float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        ids = rng.integers(0, len(x), size=len(x))
        if np.std(x[ids]) and np.std(y[ids]):
            boot.append(np.corrcoef(x[ids], y[ids])[0, 1])
    ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if boot else [float("nan"), float("nan")]
    return r, ci


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=out)
    parser.add_argument("--ridge-alpha", type=float, default=3.0)
    args = parser.parse_args()

    players = pd.read_csv(args.output / "all_player_speed_profiles.csv")
    players = players.loc[players.nfl_position.isin(["DE", "DT", "NT", "OLB"])].copy()
    players["interior_line"] = players.nfl_position.isin(["DT", "NT"]).astype(int)

    burst = pd.read_csv(args.output / "dl_rookie_blitz_game_burst.csv")
    eligible_burst = burst.loc[(burst.game_tracking_status == "usable") & burst.already_moving_before_snap.eq(0)].copy()
    burst_player = (eligible_burst.groupby("nfl_id")
                    .agg(median_speed_gain_0_1s_ydps=("speed_gain_0_1s_ydps", "median"),
                         median_net_displacement_0_1s_yd=("net_displacement_0_1s_yd", "median"),
                         median_snap_to_motion_s=("movement_onset_after_snap_s", "median"),
                         n_stationary_snap_bursts=("pressure_event", "size"))
                    .reset_index())
    pressure = pd.read_csv(args.output / "dl_rookie_blitz_outcomes.csv")
    game_player = burst_player.merge(pressure[["nfl_id", "rookie_pressure_rate", "rookie_blitz_opportunities",
                                               "median_get_off_s", "get_off_events"]], on="nfl_id", how="outer", validate="one_to_one")
    profile = players.merge(game_player, on="nfl_id", how="left", validate="one_to_one")

    drills = pd.read_csv(args.output / "player_drill_speed_profiles.csv")
    for name, (drill_name, col) in CANDIDATES.items():
        group = drills.loc[drills.drill_name.eq(drill_name), ["nfl_id", col]].rename(columns={col: name})
        if name in profile:
            profile = profile.drop(columns=[name])
        profile = profile.merge(group, on="nfl_id", how="left", validate="one_to_one")
    hoop = pd.read_csv(args.output / "dl_hoop_player_profiles.csv")[["nfl_id", "direction_change_weighted_speed_ydps", "curve_speed_retention_ratio"]]
    profile = profile.merge(hoop, on="nfl_id", how="left", validate="one_to_one")
    for name, col in HOOP_CANDIDATES.items():
        profile[name] = profile[col]
        CANDIDATES[name] = ("RUN_THE_HOOP_DRILL", col)
    profile.to_csv(args.output / "dl_transfer_player_table.csv", index=False)

    results = {"analysis_unit": "Player. Plays/frames estimate each player's rookie-year outcomes; they are not counted as independent players.",
               "pressure_definition": "Among REG rookie-year player-play rows with blitzing=True, time_to_pressure observed is a pressure event per the competition host clarification.",
               "game_burst_definition": "Median of the per-play speed gain from ball-snap speed to 1.0 s after snap, on matched rookie REG blitz plays with usable 10 Hz tracking and no speed >=1 yd/s in the 0.5 s pre-snap window.",
               "baseline": BASE, "validation": "Leave one draft cohort out. All rows for each player are aggregated before fitting. The baseline and one-feature model use the same players per comparison. Standardization is learned on training players only.",
               "ridge_alpha": args.ridge_alpha, "minimum_exposure": {name: int(val[2]) for name, val in OUTCOMES.items()}, "comparisons": []}
    comparison_rows = []
    for outcome_name, (target, exposure, min_exposure) in OUTCOMES.items():
        if target not in profile or exposure not in profile:
            continue
        eligible = profile.loc[profile[exposure].fillna(0).ge(min_exposure)].copy()
        for candidate in CANDIDATES:
            common = BASE + [candidate, target]
            data = eligible.loc[eligible[common].notna().all(axis=1)].copy()
            if len(data) < 25:
                continue
            r, ci = correlation_ci(data[candidate].to_numpy(float), data[target].to_numpy(float), seed=249 + len(data))
            oof_y, oof_base, oof_candidate = [], [], []
            folds = []
            for cohort in sorted(data.draft_year.dropna().unique()):
                train = data.loc[data.draft_year.ne(cohort)]
                test = data.loc[data.draft_year.eq(cohort)]
                row = {"held_out_draft_year": int(cohort), "n_train_players": int(len(train)), "n_test_players": int(len(test))}
                if len(train) < 20 or len(test) < 8:
                    row["status"] = "too_few_players"
                    folds.append(row)
                    continue
                pred0 = ridge_predict(train, test, BASE, target, args.ridge_alpha)
                pred1 = ridge_predict(train, test, BASE + [candidate], target, args.ridge_alpha)
                actual = test[target].to_numpy(float)
                row.update({"baseline_rmse": float(np.sqrt(np.mean((actual - pred0) ** 2))),
                            "plus_candidate_rmse": float(np.sqrt(np.mean((actual - pred1) ** 2))),
                            "rmse_change": float(np.sqrt(np.mean((actual - pred1) ** 2)) - np.sqrt(np.mean((actual - pred0) ** 2))),
                            "status": "exploratory_small_player_cohort"})
                folds.append(row)
                oof_y.extend(actual.tolist())
                oof_base.extend(pred0.tolist())
                oof_candidate.extend(pred1.tolist())
            delta = (float(np.sqrt(np.mean((np.asarray(oof_y) - np.asarray(oof_candidate)) ** 2)) -
                           np.sqrt(np.mean((np.asarray(oof_y) - np.asarray(oof_base)) ** 2))) if oof_y else None)
            result = {"outcome": outcome_name, "target_column": target, "candidate": candidate,
                      "n_players": int(len(data)), "players_by_cohort": {str(int(y)): int(n) for y, n in data.draft_year.value_counts().sort_index().items()},
                      "pearson_r": r, "player_bootstrap_95pct_ci": ci, "cohorts": folds, "pooled_oof_rmse_change": delta}
            results["comparisons"].append(result)
            comparison_rows.append({"outcome": outcome_name, "candidate": candidate, "n_players": len(data),
                                    "pearson_r": r, "ci_low": ci[0], "ci_high": ci[1],
                                    "pooled_oof_rmse_change": delta})

    (args.output / "dl_transfer_model_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    table = pd.DataFrame(comparison_rows)
    table.to_csv(args.output / "dl_transfer_candidate_summary.csv", index=False)
    lines = ["# Defensive-line Combine-to-game movement screen", "",
             "This is a candidate screen, not a final competition claim. Each model is player-level and leaves one complete draft cohort out. The pressure outcome is measured only on blitzing=True rookie regular-season plays; the early-burst outcome is derived directly from 10 Hz game tracking after ball_snap.", "",
             f"The game-tracking join covers {len(burst):,} blitz player-play keys; {int((burst.game_tracking_status == 'usable').sum()):,} have a usable ball-snap trace. The model requires at least 20 eligible stationary-snap tracking plays for burst outcomes or 50 rookie blitz opportunities for pressure rate, plus complete 40/split/size controls.", "",
             "| Game outcome | Combine trait | Players | Player correlation (95% bootstrap interval) | Pooled held-out RMSE change |", "|---|---|---:|---:|---:|"]
    for row in comparison_rows:
        lines.append(f"| {row['outcome']} | {row['candidate']} | {row['n_players']} | {row['pearson_r']:+.2f} ({row['ci_low']:+.2f}, {row['ci_high']:+.2f}) | {row['pooled_oof_rmse_change']:+.3f} |")
    lines += ["", "Positive RMSE change means the Combine movement feature made held-out predictions worse. Correlations are descriptive and must not be used to select a winner without the cohort results. Small cohorts, position/scheme effects, imperfect prospect exposure, and repeated-play measurement remain limitations.", ""]
    (args.output / "dl_transfer_model_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparisons": len(comparison_rows), "rows": comparison_rows}, indent=2))


if __name__ == "__main__":
    main()
