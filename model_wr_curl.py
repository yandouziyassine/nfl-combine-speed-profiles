"""Model WR Combine curl-break speed against rookie-season route separation.

The independent unit is the player. Draft-cohort holdout keeps a player's routes
on only one side of each validation fold. The route adjustment is learned from the
training players in each fold to avoid using held-out separation labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURES = ["forty", "ten_yd_split", "combine_height", "combine_weight"]
TRACKING_FEATURE = "speed_recovered_from_postbreak_min_by_1s_ydps"


def route_adjusted_player_outcomes(plays: pd.DataFrame, training_ids: set[int], min_plays: int = 20, min_route_plays: int = 5) -> pd.DataFrame:
    """Equal-weight a player's sufficiently observed routes after train-only route centering."""
    train = plays.loc[plays.nfl_id.isin(training_ids)]
    route_center = train.groupby("route_ran").separation_at_pass_forward.mean()
    fallback = float(train.separation_at_pass_forward.mean())
    work = plays.copy()
    work["route_center"] = work.route_ran.map(route_center).fillna(fallback)
    work["route_adjusted_separation"] = work.separation_at_pass_forward - work.route_center
    route_rows = (work.groupby(["nfl_id", "route_ran"], as_index=False)
                  .agg(route_mean_residual=("route_adjusted_separation", "mean"),
                       route_play_count=("separation_at_pass_forward", "count")))
    route_rows = route_rows.loc[route_rows.route_play_count >= min_route_plays]
    outcomes = (route_rows.groupby("nfl_id")
                .agg(route_adjusted_separation=("route_mean_residual", "mean"),
                     qualifying_route_families=("route_ran", "nunique"),
                     plays_in_qualifying_routes=("route_play_count", "sum"))
                .reset_index())
    outcomes = outcomes.loc[outcomes.plays_in_qualifying_routes >= min_plays]
    return outcomes


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, alpha: float = 2.0) -> np.ndarray:
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_train = train[target].to_numpy(float)
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0
    x_train = (x_train - mean) / sd
    x_test = (x_test - mean) / sd
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ beta


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) and np.std(y) else float("nan")


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, seed: int = 817, iterations: int = 3000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        ix = rng.integers(0, len(x), len(x))
        if np.std(x[ix]) and np.std(y[ix]):
            values.append(np.corrcoef(x[ix], y[ix])[0, 1])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [float("nan"), float("nan")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=root / "outputs")
    parser.add_argument("--minimum-plays", type=int, default=20)
    parser.add_argument("--minimum-route-plays", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    args = parser.parse_args()

    out = args.output
    features = pd.read_csv(out / "wr_curl_attempt_speed_features.csv")
    features = features.loc[features.event_status.eq("detected")].copy()
    features = (features.groupby(["nfl_id", "draft_year"], as_index=False)
                .mean(numeric_only=True))
    plays = pd.read_csv(out / "wr_rookie_route_separation.csv")
    players = pd.read_csv(root / "outputs" / "all_player_speed_profiles.csv")
    combine = players.loc[players.nfl_position.eq("WR"), ["nfl_id", "forty", "ten_yd_split", "combine_height", "combine_weight"]]
    feature_cols = ["nfl_id", "draft_year", TRACKING_FEATURE,
                    "approach_speed_ydps", "speed_lost_at_break_ydps",
                    "reacceleration_0_1_to_0_8s_after_break_ydps2",
                    "speed_1s_after_break_ydps"]
    profile = features[feature_cols].merge(combine, on="nfl_id", how="left", validate="one_to_one")
    exposure = plays.groupby("nfl_id").separation_at_pass_forward.count().rename("rookie_separation_plays")
    profile = profile.merge(exposure, on="nfl_id", how="left", validate="one_to_one")
    profile["rookie_separation_plays"] = profile.rookie_separation_plays.fillna(0).astype(int)
    profile = profile.loc[profile.rookie_separation_plays >= args.minimum_plays].copy()
    # Apply the same complete player cohort to both models; no missing test is coded as zero.
    model_cols = BASE_FEATURES + [TRACKING_FEATURE]
    profile = profile.loc[profile[model_cols].notna().all(axis=1)].copy()
    profile["nfl_id"] = profile.nfl_id.astype(int)
    plays = plays.loc[plays.nfl_id.isin(profile.nfl_id)].copy()

    cohort_records = []
    pooled_actual, pooled_base, pooled_tracking = [], [], []
    full_outcomes = route_adjusted_player_outcomes(plays, set(profile.nfl_id), args.minimum_plays, args.minimum_route_plays)
    full = profile.merge(full_outcomes, on="nfl_id", how="inner", validate="one_to_one")

    for year in sorted(profile.draft_year.unique()):
        train_ids = set(profile.loc[profile.draft_year.ne(year), "nfl_id"].astype(int))
        test_ids = set(profile.loc[profile.draft_year.eq(year), "nfl_id"].astype(int))
        train_y = route_adjusted_player_outcomes(plays.loc[plays.nfl_id.isin(train_ids) | plays.nfl_id.isin(test_ids)],
                                                 train_ids, args.minimum_plays, args.minimum_route_plays)
        data = profile.merge(train_y, on="nfl_id", how="inner", validate="one_to_one")
        train = data.loc[data.nfl_id.isin(train_ids)].copy()
        test = data.loc[data.nfl_id.isin(test_ids)].copy()
        record = {"held_out_draft_year": int(year), "n_train_players": int(len(train)), "n_test_players": int(len(test))}
        if len(train) < 20 or len(test) < 8:
            record["status"] = "Too few independent players for this fold."
            cohort_records.append(record)
            continue
        base = ridge_predict(train, test, BASE_FEATURES, "route_adjusted_separation", args.ridge_alpha)
        augmented_features = BASE_FEATURES + [TRACKING_FEATURE]
        tracking = ridge_predict(train, test, augmented_features, "route_adjusted_separation", args.ridge_alpha)
        actual = test.route_adjusted_separation.to_numpy(float)
        rmse_base = float(np.sqrt(np.mean((actual - base) ** 2)))
        rmse_tracking = float(np.sqrt(np.mean((actual - tracking) ** 2)))
        record.update({"baseline_rmse_yards": rmse_base,
                       "tracking_rmse_yards": rmse_tracking,
                       "rmse_change_yards": rmse_tracking - rmse_base,
                       "baseline_mae_yards": float(np.mean(np.abs(actual - base))),
                       "tracking_mae_yards": float(np.mean(np.abs(actual - tracking))),
                       "status": "Exploratory cohort holdout; small test cohort."})
        cohort_records.append(record)
        pooled_actual.extend(actual.tolist())
        pooled_base.extend(base.tolist())
        pooled_tracking.extend(tracking.tolist())

    primary_corr = pearson(full[TRACKING_FEATURE].to_numpy(float), full.route_adjusted_separation.to_numpy(float))
    ci = bootstrap_corr_ci(full[TRACKING_FEATURE].to_numpy(float), full.route_adjusted_separation.to_numpy(float)) if len(full) > 3 else [float("nan"), float("nan")]
    results = {
        "question": "Does Combine WR curl-route speed recovery at one second after the route break add information about rookie regular-season route separation beyond official 40 time, 10-yard split, height, and weight?",
        "outcome": "Within each validation fold, subtract training-player mean separation for each route family, average within player-route, then average equally over the player's route families with at least the minimum route exposure.",
        "primary_tracking_feature": TRACKING_FEATURE,
        "sample": {"tracked_curl_players_with_detected_break": int(features.nfl_id.nunique()),
                   "players_with_minimum_separation_exposure_and_core_tests": int(len(profile)),
                   "minimum_separation_plays": args.minimum_plays,
                   "minimum_plays_per_route_family": args.minimum_route_plays,
                   "primary_model_players_with_outcome": int(len(full)),
                   "players_by_draft_year": {str(int(k)): int(v) for k, v in profile.draft_year.value_counts().sort_index().items()}},
        "player_level_exploratory_association": {"pearson_r": primary_corr, "bootstrap_95pct_ci": ci, "n_players": int(len(full))},
        "validation": {"method": "Leave one draft cohort out; every player is wholly in train or test. Route means used to adjust the outcome are fit on training players only.",
                       "ridge_alpha_on_standardized_features": args.ridge_alpha,
                       "folds": cohort_records,
                       "pooled_oof_rmse_baseline_yards": float(np.sqrt(np.mean((np.asarray(pooled_actual)-np.asarray(pooled_base))**2))) if pooled_actual else None,
                       "pooled_oof_rmse_tracking_yards": float(np.sqrt(np.mean((np.asarray(pooled_actual)-np.asarray(pooled_tracking))**2))) if pooled_actual else None,
                       "pooled_oof_rmse_change_yards": float(np.sqrt(np.mean((np.asarray(pooled_actual)-np.asarray(pooled_tracking))**2))-np.sqrt(np.mean((np.asarray(pooled_actual)-np.asarray(pooled_base))**2))) if pooled_actual else None},
        "interpretation_limit": "A small observational sample can identify association and out-of-cohort predictive value, not a causal training effect or a draft-bust probability.",
    }
    (out / "wr_curl_model_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    full.to_csv(out / "wr_curl_player_model_table.csv", index=False)
    lines = ["# WR curl speed recovery and rookie separation", "",
             results["question"], "",
             f"The analysis contains {len(full)} players after requiring at least {args.minimum_plays} rookie separation observations, at least {args.minimum_route_plays} observations within a route family, a detected curl break, and complete official 40/split/size controls.", "",
             f"The exploratory player-level correlation between speed recovered by one second and route-adjusted separation is r={primary_corr:.3f} (player-bootstrap 95% interval {ci[0]:.3f} to {ci[1]:.3f}). This is descriptive, not a causal estimate.", "",
             "## Leave-one-draft-cohort-out prediction", "",
             "| Held-out cohort | Train players | Test players | Baseline RMSE (yd) | + tracking RMSE (yd) | Change (yd) |", "|---:|---:|---:|---:|---:|---:|"]
    for row in cohort_records:
        if "baseline_rmse_yards" in row:
            lines.append(f"| {row['held_out_draft_year']} | {row['n_train_players']} | {row['n_test_players']} | {row['baseline_rmse_yards']:.3f} | {row['tracking_rmse_yards']:.3f} | {row['rmse_change_yards']:+.3f} |")
        else:
            lines.append(f"| {row['held_out_draft_year']} | {row['n_train_players']} | {row['n_test_players']} | — | — | — |")
    lines += ["", "The baseline uses official 40 time, 10-yard split, height, and weight. The tracking model adds speed regained between the post-break minimum and one second after the detected break. Outcome route centering is learned from training players in each fold; player, not play, is the unit scored.", "", "This first model uses a single fixed tracking feature and a low-dimensional ridge regression. Cohort test sets are small. Review the event-detection sensitivity report and exposure thresholds before making a competition claim.", ""]
    (out / "wr_curl_model_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"players": len(full), "correlation": primary_corr, "ci": ci,
                      "folds": cohort_records,
                      "pooled_oof_rmse_change": results["validation"]["pooled_oof_rmse_change_yards"]}, indent=2))


if __name__ == "__main__":
    main()
