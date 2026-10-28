"""Test whether 40-yard speed phases add information about NFL career snap role.

Career snaps are divided by the number of regular seasons observed in the
competition files for each draft class. Every prospect is the independent
unit. The model holds out whole draft cohorts and compares the same players
under conventional-test and tracking-profile feature sets.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from run_analysis import DEFAULT_ARCHIVE, open_data_source

BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight", "draft_overall_pick",
        "position_DB", "position_DL", "position_OL", "position_TE"]
PHASES = ["forty_early_accel_0_1s_ydps2_mean", "forty_late_accel_10_20yd_ydps2_mean",
          "forty_speed_retention_20_40yd_ratio_mean"]
SINGLE_FEATURES = {
    "early_acceleration": PHASES[0],
    "late_acceleration": PHASES[1],
    "speed_retention": PHASES[2],
}


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, alpha: float) -> np.ndarray:
    med = train[features].median()
    tr = train[features].fillna(med)
    te = test[features].fillna(med)
    mu, sd = tr.mean(), tr.std().replace(0, 1).fillna(1)
    x = ((tr - mu) / sd).to_numpy(float)
    z = ((te - mu) / sd).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(float))
    return z @ beta


def stratified_player_bootstrap_delta(oof: pd.DataFrame, base_col: str, candidate_col: str,
                                      draws: int = 3000, seed: int = 4902) -> list[float] | None:
    if oof.empty:
        return None
    groups = [g.index.to_numpy() for _, g in oof.groupby("draft_year")]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        sample = oof.loc[idx]
        y = sample.actual.to_numpy(float)
        pred0 = sample[base_col].to_numpy(float)
        pred1 = sample[candidate_col].to_numpy(float)
        deltas.append(float(np.sqrt(np.mean((y - pred1) ** 2)) - np.sqrt(np.mean((y - pred0) ** 2))))
    return [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    args = parser.parse_args()

    profile_cols = ["nfl_id", "draft_year", "nfl_position", "combine_position", "forty", "ten_yd_split",
                    "combine_height", "combine_weight", "draft_overall_pick", *PHASES]
    profile = pd.read_csv(out / "player_speed_signatures.csv", usecols=lambda c: c in profile_cols)
    with open_data_source(args.archive) as zf:
        career = pd.read_csv(zf.open("nfl-big-data-bowl-2027/player_career_successes.csv"))
        games = pd.read_csv(zf.open("nfl-big-data-bowl-2027/games.csv"), usecols=["season", "season_type"])
    career_end_year = int(games.loc[games.season_type.eq("REG"), "season"].max())
    data = profile.merge(career, on="nfl_id", how="left", validate="one_to_one")
    offense = data.combine_position.isin(["OL", "TE", "WR"])
    data["career_role_snaps"] = np.where(offense, data.career_offensive_snaps, data.career_defensive_snaps)
    data["observed_regular_seasons"] = career_end_year - data.draft_year + 1
    data["career_role_snaps_per_season"] = data.career_role_snaps / data.observed_regular_seasons
    data["career_role_snaps_log_per_season"] = np.log1p(data.career_role_snaps_per_season)
    data["career_role_snaps_log_per_active_game"] = np.log1p(
        data.career_role_snaps / data.career_games_active.where(data.career_games_active.gt(0))
    )
    data["career_starts_per_season"] = data.career_games_started / data.observed_regular_seasons
    for pos in ["DB", "DL", "OL", "TE"]:
        data[f"position_{pos}"] = data.combine_position.eq(pos).astype(int)
    data["position_WR"] = data.combine_position.eq("WR").astype(int)
    data.to_csv(out / "career_speed_player_table.csv", index=False)

    # Complete cases for the primary candidate block ensure that both models
    # use identical prospects; absence of a tracked dash is not imputed as an
    # average athlete profile.
    needed = BASE + PHASES + ["career_role_snaps_log_per_season"]
    sample = data.loc[data[needed].notna().all(axis=1)].copy()
    specs = {"traditional_baseline": [], "all_three_speed_phases": PHASES.copy(),
             **{name: [feature] for name, feature in SINGLE_FEATURES.items()}}
    results = {"question": "Do tracked early/late 40-yard speed phases and speed retention add information about position-appropriate career snap role beyond official testing, body size, position, and draft capital?",
               "outcome": "career_role_snaps = career_offensive_snaps for Combine OL/TE/WR, otherwise career_defensive_snaps; divided by seasons represented in the competition's regular-season files (2023 cohort=3, 2024=2, 2025=1), then log1p.",
               "baseline": BASE,
               "tracking_block": PHASES,
               "validation": "Leave one complete draft cohort out. Baseline and augmented models use identical players in every comparison. Standardization and median imputation are fit on training players only. Bootstrap intervals resample fixed out-of-fold player errors and do not refit models.",
               "career_outcome_end_season": career_end_year,
               "sample": {"prospects_with_career_labels": int(data.career_role_snaps.notna().sum()),
                          "tracked_complete_case_players": int(len(sample)),
                          "players_by_cohort": {str(int(k)): int(v) for k, v in sample.draft_year.value_counts().sort_index().items()},
                          "players_by_position": {str(k): int(v) for k, v in sample.combine_position.value_counts().sort_index().items()}},
               "comparisons": []}
    prediction_tables = []
    for name, added in specs.items():
        comparison_rows = []
        fold_rows = []
        for cohort in sorted(sample.draft_year.unique()):
            train = sample.loc[sample.draft_year.ne(cohort)].copy()
            test = sample.loc[sample.draft_year.eq(cohort)].copy()
            row = {"held_out_draft_year": int(cohort), "n_train_players": int(len(train)), "n_test_players": int(len(test))}
            pred0 = ridge_predict(train, test, BASE, "career_role_snaps_log_per_season", args.ridge_alpha)
            pred1 = ridge_predict(train, test, BASE + added, "career_role_snaps_log_per_season", args.ridge_alpha)
            actual = test.career_role_snaps_log_per_season.to_numpy(float)
            row["baseline_rmse_log_snaps_per_season"] = float(np.sqrt(np.mean((actual - pred0) ** 2)))
            row["candidate_rmse_log_snaps_per_season"] = float(np.sqrt(np.mean((actual - pred1) ** 2)))
            row["rmse_change"] = row["candidate_rmse_log_snaps_per_season"] - row["baseline_rmse_log_snaps_per_season"]
            row["status"] = "scored"
            fold_rows.append(row)
            comparison_rows.extend({"nfl_id": int(pid), "draft_year": int(cohort), "actual": float(y),
                                   "pred_baseline": float(p0), "pred_candidate": float(p1)}
                                  for pid, y, p0, p1 in zip(test.nfl_id, actual, pred0, pred1))
        oof = pd.DataFrame(comparison_rows)
        delta = float(np.sqrt(np.mean((oof.actual - oof.pred_candidate) ** 2)) -
                      np.sqrt(np.mean((oof.actual - oof.pred_baseline) ** 2)))
        interval = stratified_player_bootstrap_delta(oof, "pred_baseline", "pred_candidate",
                                                      draws=args.bootstrap_draws, seed=4902 + len(added))
        results["comparisons"].append({"model": name, "added_features": added,
                                       "n_players": int(len(oof)), "pooled_oof_rmse_change": delta,
                                       "player_stratified_bootstrap_95pct_ci": interval,
                                       "folds": fold_rows})
        oof["model"] = name
        oof["outcome"] = "log_snaps_per_observed_season"
        prediction_tables.append(oof)

    # Sensitivity: role intensity conditional on being active in an NFL game.
    # This outcome is less affected by unequal follow-up time, but conditions on
    # appearing in the league and therefore answers a different question.
    active_sample = data.loc[data[BASE + PHASES + ["career_role_snaps_log_per_active_game"]].notna().all(axis=1)].copy()
    active_comparisons = []
    active_target = "career_role_snaps_log_per_active_game"
    for name, added in specs.items():
        rows = []
        fold_rows = []
        for cohort in sorted(active_sample.draft_year.unique()):
            train = active_sample.loc[active_sample.draft_year.ne(cohort)].copy()
            test = active_sample.loc[active_sample.draft_year.eq(cohort)].copy()
            pred0 = ridge_predict(train, test, BASE, active_target, args.ridge_alpha)
            pred1 = ridge_predict(train, test, BASE + added, active_target, args.ridge_alpha)
            actual = test[active_target].to_numpy(float)
            rmse0 = float(np.sqrt(np.mean((actual - pred0) ** 2)))
            rmse1 = float(np.sqrt(np.mean((actual - pred1) ** 2)))
            fold_rows.append({"held_out_draft_year": int(cohort), "n_test_players": int(len(test)),
                              "baseline_rmse": rmse0, "candidate_rmse": rmse1, "rmse_change": rmse1-rmse0})
            rows.extend({"nfl_id": int(pid), "draft_year": int(cohort), "actual": float(y),
                         "pred_baseline": float(p0), "pred_candidate": float(p1)}
                        for pid, y, p0, p1 in zip(test.nfl_id, actual, pred0, pred1))
        oof = pd.DataFrame(rows)
        delta = float(np.sqrt(np.mean((oof.actual - oof.pred_candidate) ** 2)) -
                      np.sqrt(np.mean((oof.actual - oof.pred_baseline) ** 2)))
        interval = stratified_player_bootstrap_delta(oof, "pred_baseline", "pred_candidate",
                                                     draws=args.bootstrap_draws, seed=5802 + len(added))
        active_comparisons.append({"model": name, "added_features": added, "n_players": int(len(oof)),
                                   "pooled_oof_rmse_change": delta,
                                   "player_stratified_bootstrap_95pct_ci": interval, "folds": fold_rows})
        oof["model"] = name
        oof["outcome"] = "log_snaps_per_active_game"
        prediction_tables.append(oof)
    results["active_game_sensitivity"] = {
        "outcome": "log1p(position-appropriate career snaps / career games active); conditional role intensity, not total career opportunity",
        "n_players": int(len(active_sample)),
        "players_by_cohort": {str(int(k)): int(v) for k, v in active_sample.draft_year.value_counts().sort_index().items()},
        "comparisons": active_comparisons,
    }
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions.to_csv(out / "career_speed_oof_predictions.csv", index=False)
    (out / "career_speed_transfer_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")

    lines = ["# 40-yard speed phases and career snap role", "",
             "This analysis asks whether the 10 Hz 40-yard movement profile adds information about NFL playing time beyond conventional measurements and draft capital. Career snap role is position-appropriate offensive or defensive snaps per observed regular season. The outcome is log-transformed to limit the influence of high-volume players. Because cohorts have different follow-up lengths, career snap totals are divided by the 1, 2, or 3 regular seasons represented in the supplied 2023–2025 files.", "",
             "The baseline includes official 40, 10-yard split, height, weight, draft overall pick, and Combine position. The primary comparison adds early acceleration, 10–20-yard acceleration, and 20–40-yard speed retention together. Models leave out one full draft cohort at a time. Player-bootstrap intervals resample fixed out-of-fold player errors within cohort; they do not refit models. This tests role association, not individual ability or a causal training effect.", "",
             f"Complete-case sample: {len(sample)} players across cohorts {', '.join(str(int(x)) for x in sorted(sample.draft_year.unique()))}. Outcome end season: {career_end_year}.", "",
             "| Model | Added movement features | Players | Pooled OOF RMSE Δ (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in results["comparisons"]:
        ci = item["player_stratified_bootstrap_95pct_ci"]
        delta = item["pooled_oof_rmse_change"]
        folds = {x["held_out_draft_year"]: x["rmse_change"] for x in item["folds"]}
        dtext = f"{delta:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f})"
        cells = ["NA" if y not in folds else f"{folds[y]:+.4f}" for y in [2023, 2024, 2025]]
        features = ", ".join(item["added_features"]) if item["added_features"] else "None"
        lines.append(f"| {item['model']} | {features} | {item['n_players']} | {dtext} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "Positive RMSE change means the tracking features worsened held-out predictions. The target is a playing-time/role proxy and remains affected by injury, team depth charts, and usage. No missing tracked test is assigned a speed value; the primary comparison is complete case.", ""]
    lines += ["## Sensitivity: snaps per game active", "",
              "This alternative outcome measures role intensity conditional on appearing in an NFL game. It excludes the playing-time opportunity represented by games active, so it is not interchangeable with career role volume.", "",
              "| Model | Players | Pooled OOF RMSE Δ (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |", "|---|---:|---:|---:|---:|---:|"]
    for item in active_comparisons:
        ci = item["player_stratified_bootstrap_95pct_ci"]
        folds = {x["held_out_draft_year"]: x["rmse_change"] for x in item["folds"]}
        cells = ["NA" if y not in folds else f"{folds[y]:+.4f}" for y in [2023, 2024, 2025]]
        lines.append(f"| {item['model']} | {item['n_players']} | {item['pooled_oof_rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [""]
    (out / "career_speed_transfer_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"career_end_season": career_end_year, "sample_players": int(len(sample)),
                      "sample_by_cohort": results["sample"]["players_by_cohort"],
                      "baseline_vs_phase_block_rmse_change": results["comparisons"][1]["pooled_oof_rmse_change"],
                      "report": str(out / "career_speed_transfer_report.md")}, indent=2))


if __name__ == "__main__":
    main()
