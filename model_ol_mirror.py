"""Discovery screen: offensive-line mirror movement vs rookie pass protection.

The player is the independent unit. Combine traces are summarized by player;
rookie pressure outcomes are aggregated over player-play opportunities. Results
are exploratory because this is a small, observational prospect cohort.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from run_analysis import DEFAULT_ARCHIVE, open_data_source

MEMBER = "nfl-big-data-bowl-2027/"
BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight", "position_T", "position_G", "draft_2024", "draft_2025", "mean_down", "mean_yards_to_go"]
FEATURES = {
    "mirror_abs_directional_speed_gap_ydps": "Absolute difference between mean lateral speeds toward the two path-relative directions. Requires both directions in the drill.",
    "mirror_mean_bilateral_speed_ydps": "Mean of the two path-relative lateral-speed means. Requires both directions in the drill.",
}
OUTCOMES = {
    "pressure_allowed_rate": ("pressure_allowed", "n_pressure_labels"),
    "mean_peak_pressure_probability_allowed": ("peak_pressure_probability_allowed", "n_probability_labels"),
}


def read_member(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(zf.open(MEMBER + name), **kwargs)


def design_matrices(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_parts, test_parts = [], []
    for feature in features:
        tr = pd.to_numeric(train[feature], errors="coerce")
        te = pd.to_numeric(test[feature], errors="coerce")
        median = float(tr.median()) if tr.notna().any() else 0.0
        train_parts.append(tr.fillna(median).rename(feature))
        test_parts.append(te.fillna(median).rename(feature))
        if tr.isna().any():
            train_parts.append(tr.isna().astype(float).rename(f"{feature}__missing"))
            test_parts.append(te.isna().astype(float).rename(f"{feature}__missing"))
    xdf = pd.concat(train_parts, axis=1)
    zdf = pd.concat(test_parts, axis=1)
    mu = xdf.mean()
    sd = xdf.std().replace(0, 1).fillna(1)
    return ((xdf - mu) / sd).to_numpy(float), ((zdf - mu) / sd).to_numpy(float)


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, alpha: float) -> np.ndarray:
    x, z = design_matrices(train, test, features)
    x = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(float))
    return z @ beta


def paired_player_bootstrap_rmse_delta(oof: pd.DataFrame, seed: int = 833, draws: int = 3000) -> list[float] | None:
    """Stratified paired bootstrap of candidate-minus-baseline OOF RMSE."""
    if oof.empty:
        return None
    groups = [g.index.to_numpy() for _, g in oof.groupby("draft_year")]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in groups])
        b = oof.loc[idx]
        y = b.actual.to_numpy(float)
        base = b.pred_base.to_numpy(float)
        plus = b.pred_plus.to_numpy(float)
        deltas.append(float(np.sqrt(np.mean((y - plus) ** 2)) - np.sqrt(np.mean((y - base) ** 2))))
    return [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]


def pressure_outcomes(archive: zipfile.ZipFile) -> tuple[pd.DataFrame, dict]:
    players = read_member(archive, "players.csv", usecols=["nfl_id", "draft_year", "nfl_position"])
    games = read_member(archive, "games.csv", usecols=["game_id", "season", "season_type"])
    games = games.loc[games.season_type.eq("REG"), ["game_id", "season"]]
    cols = ["game_id", "play_id", "nfl_id", "lined_up_position", "pressure_allowed",
            "peak_pressure_probability_allowed", "down", "yards_to_go", "offense_formation"]
    plays = read_member(archive, "player_play.csv", usecols=cols)
    plays = plays.merge(games, on="game_id", how="inner", validate="many_to_one")
    plays = plays.merge(players, on="nfl_id", how="inner", validate="many_to_one")
    plays = plays.loc[
        plays.season.eq(plays.draft_year)
        & plays.lined_up_position.isin(["T", "G", "C"])
        & plays.nfl_position.isin(["T", "G", "C"])
    ].copy()
    plays["pressure_allowed"] = plays.pressure_allowed.astype("boolean").astype(float)
    plays["peak_pressure_probability_allowed"] = pd.to_numeric(plays.peak_pressure_probability_allowed, errors="coerce")
    plays["down"] = pd.to_numeric(plays.down, errors="coerce")
    plays["yards_to_go"] = pd.to_numeric(plays.yards_to_go, errors="coerce")
    summary = (plays.groupby(["nfl_id", "draft_year", "nfl_position"], as_index=False)
               .agg(rookie_pass_pro_plays=("play_id", "size"),
                    n_pressure_labels=("pressure_allowed", "count"),
                    pressure_allowed_rate=("pressure_allowed", "mean"),
                    n_probability_labels=("peak_pressure_probability_allowed", "count"),
                    mean_peak_pressure_probability_allowed=("peak_pressure_probability_allowed", "mean")))
    # Context controls summarize the same player-plays that carry the primary
    # pressure label; do not mix in unlabeled rows or likely non-pass snaps.
    context = (plays.loc[plays.pressure_allowed.notna()]
               .groupby(["nfl_id", "draft_year", "nfl_position"], as_index=False)
               .agg(mean_down=("down", "mean"), mean_yards_to_go=("yards_to_go", "mean")))
    summary = summary.merge(context, on=["nfl_id", "draft_year", "nfl_position"], how="left", validate="one_to_one")
    coverage = {
        "rookie_regular_season_ol_player_play_rows": int(len(plays)),
        "rookie_ol_players": int(plays.nfl_id.nunique()),
        "pressure_label_rows": int(plays.pressure_allowed.notna().sum()),
        "pressure_probability_rows": int(plays.peak_pressure_probability_allowed.notna().sum()),
        "player_rows_with_pressure_labels": int(summary.pressure_allowed_rate.notna().sum()),
        "pressure_label_definition": "Uses the supplied player_play pressure_allowed label on rookie REG player-play rows where lined_up_position and nfl_position are T/G/C. It does not infer missing labels as no pressure.",
        "limitation": "Player outcome aggregates are opportunity-dependent and are not fully adjusted for team, opponent, protection assignment, or play context.",
    }
    return summary, coverage


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--min-plays", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    args = parser.parse_args()

    player = pd.read_csv(out / "all_player_speed_profiles.csv")
    drills = pd.read_csv(out / "player_drill_speed_profiles.csv")
    mirror = drills.loc[drills.drill_name.isin(["PASS_PRO-MIRROR_DRILL", "PASS_PRO_MIRROR_DRILL"])].copy()
    mirror["mirror_abs_directional_speed_gap_ydps"] = mirror.left_right_lateral_speed_difference_by_drill_ydps.abs()
    mirror["mirror_mean_bilateral_speed_ydps"] = (
        mirror.left_lateral_speed_mean_ydps_mean + mirror.right_lateral_speed_mean_ydps_mean
    ) / 2
    # Usually one naming variant per player. If duplicates exist, aggregate within player.
    mirror_player = (mirror.groupby("nfl_id", as_index=False)
                     .agg(mirror_abs_directional_speed_gap_ydps=("mirror_abs_directional_speed_gap_ydps", "mean"),
                          mirror_mean_bilateral_speed_ydps=("mirror_mean_bilateral_speed_ydps", "mean"),
                          mirror_early_accel_ydps2=("mean_accel_first_1s_ydps2_mean", "mean"),
                          mirror_sustained_speed_ydps=("peak_sustained_speed_ydps_mean", "mean"),
                          mirror_drill_variants=("drill_name", "nunique")))
    with open_data_source(args.archive) as archive:
        outcomes, coverage = pressure_outcomes(archive)

    data = (player.loc[player.nfl_position.isin(["T", "G", "C"])]
            .merge(mirror_player, on="nfl_id", how="left", validate="one_to_one")
            .merge(outcomes, on=["nfl_id", "draft_year", "nfl_position"], how="left", validate="one_to_one"))
    data = pd.concat([data, pd.DataFrame({
        "position_T": data.nfl_position.eq("T").astype(int),
        "position_G": data.nfl_position.eq("G").astype(int),
        "draft_2024": data.draft_year.eq(2024).astype(int),
        "draft_2025": data.draft_year.eq(2025).astype(int),
    }, index=data.index)], axis=1)
    data.to_csv(out / "ol_mirror_player_table.csv", index=False)

    model_results = {
        "analysis_unit": "Prospect/player. Every rookie player-play contributes only to that player's outcome estimate.",
        "window": "Rookie regular season; REG player-play rows with both NFL position and lined-up position in T/G/C.",
        "baseline": BASE,
        "candidate_features": FEATURES,
        "outcomes": OUTCOMES,
        "validation": "Leave-one-draft-cohort-out ridge regression. Baseline and candidate use identical complete-case players per comparison. Standardization is learned on training players only.",
        "screen_status": "Exploratory; candidate features are not independently validated, team/opponent context is not fully adjusted, and comparisons are not multiplicity-corrected.",
        "ridge_alpha": args.ridge_alpha,
        "minimum_play_sensitivity": [],
        "comparisons": [],
        "coverage": coverage,
    }
    rows = []
    for min_plays in sorted(set(args.min_plays)):
        for outcome, (_, exposure_column) in OUTCOMES.items():
            eligible = data.loc[data[exposure_column].ge(min_plays)].copy()
            for feature in FEATURES:
                cols = BASE + [feature, outcome]
                # Candidate and outcome must be observed. Baseline-test missingness
                # is imputed from training players within each cohort fold.
                sample = eligible.loc[eligible[[feature, outcome]].notna().all(axis=1)].copy()
                record = {"minimum_pressure_label_plays": min_plays, "outcome": outcome,
                          "feature": feature, "n_players": int(len(sample)),
                          "players_by_cohort": {str(int(k)): int(v) for k, v in sample.draft_year.value_counts().sort_index().items()}}
                oof_records = []
                folds = []
                for cohort in sorted(sample.draft_year.dropna().unique()):
                    train = sample.loc[sample.draft_year.ne(cohort)]
                    test = sample.loc[sample.draft_year.eq(cohort)]
                    fold = {"held_out_draft_year": int(cohort), "n_train_players": int(len(train)), "n_test_players": int(len(test))}
                    if len(train) < 15 or len(test) < 5:
                        fold["status"] = "too_few_players"
                        folds.append(fold)
                        continue
                    p0 = ridge_predict(train, test, BASE, outcome, args.ridge_alpha)
                    p1 = ridge_predict(train, test, BASE + [feature], outcome, args.ridge_alpha)
                    actual = test[outcome].to_numpy(float)
                    fold["baseline_rmse"] = float(np.sqrt(np.mean((actual - p0) ** 2)))
                    fold["candidate_rmse"] = float(np.sqrt(np.mean((actual - p1) ** 2)))
                    fold["rmse_change"] = fold["candidate_rmse"] - fold["baseline_rmse"]
                    fold["status"] = "scored"
                    folds.append(fold)
                    oof_records.extend({"nfl_id": int(pid), "draft_year": int(cohort), "actual": float(yy),
                                        "pred_base": float(a), "pred_plus": float(b)}
                                       for pid, yy, a, b in zip(test.nfl_id, actual, p0, p1))
                if oof_records:
                    oof = pd.DataFrame(oof_records)
                    y = oof.actual.to_numpy(float)
                    pred0_all = oof.pred_base.to_numpy(float)
                    pred1_all = oof.pred_plus.to_numpy(float)
                    record["pooled_oof_baseline_rmse"] = float(np.sqrt(np.mean((y - pred0_all) ** 2)))
                    record["pooled_oof_candidate_rmse"] = float(np.sqrt(np.mean((y - pred1_all) ** 2)))
                    record["pooled_oof_rmse_change"] = record["pooled_oof_candidate_rmse"] - record["pooled_oof_baseline_rmse"]
                    record["pooled_oof_rmse_change_player_bootstrap_95pct_ci"] = paired_player_bootstrap_rmse_delta(oof, seed=833 + min_plays + len(sample))
                else:
                    record["pooled_oof_rmse_change"] = None
                    record["pooled_oof_rmse_change_player_bootstrap_95pct_ci"] = None
                record["cohorts"] = folds
                model_results["comparisons"].append(record)
                fold_deltas = {str(f["held_out_draft_year"]): f.get("rmse_change") for f in folds}
                rows.append({**{k: record.get(k) for k in ["minimum_pressure_label_plays", "outcome", "feature", "n_players", "pooled_oof_rmse_change", "pooled_oof_rmse_change_player_bootstrap_95pct_ci"]},
                             "fold_rmse_changes": fold_deltas})
        model_results["minimum_play_sensitivity"].append({"minimum_pressure_label_plays": min_plays,
                                                            "players_with_outcomes": int(data.n_pressure_labels.ge(min_plays).sum())})

    (out / "ol_mirror_model_results.json").write_text(json.dumps(model_results, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "ol_mirror_candidate_summary.csv", index=False)
    lines = ["# Offensive-line mirror drill candidate screen", "",
             "Exploratory player-level screen: 10 Hz Combine pass-pro mirror movement is paired with rookie regular-season pass-protection labels. The independent unit is the player. The model leaves one full draft cohort out; it uses identical players for its baseline and candidate model. Baseline predictors include official 40/split, size, position, cohort, and each player's average down and yards-to-go context. Missing baseline measurements are imputed from training-fold medians with missingness indicators.", "",
             f"The source includes {coverage['rookie_regular_season_ol_player_play_rows']:,} rookie T/G/C player-play rows across {coverage['rookie_ol_players']} players. A higher pressure-allowed rate is worse. The mirror asymmetry is the absolute difference between lateral speeds toward the two path-relative directions; it is not anatomical left-vs-right preference.", "",
             "The score is opportunity-dependent. Context adjustment includes average down and yards-to-go on pressure-labeled plays, but not team, opponent, or protection assignment. Thresholds below restrict each player to a minimum number of observed outcome labels; no missing labels are treated as zero. Cohort folds with fewer than five test players or 15 training players are not scored.", "",
             "| Minimum labeled plays | Outcome | Tracking feature | Players | Pooled RMSE Δ (player-bootstrap 95% CI) | 2023 Δ | 2024 Δ | 2025 Δ |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        ci = r["pooled_oof_rmse_change_player_bootstrap_95pct_ci"]
        change = "NA" if r["pooled_oof_rmse_change"] is None else f"{r['pooled_oof_rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f})"
        fold_values = [r["fold_rmse_changes"].get(str(y)) for y in [2023, 2024, 2025]]
        fold_text = ["NA" if v is None else f"{v:+.4f}" for v in fold_values]
        lines.append(f"| {r['minimum_pressure_label_plays']} | {r['outcome']} | {r['feature']} | {r['n_players']} | {change} | {fold_text[0]} | {fold_text[1]} | {fold_text[2]} |")
    lines += ["", "Positive RMSE change means the tracking feature hurt held-out predictions. Bootstrap intervals resample players within draft cohort and compare paired out-of-fold errors. This candidate screen does not establish causality, a development effect, or a draft-bust probability. It should only be promoted after context adjustment, event-definition review, and a frozen follow-up specification.", ""]
    (out / "ol_mirror_model_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"players": int(data.nfl_id.nunique()), "combine_mirror_players": int(data.mirror_abs_directional_speed_gap_ydps.notna().sum()),
                      "rookie_outcome_players": int(data.n_pressure_labels.notna().sum()), "comparisons": len(rows),
                      "report": str(out / "ol_mirror_model_report.md")}, indent=2))


if __name__ == "__main__":
    main()
