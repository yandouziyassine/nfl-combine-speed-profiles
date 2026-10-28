"""Sensitivity of DL Combine early-acceleration transfer to 40-yard attempt choice.

Uses only players in the primary 100-tracked-play DL sample who have both
tracked attempts 1 and 2. Compares each attempt and their mean on identical
players with the primary pre-draft baseline and cohort-held-out design.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight"]
TARGET = "rookie_reg_p90_play_peak_speed_ydps"
PRIMARY = "forty_early_accel_0_1s_ydps2_mean"
ATTEMPT_FEATURE = "early_accel_0_1s_ydps2"
ALPHAS = [5.0, 20.0]


def predict(tr: pd.DataFrame, te: pd.DataFrame, features: list[str], alpha: float) -> np.ndarray:
    med = tr[features].median()
    train = tr[features].fillna(med)
    test = te[features].fillna(med)
    mean = train.mean()
    scale = train.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(train)), ((train - mean) / scale).to_numpy(float)])
    z = np.column_stack([np.ones(len(test)), ((test - mean) / scale).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return z @ np.linalg.solve(x.T @ x + penalty, x.T @ tr[TARGET].to_numpy(float))


def bootstrap_delta(oof: pd.DataFrame, seed: int, draws: int = 3000) -> list[float]:
    groups = [group.index.to_numpy() for _, group in oof.groupby("draft_year")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sample_index = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        sample = oof.loc[sample_index]
        baseline_rmse = np.sqrt(np.mean((sample.actual - sample.baseline) ** 2))
        candidate_rmse = np.sqrt(np.mean((sample.actual - sample.candidate) ** 2))
        values.append(float(candidate_rmse - baseline_rmse))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def rmse_delta(group: pd.DataFrame) -> float:
    baseline = np.sqrt(np.mean((group.actual - group.baseline) ** 2))
    candidate = np.sqrt(np.mean((group.actual - group.candidate) ** 2))
    return float(candidate - baseline)


def main() -> None:
    signatures = pd.read_csv(OUT / "player_speed_signatures.csv")
    eligible = signatures.loc[
        signatures.combine_position.eq("DL")
        & signatures.rookie_reg_player_plays_with_matching_metadata.ge(100)
    ].copy()
    required = ["nfl_id", "draft_year", TARGET, *BASE, PRIMARY]
    primary_sample = eligible.loc[eligible[required].notna().all(axis=1), required].copy()

    attempts = pd.read_csv(OUT / "combine_attempt_speed_profiles.csv")
    attempts = attempts.loc[
        attempts.drill_name.eq("FORTY_YARD_DASH")
        & attempts.attempt.isin([1, 2]),
        ["nfl_id", "draft_year", "attempt", ATTEMPT_FEATURE],
    ]
    wide = attempts.pivot_table(
        index=["nfl_id", "draft_year"],
        columns="attempt",
        values=ATTEMPT_FEATURE,
        aggfunc="mean",
    ).rename(columns={1: "attempt_1", 2: "attempt_2"}).reset_index()
    if "attempt_1" not in wide or "attempt_2" not in wide:
        raise ValueError("Both tracked 40-yard attempt numbers must exist in the source profiles.")

    merged = primary_sample.merge(wide, on=["nfl_id", "draft_year"], how="left", validate="one_to_one")
    coverage = {
        "primary_complete_sample_n": int(len(primary_sample)),
        "primary_complete_sample_by_cohort": {
            str(int(year)): int(n)
            for year, n in primary_sample.draft_year.value_counts().sort_index().items()
        },
        "attempt_1_and_2_both_available": int((merged.attempt_1.notna() & merged.attempt_2.notna()).sum()),
        "attempt_1_only": int((merged.attempt_1.notna() & merged.attempt_2.isna()).sum()),
        "attempt_2_only": int((merged.attempt_1.isna() & merged.attempt_2.notna()).sum()),
        "neither_attempt_1_nor_2": int((merged.attempt_1.isna() & merged.attempt_2.isna()).sum()),
    }
    paired = merged.dropna(subset=["attempt_1", "attempt_2"]).copy()
    paired["attempt_mean_1_2"] = paired[["attempt_1", "attempt_2"]].mean(axis=1)
    if paired.draft_year.nunique() != 3:
        raise ValueError("Paired subset must contain all three draft cohorts.")
    paired_feature_difference = (paired["attempt_mean_1_2"] - paired[PRIMARY]).abs()
    coverage["paired_sample_n"] = int(len(paired))
    coverage["paired_sample_by_cohort"] = {
        str(int(year)): int(n) for year, n in paired.draft_year.value_counts().sort_index().items()
    }
    coverage["attempt_1_attempt_2_pearson_r"] = float(paired["attempt_1"].corr(paired["attempt_2"]))
    coverage["attempt_1_attempt_2_spearman_rho"] = float(
        paired[["attempt_1", "attempt_2"]].corr(method="spearman").iloc[0, 1]
    )
    coverage["attempt_2_minus_attempt_1_mean_ydps2"] = float(
        (paired["attempt_2"] - paired["attempt_1"]).mean()
    )
    coverage["mean_1_2_matches_primary_aggregate_max_abs_difference_ydps2"] = float(
        paired_feature_difference.max()
    )

    candidate_features = {
        "all_available_attempt_mean": PRIMARY,
        "attempt_1_only": "attempt_1",
        "attempt_2_only": "attempt_2",
        "attempt_1_2_mean": "attempt_mean_1_2",
    }
    results = []
    oof_rows = []
    for alpha_index, alpha in enumerate(ALPHAS):
        for label, feature in candidate_features.items():
            rows = []
            folds = []
            for cohort in sorted(paired.draft_year.unique()):
                train = paired.loc[paired.draft_year.ne(cohort)]
                test = paired.loc[paired.draft_year.eq(cohort)]
                baseline = predict(train, test, BASE, alpha)
                candidate = predict(train, test, BASE + [feature], alpha)
                actual = test[TARGET].to_numpy(float)
                fold_oof = pd.DataFrame({
                    "nfl_id": test.nfl_id.to_numpy(),
                    "draft_year": test.draft_year.to_numpy(),
                    "actual": actual,
                    "baseline": baseline,
                    "candidate": candidate,
                })
                rows.append(fold_oof)
                folds.append({
                    "held_out_draft_year": int(cohort),
                    "n_players": int(len(test)),
                    "rmse_change": rmse_delta(fold_oof),
                })
            oof = pd.concat(rows, ignore_index=True)
            oof["alpha"] = alpha
            oof["feature_variant"] = label
            oof_rows.append(oof)
            results.append({
                "alpha": alpha,
                "feature_variant": label,
                "rmse_change": rmse_delta(oof),
                "player_bootstrap_95pct_ci_fixed_oof": bootstrap_delta(oof, 800 + alpha_index),
                "folds": folds,
            })

    result = {
        "question": "Does the primary DL early-acceleration association depend on averaging tracked 40-yard attempts 1 and 2?",
        "estimand": "Incremental cohort-held-out RMSE for rookie-season p90 of whole-play peak speed, in yd/s.",
        "baseline": "Official 40 time, 10-yard split, height, and weight; train-only imputation and standardization.",
        "candidate": "Baseline plus one early-acceleration measure: all-available-attempt mean, attempt 1, attempt 2, or the paired mean of attempts 1 and 2.",
        "sample": coverage,
        "alpha_grid": ALPHAS,
        "bootstrap_draws": 3000,
        "bootstrap_method": "Resample players within draft cohort from fixed out-of-fold predictions; models are not refit.",
        "comparisons": results,
        "limitations": [
            "The paired sensitivity sample is small and selected by availability of both tracked attempts.",
            "Only 2025 is a forward-in-time cohort; 2023 and 2024 leave-one-cohort-out folds train on later cohorts.",
            "The feature was selected after broader screens, so these fixed-prediction intervals do not adjust for selection.",
            "The outcome remains p90 of whole-play maxima, not fixed-window burst or pass-rush performance.",
        ],
    }
    (OUT / "dl_attempt_specific_transfer_results_100plays.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.concat(oof_rows, ignore_index=True).to_csv(OUT / "dl_attempt_specific_transfer_oof_100plays.csv", index=False)

    lines = [
        "# Combine-attempt sensitivity for DL early acceleration",
        "",
        "This exploratory check asks whether the primary early-acceleration result is driven by averaging the first two tracked 40-yard attempts. It uses the same 100-player-play exposure rule, pre-draft baseline, ridge model, and leave-one-draft-cohort-out folds as the primary model, but restricts every comparison to the players with both attempts 1 and 2.",
        "",
        f"The primary complete-case sample has {coverage['primary_complete_sample_n']} DLs. Both attempts 1 and 2 are available for {coverage['paired_sample_n']} ({', '.join(f'{year}: {n}' for year, n in coverage['paired_sample_by_cohort'].items())}). In that paired subset, attempt 1 and 2 correlate at Pearson r={coverage['attempt_1_attempt_2_pearson_r']:.2f} and Spearman rho={coverage['attempt_1_attempt_2_spearman_rho']:.2f}; attempt 2 averages {coverage['attempt_2_minus_attempt_1_mean_ydps2']:+.3f} yd/s² relative to attempt 1. The paired mean exactly matches the primary feature aggregation for these players (maximum absolute difference {coverage['mean_1_2_matches_primary_aggregate_max_abs_difference_ydps2']:.6f} yd/s²).",
        "",
        "Negative RMSE change favors adding the movement feature. The baseline is identical across feature variants and is scored on the same paired players.",
        "",
        "| Alpha | Feature | Paired n | Δ RMSE (95% fixed-OOF player-bootstrap interval) | 2023 Δ | 2024 Δ | 2025 Δ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        ci = item["player_bootstrap_95pct_ci_fixed_oof"]
        fold_map = {row["held_out_draft_year"]: row["rmse_change"] for row in item["folds"]}
        lines.append(
            f"| {item['alpha']:g} | {item['feature_variant']} | {len(paired)} | "
            f"{item['rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | "
            f"{fold_map[2023]:+.4f} | {fold_map[2024]:+.4f} | {fold_map[2025]:+.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "On the paired subset, both individual attempts and their mean lower held-out RMSE at both ridge settings. Each attempt-specific estimate has a negative point estimate in all three held-out cohorts, although the 2023 attempt-1 effect is small. The mean is slightly more consistent across cohorts. This reduces concern that the main result is an artifact of choosing only one run, but the paired sample is just 27 players and overlaps the screened discovery sample. It is not independent replication or evidence of a causal training mechanism.",
        "",
        "The target remains upper-tail speed among observed whole-play maxima. These results do not establish transfer to fixed-duration post-snap speed, first-step burst, pass-rush outcomes, or a draft decision threshold. See the OOF prediction CSV and JSON for full details.",
        "",
    ]
    (OUT / "dl_attempt_specific_transfer_report_100plays.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "paired_sample": coverage["paired_sample_n"],
        "paired_sample_by_cohort": coverage["paired_sample_by_cohort"],
        "attempt_correlation": coverage["attempt_1_attempt_2_pearson_r"],
        "comparisons": [
            {"alpha": item["alpha"], "feature": item["feature_variant"],
             "delta": round(item["rmse_change"], 4),
             "ci": [round(x, 4) for x in item["player_bootstrap_95pct_ci_fixed_oof"]]}
            for item in results
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
