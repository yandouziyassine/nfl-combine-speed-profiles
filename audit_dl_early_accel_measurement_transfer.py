"""Check primary DL transfer across defensible 40-yard signal-processing settings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight"]
TARGET = "rookie_reg_p90_play_peak_speed_ydps"
MEASURE = "early_accel_0_1s_ydps2"
MIN_PLAYS = 100
ALPHAS = [5.0, 20.0]


def predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float) -> np.ndarray:
    med = train[features].median()
    x_train = train[features].fillna(med)
    x_test = test[features].fillna(med)
    mean = x_train.mean()
    scale = x_train.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(x_train)), ((x_train - mean) / scale).to_numpy(float)])
    z = np.column_stack([np.ones(len(x_test)), ((x_test - mean) / scale).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return z @ np.linalg.solve(x.T @ x + penalty, x.T @ train[TARGET].to_numpy(float))


def main() -> None:
    signatures = pd.read_csv(OUT / "player_speed_signatures.csv")
    base = signatures.loc[
        signatures.combine_position.eq("DL")
        & signatures.rookie_reg_player_plays_with_matching_metadata.ge(MIN_PLAYS)
    ].copy()
    base = base[["nfl_id", "draft_year", TARGET, *BASE]].dropna()
    measured = pd.read_csv(OUT / "forty_feature_sensitivity_player_profiles.csv")
    measured = measured.loc[measured.combine_position.eq("DL")].copy()
    data = measured.merge(base, on=["nfl_id", "draft_year"], how="inner", validate="many_to_one")
    default_players = set(
        data.loc[
            data.onset_threshold_ydps.eq(1.0)
            & data.smoothing_frames.eq(3)
            & data[MEASURE].notna(),
            "nfl_id",
        ].astype(int)
    )
    # Freeze the complete-case sample from the production processing settings;
    # otherwise a processing variant could change the players as well as the feature.
    data = data.loc[data.nfl_id.isin(default_players)].copy()

    rows: list[dict] = []
    fold_rows: list[dict] = []
    for (threshold, width), group in data.groupby(["onset_threshold_ydps", "smoothing_frames"], sort=True):
        feature = f"measurement_{MEASURE}"
        current = group.rename(columns={MEASURE: feature}).dropna(subset=[feature, TARGET, *BASE]).copy()
        for alpha in ALPHAS:
            oof = []
            folds = []
            for cohort in sorted(current.draft_year.unique()):
                train = current.loc[current.draft_year.ne(cohort)]
                test = current.loc[current.draft_year.eq(cohort)]
                if train.empty or test.empty:
                    continue
                baseline = predict(train, test, BASE, alpha)
                candidate = predict(train, test, [*BASE, feature], alpha)
                actual = test[TARGET].to_numpy(float)
                base_rmse = float(np.sqrt(np.mean((actual - baseline) ** 2)))
                candidate_rmse = float(np.sqrt(np.mean((actual - candidate) ** 2)))
                folds.append({"draft_year": int(cohort), "n": int(len(test)), "rmse_change": candidate_rmse - base_rmse})
                oof.extend({
                    "nfl_id": int(pid), "draft_year": int(year), "actual": float(y),
                    "baseline": float(b), "candidate": float(c),
                } for pid, year, y, b, c in zip(test.nfl_id, test.draft_year, actual, baseline, candidate))
            pred = pd.DataFrame(oof)
            delta = float(
                np.sqrt(np.mean((pred.actual - pred.candidate) ** 2))
                - np.sqrt(np.mean((pred.actual - pred.baseline) ** 2))
            )
            rows.append({
                "onset_threshold_ydps": float(threshold), "smoothing_frames": int(width),
                "alpha": alpha, "n_players": int(pred.nfl_id.nunique()),
                "rmse_change_ydps": delta,
            })
            for fold in folds:
                fold_rows.append({"onset_threshold_ydps": float(threshold), "smoothing_frames": int(width), "alpha": alpha, **fold})

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "dl_early_accel_measurement_transfer_sensitivity.csv", index=False)
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(OUT / "dl_early_accel_measurement_transfer_folds.csv", index=False)
    default = results.loc[results.onset_threshold_ydps.eq(1.0) & results.smoothing_frames.eq(3)]
    summary = {
        "estimand": "Leave-one-draft-cohort-out incremental RMSE from the early 0-1 second Combine acceleration feature, conditional on at least 100 matched rookie tracked player-plays.",
        "sample_n": int(len(default_players)),
        "processing_grid": "Onset thresholds 0.5, 1.0, 1.5 yd/s crossed with centered median filters of 1, 3, 5 frames.",
        "baseline": BASE,
        "alpha_grid": ALPHAS,
        "default_results": default.to_dict(orient="records"),
        "minimum_rmse_change_by_alpha": {str(int(a)): float(g.rmse_change_ydps.min()) for a, g in results.groupby("alpha")},
        "maximum_rmse_change_by_alpha": {str(int(a)): float(g.rmse_change_ydps.max()) for a, g in results.groupby("alpha")},
        "settings_with_positive_rmse_change": int(results.rmse_change_ydps.gt(0).sum()),
        "total_settings": int(len(results)),
        "caveat": "The processing grid is a measurement sensitivity analysis, not independent validation. It does not correct broad feature-search optimism; fixed OOF intervals are not recomputed here.",
        "results": rows,
        "folds": fold_rows,
    }
    (OUT / "dl_early_accel_measurement_transfer_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Early-acceleration transfer under signal-processing alternatives", "",
        f"The same {summary['sample_n']} DL players with at least {MIN_PLAYS} matched rookie tracked plays and a complete feature under the production settings were evaluated at every setting. Each setting refits the same leave-one-draft-cohort-out ridge models using the pre-draft baseline (official 40, 10-yard split, height, weight), then adds one early-acceleration feature recomputed from the Combine trace.", "",
        "The 3 × 3 grid crosses movement-onset thresholds (0.5, 1.0, 1.5 yd/s) with centered median filters (1, 3, 5 frames). RMSE change is candidate minus baseline; negative values favor adding early acceleration.", "",
        "| Onset threshold | Filter frames | Players | Alpha 5 Δ RMSE | Alpha 20 Δ RMSE |", "|---:|---:|---:|---:|---:|"
    ]
    for (threshold, width), group in results.groupby(["onset_threshold_ydps", "smoothing_frames"], sort=True):
        by_alpha = group.set_index("alpha")
        lines.append(
            f"| {threshold:.1f} yd/s | {int(width)} | {int(group.n_players.min())} | "
            f"{by_alpha.loc[5.0, 'rmse_change_ydps']:+.4f} | {by_alpha.loc[20.0, 'rmse_change_ydps']:+.4f} |"
        )
    lines += [
        "", f"Across this grid, {summary['settings_with_positive_rmse_change']} of {summary['total_settings']} feature-model comparisons had higher OOF RMSE than baseline.", "",
        "This check asks whether the relationship persists when the feature's processing changes; it does not establish that any one processing choice is optimal. The grid is not independent validation, uses the same previously screened player sample and outcome, and does not correct for broader discovery selection. See the CSVs for player-level model-setting and fold results.", ""
    ]
    (OUT / "dl_early_accel_measurement_transfer_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["sample_n", "settings_with_positive_rmse_change", "total_settings", "default_results", "minimum_rmse_change_by_alpha", "maximum_rmse_change_by_alpha"]}, indent=2))


if __name__ == "__main__":
    main()
