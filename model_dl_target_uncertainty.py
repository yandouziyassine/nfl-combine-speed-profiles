"""Propagate game-cluster p90 sampling uncertainty through DL transfer models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TARGET = "rookie_reg_p90_play_peak_speed_ydps"
EARLY = "forty_early_accel_0_1s_ydps2_mean"
BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight"]


def predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], y_train: np.ndarray,
            alpha: float) -> np.ndarray:
    x0 = train[features]
    x1 = test[features]
    med = x0.median()
    x0 = x0.fillna(med)
    x1 = x1.fillna(med)
    mean = x0.mean()
    sd = x0.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(x0)), ((x0 - mean) / sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(x1)), ((x1 - mean) / sd).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y_train)
    return z @ beta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=500,
                    help="Number of target-uncertainty replicates; each is a full CV model refit.")
    ap.add_argument("--seed", type=int, default=9272027)
    args = ap.parse_args()

    oof = pd.read_csv(OUT / "dl_early_accel_predraft_oof_100plays.csv")
    data = pd.read_csv(OUT / "player_speed_signatures.csv")
    precision_draws = pd.read_csv(OUT / "dl_game_speed_target_bootstrap_draws_100plays.csv")
    players = oof[["nfl_id", "draft_year"]].drop_duplicates()
    data = players.merge(data, on="nfl_id", how="left", suffixes=("_oof", ""))
    data["draft_year"] = data.draft_year_oof.astype(int)
    controls = BASE
    candidate = controls + [EARLY]
    data = data.sort_values(["draft_year", "nfl_id"]).reset_index(drop=True)
    draw_map = {
        int(pid): g.p90_draw_ydps.to_numpy(float)
        for pid, g in precision_draws.groupby("nfl_id", sort=False)
    }
    if set(data.nfl_id.astype(int)) - set(draw_map):
        raise ValueError("Missing player bootstrap draws; rerun analyze_dl_game_speed_target_precision.py.")

    rng = np.random.default_rng(args.seed)
    cohorts = sorted(data.draft_year.unique())
    alphas = [5.0, 20.0]
    result = {"players": int(len(data)), "cohorts": [int(c) for c in cohorts],
              "target_uncertainty": "For each replicate, independently draw each player's rookie p90 from their game-cluster bootstrap distribution; refit the model in each leave-one-draft-cohort-out fold. Independent player draws do not preserve shared game, opponent, or weather shocks.",
              "baseline": "pre-draft official 40, 10-yard split, height, and weight; no draft capital or NFL position",
              "candidate": "baseline plus Combine 40-yard 0-1 second mean acceleration",
              "replicates": int(args.draws), "results": []}
    rows = []
    for rep in range(args.draws):
        y = np.array([rng.choice(draw_map[int(pid)]) for pid in data.nfl_id], dtype=float)
        for alpha in alphas:
            actuals, base_preds, candidate_preds = [], [], []
            for cohort in cohorts:
                tr_idx = data.draft_year.ne(cohort).to_numpy()
                te_idx = data.draft_year.eq(cohort).to_numpy()
                tr, te = data.loc[tr_idx], data.loc[te_idx]
                p0 = predict(tr, te, controls, y[tr_idx], alpha)
                p1 = predict(tr, te, candidate, y[tr_idx], alpha)
                actuals.extend(y[te_idx]); base_preds.extend(p0); candidate_preds.extend(p1)
            yv = np.asarray(actuals)
            delta = float(np.sqrt(np.mean((yv - np.asarray(candidate_preds)) ** 2)) -
                          np.sqrt(np.mean((yv - np.asarray(base_preds)) ** 2)))
            rows.append({"replicate": rep, "alpha": alpha, "rmse_change": delta})
    result_table = pd.DataFrame(rows)
    result_table.to_csv(OUT / "dl_target_uncertainty_cv_deltas_100plays.csv", index=False)
    for alpha, group in result_table.groupby("alpha"):
        vals = group.rmse_change.to_numpy()
        result["results"].append({
            "alpha": float(alpha),
            "median_rmse_change": float(np.median(vals)),
            "target_uncertainty_95pct_interval": [float(np.quantile(vals, .025)),
                                                  float(np.quantile(vals, .975))],
            "fraction_replicates_candidate_improves": float(np.mean(vals < 0)),
        })
    (OUT / "dl_target_uncertainty_results_100plays.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Game-cluster uncertainty in DL rookie game-speed prediction", "",
        "For each replicate, each player's rookie p90 was independently drawn from a bootstrap distribution created by resampling that player's observed games. The baseline and early-acceleration candidate were refit in all three leave-one-draft-cohort-out folds. This propagates player-specific outcome-estimation uncertainty; independent draws do not preserve shared game, opponent, or weather shocks, adjust for the prior discovery search, or create a future-class validation.", "",
        f"Sample: {len(data)} DLs across cohorts {', '.join(map(str, cohorts))}; {args.draws:,} outcome-uncertainty replicates.", "",
        "| Ridge alpha | Median OOF RMSE change (yd/s) | 95% outcome-uncertainty interval | Replicates improved |",
        "|---:|---:|---:|---:|"
    ]
    for row in result["results"]:
        ci = row["target_uncertainty_95pct_interval"]
        lines.append(f"| {row['alpha']:g} | {row['median_rmse_change']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['fraction_replicates_candidate_improves']:.1%} |")
    lines += ["", "A negative change favors adding early acceleration. The draw distribution is conditional on each player's observed games, tracked role, and available data; draws are independent across players.", ""]
    (OUT / "dl_target_uncertainty_report_100plays.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
