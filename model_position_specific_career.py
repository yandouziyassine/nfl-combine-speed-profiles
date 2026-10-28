"""Robustness test: do 40-yard phases transfer differently by position?

Uses position-specific slopes for the conventional-testing baseline and adds
position-specific 40-yard phase slopes. Whole draft cohorts are held out. Every
imputation/scaling estimate is fit on training players only. No plays or frames
are treated as independent observations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
POSITIONS = ["DB", "DL", "OL", "TE", "WR"]
CONVENTIONAL = ["forty", "ten_yd_split", "combine_height", "combine_weight", "draft_pick_log"]
PHASES = ["forty_early_accel_0_1s_ydps2_mean", "forty_late_accel_10_20yd_ydps2_mean",
          "forty_speed_retention_20_40yd_ratio_mean"]
ALPHAS = [0.1, 1.0, 5.0, 20.0]


def predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, alpha: float) -> np.ndarray:
    med = train[features].median()
    a = train[features].fillna(med)
    b = test[features].fillna(med)
    mu = a.mean()
    sd = a.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(a)), ((a - mu) / sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(b)), ((b - mu) / sd).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return z @ np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(float))


def bootstrap_delta(oof: pd.DataFrame, draws: int = 2000, seed: int = 20271028) -> list[float]:
    groups = [g.index.to_numpy() for _, g in oof.groupby("draft_year")]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        ix = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        d = oof.loc[ix]
        y, p0, p1 = d.actual.to_numpy(), d.baseline.to_numpy(), d.candidate.to_numpy()
        vals.append(np.sqrt(np.mean((y-p1)**2)) - np.sqrt(np.mean((y-p0)**2)))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main() -> None:
    data = pd.read_csv(OUT / "career_speed_player_table.csv")
    data["draft_pick_log"] = np.log1p(data.draft_overall_pick)
    for pos in POSITIONS:
        data[f"position_{pos}"] = data.combine_position.eq(pos).astype(int)

    # Fully position-specific linear slopes, with position indicators as intercepts.
    # Interactions are constructed before fold scaling; all parameters remain
    # inside the ridge pipeline's training-fold standardization.
    for pos in POSITIONS:
        for f in CONVENTIONAL + PHASES:
            data[f"{f}_by_{pos}"] = data[f] * data[f"position_{pos}"]
    pos_intercepts = [f"position_{p}" for p in POSITIONS[:-1]]
    flexible_base = pos_intercepts + [f"{f}_by_{p}" for f in CONVENTIONAL for p in POSITIONS]
    phase_block = [f"{f}_by_{p}" for f in PHASES for p in POSITIONS]
    sample_base = ["nfl_id", "draft_year", "combine_position", *CONVENTIONAL, *PHASES]
    outcomes = {
        "annual_role_snaps": "career_role_snaps_log_per_season",
        "snaps_per_active_game": "career_role_snaps_log_per_active_game",
    }
    result = {
        "question": "Do position-specific 40-yard phase slopes add out-of-cohort information beyond position-specific conventional testing, body size, and draft capital?",
        "specification": {"positions": POSITIONS, "conventional_slopes": CONVENTIONAL,
                          "phase_slopes": PHASES, "alpha_grid": ALPHAS,
                          "validation": "leave one entire draft cohort out; train-fold median imputation and scaling; paired player bootstrap stratified by cohort, conditional on fixed out-of-fold predictions (no model refit)"},
        "outcomes": {},
    }
    prediction_rows = []
    for outcome_name, target in outcomes.items():
        required = sample_base + [target]
        sample = data.loc[data[required].notna().all(axis=1)].copy()
        outcome_result = {"target": target, "n_players": int(len(sample)),
                          "players_by_cohort": {str(int(k)): int(v) for k, v in sample.draft_year.value_counts().sort_index().items()},
                          "players_by_position": {str(k): int(v) for k, v in sample.combine_position.value_counts().sort_index().items()},
                          "alpha_results": []}
        for ai, alpha in enumerate(ALPHAS):
            oof_rows = []
            fold_rows = []
            for cohort in sorted(sample.draft_year.unique()):
                tr = sample.loc[sample.draft_year.ne(cohort)]
                te = sample.loc[sample.draft_year.eq(cohort)]
                p0 = predict(tr, te, flexible_base, target, alpha)
                p1 = predict(tr, te, flexible_base + phase_block, target, alpha)
                y = te[target].to_numpy(float)
                rm0 = float(np.sqrt(np.mean((y-p0)**2)))
                rm1 = float(np.sqrt(np.mean((y-p1)**2)))
                fold_rows.append({"held_out_draft_year": int(cohort), "n_train": int(len(tr)), "n_test": int(len(te)),
                                  "baseline_rmse": rm0, "candidate_rmse": rm1, "rmse_change": rm1-rm0})
                oof_rows.extend({"nfl_id": int(pid), "draft_year": int(cohort), "actual": float(v),
                                 "baseline": float(b), "candidate": float(c), "alpha": alpha,
                                 "outcome": outcome_name}
                                for pid, v, b, c in zip(te.nfl_id, y, p0, p1))
            oof = pd.DataFrame(oof_rows)
            delta = float(np.sqrt(np.mean((oof.actual-oof.candidate)**2)) - np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
            ci = bootstrap_delta(oof, seed=20271028 + ai + (100 if outcome_name != "annual_role_snaps" else 0))
            outcome_result["alpha_results"].append({"alpha": alpha, "pooled_oof_rmse_change": delta,
                                                     "player_bootstrap_95pct_ci": ci, "folds": fold_rows})
            prediction_rows.extend(oof_rows)
        result["outcomes"][outcome_name] = outcome_result

    pd.DataFrame(prediction_rows).to_csv(OUT / "position_specific_career_oof_predictions.csv", index=False)
    (OUT / "position_specific_career_results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Does the speed-phase link vary by position?", "",
             "This sensitivity model addresses an important weakness of a pooled baseline: football positions can use speed differently. The baseline gives each Combine position its own intercept and its own slopes for official 40, 10-yard split, height, weight, and log draft pick. The candidate adds position-specific slopes for early acceleration, late acceleration, and 20–40 yard speed retention. It uses the same complete-case players in both models, holds out full draft cohorts, and fits scaling/imputation on training cohorts only. Ridge strength is varied over a prespecified grid; no alpha is selected using the held-out cohort. Bootstrap intervals resample fixed out-of-fold player errors within cohort; models are not refit in bootstrap replicates.", "",
             "Positive RMSE change means the tracked phase block worsened held-out prediction. Cohort-level results matter because there are only three draft classes.", ""]
    for outcome_name, info in result["outcomes"].items():
        lines += [f"## {outcome_name.replace('_', ' ').title()}", "",
                  f"Sample: {info['n_players']} players; by position: {info['players_by_position']}.", "",
                  "| Ridge alpha | Pooled OOF RMSE Δ (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |", "|---:|---:|---:|---:|---:|"]
        for row in info["alpha_results"]:
            folds = {f["held_out_draft_year"]: f["rmse_change"] for f in row["folds"]}
            ci = row["player_bootstrap_95pct_ci"]
            cells = ["NA" if y not in folds else f"{folds[y]:+.4f}" for y in [2023, 2024, 2025]]
            lines.append(f"| {row['alpha']:g} | {row['pooled_oof_rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
    lines += ["This is a robustness analysis, not a confirmatory discovery: there are only three held-out cohorts, within-position samples are modest, and the phase block has 15 position-specific slopes. A stable improvement across ridge strengths and cohorts would justify a narrower follow-up; an unstable result would not support a position-specific scouting rule.", ""]
    (OUT / "position_specific_career_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: {"n": v["n_players"], "rmse_delta_by_alpha": [round(x["pooled_oof_rmse_change"], 5) for x in v["alpha_results"]]} for k,v in result["outcomes"].items()}, indent=2))


if __name__ == "__main__":
    main()
