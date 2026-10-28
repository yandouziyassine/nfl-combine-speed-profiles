"""Exploratory audit of edge/interior differences in DL movement associations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight", "interior_line"]
FEATURES = ["hoop_turn_weighted_speed", "pass_rush_sustained_speed"]
OUTCOMES = {
    "blitz_pressure_rate": ("rookie_pressure_rate", "rookie_blitz_opportunities", 50),
    "game_early_speed_gain": ("median_speed_gain_0_1s_ydps", "n_stationary_snap_bursts", 20),
}


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, alpha: float) -> np.ndarray:
    med = train[features].median()
    tr = train[features].fillna(med)
    te = test[features].fillna(med)
    # Explicit missingness flags retain availability information without using
    # any test-fold statistics.
    for f in features:
        if train[f].isna().any():
            tr[f"{f}__missing"] = train[f].isna().astype(float)
            te[f"{f}__missing"] = test[f].isna().astype(float)
    mu, sd = tr.mean(), tr.std().replace(0, 1).fillna(1)
    x = ((tr - mu) / sd).to_numpy(float)
    z = ((te - mu) / sd).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(float))
    return z @ beta


def bootstrap_r(x: np.ndarray, y: np.ndarray, seed: int, n_boot: int = 3000) -> tuple[float, list[float]]:
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) and np.std(y) else float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ids = rng.integers(0, len(x), size=len(x))
        if np.std(x[ids]) and np.std(y[ids]):
            vals.append(float(np.corrcoef(x[ids], y[ids])[0, 1]))
    return r, [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if vals else [float("nan"), float("nan")]


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    p = pd.read_csv(out / "dl_transfer_player_table.csv")
    p = p.loc[p.nfl_position.isin(["DE", "DT", "NT", "OLB"])].copy()
    p["interior_line"] = p.nfl_position.isin(["DT", "NT"]).astype(int)
    alpha = 3.0
    results = {"status": "exploratory position-heterogeneity audit after an all-DL candidate screen; no confirmatory status",
               "role_definition": {"edge": ["DE", "OLB"], "interior": ["DT", "NT"]},
               "comparisons": []}
    for outcome_name, (target, exposure, minimum) in OUTCOMES.items():
        eligible = p.loc[p[exposure].fillna(0).ge(minimum)].copy()
        for feature in FEATURES:
            sample = eligible.loc[eligible[[target, feature]].notna().all(axis=1)].copy()
            role_stats = {}
            for role, group in sample.groupby("interior_line"):
                x, y = group[feature].to_numpy(float), group[target].to_numpy(float)
                r, ci = bootstrap_r(x, y, seed=701 + len(group) + int(role))
                role_stats["interior" if role else "edge"] = {"n_players": int(len(group)), "pearson_r": r, "player_bootstrap_95pct_ci": ci}

            models = {"main_effect": {"pooled_oof_y": [], "base": [], "plus": []},
                      "role_interaction": {"pooled_oof_y": [], "base": [], "plus": []}}
            fold_rows = []
            for cohort in sorted(sample.draft_year.dropna().unique()):
                tr = sample.loc[sample.draft_year.ne(cohort)].copy()
                te = sample.loc[sample.draft_year.eq(cohort)].copy()
                fold = {"held_out_draft_year": int(cohort), "n_train_players": int(len(tr)), "n_test_players": int(len(te))}
                if len(tr) < 20 or len(te) < 8:
                    fold["status"] = "too_few_players"
                    fold_rows.append(fold)
                    continue
                interaction = f"{feature}_x_interior"
                tr[interaction] = tr[feature] * tr.interior_line
                te[interaction] = te[feature] * te.interior_line
                pred_base = ridge_predict(tr, te, BASE, target, alpha)
                pred_main = ridge_predict(tr, te, BASE + [feature], target, alpha)
                pred_interaction = ridge_predict(tr, te, BASE + [feature, interaction], target, alpha)
                actual = te[target].to_numpy(float)
                fold["base_rmse"] = float(np.sqrt(np.mean((actual - pred_base) ** 2)))
                fold["main_effect_rmse"] = float(np.sqrt(np.mean((actual - pred_main) ** 2)))
                fold["role_interaction_rmse"] = float(np.sqrt(np.mean((actual - pred_interaction) ** 2)))
                fold["main_effect_rmse_change"] = fold["main_effect_rmse"] - fold["base_rmse"]
                fold["role_interaction_rmse_change"] = fold["role_interaction_rmse"] - fold["base_rmse"]
                fold["status"] = "scored"
                fold_rows.append(fold)
                models["main_effect"]["pooled_oof_y"].extend(actual.tolist())
                models["main_effect"]["base"].extend(pred_base.tolist())
                models["main_effect"]["plus"].extend(pred_main.tolist())
                models["role_interaction"]["pooled_oof_y"].extend(actual.tolist())
                models["role_interaction"]["base"].extend(pred_base.tolist())
                models["role_interaction"]["plus"].extend(pred_interaction.tolist())

            pooled = {}
            for model_name, v in models.items():
                if v["pooled_oof_y"]:
                    yy = np.asarray(v["pooled_oof_y"])
                    base_rmse = float(np.sqrt(np.mean((yy - np.asarray(v["base"])) ** 2)))
                    plus_rmse = float(np.sqrt(np.mean((yy - np.asarray(v["plus"])) ** 2)))
                    pooled[model_name] = {"baseline_rmse": base_rmse, "candidate_rmse": plus_rmse, "rmse_change": plus_rmse - base_rmse}
            results["comparisons"].append({"outcome": outcome_name, "target": target, "candidate": feature,
                                           "n_players": int(len(sample)), "players_by_cohort": {str(int(k)): int(v) for k, v in sample.draft_year.value_counts().sort_index().items()},
                                           "role_correlations": role_stats, "pooled_oof": pooled, "folds": fold_rows})

    (out / "dl_position_heterogeneity_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# DL role-specific association audit", "",
             "This is an exploratory heterogeneity audit motivated by the football difference between edge and interior rushers. It follows the broader DL candidate screen, so the results are discovery evidence only. Each row is a player-level summary; the role-specific correlations are unadjusted within-role descriptions. The cohort-held-out models use the same 40/split/size/role baseline and compare either a common candidate slope or a candidate-by-interior interaction.", "",
             "| Outcome | Combine feature | Players | Edge r (n) | Interior r (n) | Main-effect OOF RMSE Δ | Interaction OOF RMSE Δ |", "|---|---|---:|---:|---:|---:|---:|"]
    for x in results["comparisons"]:
        edge = x["role_correlations"].get("edge", {})
        interior = x["role_correlations"].get("interior", {})
        main = x["pooled_oof"].get("main_effect", {}).get("rmse_change")
        inter = x["pooled_oof"].get("role_interaction", {}).get("rmse_change")
        fm = "NA" if main is None else f"{main:+.4f}"
        fi = "NA" if inter is None else f"{inter:+.4f}"
        er = "NA" if "pearson_r" not in edge else f"{edge['pearson_r']:+.2f} (n={edge['n_players']})"
        ir = "NA" if "pearson_r" not in interior else f"{interior['pearson_r']:+.2f} (n={interior['n_players']})"
        lines.append(f"| {x['outcome']} | {x['candidate']} | {x['n_players']} | {er} | {ir} | {fm} | {fi} |")
    lines += ["", "A negative RMSE change is better. These comparisons are small and were selected after a broader screen; even an apparent role interaction needs a separately locked validation before it can support a final claim.", ""]
    (out / "dl_position_heterogeneity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparisons": len(results["comparisons"]), "report": str(out / "dl_position_heterogeneity_report.md")}, indent=2))


if __name__ == "__main__":
    main()
