"""Nested cohort-held-out check for selecting among 40-yard speed phases.

This is an internal selection-bias sensitivity, not independent validation.
The outer test cohort never participates in choosing the phase feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TARGET = "rookie_reg_p90_play_peak_speed_ydps"
BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight"]
PHASES = {
    "early_0_1s": "forty_early_accel_0_1s_ydps2_mean",
    "late_10_20yd": "forty_late_accel_10_20yd_ydps2_mean",
    "retention_20_40yd": "forty_speed_retention_20_40yd_ratio_mean",
}


def predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float) -> np.ndarray:
    med = train[features].median()
    a = train[features].fillna(med)
    b = test[features].fillna(med)
    mu = a.mean()
    sd = a.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(a)), ((a - mu) / sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(b)), ((b - mu) / sd).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return z @ np.linalg.solve(x.T @ x + penalty, x.T @ train[TARGET].to_numpy(float))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def paired_player_bootstrap(rows: pd.DataFrame, seed: int = 20271024, draws: int = 5000) -> list[float]:
    groups = [g.index.to_numpy() for _, g in rows.groupby("outer_year")]
    rng = np.random.default_rng(seed)
    delta = np.empty(draws)
    for b in range(draws):
        ix = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        sample = rows.loc[ix]
        delta[b] = rmse(sample.actual.to_numpy(), sample.selected.to_numpy()) - rmse(
            sample.actual.to_numpy(), sample.baseline.to_numpy()
        )
    return [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-game-plays", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    data = pd.read_csv(OUT / "player_speed_signatures.csv")
    sample = data.loc[
        data.combine_position.eq("DL")
        & data.rookie_reg_player_plays_with_matching_metadata.ge(args.min_game_plays)
    ].copy()
    features = list(PHASES.values())
    required = ["nfl_id", "draft_year", TARGET, *BASE, *features]
    sample = sample.dropna(subset=required).copy()
    cohorts = sorted(sample.draft_year.astype(int).unique())
    if len(cohorts) != 3:
        raise ValueError(f"Expected three cohorts, found {cohorts}")

    outer_rows = []
    fold_summaries = []
    for outer_year in cohorts:
        train_outer = sample.loc[sample.draft_year.ne(outer_year)].copy()
        test_outer = sample.loc[sample.draft_year.eq(outer_year)].copy()
        inner_scores: dict[str, list[float]] = {name: [] for name in PHASES}
        inner_details = []
        for inner_year in sorted(train_outer.draft_year.astype(int).unique()):
            train_inner = train_outer.loc[train_outer.draft_year.ne(inner_year)]
            valid_inner = train_outer.loc[train_outer.draft_year.eq(inner_year)]
            y_inner = valid_inner[TARGET].to_numpy(float)
            baseline_rmse = rmse(y_inner, predict(train_inner, valid_inner, BASE, args.alpha))
            scores = {}
            for name, feature in PHASES.items():
                candidate = rmse(y_inner, predict(train_inner, valid_inner, BASE + [feature], args.alpha))
                delta = candidate - baseline_rmse
                inner_scores[name].append(delta)
                scores[name] = delta
            inner_details.append({"inner_held_out_year": int(inner_year), "baseline_rmse": baseline_rmse,
                                  "phase_rmse_change": scores})
        mean_inner_delta = {name: float(np.mean(vals)) for name, vals in inner_scores.items()}
        chosen_name = min(mean_inner_delta, key=mean_inner_delta.get)
        chosen_feature = PHASES[chosen_name]
        y = test_outer[TARGET].to_numpy(float)
        p0 = predict(train_outer, test_outer, BASE, args.alpha)
        p_selected = predict(train_outer, test_outer, BASE + [chosen_feature], args.alpha)
        p_early = predict(train_outer, test_outer, BASE + [PHASES["early_0_1s"]], args.alpha)
        for pid, year, actual, baseline, selected, early in zip(
            test_outer.nfl_id, test_outer.draft_year, y, p0, p_selected, p_early
        ):
            outer_rows.append({"nfl_id": int(pid), "outer_year": int(year), "actual": actual,
                               "baseline": baseline, "selected": selected, "selected_phase": chosen_name,
                               "early_locked": early})
        fold_summaries.append({
            "outer_held_out_year": int(outer_year),
            "n_test_players": int(len(test_outer)),
            "inner_mean_rmse_change": mean_inner_delta,
            "selected_phase": chosen_name,
            "inner_folds": inner_details,
            "baseline_rmse": rmse(y, p0),
            "nested_selected_rmse": rmse(y, p_selected),
            "nested_selected_rmse_change": rmse(y, p_selected) - rmse(y, p0),
            "locked_early_rmse": rmse(y, p_early),
            "locked_early_rmse_change": rmse(y, p_early) - rmse(y, p0),
        })

    oof = pd.DataFrame(outer_rows)
    selected_delta = rmse(oof.actual.to_numpy(), oof.selected.to_numpy()) - rmse(
        oof.actual.to_numpy(), oof.baseline.to_numpy()
    )
    early_delta = rmse(oof.actual.to_numpy(), oof.early_locked.to_numpy()) - rmse(
        oof.actual.to_numpy(), oof.baseline.to_numpy()
    )
    result = {
        "design": "outer leave-one-draft-cohort-out; within each outer training set, choose the phase with lowest mean inner leave-one-cohort-out RMSE change; refit on outer training cohorts",
        "interpretation": "internal selection-aware sensitivity only; not independent validation, and the phase family is limited to the three previously examined 40-yard phases",
        "minimum_rookie_plays": args.min_game_plays,
        "ridge_alpha_fixed": args.alpha,
        "n_complete_case_players": int(len(sample)),
        "players_by_cohort": {str(int(k)): int(v) for k, v in sample.draft_year.value_counts().sort_index().items()},
        "features": {"baseline": BASE, "candidate_phase_features": PHASES},
        "nested_selected_pooled_oof_rmse_change": selected_delta,
        "nested_selected_player_bootstrap_95pct_interval": paired_player_bootstrap(oof),
        "locked_early_pooled_oof_rmse_change_same_players": early_delta,
        "locked_early_player_bootstrap_95pct_interval_same_players": paired_player_bootstrap(oof.assign(selected=oof.early_locked), seed=20271025),
        "phase_selection_counts": oof.groupby("selected_phase").outer_year.nunique().astype(int).to_dict(),
        "folds": fold_summaries,
    }
    tag = f"{args.min_game_plays}plays"
    oof.to_csv(OUT / f"dl_nested_phase_selection_oof_{tag}.csv", index=False)
    (OUT / f"dl_nested_phase_selection_{tag}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Nested selection among 40-yard speed phases",
        "",
        result["design"].capitalize() + ".",
        "",
        f"The complete-case sample contains {len(sample)} defensive linemen with at least {args.min_game_plays} matched rookie tracked plays: "
        + ", ".join(f"{year}: {n}" for year, n in result["players_by_cohort"].items()) + ".",
        "",
        f"Nested selection RMSE change: {selected_delta:+.4f} yd/s (paired fixed-OOF player-bootstrap 95% interval "
        f"{result['nested_selected_player_bootstrap_95pct_interval'][0]:+.4f}, {result['nested_selected_player_bootstrap_95pct_interval'][1]:+.4f}).",
        f"Locked early-phase RMSE change on the same players: {early_delta:+.4f} yd/s (95% interval "
        f"{result['locked_early_player_bootstrap_95pct_interval_same_players'][0]:+.4f}, {result['locked_early_player_bootstrap_95pct_interval_same_players'][1]:+.4f}).",
        "",
        "| Outer held-out cohort | n | Inner-selected phase | Nested selected Δ RMSE | Locked early Δ RMSE |",
        "|---:|---:|---|---:|---:|",
    ]
    for fold in fold_summaries:
        lines.append(
            f"| {fold['outer_held_out_year']} | {fold['n_test_players']} | {fold['selected_phase']} | "
            f"{fold['nested_selected_rmse_change']:+.4f} | {fold['locked_early_rmse_change']:+.4f} |"
        )
    lines += [
        "",
        "This sensitivity tests whether the early phase remains useful when the choice among these three already-screened phases is made using training cohorts only. It does not correct the broader search across positions, drills, outcomes, and thresholds; only three cohorts are available, so no outer fold is a genuinely independent future replication. Bootstrap intervals condition on the fixed OOF predictions and do not refit or repeat feature selection. Interpret this as an optimism check, not a scouting model or proof of transfer.",
        "",
    ]
    (OUT / f"dl_nested_phase_selection_report_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(sample), "players_by_cohort": result["players_by_cohort"],
                      "nested_selected_delta": selected_delta, "locked_early_delta": early_delta,
                      "phase_by_outer_cohort": {str(x["outer_held_out_year"]): x["selected_phase"] for x in fold_summaries}}, indent=2))


if __name__ == "__main__":
    main()
