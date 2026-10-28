"""Nested cohort-held-out selection across 40-yard phases and position groups.

The fixed family is 3 movement phases x 4 adequately sized Combine groups.
This does not correct discovery across other drills or NFL outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TARGET = "rookie_reg_p90_play_peak_speed_ydps"
POSITIONS = ["DB", "DL", "OL", "WR"]
BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight"]
PHASES = {
    "early_0_1s": "forty_early_accel_0_1s_ydps2_mean",
    "late_10_20yd": "forty_late_accel_10_20yd_ydps2_mean",
    "retention_20_40yd": "forty_speed_retention_20_40yd_ratio_mean",
}
ALPHA = 5.0


def make_design(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Position-specific conventional baseline and one-hot phase candidates."""
    d = data.copy()
    baseline: list[str] = []
    candidates: dict[str, str] = {}
    for pos in POSITIONS:
        indicator = d.combine_position.eq(pos).astype(float)
        if pos != POSITIONS[-1]:
            name = f"position_{pos}"
            d[name] = indicator
            baseline.append(name)
        for feature in BASE:
            name = f"{feature}_by_{pos}"
            d[name] = d[feature] * indicator
            baseline.append(name)
        for phase_name, feature in PHASES.items():
            name = f"{phase_name}_by_{pos}"
            d[name] = d[feature] * indicator
            candidates[f"{pos}:{phase_name}"] = name
    return d, {"baseline": baseline, **candidates}


def predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float = ALPHA) -> np.ndarray:
    med = train[features].median()
    a = train[features].fillna(med)
    b = test[features].fillna(med)
    mu = a.mean()
    sd = a.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(a)), ((a - mu) / sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(b)), ((b - mu) / sd).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ train[TARGET].to_numpy(float))
    return z @ coef


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def paired_bootstrap(oof: pd.DataFrame, prediction: str = "selected", draws: int = 3000,
                     seed: int = 20271028) -> list[float]:
    groups = [g.index.to_numpy() for _, g in oof.groupby("draft_year")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        d = oof.loc[ix]
        values.append(rmse(d.actual.to_numpy(), d[prediction].to_numpy()) -
                      rmse(d.actual.to_numpy(), d.baseline.to_numpy()))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def main() -> None:
    raw = pd.read_csv(OUT / "player_speed_signatures.csv")
    raw["draft_pick_log"] = np.log1p(pd.to_numeric(raw.draft_overall_pick, errors="coerce"))
    phase_cols = list(PHASES.values())
    required = ["nfl_id", "draft_year", "combine_position", TARGET,
                "rookie_reg_player_plays_with_matching_metadata", *BASE, *phase_cols]
    raw = raw.loc[raw.combine_position.isin(POSITIONS)
                  & raw.rookie_reg_player_plays_with_matching_metadata.ge(100)
                  & raw[required].notna().all(axis=1)].copy()
    data, feature_map = make_design(raw)
    baseline = feature_map["baseline"]
    candidate_keys = [f"{pos}:{phase}" for pos in POSITIONS for phase in PHASES]
    years = sorted(data.draft_year.astype(int).unique())
    if len(years) != 3:
        raise ValueError(f"Expected three draft cohorts; found {years}")

    outer_rows, fold_results = [], []
    for outer_year in years:
        train_outer = data.loc[data.draft_year.ne(outer_year)].copy()
        test_outer = data.loc[data.draft_year.eq(outer_year)].copy()
        inner_scores = {key: [] for key in candidate_keys}
        for inner_year in sorted(train_outer.draft_year.astype(int).unique()):
            tr = train_outer.loc[train_outer.draft_year.ne(inner_year)]
            va = train_outer.loc[train_outer.draft_year.eq(inner_year)]
            p0 = predict(tr, va, baseline)
            y = va[TARGET].to_numpy(float)
            for key in candidate_keys:
                p1 = predict(tr, va, baseline + [feature_map[key]])
                inner_scores[key].extend((y - p1).tolist())
        # Select by pooled inner-fold RMSE; candidate family, alpha, cohort rules,
        # target and baseline were fixed before reading the outer test cohort.
        score_by_candidate = {key: float(np.sqrt(np.mean(np.square(inner_scores[key]))))
                              for key in candidate_keys}
        selected_key = min(score_by_candidate, key=score_by_candidate.get)
        p0 = predict(train_outer, test_outer, baseline)
        p_selected = predict(train_outer, test_outer, baseline + [feature_map[selected_key]])
        locked_key = "DL:early_0_1s"
        p_locked = predict(train_outer, test_outer, baseline + [feature_map[locked_key]])
        for pid, year, actual, base_pred, selected_pred, locked_pred in zip(
                test_outer.nfl_id, test_outer.draft_year, test_outer[TARGET], p0, p_selected, p_locked):
            outer_rows.append({"nfl_id": int(pid), "draft_year": int(year), "actual": float(actual),
                               "baseline": float(base_pred), "selected": float(selected_pred),
                               "selected_candidate": selected_key, "locked_dl_early": float(locked_pred)})
        fold_results.append({
            "outer_held_out_year": int(outer_year), "n_test_players": int(len(test_outer)),
            "selected_candidate": selected_key,
            "inner_rmse_by_candidate": {key: float(value) for key, value in score_by_candidate.items()},
            "baseline_rmse": rmse(test_outer[TARGET].to_numpy(float), p0),
            "nested_selected_rmse": rmse(test_outer[TARGET].to_numpy(float), p_selected),
            "nested_rmse_change": rmse(test_outer[TARGET].to_numpy(float), p_selected) - rmse(test_outer[TARGET].to_numpy(float), p0),
            "locked_dl_early_rmse": rmse(test_outer[TARGET].to_numpy(float), p_locked),
            "locked_dl_early_rmse_change": rmse(test_outer[TARGET].to_numpy(float), p_locked) - rmse(test_outer[TARGET].to_numpy(float), p0),
        })

    oof = pd.DataFrame(outer_rows)
    nested_delta = rmse(oof.actual.to_numpy(), oof.selected.to_numpy()) - rmse(oof.actual.to_numpy(), oof.baseline.to_numpy())
    locked_delta = rmse(oof.actual.to_numpy(), oof.locked_dl_early.to_numpy()) - rmse(oof.actual.to_numpy(), oof.baseline.to_numpy())
    result = {
        "question": "Does selecting one position-specific 40-yard phase inside the training cohorts improve held-out rookie game-speed prediction?",
        "target": TARGET,
        "minimum_tracked_player_plays": 100,
        "positions": POSITIONS,
        "candidate_family": candidate_keys,
        "baseline": "position-specific official 40, 10-yard split, height, and weight; no draft capital or NFL subposition",
        "selection": "Outer leave-one-draft-cohort-out. In each outer training set, choose the position-phase candidate with lowest pooled inner leave-one-cohort-out RMSE; refit on the outer training cohorts, then score the outer cohort.",
        "fixed_ridge_alpha": ALPHA,
        "sample": {"n_players": int(len(data)),
                   "by_cohort": {str(int(k)): int(v) for k, v in data.draft_year.value_counts().sort_index().items()},
                   "by_position": {str(k): int(v) for k, v in data.combine_position.value_counts().sort_index().items()}},
        "nested_selected_pooled_oof_rmse_change": nested_delta,
        "nested_selected_fixed_oof_player_bootstrap_95pct_interval": paired_bootstrap(oof),
        "locked_dl_early_pooled_oof_rmse_change_same_players": locked_delta,
        "locked_dl_early_fixed_oof_player_bootstrap_95pct_interval_same_players": paired_bootstrap(
            oof, prediction="locked_dl_early", seed=20271029),
        "phase_selection_counts": oof.groupby("selected_candidate").draft_year.nunique().astype(int).to_dict(),
        "folds": fold_results,
        "limitation": "This nested audit corrects selection only within the fixed 12-candidate 40-yard phase-by-position family and the common complete-case sample. It does not nest the broader search across drills, outcomes, exposure cutoffs, or all processing choices, and three cohorts cannot provide independent future validation. Bootstrap intervals condition on fixed outer predictions and do not repeat selection.",
    }
    oof.to_csv(OUT / "position_phase_nested_selection_oof_100plays.csv", index=False)
    (OUT / "position_phase_nested_selection_100plays.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Nested selection across 40-yard phase and position", "",
             result["selection"], "",
             f"Common complete-case sample: {len(data)} players; position counts {result['sample']['by_position']}; cohort counts {result['sample']['by_cohort']}.", "",
             f"Nested 12-candidate procedure: pooled OOF RMSE change {nested_delta:+.4f} yd/s (fixed-OOF player-bootstrap interval {result['nested_selected_fixed_oof_player_bootstrap_95pct_interval'][0]:+.4f} to {result['nested_selected_fixed_oof_player_bootstrap_95pct_interval'][1]:+.4f}).",
             f"Locked DL early-acceleration term on the same players: {locked_delta:+.4f} yd/s (fixed-OOF player-bootstrap interval {result['locked_dl_early_fixed_oof_player_bootstrap_95pct_interval_same_players'][0]:+.4f} to {result['locked_dl_early_fixed_oof_player_bootstrap_95pct_interval_same_players'][1]:+.4f}).", "",
             "| Outer held-out cohort | n | Selected position:phase | Nested RMSE change | Locked DL early change |", "|---:|---:|---|---:|---:|"]
    for fold in fold_results:
        lines.append(f"| {fold['outer_held_out_year']} | {fold['n_test_players']} | {fold['selected_candidate']} | {fold['nested_rmse_change']:+.4f} | {fold['locked_dl_early_rmse_change']:+.4f} |")
    lines += ["", result["limitation"], ""]
    (OUT / "position_phase_nested_selection_report_100plays.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"sample": result["sample"], "nested_delta": nested_delta, "locked_dl_early_delta": locked_delta,
                      "selected_by_outer_year": {str(x["outer_held_out_year"]): x["selected_candidate"] for x in fold_results}}, indent=2))


if __name__ == "__main__":
    main()
