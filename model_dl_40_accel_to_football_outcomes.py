"""Test whether 40-yard early acceleration transfers to DL blitz movement/outcomes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight"]
FEATURE = "forty_early_accel_0_1s_ydps2_mean"
TARGETS = {
    "post_snap_speed_gain": ("median_speed_gain_0_1s_ydps", "n_stationary_snap_bursts", 20),
    "post_snap_displacement": ("median_net_displacement_0_1s_yd", "n_stationary_snap_bursts", 20),
    "blitz_pressure_rate": ("rookie_pressure_rate", "rookie_blitz_opportunities", 50),
    "get_off_time": ("median_get_off_s", "get_off_events", 20),
}
ALPHAS = [5.0, 20.0]


def predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str,
            alpha: float) -> np.ndarray:
    med = train[features].median()
    a = train[features].fillna(med)
    b = test[features].fillna(med)
    mu = a.mean()
    sd = a.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(a)), ((a-mu)/sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(b)), ((b-mu)/sd).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return z @ np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(float))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y-p)**2)))


def delta_ci(oof: pd.DataFrame, seed: int, draws: int = 3000) -> list[float]:
    groups = [g.index.to_numpy() for _, g in oof.groupby("draft_year")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ix = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        sample = oof.loc[ix]
        values.append(rmse(sample.actual.to_numpy(), sample.candidate.to_numpy()) -
                      rmse(sample.actual.to_numpy(), sample.baseline.to_numpy()))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-burst-plays", type=int, default=50,
                    help="Minimum eligible stationary-snap blitz bursts; pressure/get-off use their own opportunity floors.")
    args = ap.parse_args()
    data = pd.read_csv(OUT / "dl_transfer_player_table.csv")
    data = data.loc[data.nfl_position.isin(["DE", "DT", "NT", "OLB"])].copy()
    results = {"question": "Does Combine 40-yard first-second acceleration add out-of-cohort information for rookie DL post-snap movement, blitz pressure rate, or get-off time?",
               "baseline": "official 40, 10-yard split, height, and weight only; all measured pre-draft",
               "candidate": FEATURE,
               "validation": "leave one full draft cohort out; within-player aggregation; training-fold imputation/scaling; fixed-prediction player bootstrap within cohort; no player or play leakage",
               "minimum_eligible_burst_plays": int(args.min_burst_plays), "outcomes": []}
    prediction_tables = []
    for target_i, (name, (target, exposure, floor)) in enumerate(TARGETS.items()):
        threshold = max(floor, args.min_burst_plays if exposure == "n_stationary_snap_bursts" else floor)
        need = ["nfl_id", "draft_year", target, exposure, *BASE, FEATURE]
        eligible = data.loc[data[exposure].fillna(0).ge(threshold)]
        sample = eligible.loc[eligible[need].notna().all(axis=1)].copy()
        if sample.draft_year.nunique() < 3 or len(sample) < 25:
            continue
        outcome_result = {"name": name, "target": target, "exposure_column": exposure,
                          "minimum_exposure": int(threshold), "n_players": int(len(sample)),
                          "by_cohort": {str(int(k)): int(v) for k,v in sample.draft_year.value_counts().sort_index().items()},
                          "comparisons": []}
        for alpha_i, alpha in enumerate(ALPHAS):
            rows = []; folds = []
            for cohort in sorted(sample.draft_year.unique()):
                train = sample.loc[sample.draft_year.ne(cohort)]
                test = sample.loc[sample.draft_year.eq(cohort)]
                p0 = predict(train, test, BASE, target, alpha)
                p1 = predict(train, test, BASE+[FEATURE], target, alpha)
                y = test[target].to_numpy(float)
                folds.append({"held_out_draft_year": int(cohort), "n_players": int(len(test)),
                              "baseline_rmse": rmse(y,p0), "candidate_rmse": rmse(y,p1),
                              "rmse_change": rmse(y,p1)-rmse(y,p0)})
                rows.extend({"nfl_id":int(pid),"draft_year":int(year),"actual":float(actual),
                             "baseline":float(b),"candidate":float(c),"alpha":alpha,"outcome":name}
                            for pid,year,actual,b,c in zip(test.nfl_id,test.draft_year,y,p0,p1))
            oof = pd.DataFrame(rows)
            change = rmse(oof.actual.to_numpy(),oof.candidate.to_numpy())-rmse(oof.actual.to_numpy(),oof.baseline.to_numpy())
            ci = delta_ci(oof, seed=7811+target_i*10+alpha_i)
            outcome_result["comparisons"].append({"alpha":alpha,"rmse_change":change,
                                                   "player_bootstrap_95pct_ci":ci,"folds":folds})
            prediction_tables.extend(rows)
        results["outcomes"].append(outcome_result)
    pd.DataFrame(prediction_tables).to_csv(OUT / f"dl_40_accel_football_oof_{args.min_burst_plays}plays.csv",index=False)
    (OUT / f"dl_40_accel_football_results_{args.min_burst_plays}plays.json").write_text(
        json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
    lines = ["# Does 40-yard early acceleration transfer to DL football outcomes?", "",
             "The baseline uses only pre-draft official 40 time, 10-yard split, height, and weight. The candidate adds Combine 0–1 second acceleration. Outcomes are player-level rookie-season aggregates: 1-second post-snap speed gain/displacement on stationary-snap blitz plays, observed blitz pressure rate, and get-off time. Complete draft cohorts are held out; scores weight players equally. Pressure and get-off remain sensitive to role, opportunity, and team context.", "",
             "A negative RMSE change favors adding early acceleration. Fixed-prediction player-bootstrap intervals do not refit models or account for the wider feature search.", "",
             "| Outcome | Players | Minimum exposure | Alpha | RMSE change (95% player-bootstrap interval) | 2023 | 2024 | 2025 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for outcome in results["outcomes"]:
        for comp in outcome["comparisons"]:
            ci=comp["player_bootstrap_95pct_ci"]; f={x["held_out_draft_year"]:x["rmse_change"] for x in comp["folds"]}
            lines.append(f"| {outcome['name']} | {outcome['n_players']} | {outcome['minimum_exposure']} | {comp['alpha']:g} | {comp['rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {f[2023]:+.4f} | {f[2024]:+.4f} | {f[2025]:+.4f} |")
    lines += ["", "This is a direct football-outcome follow-up, not a causal or opportunity-adjusted model. Burst outcomes require the stated number of eligible stationary-snap blitz plays; pressure and get-off use minimums of 50 blitz opportunities and 20 get-off events. Interpret these outcomes cautiously because assignment and exposure vary by team and season.", ""]
    (OUT / f"dl_40_accel_football_report_{args.min_burst_plays}plays.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({o["name"]:{"n":o["n_players"],"deltas":[round(x["rmse_change"],4) for x in o["comparisons"]]} for o in results["outcomes"]},indent=2))


if __name__ == "__main__":
    main()
