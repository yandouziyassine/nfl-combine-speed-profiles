"""Test 40-yard speed phases against rookie DB tackle probability.

Player-play labels are aggregated to one binomial outcome per Combine DB. The
model is fit to tackle successes/opportunities with leave-one-draft-cohort-out
validation; test scoring and bootstrap uncertainty are player-weighted.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from project_config import DEFAULT_ARCHIVE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
POSITIONS = ["CB", "FS", "SS"]
BASE_NUMERIC = ["forty", "ten_yd_split", "combine_height", "combine_weight", "draft_pick_log"]
PHASES = ["forty_early_accel_0_1s_ydps2_mean", "forty_late_accel_10_20yd_ydps2_mean",
          "forty_speed_retention_20_40yd_ratio_mean"]
ALPHAS = [0.1, 1.0, 5.0]


def read_file(source: Path, name: str, usecols=None) -> pd.DataFrame:
    if source.is_dir():
        found = list(source.rglob(name))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one {name} under {source}; found {len(found)}")
        return pd.read_csv(found[0], usecols=usecols)
    with zipfile.ZipFile(source) as z:
        match = next((n for n in z.namelist() if n.endswith("/" + name)), None)
        if match is None:
            raise FileNotFoundError(name)
        return pd.read_csv(z.open(match), usecols=usecols)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float) -> np.ndarray:
    med = train[features].median()
    a, b = train[features].fillna(med), test[features].fillna(med)
    mu, sd = a.mean(), a.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(a)), ((a-mu)/sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(b)), ((b-mu)/sd).to_numpy(float)])
    y = train.tackles.to_numpy(float) / train.opportunities.to_numpy(float)
    n = train.opportunities.to_numpy(float)
    prior = np.clip(np.sum(train.tackles)/np.sum(n), 1e-5, 1-1e-5)
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(prior/(1-prior))
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    for _ in range(100):
        eta = x @ beta
        p = sigmoid(eta)
        w = np.maximum(n*p*(1-p), 1e-8)
        working = eta + (y-p)/np.maximum(p*(1-p), 1e-8)
        updated = np.linalg.solve(x.T @ (w[:, None]*x) + penalty, x.T @ (w*working))
        if np.max(np.abs(updated-beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    return sigmoid(z @ beta)


def player_boot_ci(delta: np.ndarray, cohorts: np.ndarray, draws: int = 3000, seed: int = 7721) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cohorts == c) for c in np.unique(cohorts)]
    vals = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals.append(float(delta[idx].mean()))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--min-opportunities", type=int, default=50)
    args = parser.parse_args()
    pp = read_file(args.archive, "player_play.csv", ["game_id","play_id","nfl_id","tackle","assist","lined_up_position"])
    players = read_file(args.archive, "players.csv", ["nfl_id","nfl_position"])
    games = read_file(args.archive, "games.csv", ["game_id","season","season_type"])
    sig_cols = ["nfl_id","draft_year","combine_position","forty","ten_yd_split","combine_height","combine_weight","draft_overall_pick",*PHASES]
    signatures = pd.read_csv(OUT / "player_speed_signatures.csv", usecols=lambda c: c in set(sig_cols))
    rows = pp.merge(players, on="nfl_id", how="left", validate="many_to_one").merge(games, on="game_id", how="left", validate="many_to_one").merge(signatures, on="nfl_id", how="inner", validate="many_to_one")
    rows = rows.loc[rows.combine_position.eq("DB") & rows.nfl_position.isin(POSITIONS) & rows.season.eq(rows.draft_year) & rows.season_type.eq("REG")].copy()
    rows["tackle_event"] = rows.tackle.fillna(False).astype(bool)
    rows["assist_event"] = rows.assist.fillna(False).astype(bool)
    player = (rows.groupby(["nfl_id","draft_year","nfl_position",*sig_cols[2:]], dropna=False)
              .agg(opportunities=("tackle_event","size"), tackles=("tackle_event","sum"), assists=("assist_event","sum"))
              .reset_index())
    player = player.loc[player.opportunities.ge(args.min_opportunities)].copy()
    player["draft_pick_log"] = np.log1p(player.draft_overall_pick)
    for pos in POSITIONS:
        player[f"position_{pos}"] = player.nfl_position.eq(pos).astype(int)
    player["smoothed_tackle_rate"] = (player.tackles + .5)/(player.opportunities + 1)
    tag = f"{args.min_opportunities}plus"
    player.to_csv(OUT / f"db_tackle_player_table_{tag}.csv", index=False)

    position_features = [f"position_{p}" for p in POSITIONS[:-1]]
    base = BASE_NUMERIC + position_features
    required = base + PHASES + ["smoothed_tackle_rate"]
    sample = player.loc[player[required].notna().all(axis=1)].copy()
    result = {"question": "Do 10 Hz 40-yard speed phases improve prediction of rookie DB tackle probability beyond official testing, size, draft capital, and DB subposition?",
              "outcome": "tackles per observed rookie regular-season DB player-play; one binomial row per player with play appearances as opportunities",
              "sample": {"min_opportunities_per_player": args.min_opportunities,"n_players": int(len(sample)),
                         "players_by_cohort": {str(int(k)):int(v) for k,v in sample.draft_year.value_counts().sort_index().items()},
                         "players_by_position": {str(k):int(v) for k,v in sample.nfl_position.value_counts().sort_index().items()},
                         "total_tackles": int(sample.tackles.sum()),"total_opportunities":int(sample.opportunities.sum())},
              "validation": "leave-one-draft-cohort-out; binomial ridge logistic regression; training-fold imputation/scaling; equal player-weighted test scoring; fixed-prediction bootstrap resamples players within cohort",
              "comparisons": []}
    predictions = []
    for alpha in ALPHAS:
        oof=[]; folds=[]
        for cohort in sorted(sample.draft_year.unique()):
            tr=sample.loc[sample.draft_year.ne(cohort)].copy(); te=sample.loc[sample.draft_year.eq(cohort)].copy()
            p0=fit_predict(tr,te,base,alpha); p1=fit_predict(tr,te,base+PHASES,alpha)
            obs=te.tackles.to_numpy(float)/te.opportunities.to_numpy(float)
            brier0=(obs-p0)**2; brier1=(obs-p1)**2
            eps=1e-6
            ll0=-(obs*np.log(np.clip(p0,eps,1-eps))+(1-obs)*np.log(np.clip(1-p0,eps,1-eps)))
            ll1=-(obs*np.log(np.clip(p1,eps,1-eps))+(1-obs)*np.log(np.clip(1-p1,eps,1-eps)))
            folds.append({"held_out_draft_year":int(cohort),"n_players":int(len(te)),
                          "baseline_player_weighted_brier":float(brier0.mean()),"phase_player_weighted_brier":float(brier1.mean()),
                          "brier_change":float((brier1-brier0).mean()),"log_loss_change":float((ll1-ll0).mean())})
            oof.extend({"nfl_id":int(pid),"draft_year":int(c),"actual_tackle_rate":float(y),"opportunities":int(n),
                        "pred_baseline":float(a),"pred_phase":float(b),"brier_delta":float(e),"logloss_delta":float(f),"ridge_alpha":alpha}
                       for pid,c,y,n,a,b,e,f in zip(te.nfl_id,te.draft_year,obs,te.opportunities,p0,p1,brier1-brier0,ll1-ll0))
        oof_df=pd.DataFrame(oof)
        ci=player_boot_ci(oof_df.brier_delta.to_numpy(),oof_df.draft_year.to_numpy(),seed=7721+int(alpha*10))
        result["comparisons"].append({"ridge_alpha":alpha,"player_weighted_brier_change":float(oof_df.brier_delta.mean()),
                                      "player_bootstrap_95pct_ci":ci,"player_weighted_log_loss_change":float(oof_df.logloss_delta.mean()),"folds":folds})
        predictions.extend(oof)
    prediction_path = OUT / f"db_tackle_oof_predictions_{tag}.csv"
    results_path = OUT / f"db_tackle_transfer_results_{tag}.json"
    report_path = OUT / f"db_tackle_transfer_report_{tag}.md"
    pd.DataFrame(predictions).to_csv(prediction_path,index=False)
    results_path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# Do Combine 40-yard phases transfer to DB tackling?","",
           "The outcome is each defensive back's rookie regular-season tackle count divided by the DB player-play rows supplied for that player. This measures tackle frequency in the observed player-play sample; it is affected by alignment, role, opportunity, and team scheme, so it is not a tackle-success probability conditional on a verified tackle opportunity. The baseline includes official 40, 10-yard split, height, weight, log draft pick, and CB/FS/SS role. The candidate adds early acceleration, late acceleration, and speed retention. Models are binomial ridge regressions and leave out one entire draft class. Test scores average players equally; bootstrap intervals resample fixed OOF player scores within class and do not refit models.","",
           f"Complete sample: {len(sample)} players with at least {args.min_opportunities} rookie DB player-play rows; {int(sample.tackles.sum())} tackle labels over {int(sample.opportunities.sum())} player-play rows.","",
           "| Ridge alpha | Player-weighted Brier Δ (95% CI) | Player-weighted log-loss Δ | 2023 Brier Δ | 2024 Brier Δ | 2025 Brier Δ |","|---:|---:|---:|---:|---:|---:|"]
    for item in result["comparisons"]:
        ci=item["player_bootstrap_95pct_ci"]; f={x["held_out_draft_year"]:x for x in item["folds"]}
        cells=["NA" if y not in f else f"{f[y]['brier_change']:+.4f}" for y in [2023,2024,2025]]
        lines.append(f"| {item['ridge_alpha']:g} | {item['player_weighted_brier_change']:+.5f} ({ci[0]:+.5f}, {ci[1]:+.5f}) | {item['player_weighted_log_loss_change']:+.5f} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["","Negative changes favor the 40-phase model. The sample is limited to prospects whose NFL position and rookie-season player-play outcomes are observed in this competition data. This is a candidate transfer screen; a positive result would still need a sensitivity analysis for role/opportunity and the choice of tackle versus tackle-plus-assist.",""]
    report_path.write_text("\n".join(lines),encoding="utf-8")
    if args.min_opportunities == 50:
        player.to_csv(OUT / "db_tackle_player_table.csv", index=False)
        pd.DataFrame(predictions).to_csv(OUT / "db_tackle_oof_predictions.csv",index=False)
        (OUT / "db_tackle_transfer_results.json").write_text(results_path.read_text(encoding="utf-8"),encoding="utf-8")
        (OUT / "db_tackle_transfer_report.md").write_text(report_path.read_text(encoding="utf-8"),encoding="utf-8")
    print(json.dumps({"sample":result["sample"],"results":[{"alpha":r["ridge_alpha"],"brier_delta":r["player_weighted_brier_change"],"ci":r["player_bootstrap_95pct_ci"],"logloss_delta":r["player_weighted_log_loss_change"]} for r in result["comparisons"]]},indent=2))


if __name__ == "__main__":
    main()
