"""Context-adjusted, player-balanced DL blitz-pressure screen for 40 early accel."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"
CONTROLS=["forty","ten_yd_split","combine_height","combine_weight","down","yards_to_go"]
EARLY="forty_early_accel_0_1s_ydps2_mean"


def design(train:pd.DataFrame,test:pd.DataFrame,features:list[str]):
    med=train[features].median()
    a=train[features].fillna(med); b=test[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    ax=((a-mu)/sd).to_numpy(float); bx=((b-mu)/sd).to_numpy(float)
    ca=pd.get_dummies(train[["lined_up_position","distance_bin"]].fillna("UNK").astype(str),dtype=float)
    cb=pd.get_dummies(test[["lined_up_position","distance_bin"]].fillna("UNK").astype(str),dtype=float).reindex(columns=ca.columns,fill_value=0)
    x=np.column_stack([np.ones(len(train)),ax,ca.to_numpy(float)])
    z=np.column_stack([np.ones(len(test)),bx,cb.to_numpy(float)])
    return x,z


def fit_logistic(x:np.ndarray,y:np.ndarray,w:np.ndarray,alpha:float)->np.ndarray:
    beta=np.zeros(x.shape[1]); penalty=np.eye(x.shape[1]); penalty[0,0]=0
    for _ in range(100):
        eta=np.clip(x@beta,-20,20); p=1/(1+np.exp(-eta)); v=np.maximum(p*(1-p),1e-7)
        grad=x.T@(w*(y-p))-alpha*(penalty@beta)
        hess=x.T@((w*v)[:,None]*x)+alpha*penalty
        step=np.linalg.solve(hess,grad); beta+=step
        if np.max(np.abs(step))<1e-8: break
    return beta


def main():
    raw=pd.read_csv(OUT/"dl_rookie_blitz_opportunities.csv")
    sig=pd.read_csv(OUT/"player_speed_signatures.csv")
    data=raw.merge(sig[["nfl_id","combine_position","forty","ten_yd_split","combine_height","combine_weight",EARLY]],on="nfl_id",how="inner",validate="many_to_one")
    data=data.loc[data.combine_position.eq("DL")&data.draft_year.isin([2023,2024,2025])].copy()
    data["down"]=pd.to_numeric(data.down,errors="coerce")
    data["yards_to_go"]=pd.to_numeric(data.yards_to_go,errors="coerce")
    data["pressure_event"]=pd.to_numeric(data.pressure_event,errors="coerce")
    data["distance_bin"]=pd.cut(data.yards_to_go,[-np.inf,3,6,10,np.inf],labels=["1-3","4-6","7-10","11+"]).astype(str)
    counts=data.groupby("nfl_id").size()
    eligible=set(counts[counts.ge(50)].index.astype(int))
    need=["nfl_id","draft_year","pressure_event",*CONTROLS,EARLY,"lined_up_position","distance_bin"]
    data=data.loc[data.nfl_id.isin(eligible)&data[need].notna().all(axis=1)].copy()
    players=data[["nfl_id","draft_year"]].drop_duplicates()
    result={"question":"Does Combine 40-yard 0-1 second acceleration improve rookie blitz-pressure prediction after controlling for play down, distance, and DL lineup position?",
            "baseline":"pre-draft official 40, 10-yard split, height, and weight, plus play down, yards-to-go, binned distance, and lined-up position",
            "candidate":"baseline plus Combine 40-yard early acceleration",
            "sample":{"players":int(players.nfl_id.nunique()),"player_plays":int(len(data)),"by_cohort":{str(int(k)):int(v) for k,v in players.draft_year.value_counts().sort_index().items()},"minimum_blitz_opportunities":50},
            "training":"weighted ridge logistic likelihood; each player receives total training weight 1 regardless of number of blitz plays",
            "validation":"leave one entire draft cohort out; test Brier scores are computed per player then equally averaged; confidence intervals bootstrap fixed player-level OOF Brier differences within cohort",
            "comparisons":[]}
    out=[]
    for ai,alpha in enumerate([0.1,1.0,5.0]):
        player_scores=[]; folds=[]
        for cohort in sorted(data.draft_year.unique()):
            tr=data.loc[data.draft_year.ne(cohort)].copy(); te=data.loc[data.draft_year.eq(cohort)].copy()
            # Continuous design matrices receive identical train-fitted scaling and one-hot context columns.
            xb,zb=design(tr,te,CONTROLS); xc,zc=design(tr,te,CONTROLS+[EARLY])
            ytr=tr.pressure_event.to_numpy(float); w=1/tr.groupby("nfl_id").nfl_id.transform("size").to_numpy(float)
            b0=fit_logistic(xb,ytr,w,alpha); b1=fit_logistic(xc,ytr,w,alpha)
            p0=np.clip(1/(1+np.exp(-np.clip(zb@b0,-20,20))),1e-6,1-1e-6)
            p1=np.clip(1/(1+np.exp(-np.clip(zc@b1,-20,20))),1e-6,1-1e-6)
            te["brier_base"]=(te.pressure_event.to_numpy(float)-p0)**2
            te["brier_candidate"]=(te.pressure_event.to_numpy(float)-p1)**2
            sc=te.groupby(["nfl_id","draft_year"],as_index=False).agg(baseline_brier=("brier_base","mean"),candidate_brier=("brier_candidate","mean"))
            sc["delta_brier"]=sc.candidate_brier-sc.baseline_brier; sc["alpha"]=alpha
            player_scores.extend(sc.to_dict("records"))
            folds.append({"held_out_draft_year":int(cohort),"n_players":int(len(sc)),"n_player_plays":int(len(te)),
                          "baseline_brier":float(sc.baseline_brier.mean()),"candidate_brier":float(sc.candidate_brier.mean()),
                          "brier_change":float(sc.delta_brier.mean())})
        scores=pd.DataFrame([r for r in player_scores if r["alpha"]==alpha])
        groups=[g.index.to_numpy() for _,g in scores.groupby("draft_year")]; rng=np.random.default_rng(719+ai); boot=[]
        for _ in range(3000):
            ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); boot.append(float(scores.loc[ix].delta_brier.mean()))
        delta=float(scores.delta_brier.mean()); ci=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
        result["comparisons"].append({"alpha":alpha,"player_balanced_brier_change":delta,"player_bootstrap_95pct_ci":ci,"folds":folds})
    pd.DataFrame(player_scores).to_csv(OUT/"dl_pressure_context_oof_player_scores.csv",index=False)
    (OUT/"dl_pressure_context_results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# Context-adjusted DL blitz-pressure model","",
           "This player-play sensitivity uses only rookie regular-season DL blitz rows with at least 50 opportunities per player. The baseline includes pre-draft Combine test/size features and play context (down, yards-to-go, distance bin, and lined-up position). A weighted ridge-logistic model gives each player total training weight 1. Held-out Brier scores are averaged within player and then across players, so high-volume players do not dominate evaluation. Entire draft cohorts are held out.","",
           "Negative Brier change favors the early-acceleration candidate. Player-bootstrap intervals resample fixed OOF player-level loss differences; they do not refit models or adjust for broader screening.","",
           "| Alpha | Players | Player-play rows | Player-balanced Brier change (95% CI) | 2023 | 2024 | 2025 |","|---:|---:|---:|---:|---:|---:|---:|"]
    for comp in result["comparisons"]:
        ci=comp["player_bootstrap_95pct_ci"]; f={x["held_out_draft_year"]:x for x in comp["folds"]}
        lines.append(f"| {comp['alpha']:g} | {result['sample']['players']} | {result['sample']['player_plays']:,} | {comp['player_balanced_brier_change']:+.5f} ({ci[0]:+.5f}, {ci[1]:+.5f}) | {f[2023]['brier_change']:+.5f} | {f[2024]['brier_change']:+.5f} | {f[2025]['brier_change']:+.5f} |")
    lines += ["","This does not isolate individual pass-rush assignment or opponent quality. Blitz pressure is an opportunity-dependent outcome and the cohort is small.",""]
    (OUT/"dl_pressure_context_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"sample":result["sample"],"comparisons":[{"alpha":x["alpha"],"delta":round(x["player_balanced_brier_change"],5),"ci":x["player_bootstrap_95pct_ci"]} for x in result["comparisons"]]},indent=2))


if __name__=="__main__": main()
