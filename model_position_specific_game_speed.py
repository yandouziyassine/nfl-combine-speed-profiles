"""Test whether 40-yard speed phases transfer to rookie game speed by position."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"
POSITIONS=["DB","DL","OL","TE","WR"]
CONVENTIONAL=["forty","ten_yd_split","combine_height","combine_weight","draft_pick_log"]
PHASES=["forty_early_accel_0_1s_ydps2_mean","forty_late_accel_10_20yd_ydps2_mean","forty_speed_retention_20_40yd_ratio_mean"]
ALPHAS=[0.1,1.0,5.0,20.0]
TARGET="rookie_reg_p90_play_peak_speed_ydps"


def predict(train,test,features,target,alpha):
    med=train[features].median(); a=train[features].fillna(med); b=test[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    x=np.column_stack([np.ones(len(a)),((a-mu)/sd).to_numpy(float)])
    z=np.column_stack([np.ones(len(b)),((b-mu)/sd).to_numpy(float)])
    penalty=np.eye(x.shape[1])*alpha; penalty[0,0]=0
    beta=np.linalg.solve(x.T@x+penalty,x.T@train[target].to_numpy(float))
    return z@beta


def bootstrap_delta(oof,draws=3000,seed=1311):
    groups=[g.index.to_numpy() for _,g in oof.groupby('draft_year')]; rng=np.random.default_rng(seed); vals=[]
    for _ in range(draws):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); d=oof.loc[ix]
        vals.append(float(np.sqrt(np.mean((d.actual-d.candidate)**2))-np.sqrt(np.mean((d.actual-d.baseline)**2))))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-game-plays',type=int,default=25)
    args=ap.parse_args(); tag=f'{args.min_game_plays}plays'
    data=pd.read_csv(OUT/'player_speed_signatures.csv')
    derived={'draft_pick_log':np.log1p(data.draft_overall_pick)}
    for p in POSITIONS: derived[f'position_{p}']=data.combine_position.eq(p).astype(int)
    for p in POSITIONS:
        for f in CONVENTIONAL+PHASES:
            source=derived[f] if f in derived else data[f]
            derived[f'{f}_by_{p}']=source*derived[f'position_{p}']
    data=pd.concat([data,pd.DataFrame(derived,index=data.index)],axis=1)
    pos_intercepts=[f'position_{p}' for p in POSITIONS[:-1]]
    flexible=[*pos_intercepts,*[f'{f}_by_{p}' for f in CONVENTIONAL for p in POSITIONS]]
    phase=[f'{f}_by_{p}' for f in PHASES for p in POSITIONS]
    required=["nfl_id","draft_year","combine_position",TARGET,"rookie_reg_player_plays_with_matching_metadata",*CONVENTIONAL,*PHASES]
    sample=data.loc[data["rookie_reg_player_plays_with_matching_metadata"].ge(args.min_game_plays)&data[required].notna().all(axis=1)].copy()
    result={'question':'Do position-specific 40-yard phase slopes improve prediction of rookie regular-season game-speed capacity beyond position-specific official tests, body size, and draft capital?',
            'target':f'player-level 90th percentile of per-play rookie-season peak speed, minimum {args.min_game_plays} matched tracked player-plays',
            'sample':{'minimum_tracked_plays':args.min_game_plays,'n_players':int(len(sample)),
                      'by_cohort':{str(int(k)):int(v) for k,v in sample.draft_year.value_counts().sort_index().items()},
                      'by_position':{str(k):int(v) for k,v in sample.combine_position.value_counts().sort_index().items()}},
            'validation':'leave one draft cohort out; separate conventional slopes and phase slopes by Combine position; fold-only imputation/scaling; ridge alpha grid not selected on held-out cohorts; paired player bootstrap of fixed OOF predictions, no model refit.',
            'comparisons':[]}
    prediction_rows=[]
    for ai,alpha in enumerate(ALPHAS):
        oof=[]; folds=[]
        for cohort in sorted(sample.draft_year.unique()):
            tr=sample.loc[sample.draft_year.ne(cohort)]; te=sample.loc[sample.draft_year.eq(cohort)]
            base=predict(tr,te,flexible,TARGET,alpha); cand=predict(tr,te,flexible+phase,TARGET,alpha); y=te[TARGET].to_numpy(float)
            rm0=float(np.sqrt(np.mean((y-base)**2))); rm1=float(np.sqrt(np.mean((y-cand)**2)))
            folds.append({'held_out_draft_year':int(cohort),'n_players':int(len(te)),'baseline_rmse':rm0,'candidate_rmse':rm1,'rmse_change':rm1-rm0})
            oof.extend({'nfl_id':int(pid),'draft_year':int(year),'actual':float(actual),'baseline':float(p0),'candidate':float(p1),'alpha':alpha}
                       for pid,year,actual,p0,p1 in zip(te.nfl_id,te.draft_year,y,base,cand))
        oof=pd.DataFrame(oof); delta=float(np.sqrt(np.mean((oof.actual-oof.candidate)**2))-np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
        ci=bootstrap_delta(oof,seed=1311+ai)
        result['comparisons'].append({'alpha':alpha,'rmse_change':delta,'player_bootstrap_95pct_ci':ci,'folds':folds})
        prediction_rows.extend(oof.to_dict('records'))
    pd.DataFrame(prediction_rows).to_csv(OUT/f'position_specific_game_speed_oof_{tag}.csv',index=False)
    result_path=OUT/f'position_specific_game_speed_results_{tag}.json'; report_path=OUT/f'position_specific_game_speed_report_{tag}.md'
    result_path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Do 40-yard speed phases transfer to rookie game speed by position?','',
           'The outcome is each prospect’s rookie regular-season 90th percentile of per-play peak speed, calculated from matched 10 Hz game tracking; players need a minimum number of matched player-play observations. The baseline allows each Combine position its own slopes for official 40, 10-yard split, height, weight, and log draft pick. The candidate adds early acceleration, 10–20 yard acceleration, and 20–40 yard speed retention with position-specific slopes. The phase block therefore adds 15 interaction terms.','',
           f"Complete sample: {len(sample)} players, minimum {args.min_game_plays} tracked rookie player-plays. Position coverage: {result['sample']['by_position']}.",'',
           'Whole draft cohorts are held out. Imputation and scaling use training cohorts only. Ridge alpha varies on a fixed grid; no alpha is selected on the held-out classes. Player-bootstrap intervals resample fixed out-of-fold errors within cohort and do not refit models. The target is speed capacity among observed tracked plays, not route-specific performance.','',
           '| Ridge alpha | Pooled OOF RMSE Δ (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |','|---:|---:|---:|---:|---:|']
    for x in result['comparisons']:
        f={z['held_out_draft_year']:z['rmse_change'] for z in x['folds']}; ci=x['player_bootstrap_95pct_ci']
        vals=[f'{f[y]:+.4f}' if y in f else 'NA' for y in [2023,2024,2025]]
        lines.append(f"| {x['alpha']:g} | {x['rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines+=['','Negative change favors the 40-phase model. The full-position interaction block is intentionally stringent but can be underpowered within small position/cohort cells. A stable improvement would motivate a position-specific follow-up; an unstable or null result does not show that every use of tracking is unhelpful.','']
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    if args.min_game_plays==25:
        (OUT/'position_specific_game_speed_results.json').write_text(result_path.read_text(encoding='utf-8'),encoding='utf-8')
        (OUT/'position_specific_game_speed_report.md').write_text(report_path.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'sample':result['sample'],'deltas':[{'alpha':x['alpha'],'rmse_change':x['rmse_change'],'ci':x['player_bootstrap_95pct_ci']} for x in result['comparisons']]},indent=2))


if __name__=='__main__': main()
