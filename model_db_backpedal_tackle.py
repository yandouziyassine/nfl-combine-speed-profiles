"""Test Combine backpedal-to-45-degree movement against rookie DB tackle rate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
BASE_NUMERIC=["forty","ten_yd_split","combine_height","combine_weight","draft_pick_log"]
PHASES=["bp_early_accel","bp_bilateral_lateral_speed","bp_abs_lateral_gap"]
ALPHAS=[0.1,1.0,5.0]
DRILL="BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"


def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))


def fit_predict(train,test,features,alpha):
    med=train[features].median(); a=train[features].fillna(med); b=test[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    x=np.column_stack([np.ones(len(a)),((a-mu)/sd).to_numpy(float)])
    z=np.column_stack([np.ones(len(b)),((b-mu)/sd).to_numpy(float)])
    y=train.tackles.to_numpy(float)/train.opportunities.to_numpy(float)
    n=train.opportunities.to_numpy(float)
    prior=np.clip(train.tackles.sum()/n.sum(),1e-5,1-1e-5)
    beta=np.zeros(x.shape[1]); beta[0]=np.log(prior/(1-prior))
    penalty=np.eye(x.shape[1])*alpha; penalty[0,0]=0
    for _ in range(100):
        p=sigmoid(x@beta); w=np.maximum(n*p*(1-p),1e-8)
        working=x@beta+(y-p)/np.maximum(p*(1-p),1e-8)
        update=np.linalg.solve(x.T@(w[:,None]*x)+penalty,x.T@(w*working))
        if np.max(np.abs(update-beta))<1e-8: beta=update; break
        beta=update
    return sigmoid(z@beta)


def ci_player(delta, cohort, draws=3000, seed=9235):
    rng=np.random.default_rng(seed); groups=[np.flatnonzero(cohort==c) for c in np.unique(cohort)]; vals=[]
    for _ in range(draws):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); vals.append(float(delta[ix].mean()))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-opportunities',type=int,default=50)
    args=ap.parse_args(); tag=f'{args.min_opportunities}plus'
    players=pd.read_csv(OUT/f'db_tackle_player_table_{tag}.csv')
    drill=pd.read_csv(OUT/'player_drill_speed_profiles.csv')
    bp=drill.loc[drill.drill_name.eq(DRILL),['nfl_id','early_accel_0_1s_ydps2_mean','left_lateral_speed_mean_ydps_mean','right_lateral_speed_mean_ydps_mean']].copy()
    bp['bp_early_accel']=bp.early_accel_0_1s_ydps2_mean
    bp['bp_bilateral_lateral_speed']=(bp.left_lateral_speed_mean_ydps_mean+bp.right_lateral_speed_mean_ydps_mean)/2
    bp['bp_abs_lateral_gap']=(bp.left_lateral_speed_mean_ydps_mean-bp.right_lateral_speed_mean_ydps_mean).abs()
    data=players.merge(bp[['nfl_id',*PHASES]],on='nfl_id',how='left',validate='one_to_one')
    data['draft_pick_log']=np.log1p(data.draft_overall_pick)
    for pos in ['CB','FS']:
        data[f'position_{pos}']=data.nfl_position.eq(pos).astype(int)
    controls=BASE_NUMERIC+['position_CB','position_FS']
    needed=controls+PHASES
    sample=data.loc[data[needed].notna().all(axis=1)].copy()
    result={'question':'Do Combine backpedal-to-45-degree transition features improve held-out prediction of rookie DB tackle rate beyond official sprint testing, size, draft capital, and CB/FS/SS role?',
            'drill':DRILL,
            'features':{'bp_early_accel':'mean 0–1 second acceleration after detected movement onset',
                        'bp_bilateral_lateral_speed':'mean of the tracked path-relative left and right lateral speeds',
                        'bp_abs_lateral_gap':'absolute difference in path-relative left and right lateral speeds'},
            'interpretation_limit':'Path-relative left/right is not anatomical laterality; tackle rate per observed player-play is role/opportunity dependent, not success conditional on a verified tackle opportunity.',
            'sample':{'min_opportunities':args.min_opportunities,'n_players':int(len(sample)),
                      'by_cohort':{str(int(k)):int(v) for k,v in sample.draft_year.value_counts().sort_index().items()},
                      'by_position':{str(k):int(v) for k,v in sample.nfl_position.value_counts().sort_index().items()},
                      'tackles':int(sample.tackles.sum()),'player_play_rows':int(sample.opportunities.sum())},
            'validation':'leave one draft cohort out, binomial ridge, training-fold imputation/scaling; equal-player-weighted score; bootstrap fixed OOF player errors within cohort; no model refit in bootstrap.',
            'comparisons':[]}
    pred_rows=[]
    for ai,alpha in enumerate(ALPHAS):
        oof=[]; folds=[]
        for cohort in sorted(sample.draft_year.unique()):
            tr=sample.loc[sample.draft_year.ne(cohort)]; te=sample.loc[sample.draft_year.eq(cohort)]
            p0=fit_predict(tr,te,controls,alpha); p1=fit_predict(tr,te,controls+PHASES,alpha)
            y=te.tackles.to_numpy(float)/te.opportunities.to_numpy(float)
            b0=(y-p0)**2; b1=(y-p1)**2
            folds.append({'held_out_draft_year':int(cohort),'n_players':int(len(te)),
                          'baseline_brier':float(b0.mean()),'candidate_brier':float(b1.mean()),'brier_change':float((b1-b0).mean())})
            oof.extend({'nfl_id':int(pid),'draft_year':int(year),'actual_tackle_rate':float(actual),'opportunities':int(n),
                        'pred_baseline':float(base),'pred_candidate':float(cand),'brier_delta':float(delta),'alpha':alpha}
                       for pid,year,actual,n,base,cand,delta in zip(te.nfl_id,te.draft_year,y,te.opportunities,p0,p1,b1-b0))
        oof=pd.DataFrame(oof); interval=ci_player(oof.brier_delta.to_numpy(),oof.draft_year.to_numpy(),seed=9235+ai)
        result['comparisons'].append({'alpha':alpha,'brier_change':float(oof.brier_delta.mean()),'brier_ci':interval,'folds':folds})
        pred_rows.extend(oof.to_dict('records'))
    pd.DataFrame(pred_rows).to_csv(OUT/f'db_backpedal_oof_predictions_{tag}.csv',index=False)
    report_path=OUT/f'db_backpedal_transfer_report_{tag}.md'
    (OUT/f'db_backpedal_transfer_results_{tag}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Can the DB backpedal transition profile add rookie tackle-rate information?','',
           'The candidate comes from the Combine BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION drill. It adds 0–1 second acceleration, mean lateral speed across the measured path-relative left/right directions, and the absolute left/right lateral-speed gap. These directions are not anatomical sides. The model predicts each DB’s tackle count per observed rookie regular-season player-play row; the denominator is not a verified tackle opportunity. The baseline uses official 40, 10-yard split, size, draft capital, and CB/FS/SS subposition.', '',
           f"Complete sample: {len(sample)} players with at least {args.min_opportunities} observed rookie player-play rows; {int(sample.tackles.sum())} tackles over {int(sample.opportunities.sum())} rows.",'',
           'Whole draft cohorts are held out. Ridge alpha is varied over a fixed grid; no hyperparameter is selected using the held-out class. Test Brier scores weight players equally. Bootstrap intervals resample fixed out-of-fold player errors within cohort and do not refit the model.','',
           '| Alpha | Player-weighted Brier Δ (95% CI) | 2023 Δ | 2024 Δ | 2025 Δ |','|---:|---:|---:|---:|---:|']
    for x in result['comparisons']:
        f={q['held_out_draft_year']:q['brier_change'] for q in x['folds']}; ci=x['brier_ci']
        y=[f'{f[c]:+.5f}' if c in f else 'NA' for c in [2023,2024,2025]]
        lines.append(f"| {x['alpha']:g} | {x['brier_change']:+.5f} ({ci[0]:+.5f}, {ci[1]:+.5f}) | {y[0]} | {y[1]} | {y[2]} |")
    lines+=['','Negative change favors the movement-profile model. Treat any lift as a hypothesis for validation: the sample contains three cohorts, position-specific drill summaries may be noisy, and tackle rates mix player skill with alignment, scheme, and opportunity. The result is not a player score.','']
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'n_players':len(sample),'results':[{'alpha':x['alpha'],'brier_change':x['brier_change'],'ci':x['brier_ci']} for x in result['comparisons']]},indent=2))


if __name__=='__main__': main()
