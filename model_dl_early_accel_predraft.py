"""Evaluate DL early acceleration using a genuinely pre-draft baseline."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'
BASE=['forty','ten_yd_split','combine_height','combine_weight']
EARLY='forty_early_accel_0_1s_ydps2_mean'
TARGET='rookie_reg_p90_play_peak_speed_ydps'; ALPHAS=[5.0,20.0]

def predict(tr,te,features,alpha):
    med=tr[features].median(); a=tr[features].fillna(med); b=te[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    x=np.column_stack([np.ones(len(a)),((a-mu)/sd).to_numpy(float)])
    z=np.column_stack([np.ones(len(b)),((b-mu)/sd).to_numpy(float)])
    penalty=np.eye(x.shape[1])*alpha; penalty[0,0]=0
    return z@np.linalg.solve(x.T@x+penalty,x.T@tr[TARGET].to_numpy(float))

def bootstrap(oof,seed):
    groups=[g.index.to_numpy() for _,g in oof.groupby('draft_year')]
    rng=np.random.default_rng(seed); values=[]
    for _ in range(3000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); d=oof.loc[ix]
        values.append(float(np.sqrt(np.mean((d.actual-d.candidate)**2))-np.sqrt(np.mean((d.actual-d.baseline)**2))))
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--min-game-plays',type=int,default=100); args=ap.parse_args(); tag=f'{args.min_game_plays}plays'
    d=pd.read_csv(OUT/'player_speed_signatures.csv'); d=d.loc[d.combine_position.eq('DL')&d.rookie_reg_player_plays_with_matching_metadata.ge(args.min_game_plays)].copy()
    need=['nfl_id','draft_year',TARGET,*BASE,EARLY]
    s=d.loc[d[need].notna().all(axis=1)].copy()
    if s.draft_year.nunique()!=3: raise ValueError('Need complete players in each of the three draft cohorts.')
    result={'question':'Does early 0–1 second 40-yard acceleration improve prediction of rookie-season game-speed capacity using only information available before the draft?',
            'baseline':'official 40, 10-yard split, height, and weight; excludes draft capital and NFL position',
            'target':TARGET,'min_game_plays':args.min_game_plays,'sample':{'n':int(len(s)),'exposure_eligible':int(len(d)),'by_cohort':{str(int(k)):int(v) for k,v in s.draft_year.value_counts().sort_index().items()}},'comparisons':[]}
    oof_rows=[]
    for ai,alpha in enumerate(ALPHAS):
        rows=[]; folds=[]
        for cohort in sorted(s.draft_year.unique()):
            tr=s.loc[s.draft_year.ne(cohort)]; te=s.loc[s.draft_year.eq(cohort)]
            baseline=predict(tr,te,BASE,alpha); candidate=predict(tr,te,BASE+[EARLY],alpha); y=te[TARGET].to_numpy(float)
            r0=float(np.sqrt(np.mean((y-baseline)**2))); r1=float(np.sqrt(np.mean((y-candidate)**2)))
            folds.append({'held_out_draft_year':int(cohort),'n_players':int(len(te)),'baseline_rmse':r0,'candidate_rmse':r1,'rmse_change':r1-r0})
            rows.extend({'nfl_id':int(pid),'draft_year':int(year),'actual':float(actual),'baseline':float(b),'candidate':float(c),'alpha':alpha}
                        for pid,year,actual,b,c in zip(te.nfl_id,te.draft_year,y,baseline,candidate))
        oof=pd.DataFrame(rows); delta=float(np.sqrt(np.mean((oof.actual-oof.candidate)**2))-np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
        forward = oof.loc[oof.draft_year.eq(2025)].copy()
        forward_delta = float(np.sqrt(np.mean((forward.actual-forward.candidate)**2))-
                              np.sqrt(np.mean((forward.actual-forward.baseline)**2))) if len(forward) else np.nan
        result['comparisons'].append({'alpha':alpha,'rmse_change':delta,
                                      'player_bootstrap_95pct_ci':bootstrap(oof,800+ai),'folds':folds,
                                      'forward_2025':{'n_players':int(len(forward)),
                                                      'baseline_rmse':float(np.sqrt(np.mean((forward.actual-forward.baseline)**2))) if len(forward) else np.nan,
                                                      'candidate_rmse':float(np.sqrt(np.mean((forward.actual-forward.candidate)**2))) if len(forward) else np.nan,
                                                      'rmse_change':forward_delta,
                                                      'player_bootstrap_95pct_ci':bootstrap(forward,1800+ai) if len(forward) else [np.nan,np.nan]}})
        oof_rows.extend(rows)
    pd.DataFrame(oof_rows).to_csv(OUT/f'dl_early_accel_predraft_oof_{tag}.csv',index=False)
    (OUT/f'dl_early_accel_predraft_results_{tag}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# DL early acceleration with a pre-draft baseline','',
           'The baseline uses only information available before the draft: official 40, 10-yard split, height, and weight. It excludes draft capital and NFL position. The candidate adds tracked 0–1 second acceleration.','',
           f"Sample: {len(s)} complete DL players with at least {args.min_game_plays} matched rookie-season tracked player-plays; exposure-eligible before Combine-data completeness: {len(d)}.",'',
           'Whole draft cohorts are held out. Imputation and scaling use training players only. Intervals bootstrap fixed OOF player predictions within cohort and do not refit models.','',
           '| Alpha | Baseline RMSE | Candidate RMSE | RMSE change (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |','|---:|---:|---:|---:|---:|---:|---:|']
    for x in result['comparisons']:
        f={z['held_out_draft_year']:z for z in x['folds']}; ci=x['player_bootstrap_95pct_ci']
        lines.append(f"| {x['alpha']:g} | {np.mean([z['baseline_rmse']**2 for z in x['folds']])**.5:.4f} | {np.mean([z['candidate_rmse']**2 for z in x['folds']])**.5:.4f} | {x['rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {f[2023]['rmse_change']:+.4f} | {f[2024]['rmse_change']:+.4f} | {f[2025]['rmse_change']:+.4f} |")
    lines += ['', '## Forward-in-time 2025 holdout', '',
              'This fold trains on 2023–2024 and tests on the 2025 class. Because the feature emerged from a wider screen that included all three cohorts, treat this as an exploratory temporal check rather than independent confirmation.', '',
              '| Alpha | Test players | Baseline RMSE | Candidate RMSE | RMSE change (95% fixed-prediction player-bootstrap CI) |',
              '|---:|---:|---:|---:|---:|']
    for x in result['comparisons']:
        fwd=x['forward_2025']; ci=fwd['player_bootstrap_95pct_ci']
        lines.append(f"| {x['alpha']:g} | {fwd['n_players']} | {fwd['baseline_rmse']:.4f} | {fwd['candidate_rmse']:.4f} | {fwd['rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) |")
    lines+=['','This is a candidate screening result, not a validated scouting rule. Only three draft cohorts are available; broad exploratory model selection and limited sample size may inflate apparent value.','']
    (OUT/f'dl_early_accel_predraft_report_{tag}.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'sample':result['sample'],'comparisons':[{'alpha':x['alpha'],'delta':round(x['rmse_change'],4),'ci':x['player_bootstrap_95pct_ci']} for x in result['comparisons']]},indent=2))
if __name__=='__main__': main()

