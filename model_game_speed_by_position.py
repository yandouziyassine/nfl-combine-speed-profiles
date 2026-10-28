"""Position-wise follow-up to the all-position game-speed transfer model."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"
POSITIONS=["DB","DL","OL","WR"]
BASE=["forty","ten_yd_split","combine_height","combine_weight","draft_pick_log"]
PHASES=["forty_early_accel_0_1s_ydps2_mean","forty_late_accel_10_20yd_ydps2_mean","forty_speed_retention_20_40yd_ratio_mean"]
TARGET="rookie_reg_p90_play_peak_speed_ydps"
ALPHAS=[0.1,1.0,5.0,20.0]

def pred(tr,te,features,alpha):
    med=tr[features].median(); a=tr[features].fillna(med); b=te[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    x=np.column_stack([np.ones(len(a)),((a-mu)/sd).to_numpy(float)])
    z=np.column_stack([np.ones(len(b)),((b-mu)/sd).to_numpy(float)])
    lam=np.eye(x.shape[1])*alpha; lam[0,0]=0
    return z@np.linalg.solve(x.T@x+lam,x.T@tr[TARGET].to_numpy(float))

def ci(oof,draws=3000,seed=430):
    gr=[g.index.to_numpy() for _,g in oof.groupby('draft_year')]; rng=np.random.default_rng(seed); v=[]
    for _ in range(draws):
        idx=np.concatenate([rng.choice(x,len(x),replace=True) for x in gr]); d=oof.loc[idx]
        v.append(float(np.sqrt(np.mean((d.actual-d.candidate)**2))-np.sqrt(np.mean((d.actual-d.baseline)**2))))
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--min-game-plays',type=int,default=25); args=ap.parse_args(); tag=f'{args.min_game_plays}plays'
    d=pd.read_csv(OUT/'player_speed_signatures.csv'); d['draft_pick_log']=np.log1p(d.draft_overall_pick)
    required=['nfl_id','draft_year','combine_position',TARGET,'rookie_reg_player_plays_with_matching_metadata',*BASE,*PHASES]
    d=d.loc[d.rookie_reg_player_plays_with_matching_metadata.ge(args.min_game_plays)&d[required].notna().all(axis=1)]
    result={'target':TARGET,'minimum_tracked_plays':args.min_game_plays,'method':'Separate within-position ridge models; leave one draft cohort out; train-only imputation/scaling; paired player-bootstrap of fixed OOF errors, no refit.','positions':{}}
    predrows=[]
    for pi,pos in enumerate(POSITIONS):
        s=d.loc[d.combine_position.eq(pos)].copy()
        info={'n_players':int(len(s)),'by_cohort':{str(int(k)):int(v) for k,v in s.draft_year.value_counts().sort_index().items()},'alphas':[]}
        # Avoid a model claim when a position/cohort cell is too small for a stable holdout.
        if len(s)<30 or len(s.draft_year.unique())<3:
            info['status']='insufficient players for a three-cohort model screen'
            result['positions'][pos]=info; continue
        for ai,alpha in enumerate(ALPHAS):
            rows=[]; folds=[]
            for cohort in sorted(s.draft_year.unique()):
                tr=s.loc[s.draft_year.ne(cohort)]; te=s.loc[s.draft_year.eq(cohort)]
                b=pred(tr,te,BASE,alpha); c=pred(tr,te,BASE+PHASES,alpha); y=te[TARGET].to_numpy(float)
                r0=float(np.sqrt(np.mean((y-b)**2))); r1=float(np.sqrt(np.mean((y-c)**2)))
                folds.append({'held_out_draft_year':int(cohort),'n_players':int(len(te)),'baseline_rmse':r0,'phase_rmse':r1,'rmse_change':r1-r0})
                rows.extend({'nfl_id':int(pid),'draft_year':int(year),'actual':float(actual),'baseline':float(p0),'candidate':float(p1),'alpha':alpha,'position':pos}
                            for pid,year,actual,p0,p1 in zip(te.nfl_id,te.draft_year,y,b,c))
            oof=pd.DataFrame(rows); delta=float(np.sqrt(np.mean((oof.actual-oof.candidate)**2))-np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
            interval=ci(oof,seed=430+pi*100+ai)
            info['alphas'].append({'alpha':alpha,'rmse_change':delta,'player_bootstrap_95pct_ci':interval,'folds':folds})
            predrows.extend(rows)
        result['positions'][pos]=info
    pd.DataFrame(predrows).to_csv(OUT/f'game_speed_by_position_oof_{tag}.csv',index=False)
    rp=OUT/f'game_speed_by_position_results_{tag}.json'; rp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Which positions show 40-phase transfer to rookie game speed?','',
           'This is a position-wise follow-up to the interaction model. For each position, train a separate ridge model using conventional testing and draft capital, then add the three 40-yard phase features. Every player remains within one draft cohort; evaluation holds out a full cohort. Minimum tracked-play exposure is applied before modeling. Imputation and scaling are fit within each training fold. Ridge alpha is varied without selecting on the held-out classes. Bootstrap intervals resample fixed out-of-fold player errors within cohort; models are not refit.','',
           'Positions with fewer than 30 complete players are not scored. RMSE change below zero favors adding the 40-yard phase block. These position-specific estimates are exploratory and can be noisy with only three cohorts.','']
    for pos,info in result['positions'].items():
        lines += [f"## {pos}","",f"Players: {info['n_players']}; cohorts: {info['by_cohort']}.",""]
        if 'status' in info: lines += [info['status'],""]; continue
        lines += ['| Alpha | Pooled RMSE Δ (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |','|---:|---:|---:|---:|---:|']
        for x in info['alphas']:
            f={z['held_out_draft_year']:z['rmse_change'] for z in x['folds']}; ci0=x['player_bootstrap_95pct_ci']; v=[f'{f[y]:+.4f}' if y in f else 'NA' for y in [2023,2024,2025]]
            lines.append(f"| {x['alpha']:g} | {x['rmse_change']:+.4f} ({ci0[0]:+.4f}, {ci0[1]:+.4f}) | {v[0]} | {v[1]} | {v[2]} |")
        lines.append('')
    rp.with_name(f'game_speed_by_position_report_{tag}.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({pos:{'n':x['n_players'],'deltas':[round(a['rmse_change'],4) for a in x['alphas']]} for pos,x in result['positions'].items()},indent=2))

if __name__=='__main__': main()
