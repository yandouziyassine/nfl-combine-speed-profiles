"""Check whether early 40-yard acceleration drives DL game-speed transfer."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'
BASE=['forty','ten_yd_split','combine_height','combine_weight','draft_pick_log']
EARLY='forty_early_accel_0_1s_ydps2_mean'
LATE='forty_late_accel_10_20yd_ydps2_mean'; RET='forty_speed_retention_20_40yd_ratio_mean'
TARGET='rookie_reg_p90_play_peak_speed_ydps'; ALPHAS=[5.0,20.0]

def predict(tr,te,features,alpha):
    med=tr[features].median(); a=tr[features].fillna(med); b=te[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    x=np.column_stack([np.ones(len(a)),((a-mu)/sd).to_numpy(float)])
    z=np.column_stack([np.ones(len(b)),((b-mu)/sd).to_numpy(float)])
    p=np.eye(x.shape[1])*alpha; p[0,0]=0
    return z@np.linalg.solve(x.T@x+p,x.T@tr[TARGET].to_numpy(float))

def ci(oof,seed):
    gr=[g.index.to_numpy() for _,g in oof.groupby('draft_year')]; rng=np.random.default_rng(seed); vals=[]
    for _ in range(3000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in gr]); d=oof.loc[ix]
        vals.append(float(np.sqrt(np.mean((d.actual-d.candidate)**2))-np.sqrt(np.mean((d.actual-d.baseline)**2))))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--min-game-plays',type=int,default=50); args=ap.parse_args(); tag=f'{args.min_game_plays}plays'
    d=pd.read_csv(OUT/'player_speed_signatures.csv'); d['draft_pick_log']=np.log1p(d.draft_overall_pick)
    d=d.loc[d.combine_position.eq('DL')&d.rookie_reg_player_plays_with_matching_metadata.ge(args.min_game_plays)].copy()
    roles=sorted(d.nfl_position.dropna().unique())
    derived={f'role_{r}':d.nfl_position.eq(r).astype(int) for r in roles[1:]}
    d=pd.concat([d,pd.DataFrame(derived,index=d.index)],axis=1)
    controls=BASE+[f'role_{r}' for r in roles[1:]]
    specs={'early_acceleration_only':[EARLY],'all_three_phases':[EARLY,LATE,RET]}
    result={'min_game_plays':args.min_game_plays,'exposure_eligible_players':int(len(d)),
            'baseline':'official 40, 10-yard split, size, log draft pick, and observed NFL subposition indicators (sensitivity control only)',
            'specifications':{},'samples':{}}
    prediction_rows=[]
    for si,(name,features) in enumerate(specs.items()):
        result['specifications'][name]=[]
        required=['nfl_id','draft_year','nfl_position',TARGET,*controls,*features]
        s=d.loc[d[required].notna().all(axis=1)].copy()
        if s.draft_year.nunique()!=3: raise ValueError(f'{name} has fewer than three complete draft cohorts')
        result['samples'][name]={'n':int(len(s)),'by_cohort':{str(int(k)):int(v) for k,v in s.draft_year.value_counts().sort_index().items()},'by_nfl_position':{str(k):int(v) for k,v in s.nfl_position.value_counts().items()}}
        for ai,alpha in enumerate(ALPHAS):
            rows=[]; folds=[]
            for cohort in sorted(s.draft_year.unique()):
                tr=s.loc[s.draft_year.ne(cohort)]; te=s.loc[s.draft_year.eq(cohort)]
                p0=predict(tr,te,controls,alpha); p1=predict(tr,te,controls+features,alpha); y=te[TARGET].to_numpy(float)
                r0=float(np.sqrt(np.mean((y-p0)**2))); r1=float(np.sqrt(np.mean((y-p1)**2)))
                folds.append({'held_out_draft_year':int(cohort),'n_players':int(len(te)),'baseline_rmse':r0,'candidate_rmse':r1,'rmse_change':r1-r0})
                rows.extend({'nfl_id':int(pid),'draft_year':int(year),'actual':float(y0),'baseline':float(b),'candidate':float(c),'alpha':alpha,'spec':name}
                            for pid,year,y0,b,c in zip(te.nfl_id,te.draft_year,y,p0,p1))
            oof=pd.DataFrame(rows); delta=float(np.sqrt(np.mean((oof.actual-oof.candidate)**2))-np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
            interval=ci(oof,seed=456+si*10+ai)
            result['specifications'][name].append({'alpha':alpha,'rmse_change':delta,'player_bootstrap_95pct_ci':interval,'folds':folds})
            prediction_rows.extend(rows)
    pd.DataFrame(prediction_rows).to_csv(OUT/f'dl_early_accel_oof_{tag}.csv',index=False)
    (OUT/f'dl_early_accel_results_{tag}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Does early 40-yard acceleration explain DL rookie game speed?','',
           'The early-only and all-phase specifications use separate complete-case samples: a player need not have late-phase or 20–40-yard data to enter the early-only comparison. The baseline includes official 40, 10-yard split, size, draft capital, and NFL DE/DT/NT/OLB indicators. Since NFL role is post-draft, this model is a sensitivity check for role confounding, not a draft-time decision model. The outcome is rookie regular-season p90 per-play peak speed.','',
           f"Exposure-eligible: {len(d)} DLs with at least {args.min_game_plays} tracked player-plays. Complete-case sample sizes vary by feature block.",'',
           'Whole cohorts are held out, transformations are fit on training players only, and bootstrap intervals resample fixed OOF player errors within cohort. The component ablation is exploratory and does not identify a causal mechanism.','',
           '| Added feature(s) | Alpha | RMSE Δ (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |','|---|---:|---:|---:|---:|---:|']
    for name,items in result['specifications'].items():
        for x in items:
            f={z['held_out_draft_year']:z['rmse_change'] for z in x['folds']}; ci0=x['player_bootstrap_95pct_ci']; vals=[f'{f[y]:+.4f}' for y in [2023,2024,2025]]
            lines.append(f"| {name} (n={result['samples'][name]['n']}) | {x['alpha']:g} | {x['rmse_change']:+.4f} ({ci0[0]:+.4f}, {ci0[1]:+.4f}) | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines+=['','Negative RMSE changes favor the added Combine phase. The small DL sample, post-draft role control, and broad prior screen limit conclusions; future-cohort replication is needed.','']
    (OUT/f'dl_early_accel_report_{tag}.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'exposure_eligible':len(d),'samples':result['samples'],'deltas':{k:[round(x['rmse_change'],4) for x in v] for k,v in result['specifications'].items()}},indent=2))

if __name__=='__main__': main()

