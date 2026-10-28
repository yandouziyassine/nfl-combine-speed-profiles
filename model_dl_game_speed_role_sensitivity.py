"""Check that the DL phase-to-game-speed lift is not just edge/interior role."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"
BASE=["forty","ten_yd_split","combine_height","combine_weight","draft_pick_log"]
PHASES=["forty_early_accel_0_1s_ydps2_mean","forty_late_accel_10_20yd_ydps2_mean","forty_speed_retention_20_40yd_ratio_mean"]
TARGET="rookie_reg_p90_play_peak_speed_ydps"; ALPHAS=[0.1,1.0,5.0,20.0]

def predict(tr,te,features,alpha):
    med=tr[features].median(); a=tr[features].fillna(med); b=te[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    x=np.column_stack([np.ones(len(a)),((a-mu)/sd).to_numpy(float)])
    z=np.column_stack([np.ones(len(b)),((b-mu)/sd).to_numpy(float)])
    p=np.eye(x.shape[1])*alpha; p[0,0]=0
    return z@np.linalg.solve(x.T@x+p,x.T@tr[TARGET].to_numpy(float))

def boot(oof,seed):
    groups=[g.index.to_numpy() for _,g in oof.groupby('draft_year')]; rng=np.random.default_rng(seed); vals=[]
    for _ in range(3000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); d=oof.loc[ix]
        vals.append(float(np.sqrt(np.mean((d.actual-d.candidate)**2))-np.sqrt(np.mean((d.actual-d.baseline)**2))))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--min-game-plays',type=int,default=50); args=ap.parse_args(); tag=f'{args.min_game_plays}plays'
    d=pd.read_csv(OUT/'player_speed_signatures.csv'); d['draft_pick_log']=np.log1p(d.draft_overall_pick)
    d=d.loc[d.combine_position.eq('DL')&d.rookie_reg_player_plays_with_matching_metadata.ge(args.min_game_plays)].copy()
    d=d.loc[d[["nfl_id","draft_year","nfl_position",TARGET,*BASE,*PHASES]].notna().all(axis=1)].copy()
    roles=sorted(d.nfl_position.unique())
    for role in roles[1:]: d[f'nfl_role_{role}']=d.nfl_position.eq(role).astype(int)
    role_cols=[f'nfl_role_{r}' for r in roles[1:]]
    controls={"pre_draft_baseline":BASE,"subposition_adjusted_baseline":BASE+role_cols}
    outputs={}
    for baseline_name,base in controls.items():
        items=[]; prediction_rows=[]
        for ai,alpha in enumerate(ALPHAS):
            oof=[]; folds=[]
            for cohort in sorted(d.draft_year.unique()):
                tr=d.loc[d.draft_year.ne(cohort)]; te=d.loc[d.draft_year.eq(cohort)]
                p0=predict(tr,te,base,alpha); p1=predict(tr,te,base+PHASES,alpha); y=te[TARGET].to_numpy(float)
                r0=float(np.sqrt(np.mean((y-p0)**2))); r1=float(np.sqrt(np.mean((y-p1)**2)))
                folds.append({'held_out_draft_year':int(cohort),'n_players':int(len(te)),'baseline_rmse':r0,'phase_rmse':r1,'rmse_change':r1-r0})
                oof.extend({'nfl_id':int(pid),'draft_year':int(year),'actual':float(yy),'baseline':float(b),'candidate':float(c),'alpha':alpha}
                           for pid,year,yy,b,c in zip(te.nfl_id,te.draft_year,y,p0,p1))
            oof=pd.DataFrame(oof); delta=float(np.sqrt(np.mean((oof.actual-oof.candidate)**2))-np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
            interval=boot(oof,seed=996+ai+(100 if baseline_name!='pre_draft_baseline' else 0))
            items.append({'alpha':alpha,'rmse_change':delta,'player_bootstrap_95pct_ci':interval,'folds':folds})
            for row in oof.to_dict('records'): row['baseline_specification']=baseline_name; prediction_rows.append(row)
        outputs[baseline_name]=items
        pd.DataFrame(prediction_rows).to_csv(OUT/f'dl_game_speed_role_oof_{baseline_name}_{tag}.csv',index=False)
    result={'question':'Does the DL phase block still add rookie game-speed information after controlling for observed NFL edge/interior position?','sample':{'min_game_plays':args.min_game_plays,'n':int(len(d)),'by_cohort':{str(int(k)):int(v) for k,v in d.draft_year.value_counts().sort_index().items()},'by_nfl_position':{str(k):int(v) for k,v in d.nfl_position.value_counts().sort_index().items()}},'role_columns':roles,'validation':'LOCO by draft cohort; train-only scaling/imputation; bootstrap fixed OOF player errors within cohort; no refit.','specifications':outputs}
    (OUT/f'dl_game_speed_role_results_{tag}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Does the DL game-speed result survive subposition controls?','',
           'The main DL screen found that 40-yard phases improved held-out prediction of rookie 90th-percentile game speed. Because Combine DLs later play different roles, this sensitivity adds NFL subposition indicators (DE, DT, NT, OLB) to the conventional-test baseline before adding the phase block. NFL position is post-draft context and is used only to check whether the phase lift is a proxy for edge/interior role.','',
           f"Sample: {len(d)} DLs with at least {args.min_game_plays} tracked rookie player-plays; role coverage: {result['sample']['by_nfl_position']}.",'',
           'Negative RMSE change favors the phase block. Models leave one draft cohort out; intervals resample fixed out-of-fold player errors within cohort. The cohort holdouts and position sample remain small.','']
    for spec,items in outputs.items():
        lines += [f'## {spec.replace("_"," ").title()}','', '| Alpha | RMSE Δ (95% CI) | 2023 Δ | 2024 Δ | 2025 Δ |','|---:|---:|---:|---:|---:|']
        for x in items:
            f={z['held_out_draft_year']:z['rmse_change'] for z in x['folds']}; ci=x['player_bootstrap_95pct_ci']; vals=[f'{f[y]:+.4f}' if y in f else 'NA' for y in [2023,2024,2025]]
            lines.append(f"| {x['alpha']:g} | {x['rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines.append('')
    (OUT/f'dl_game_speed_role_report_{tag}.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'n':len(d),'roles':result['sample']['by_nfl_position'],'deltas':{k:[round(x['rmse_change'],4) for x in v] for k,v in outputs.items()}},indent=2))

if __name__=='__main__': main()
