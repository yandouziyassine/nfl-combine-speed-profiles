"""Explore which 40-yard phase carries the DL game-speed transfer signal."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'
BASE=['forty','ten_yd_split','combine_height','combine_weight','draft_pick_log']
PHASES={'early_acceleration':'forty_early_accel_0_1s_ydps2_mean',
        'late_acceleration':'forty_late_accel_10_20yd_ydps2_mean',
        'speed_retention':'forty_speed_retention_20_40yd_ratio_mean'}
TARGET='rookie_reg_p90_play_peak_speed_ydps'
ALPHAS=[5.0,20.0]

def predict(tr,te,features,alpha):
    med=tr[features].median(); a=tr[features].fillna(med); b=te[features].fillna(med)
    mu=a.mean(); sd=a.std().replace(0,1).fillna(1)
    x=np.column_stack([np.ones(len(a)),((a-mu)/sd).to_numpy(float)])
    z=np.column_stack([np.ones(len(b)),((b-mu)/sd).to_numpy(float)])
    p=np.eye(x.shape[1])*alpha; p[0,0]=0
    return z@np.linalg.solve(x.T@x+p,x.T@tr[TARGET].to_numpy(float))

def bootstrap(oof,seed):
    # Player bootstrap within held-out cohort, vectorized over draws. This is
    # mathematically the same paired resampling as the former DataFrame loop,
    # without tens of thousands of small pandas copies.
    rng=np.random.default_rng(seed); groups=list(oof.groupby('draft_year',sort=True)); draws=3000
    candidate_mse=np.zeros(draws); baseline_mse=np.zeros(draws); total_n=len(oof)
    for _,g in groups:
        yc=g.actual.to_numpy(float); ec=yc-g.candidate.to_numpy(float); eb=yc-g.baseline.to_numpy(float); n=len(g)
        ix=rng.integers(0,n,size=(draws,n))
        weight=n/total_n
        candidate_mse+=weight*np.mean(ec[ix]**2,axis=1)
        baseline_mse+=weight*np.mean(eb[ix]**2,axis=1)
    delta=np.sqrt(candidate_mse)-np.sqrt(baseline_mse)
    return [float(np.quantile(delta,.025)),float(np.quantile(delta,.975))]

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--min-game-plays',type=int,default=50); args=ap.parse_args(); tag=f'{args.min_game_plays}plays'
    d=pd.read_csv(OUT/'player_speed_signatures.csv'); d['draft_pick_log']=np.log1p(d.draft_overall_pick)
    exposure=d.combine_position.eq('DL')&d.rookie_reg_player_plays_with_matching_metadata.ge(args.min_game_plays)
    d=d.loc[exposure].copy()
    single={k:[v] for k,v in PHASES.items()}
    specs={'all_three_phases':list(PHASES.values()),**single,
           **{f'{k}_common_sample':v for k,v in single.items()}}
    results={'minimum_tracked_plays':args.min_game_plays,'exposure_eligible_players':int(len(d)),'specifications':{}}
    oof_all=[]
    for si,(name,features) in enumerate(specs.items()):
        required=['nfl_id','draft_year',TARGET,*BASE,*features]
        if name.endswith('_common_sample'):
            required.extend(PHASES.values())
        s=d.loc[d[required].notna().all(axis=1)].copy()
        if s.draft_year.nunique()!=3: raise ValueError(f'{name} has fewer than three complete draft cohorts')
        results['specifications'][name]=[]
        results.setdefault('samples',{})[name]={'n_players':int(len(s)),'by_cohort':{str(int(k)):int(v) for k,v in s.draft_year.value_counts().sort_index().items()}}
        for ai,alpha in enumerate(ALPHAS):
            rows=[]; folds=[]
            for cohort in sorted(s.draft_year.unique()):
                tr=s.loc[s.draft_year.ne(cohort)]; te=s.loc[s.draft_year.eq(cohort)]
                p0=predict(tr,te,BASE,alpha); p1=predict(tr,te,BASE+features,alpha); y=te[TARGET].to_numpy(float)
                r0=float(np.sqrt(np.mean((y-p0)**2))); r1=float(np.sqrt(np.mean((y-p1)**2)))
                folds.append({'held_out_draft_year':int(cohort),'n_players':int(len(te)),'baseline_rmse':r0,'candidate_rmse':r1,'rmse_change':r1-r0})
                rows.extend({'nfl_id':int(pid),'draft_year':int(year),'actual':float(actual),'baseline':float(b),'candidate':float(c),'phase_spec':name,'alpha':alpha}
                            for pid,year,actual,b,c in zip(te.nfl_id,te.draft_year,y,p0,p1))
            oof=pd.DataFrame(rows); delta=float(np.sqrt(np.mean((oof.actual-oof.candidate)**2))-np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
            interval=bootstrap(oof,seed=4321+si*10+ai)
            results['specifications'][name].append({'alpha':alpha,'rmse_change':delta,'player_bootstrap_95pct_ci':interval,'folds':folds})
            oof_all.extend(rows)
    pd.DataFrame(oof_all).to_csv(OUT/f'dl_game_speed_ablation_oof_{tag}.csv',index=False)
    (OUT/f'dl_game_speed_ablation_results_{tag}.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Which 40-yard phase contributes to DL game-speed transfer?','',
           'This is a discovery-stage ablation of the DL-only result. The target and cohort holdouts are unchanged. Each model compares the same players within its specification and adds either one 40-yard phase or all three to the traditional-test/draft baseline. This does not identify causal mechanisms; correlated phase features can substitute for one another. Feature-wise results are exploratory and are not multiplicity-adjusted.','',
           f"Exposure-eligible: {len(d)} DLs with at least {args.min_game_plays} matched rookie tracked plays. Complete-case sample sizes vary by feature block because 40-yard distance features have different coverage.",'',
           '| Added feature(s) | Alpha | RMSE Δ (95% player-bootstrap CI) | 2023 Δ | 2024 Δ | 2025 Δ |','|---|---:|---:|---:|---:|---:|']
    for name,items in results['specifications'].items():
        for x in items:
            f={z['held_out_draft_year']:z['rmse_change'] for z in x['folds']}; ci=x['player_bootstrap_95pct_ci']; vals=[f'{f[y]:+.4f}' for y in [2023,2024,2025]]
            n=results['samples'][name]['n_players']
            lines.append(f"| {name} (n={n}) | {x['alpha']:g} | {x['rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines+=['','A negative change favors the phase feature. The interval reflects player resampling conditional on fixed OOF predictions; it does not refit models or account for selecting this position/feature family from prior screens. A feature should only be promoted if the grouped signal is robust and the final claim acknowledges exploratory selection.','']
    (OUT/f'dl_game_speed_ablation_report_{tag}.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:[{'alpha':x['alpha'],'delta':round(x['rmse_change'],4),'ci':x['player_bootstrap_95pct_ci']} for x in v] for k,v in results['specifications'].items()},indent=2))

if __name__=='__main__': main()
