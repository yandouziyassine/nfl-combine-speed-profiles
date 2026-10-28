from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'

def rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a,dtype=float)-np.asarray(b,dtype=float))**2)))

def main():
    preds=pd.read_csv(OUT/'dl_game_speed_ablation_oof_100plays.csv')
    stored=json.loads((OUT/'dl_game_speed_ablation_results_100plays.json').read_text(encoding='utf-8'))
    checks=[]
    def add(name,ok,detail): checks.append({'check':name,'passed':bool(ok),'detail':detail})
    expected_specs={'early_acceleration','late_acceleration','speed_retention','all_three_phases','early_acceleration_common_sample','late_acceleration_common_sample','speed_retention_common_sample'}
    add('Expected specifications',set(preds.phase_spec.unique())==expected_specs,str(sorted(preds.phase_spec.unique())))
    add('No duplicate player predictions per model',not preds.duplicated(['phase_spec','alpha','nfl_id']).any(),f'{int(preds.duplicated(["phase_spec","alpha","nfl_id"]).sum())} duplicates')
    add('One outcome per player',preds.groupby('nfl_id').actual.nunique().max()==1,'player outcome is constant across specifications')
    add('Baseline and candidate paired within each feature sample',preds[['actual','baseline','candidate']].notna().all(axis=1).all(),'all scored rows contain the same player outcome and both predictions; feature blocks can have different eligible-player samples')
    pooled=[]
    for spec,group in preds.groupby('phase_spec'):
        for alpha,g in group.groupby('alpha'):
            expected=next(x for x in stored['specifications'][spec] if x['alpha']==float(alpha))
            delta=rmse(g.actual,g.candidate)-rmse(g.actual,g.baseline)
            expected_n=stored['samples'][spec]['n_players']
            same_n=g.nfl_id.nunique()==expected_n
            add(f'{spec} alpha={alpha:g}: pooled RMSE matches JSON',abs(delta-expected['rmse_change'])<1e-10,f'recomputed {delta:+.10f}; stored {expected["rmse_change"]:+.10f}')
            add(f'{spec} alpha={alpha:g}: sample size matches metadata',same_n,f'n={g.nfl_id.nunique()}, expected={expected_n}')
            for fold in expected['folds']:
                h=g.loc[g.draft_year.eq(fold['held_out_draft_year'])]
                d=rmse(h.actual,h.candidate)-rmse(h.actual,h.baseline)
                add(f'{spec} alpha={alpha:g}: {fold["held_out_draft_year"]} fold matches JSON',abs(d-fold['rmse_change'])<1e-10,f'recomputed {d:+.10f}; stored {fold["rmse_change"]:+.10f}')
            pooled.append({'specification':spec,'alpha':float(alpha),'n_players':int(g.nfl_id.nunique()),'baseline_rmse':rmse(g.actual,g.baseline),'candidate_rmse':rmse(g.actual,g.candidate),'rmse_change':delta})
    result={'audit_type':'independent arithmetic and OOF table integrity check; does not independently refit models or re-extract source tracking','passed':all(x['passed'] for x in checks),'n_checks':len(checks),'n_passed':sum(x['passed'] for x in checks),'checks':checks,'pooled_scores':pooled}
    (OUT/'dl_early_accel_score_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# OOF score audit: DL 40-yard phase ablation','',
           'This audit independently recomputes pooled and cohort-level RMSE changes from the saved player-level out-of-fold prediction table and compares them with the model result JSON. It also checks player uniqueness, outcome consistency, complete baseline/candidate pairs, and feature-specific sample sizes. Early-only specifications do not require late-phase or retention data. It does not re-fit the ridge models or independently reconstruct the source features.','',
           f"Result: **{'PASS' if result['passed'] else 'FAIL'}** ({result['n_passed']}/{result['n_checks']} arithmetic and table-integrity checks passed).",'',
           '| Added feature | Alpha | n players | Baseline RMSE | Candidate RMSE | RMSE change |','|---|---:|---:|---:|---:|---:|']
    for x in pooled:
        lines.append(f"| {x['specification']} | {x['alpha']:g} | {x['n_players']} | {x['baseline_rmse']:.4f} | {x['candidate_rmse']:.4f} | {x['rmse_change']:+.4f} |")
    lines+=['','## Interpretation boundary','',
            'This confirms that the published score table agrees with its saved OOF predictions. It is not an independent replication of feature extraction, training, or validation design. The fixed-prediction bootstrap does not include model refitting, and the 34-player result remains exploratory after broader screens.','']
    (OUT/'dl_early_accel_score_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'n_checks':result['n_checks'],'n_passed':result['n_passed']}))
if __name__=='__main__': main()

