"""Position-stratified 40-yard attempt repeatability for key phase measures."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_40_repeatability import icc_absolute_single

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'
FEATURES={'early_acceleration':'early_accel_0_1s_ydps2','late_acceleration':'late_accel_10_20yd_ydps2','speed_retention':'speed_retention_20_40yd_ratio'}

def main():
    attempts=pd.read_csv(OUT/'combine_attempt_speed_profiles.csv')
    player=pd.read_csv(OUT/'player_speed_signatures.csv',usecols=['nfl_id','combine_position'])
    attempts=attempts.loc[attempts.drill_name.eq('FORTY_YARD_DASH')&attempts.attempt.isin([1,2])].merge(player,on='nfl_id',how='left',validate='many_to_one')
    rows=[]; rng=np.random.default_rng(20271030)
    for pos in ['DB','DL','OL','TE','WR']:
        sub=attempts.loc[attempts.combine_position.eq(pos)]
        for label,feature in FEATURES.items():
            wide=sub.pivot_table(index='nfl_id',columns='attempt',values=feature,aggfunc='mean')
            if 1 not in wide or 2 not in wide: continue
            pair=wide[[1,2]].dropna().to_numpy(float); n=len(pair)
            if n<10: continue
            icc=icc_absolute_single(pair); b=[]
            for _ in range(3000): b.append(icc_absolute_single(pair[rng.integers(0,n,size=n)]))
            ci=np.quantile(b,[.025,.975]); diff=pair[:,1]-pair[:,0]
            rows.append({'combine_position':pos,'measure':label,'players_with_both_attempts':n,
                         'attempt_2_minus_1_mean':float(diff.mean()),'typical_error':float(diff.std(ddof=1)/np.sqrt(2)),
                         'icc_absolute_agreement_single_attempt':float(icc),'icc_bootstrap_95pct_ci':[float(ci[0]),float(ci[1])],
                         'icc_mean_of_two_attempts':float(2*icc/(1+icc))})
    result={'design':'Attempts 1 and 2 paired within player and Combine position; ICC(A,1) absolute agreement; 3000 player-bootstrap draws; same-session estimates.',
            'attempt_rows':int(len(attempts)),'results':rows,
            'caveat':'Small position-feature samples, especially speed retention; bootstrap intervals conditional on the measured players; repeatability does not establish NFL transfer.'}
    (OUT/'forty_repeatability_by_position_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT/'forty_repeatability_by_position.csv',index=False)
    lines=['# Does 40-yard phase repeatability differ by position?','',
           'Each athlete’s Combine 40-yard attempts 1 and 2 are paired. ICC(A,1) is two-way random-effects absolute agreement for one attempt; the player-bootstrap resamples athletes. ICC(A,2) is the estimated reliability of the average of two attempts. These are within-session estimates.','',
           '| Combine position | Measure | Players with both attempts | Typical error | ICC(A,1) (95% player-bootstrap CI) | ICC(A,2) |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        ci=r['icc_bootstrap_95pct_ci']; lines.append(f"| {r['combine_position']} | {r['measure']} | {r['players_with_both_attempts']} | {r['typical_error']:.3f} | {r['icc_absolute_agreement_single_attempt']:.3f} ({ci[0]:.3f}, {ci[1]:.3f}) | {r['icc_mean_of_two_attempts']:.3f} |")
    lines += ['',result['caveat'],'']
    (OUT/'forty_repeatability_by_position_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'results':rows},indent=2))

if __name__=='__main__': main()
