"""Same-session repeatability of DB backpedal-transition movement features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_40_repeatability import icc_absolute_single

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"
DRILL="BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"
FEATURES={"Early acceleration (0–1 s)":"early_accel_0_1s_ydps2",
          "Sustained peak speed":"peak_sustained_speed_ydps",
          "Absolute path-relative lateral speed gap":"left_right_lateral_speed_difference_ydps"}


def main():
    attempts=pd.read_csv(OUT/"combine_attempt_speed_profiles.csv")
    player=pd.read_csv(OUT/"player_speed_signatures.csv",usecols=["nfl_id","combine_position"])
    attempts=attempts.loc[attempts.drill_name.eq(DRILL)].merge(player,on="nfl_id",how="left",validate="many_to_one")
    attempts=attempts.loc[attempts.combine_position.eq("DB")].copy()
    attempts=attempts.sort_values(["nfl_id","attempt","event_id"])
    attempts["repeat_index"]=attempts.groupby("nfl_id").cumcount()+1
    results=[]; rng=np.random.default_rng(20271029)
    for label,feature in FEATURES.items():
        x=attempts[feature].abs() if feature=="left_right_lateral_speed_difference_ydps" else attempts[feature]
        tab=attempts.assign(value=x).pivot_table(index="nfl_id",columns="repeat_index",values="value",aggfunc="mean")
        valid=tab.reindex(columns=[1,2]).dropna()
        pairs=valid.to_numpy(float); n=len(pairs)
        draws=[]
        for _ in range(3000):
            s=pairs[rng.integers(0,n,size=n)]
            draws.append(icc_absolute_single(s))
        ci=np.quantile(np.asarray(draws),[.025,.975])
        diff=pairs[:,1]-pairs[:,0]
        results.append({"measure":label,"feature":feature,"players_with_two_valid_attempts":n,
                        "attempt_labels":sorted(int(v) for v in valid.columns),
                        "attempt_2_minus_1_mean":float(diff.mean()),"typical_error":float(diff.std(ddof=1)/np.sqrt(2)),
                        "icc_absolute_agreement_single_attempt":float(icc_absolute_single(pairs)),
                        "icc_bootstrap_95pct_ci":[float(ci[0]),float(ci[1])],
                        "icc_mean_of_two_attempts":float(2*icc_absolute_single(pairs)/(1+icc_absolute_single(pairs)))})
    out={"drill":DRILL,"combine_position":"DB","attempt_rows":int(len(attempts)),"players":int(attempts.nfl_id.nunique()),
         "design":"Within-player repeated attempts 3 vs 4; two-way random-effects absolute-agreement ICC(A,1); 3000 player-bootstrap resamples",
         "caveat":"Same Combine session; path-relative lateral gap is not anatomical left/right. Repeatability does not establish football transfer.","measures":results}
    (OUT/"db_backpedal_repeatability_results.json").write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(results).to_csv(OUT/"db_backpedal_repeatability.csv",index=False)
    lines=["# Are DB backpedal-transition features repeatable?","",
           "This audit checks the movement features used in the rookie DB tackle-rate candidate. Combine attempts are paired within player; the supplied attempt labels for this drill are 3 and 4. ICC(A,1) estimates absolute agreement for one attempt. Uncertainty resamples players. The lateral gap is the absolute path-relative left-minus-right lateral-speed difference when both directions are present within an attempt; it does not identify anatomical laterality.","",
           f"Coverage: {len(attempts)} backpedal-transition attempts from {attempts.nfl_id.nunique()} Combine DBs.","",
           "| Feature | Players with both valid attempts | Attempt 2 − 1 mean | Typical error | ICC(A,1) (95% player-bootstrap CI) | ICC(A,2) |","|---|---:|---:|---:|---:|---:|"]
    for r in results:
        ci=r['icc_bootstrap_95pct_ci']
        lines.append(f"| {r['measure']} | {r['players_with_two_valid_attempts']} | {r['attempt_2_minus_1_mean']:+.3f} | {r['typical_error']:.3f} | {r['icc_absolute_agreement_single_attempt']:.3f} ({ci[0]:.3f}, {ci[1]:.3f}) | {r['icc_mean_of_two_attempts']:.3f} |")
    lines += ["","These are same-session repeats, not separate-day reliability. The gap feature additionally requires both path-relative directions within each attempt, which reduces its repeat-pair sample. A reliable score is not necessarily useful for predicting NFL performance.",""]
    (OUT/"db_backpedal_repeatability_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"players":out['players'],"results":results},indent=2))


if __name__=="__main__": main()
