"""Assemble a descriptive, position-aware speed signature for all 510 prospects.

No overall talent score is produced. Metrics remain separated by the test/drill
that generated them, and position percentiles are descriptive peer comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = {
    "forty_early_accel_0_1s_ydps2_mean": "Build",
    "forty_late_accel_10_20yd_ydps2_mean": "Build later in dash",
    "forty_peak_sustained_speed_ydps_mean": "Carry",
    "forty_speed_change_20_40yd_ydps_mean": "Maintain",
    "forty_speed_retention_20_40yd_ratio_mean": "Maintain",
    "short_shuttle_early_accel_0_1s_ydps2_mean": "Redirect",
    "short_shuttle_peak_sustained_speed_ydps_mean": "Redirect",
    "short_shuttle_braking_magnitude_ydps2": "Brake",
    "three_cone_early_accel_0_1s_ydps2_mean": "Redirect",
    "three_cone_peak_sustained_speed_ydps_mean": "Redirect",
    "three_cone_braking_magnitude_ydps2": "Brake",
    "short_shuttle_lateral_speed_balance_abs_ydps": "Directional balance",
    "three_cone_lateral_speed_balance_abs_ydps": "Directional balance",
}


def add_peer_scores(data: pd.DataFrame, feature: str) -> tuple[pd.Series, pd.Series]:
    z = pd.Series(np.nan, index=data.index, dtype=float)
    pct = pd.Series(np.nan, index=data.index, dtype=float)
    for _, idx in data.groupby("combine_position", dropna=False).groups.items():
        x = pd.to_numeric(data.loc[idx, feature], errors="coerce")
        valid = x.dropna()
        if len(valid) < 3:
            continue
        med = float(valid.median())
        mad = float((valid - med).abs().median())
        scale = 1.4826 * mad
        if scale == 0:
            scale = float(valid.std(ddof=0))
        if scale > 0:
            z.loc[valid.index] = (valid - med) / scale
        # Neutral empirical rank: this is a measurement comparison, not a
        # desirability score. Interpretation depends on role and outcome.
        pct.loc[valid.index] = valid.rank(method="average", pct=True, ascending=True)
    return z.clip(-4, 4), pct


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    base = pd.read_csv(out / "all_player_speed_profiles.csv")
    drill = pd.read_csv(out / "player_drill_speed_profiles.csv")
    # Preserve official position and outcome fields but add explicit movement-domain summaries.
    short = drill.loc[drill.drill_name.eq("SHORT_SHUTTLE")].set_index("nfl_id")
    cone = drill.loc[drill.drill_name.eq("THREE_CONE_DRILL")].set_index("nfl_id")
    additions = {}
    for src, dst in [(short, "short_shuttle"), (cone, "three_cone")]:
        additions[f"{dst}_early_accel_0_1s_ydps2_mean"] = base.nfl_id.map(src.mean_accel_first_1s_ydps2_mean)
        additions[f"{dst}_peak_sustained_speed_ydps_mean"] = base.nfl_id.map(src.peak_sustained_speed_ydps_mean)
        additions[f"{dst}_braking_magnitude_ydps2"] = base.nfl_id.map(-pd.to_numeric(src.peak_braking_ydps2_mean, errors="coerce"))
        additions[f"{dst}_lateral_speed_balance_abs_ydps"] = base.nfl_id.map(src.left_right_lateral_speed_difference_by_drill_ydps.abs())
    base = pd.concat([base, pd.DataFrame(additions, index=base.index)], axis=1)

    skill_counts = (drill.loc[drill.drill_type.astype(str).str.startswith("SKILL_DRILLS_")]
                    .groupby("nfl_id").drill_name.nunique())
    base = pd.concat([base, pd.DataFrame({
        "forty_tracked_attempts": base["forty_trace_duration_s_count"].fillna(0).astype(int),
        "tracked_skill_drill_count": base.nfl_id.map(skill_counts).fillna(0).astype(int),
    }, index=base.index)], axis=1)

    score_info = {}
    score_columns = {}
    for feature, domain in FEATURES.items():
        if feature not in base:
            base = pd.concat([base, pd.Series(np.nan, index=base.index, name=feature)], axis=1)
        z, pct = add_peer_scores(base, feature)
        score_columns[f"{feature}_position_robust_z"] = z
        score_columns[f"{feature}_position_percentile"] = pct
        score_info[feature] = {"domain": domain,
                               "percentile_interpretation": "Higher percentile means a numerically higher measurement, not better performance.",
                               "football_interpretation": "Context-dependent; evaluate with position, drill, role, and NFL outcome.",
                               "n_available": int(base[feature].notna().sum()),
                               "available_by_position": {str(k): int(v) for k, v in base.groupby("combine_position")[feature].count().items()}}
    base = pd.concat([base, pd.DataFrame(score_columns, index=base.index)], axis=1)
    base.to_csv(out / "player_speed_signatures.csv", index=False)
    coverage = {
        "prospect_rows": int(base.nfl_id.nunique()),
        "profile_note": "One row per supplied prospect. Missing drill measurements remain null. Raw values are retained; no cross-drill composite talent score is created.",
        "normalization": "Robust z-score and empirical percentile within Combine position across the supplied 2023-2025 cohorts. Percentiles rank the numeric measurement only; high or low is not inherently better and is not a performance probability.",
        "features": score_info,
    }
    (out / "speed_signature_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps({"players": coverage["prospect_rows"], "columns": len(base.columns),
                      "features_with_coverage": sum(v["n_available"] > 0 for v in score_info.values()),
                      "rows_by_position": {str(k): int(v) for k, v in base.groupby("combine_position").nfl_id.nunique().items()}}, indent=2))


if __name__ == "__main__":
    main()
