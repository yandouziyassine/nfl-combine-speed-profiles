"""Quantify same-session repeatability of tracked 40-yard dash traits.

Each athlete is the inference unit. Attempt 1 and attempt 2 are paired within
the Combine. ICC(A,1) estimates absolute agreement for one attempt; bootstrap
intervals resample athletes, not frames. This is measurement evidence, not a
test of NFL performance transfer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FEATURES = {
    "Early acceleration (0–1 s)": "early_accel_0_1s_ydps2",
    "Late acceleration (10–20 yd)": "late_accel_10_20yd_ydps2",
    "Speed retention (20–40 yd)": "speed_retention_20_40yd_ratio",
    "Time to 10 yd": "time_to_10yd_s",
    "Sustained peak speed": "peak_sustained_speed_ydps",
}


def icc_absolute_single(x: np.ndarray) -> float:
    """Two-way random effects, absolute agreement, single measure ICC(A,1)."""
    n, k = x.shape
    if n < 3 or k != 2:
        return float("nan")
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ss_rows = k * np.square(row_means - grand).sum()
    ss_cols = n * np.square(col_means - grand).sum()
    residual = x - row_means[:, None] - col_means[None, :] + grand
    ss_error = np.square(residual).sum()
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    return float((ms_rows - ms_error) / denom) if denom != 0 else float("nan")


def main() -> None:
    attempts = pd.read_csv(OUT / "combine_attempt_speed_profiles.csv")
    attempts = attempts.loc[attempts.drill_name.eq("FORTY_YARD_DASH") & attempts.attempt.isin([1, 2])]
    results = []
    rng = np.random.default_rng(20271027)
    for label, feature in FEATURES.items():
        wide = attempts.pivot_table(index="nfl_id", columns="attempt", values=feature, aggfunc="mean")
        if 1 not in wide or 2 not in wide:
            continue
        pair = wide[[1, 2]].dropna().to_numpy(float)
        n = len(pair)
        estimates = []
        for _ in range(3000):
            sample = pair[rng.integers(0, n, size=n)]
            estimates.append(icc_absolute_single(sample))
        icc = icc_absolute_single(pair)
        valid = np.asarray(estimates)
        valid = valid[np.isfinite(valid)]
        q_low, q_high = np.quantile(valid, [.025, .975])
        mean_two_icc = (2 * icc / (1 + icc)) if np.isfinite(icc) and (1 + icc) != 0 else float("nan")
        mean_two_ci = [(2 * q / (1 + q)) if (1 + q) != 0 else float("nan") for q in [q_low, q_high]]
        diff = pair[:, 1] - pair[:, 0]
        results.append({
            "measure": label,
            "feature": feature,
            "players_with_both_attempts": int(n),
            "attempt_2_minus_1_mean": float(np.mean(diff)),
            "paired_difference_sd": float(np.std(diff, ddof=1)),
            "typical_error": float(np.std(diff, ddof=1) / np.sqrt(2)),
            "pearson_attempt_correlation": float(np.corrcoef(pair.T)[0, 1]),
            "icc_absolute_agreement_single_attempt": float(icc),
            "icc_bootstrap_95pct_ci": [float(q_low), float(q_high)],
            "icc_mean_of_two_attempts": float(mean_two_icc),
            "icc_mean_of_two_bootstrap_95pct_ci": [float(mean_two_ci[0]), float(mean_two_ci[1])],
        })

    summary = {"design": "Combine 40-yard dash attempt 1 vs attempt 2; paired within athlete",
               "n_attempts": int(len(attempts)),
               "players_with_two_attempt_rows": int(attempts.groupby("nfl_id").filter(lambda x: x.attempt.nunique() == 2).nfl_id.nunique()),
               "reliability_estimand": "ICC(A,1), two-way random effects, absolute agreement; same-session repeats",
               "bootstrap_unit": "athlete, 3000 resamples",
               "interpretation_limit": "Repeatability within one Combine session does not establish between-day reliability, predictive value, or coaching causality.",
               "measures": results}
    (OUT / "combine_40_repeatability_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(results).to_csv(OUT / "combine_40_repeatability.csv", index=False)

    report = ["# Are the 40-yard speed phases repeatable?", "",
              "The competition provides up to two tracked 40-yard attempts per prospect. This audit asks whether the derived measures agree across attempts before interpreting small differences as player traits. Attempt 1 and attempt 2 are paired within athlete. ICC(A,1) is a two-way random-effects, absolute-agreement intraclass correlation for one attempt; uncertainty resamples athletes. Typical error is the standard deviation of paired differences divided by √2.", "",
              f"The file contains {len(attempts)} tracked attempts, including {summary['players_with_two_attempt_rows']} prospects with both attempts.", "",
              "| Measure | n paired | Attempt 2 − 1 mean | Typical error | ICC(A,1) (95% player-bootstrap CI) | ICC(A,2) of 2-attempt mean |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        ci = row["icc_bootstrap_95pct_ci"]
        report.append(f"| {row['measure']} | {row['players_with_both_attempts']} | {row['attempt_2_minus_1_mean']:+.3f} | {row['typical_error']:.3f} | {row['icc_absolute_agreement_single_attempt']:.3f} ({ci[0]:.3f}, {ci[1]:.3f}) | {row['icc_mean_of_two_attempts']:.3f} |")
    report += ["", "A high ICC means athletes retain similar relative standing across attempts and the attempt-level score agrees reasonably well in absolute terms. ICC depends on between-athlete variation; typical error remains in the measure's units. A systematic attempt-2 shift can reflect warm-up, fatigue, or drill order and is not automatically noise. These are same-session repeats, not separate-day reliability. Repeatability alone does not show that a metric predicts NFL performance.", "",
               "Movement onset in this dataset is detected from the first two consecutive frames above a speed threshold. It is not reaction time because no cue timestamp is supplied.", ""]
    (OUT / "combine_40_repeatability_report.md").write_text("\n".join(report), encoding="utf-8")

    # Minimal self-contained SVG avoids a plotting-library dependency.
    width, height = 1000, 470
    left, right, top, row_h = 290, 940, 100, 62
    sx = lambda value: left + (value + .05) / 1.05 * (right - left)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#fbfcfc"/>',
                '<text x="40" y="42" font-family="Arial,sans-serif" font-size="23" font-weight="700" fill="#17323a">Do 40-yard movement traits repeat across attempts?</text>',
                '<text x="40" y="68" font-family="Arial,sans-serif" font-size="13" fill="#587078">Single-attempt ICC(A,1), absolute agreement · 95% athlete-bootstrap intervals</text>']
    for tick in [0, .2, .4, .6, .8, 1.0]:
        x = sx(tick)
        elements.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{top+row_h*len(results)-12}" stroke="#dce5e5"/>')
        elements.append(f'<text x="{x:.1f}" y="{top+row_h*len(results)+10}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#587078">{tick:.1f}</text>')
    xref = sx(.7)
    elements.append(f'<line x1="{xref:.1f}" y1="{top-14}" x2="{xref:.1f}" y2="{top+row_h*len(results)-12}" stroke="#b7643d" stroke-width="2" stroke-dasharray="6,5"/>')
    for i, row in enumerate(results):
        y = top + i * row_h + 16
        value = row["icc_absolute_agreement_single_attempt"]
        lo, hi = row["icc_bootstrap_95pct_ci"]
        elements.append(f'<text x="40" y="{y+4}" font-family="Arial,sans-serif" font-size="14" fill="#17323a">{row["measure"]}</text>')
        elements.append(f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="#70b8b5" stroke-width="5" stroke-linecap="round"/>')
        elements.append(f'<circle cx="{sx(value):.1f}" cy="{y}" r="7" fill="#147d82"/>')
        elements.append(f'<text x="{right+8}" y="{y+4}" font-family="Arial,sans-serif" font-size="12" fill="#17323a">{value:.2f} (n={row["players_with_both_attempts"]})</text>')
    elements += [f'<text x="{left}" y="{height-34}" font-family="Arial,sans-serif" font-size="12" fill="#587078">Lower agreement</text>',
                 f'<text x="{right}" y="{height-34}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#587078">Higher agreement</text>',
                 f'<text x="{xref+8:.1f}" y="{height-55}" font-family="Arial,sans-serif" font-size="11" fill="#9c4a2d">0.70 reference; not a universal cutoff</text>',
                 '<text x="40" y="450" font-family="Arial,sans-serif" font-size="11" fill="#587078">Same Combine session; repeatability does not establish NFL predictive value or between-day reliability.</text>', '</svg>']
    (OUT / "combine_40_repeatability.svg").write_text("\n".join(elements), encoding="utf-8")
    print(json.dumps({"report": str(OUT / "combine_40_repeatability_report.md"), "results": results}, indent=2))


if __name__ == "__main__":
    main()
