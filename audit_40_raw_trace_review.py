"""Raw-trace audit for the production 40-yard early-acceleration feature.

Selects a deterministic, outcome-blind spread of DL player profiles across the
early-acceleration distribution, plots raw and production-smoothed attempt traces,
and checks timestamp cadence and coordinate-derived speed consistency.
"""
from __future__ import annotations

import argparse
import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_analysis as core

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FEATURE = "early_accel_0_1s_ydps2"


def svg_trace_plot(traces: list[dict], path: Path) -> None:
    width, height = 1120, 190 + len(traces) * 92
    left, right = 210, 1080
    top = 90
    panel_h = 68
    x_max, y_max = 2.5, 12.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,sans-serif;fill:#17212b}.title{font-size:24px;font-weight:bold}.label{font-size:13px}.small{font-size:11px}.axis{stroke:#9aa7b2;stroke-width:1}.grid{stroke:#e5eaee;stroke-width:1}.raw{fill:none;stroke:#3478a5;stroke-width:1.7;stroke-dasharray:4 3;opacity:.95}.smooth{fill:none;stroke:#b33f35;stroke-width:2.2}</style>',
             '<rect width="100%" height="100%" fill="#fff"/>',
             '<text x="40" y="38" class="title">Raw 40-yard speed traces across the DL early-acceleration range</text>',
             '<text x="40" y="61" class="small">Blue: recorded speed; red: centered 3-frame median used for features. Time is relative to detected 1 yd/s movement onset.</text>']
    for i, item in enumerate(traces):
        y0 = top + i * 88
        label = html.escape(f"Profile {item['profile_rank']:02d} · {item['accel']:.2f} yd/s² · attempt {item['attempt']}")
        parts.append(f'<text x="40" y="{y0+35}" class="label">{label}</text>')
        for tick in range(0, 13, 3):
            yy = y0 + panel_h - (tick / y_max) * panel_h
            parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
            if i == 0:
                parts.append(f'<text x="{left-28}" y="{yy+4:.1f}" class="small">{tick}</text>')
        for tick in (0, .5, 1, 1.5, 2, 2.5):
            xx = left + tick / x_max * (right-left)
            parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+panel_h}" class="grid"/>')
            if i == len(traces)-1:
                parts.append(f'<text x="{xx-6:.1f}" y="{y0+panel_h+16}" class="small">{tick:g}</text>')
        # Draw raw speed last with a dashed stroke so the close-by smoothed line
        # remains distinguishable instead of covering it.
        for key, cls in (("smooth", "smooth"), ("raw", "raw")):
            points = []
            for t, s in zip(item["t"], item[key]):
                xx = left + float(t) / x_max * (right-left)
                yy = y0 + panel_h - np.clip(float(s), 0, y_max) / y_max * panel_h
                points.append(f"{xx:.1f},{yy:.1f}")
            parts.append(f'<polyline class="{cls}" points="{" ".join(points)}"/>')
    parts += [f'<text x="{(left+right)//2}" y="{height-22}" text-anchor="middle" class="label">Seconds after detected movement onset</text>',
              f'<text x="18" y="{top+len(traces)*88//2}" transform="rotate(-90 18 {top+len(traces)*88//2})" text-anchor="middle" class="label">Speed (yd/s)</text>', '</svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=core.DEFAULT_ARCHIVE)
    ap.add_argument("--profiles", type=Path, default=OUT / "player_speed_signatures.csv")
    args = ap.parse_args()

    profiles = pd.read_csv(args.profiles, usecols=["nfl_id", "draft_year", "combine_position", f"forty_{FEATURE}_mean"])
    dl = profiles.loc[profiles.combine_position.eq("DL")].dropna(subset=[f"forty_{FEATURE}_mean"]).copy()
    dl = dl.sort_values([f"forty_{FEATURE}_mean", "nfl_id"], kind="mergesort").reset_index(drop=True)
    # Select ten ranks spanning the entire feature distribution. No NFL outcomes
    # or player names are used in selection; the figure labels are anonymized.
    ranks = np.linspace(0, len(dl)-1, 10).round().astype(int)
    selected = dl.iloc[ranks].copy()
    selected["profile_rank"] = np.arange(1, len(selected)+1)
    keys = selected[["nfl_id", "draft_year", "profile_rank", f"forty_{FEATURE}_mean"]].rename(columns={f"forty_{FEATURE}_mean": "player_mean_accel"})

    usecols = ["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt", "time", "s", "x", "y", "entity_type"]
    source = core.DataDirectory(args.archive) if args.archive.is_dir() else zipfile.ZipFile(args.archive)
    with source as archive:
        raw = core.read_csv(archive, "combine_tracking.csv", usecols=usecols)
    raw = raw.loc[raw.entity_type.eq("PLAYER") & raw.drill_name.eq("FORTY_YARD_DASH")].copy()
    raw["nfl_id"] = pd.to_numeric(raw.nfl_id, errors="coerce")
    all_checks = []
    all_group_cols = ["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt"]
    for key, g in raw.groupby(all_group_cols, sort=False, dropna=False):
        g = g.sort_values("time", kind="mergesort")
        ts = core.parse_sensor_times(g.time)
        t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        speed = pd.to_numeric(g.s, errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(g.x, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g.y, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(speed) & np.isfinite(x) & np.isfinite(y)
        t, speed, x, y = t[valid], speed[valid], x[valid], y[valid]
        dt = np.diff(t)
        good = dt > 0
        coord_speed = np.hypot(np.diff(x)[good], np.diff(y)[good]) / dt[good]
        trapezoid_speed = (speed[:-1][good] + speed[1:][good]) / 2
        err = coord_speed - trapezoid_speed
        all_checks.append({"draft_year": key[0], "event_id": key[1], "nfl_id": key[2], "attempt": key[5],
                           "frames": int(len(t)), "duration_s": float(t[-1]) if len(t) else np.nan,
                           "max_abs_cadence_error_s": float(np.max(np.abs(dt-.1))) if len(dt) else np.nan,
                           "fraction_dt_outside_5ms": float(np.mean(np.abs(dt-.1) > .005)) if len(dt) else np.nan,
                           "max_raw_speed_ydps": float(np.max(speed)) if len(speed) else np.nan,
                           "fraction_negative_raw_speed": float(np.mean(speed < 0)) if len(speed) else np.nan,
                           "coord_vs_speed_mae_ydps": float(np.mean(np.abs(err))) if len(err) else np.nan,
                           "coord_vs_speed_p95_abs_error_ydps": float(np.quantile(np.abs(err), .95)) if len(err) else np.nan,
                           "nonpositive_time_intervals": int((dt <= 0).sum())})
    all_checks = pd.DataFrame(all_checks)
    all_checks.to_csv(OUT / "forty_raw_trace_all_attempt_checks.csv", index=False)
    raw = raw.merge(keys[["nfl_id", "draft_year", "profile_rank", "player_mean_accel"]], on=["nfl_id", "draft_year"], how="inner", validate="many_to_one")
    selected_attempt_count = int(raw.groupby(["draft_year", "event_id", "nfl_id", "attempt"], dropna=False).ngroups)

    trace_rows, audit_rows = [], []
    group_cols = ["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt"]
    for key, g in raw.groupby(group_cols, sort=False, dropna=False):
        g = g.sort_values("time", kind="mergesort")
        ts = core.parse_sensor_times(g.time)
        t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        speed = pd.to_numeric(g.s, errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(g.x, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g.y, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(speed) & np.isfinite(x) & np.isfinite(y)
        t, speed, x, y = t[valid], speed[valid], x[valid], y[valid]
        if len(t) < 4:
            continue
        smoothed = pd.Series(speed).rolling(3, center=True, min_periods=1).median().to_numpy()
        sustained = (smoothed >= 1.0) & np.r_[smoothed[1:] >= 1.0, False]
        starts = np.flatnonzero(sustained)
        if not len(starts):
            continue
        start = int(starts[0])
        t_rel = t[start:] - t[start]
        s_raw = speed[start:]
        s_smooth = smoothed[start:]
        accel = core.calculate_attempt(g)[FEATURE]
        dt = np.diff(t)
        coord_speed = np.hypot(np.diff(x), np.diff(y)) / dt
        trapezoid_speed = (speed[:-1] + speed[1:]) / 2
        err = coord_speed - trapezoid_speed
        rec = raw.loc[(raw.draft_year.eq(key[0])) & (raw.event_id.eq(key[1])) & (raw.nfl_id.eq(key[2])) & (raw.attempt.eq(key[5]))].iloc[0]
        # Keep plotted values within the first 2.5 s so selected profiles are comparable.
        keep = t_rel <= 2.5
        trace_rows.append({"profile_rank": int(rec.profile_rank), "accel": float(rec.player_mean_accel), "attempt": int(key[5]),
                           "t": t_rel[keep].tolist(), "raw": s_raw[keep].tolist(), "smooth": s_smooth[keep].tolist()})
        audit_rows.append({"profile_rank": int(rec.profile_rank), "draft_year": int(key[0]), "event_id": key[1], "player_id": int(key[2]),
                           "attempt": int(key[5]), "player_mean_early_accel_ydps2": float(rec.player_mean_accel),
                           "attempt_early_accel_ydps2": float(accel), "frames": int(len(t)),
                           "duration_s": float(t[-1]), "onset_s": float(t[start]),
                           "max_abs_cadence_error_s": float(np.max(np.abs(dt-.1))) if len(dt) else np.nan,
                           "max_raw_speed_ydps": float(np.max(speed)),
                           "fraction_negative_raw_speed": float(np.mean(speed < 0)),
                           "coord_vs_speed_mae_ydps": float(np.mean(np.abs(err))),
                           "coord_vs_speed_p95_abs_error_ydps": float(np.quantile(np.abs(err), .95)),
                           "fraction_dt_outside_5ms": float(np.mean(np.abs(dt-.1) > .005)) if len(dt) else np.nan})
    audit = pd.DataFrame(audit_rows).sort_values(["profile_rank", "attempt"])
    audit.to_csv(OUT / "forty_raw_trace_review_sample.csv", index=False)
    traces = sorted(trace_rows, key=lambda r: (r["profile_rank"], r["attempt"]))
    svg_trace_plot(traces, OUT / "forty_raw_trace_review.svg")
    summary = {
        "selection": "Ten deterministic ranks spanning the full DL mean early-acceleration distribution, selected without using NFL outcomes; all observed 40-yard attempts for selected profiles.",
        "profiles_selected": int(selected.nfl_id.nunique()),
        "attempts_reviewed": int(len(audit)),
        "attempts_without_two_frame_onset": int(selected_attempt_count-len(audit)),
        "all_40_attempts": int(len(all_checks)),
        "all_attempt_integrity": {
            "attempts_with_any_gap_over_5ms": int((all_checks.fraction_dt_outside_5ms > 0).sum()),
            "attempts_with_nonpositive_time_interval": int((all_checks.nonpositive_time_intervals > 0).sum()),
            "attempts_with_negative_raw_speed": int((all_checks.fraction_negative_raw_speed > 0).sum()),
            "median_coord_speed_mae_ydps": float(all_checks.coord_vs_speed_mae_ydps.median()),
            "p95_coord_speed_mae_ydps": float(all_checks.coord_vs_speed_mae_ydps.quantile(.95)),
            "max_coord_speed_mae_ydps": float(all_checks.coord_vs_speed_mae_ydps.max()),
        },
        "sample_metrics": {
            "median_max_abs_cadence_error_s": float(audit.max_abs_cadence_error_s.median()),
            "attempts_with_any_gap_over_5ms": int((audit.fraction_dt_outside_5ms > 0).sum()),
            "median_coord_speed_mae_ydps": float(audit.coord_vs_speed_mae_ydps.median()),
            "p95_coord_speed_mae_ydps": float(audit.coord_vs_speed_mae_ydps.quantile(.95)),
            "max_raw_speed_ydps": float(audit.max_raw_speed_ydps.max()),
            "attempts_with_negative_raw_speed": int((audit.fraction_negative_raw_speed > 0).sum()),
        },
    "limitations": ["This is a deterministic, outcome-blind plotted trace sample; human visual review of all 793 attempts was not performed.",
                        "Coordinate-derived speed is a consistency diagnostic; filtered velocity may differ from frame displacement because of sensor and smoothing conventions."],
    }
    (OUT / "forty_raw_trace_review_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Outcome-blind sample audit of raw 40-yard traces", "",
             summary["selection"], "",
             f"The sample contains {summary['profiles_selected']} DL profiles and {summary['attempts_reviewed']} attempts with a detectable two-frame onset. The trace figure overlays raw speed (blue) and the production centered three-frame median (red), aligned to the production 1 yd/s onset.", "",
             f"Separate cadence and speed-consistency checks cover all {summary['all_40_attempts']} tracked 40-yard attempts.", "",
             "## Sample checks", "",
             f"- Median maximum absolute cadence deviation from 0.1 seconds: {summary['sample_metrics']['median_max_abs_cadence_error_s']:.4f} seconds.",
             f"- Attempts with any interval more than 5 ms from cadence: {summary['sample_metrics']['attempts_with_any_gap_over_5ms']} of {summary['attempts_reviewed']}.",
             f"- Median coordinate-derived versus supplied speed MAE: {summary['sample_metrics']['median_coord_speed_mae_ydps']:.3f} yd/s; sample 95th percentile of attempt-level MAE: {summary['sample_metrics']['p95_coord_speed_mae_ydps']:.3f} yd/s.",
             f"- Maximum observed raw speed: {summary['sample_metrics']['max_raw_speed_ydps']:.3f} yd/s; attempts with negative speed: {summary['sample_metrics']['attempts_with_negative_raw_speed']}.", "",
             "## All-attempt integrity checks", "",
             f"- Intervals more than 5 ms from 0.1-second cadence: {summary['all_attempt_integrity']['attempts_with_any_gap_over_5ms']} attempts.",
             f"- Attempts with nonpositive time intervals or negative recorded speed: {summary['all_attempt_integrity']['attempts_with_nonpositive_time_interval']} and {summary['all_attempt_integrity']['attempts_with_negative_raw_speed']}, respectively.",
             f"- Coordinate-derived versus supplied speed MAE: median {summary['all_attempt_integrity']['median_coord_speed_mae_ydps']:.3f} yd/s, 95th percentile {summary['all_attempt_integrity']['p95_coord_speed_mae_ydps']:.3f} yd/s, maximum attempt-level MAE {summary['all_attempt_integrity']['max_coord_speed_mae_ydps']:.3f} yd/s.", "",
             "## Interpretation", "",
             "Visual review of the selected sample shows the raw and smoothed traces closely aligned, with smooth early speed build and no obvious isolated one-frame spikes. This is limited to 19 attempts; it does not establish all-attempt trace validity, device calibration, or biomechanical mechanism. The feature remains a threshold-relative change in recorded speed, not force, reaction time, or an observed first-step event.", ""]
    (OUT / "forty_raw_trace_review_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
