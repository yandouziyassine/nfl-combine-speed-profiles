"""Compare traveled-path and course-aligned distance for 40-yard phase features."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_analysis import DEFAULT_ARCHIVE, interpolate_at, open_data_source, parse_sensor_times

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def interp_at(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if len(x) < 2 or target < x[0] or target > x[-1] or np.any(np.diff(x) <= 0):
        return float("nan")
    return float(np.interp(target, x, y))


def forward_features(group: pd.DataFrame) -> dict[str, float]:
    g = group.sort_values("time")
    t0 = parse_sensor_times(g.time)
    t = (t0 - t0.iloc[0]).dt.total_seconds().to_numpy(float)
    xy = g[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    speed = pd.to_numeric(g.s, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(xy).all(axis=1) & np.isfinite(speed)
    t, xy, speed = t[valid], xy[valid], speed[valid]
    if len(t) < 5 or np.any(np.diff(t) <= 0):
        return {}
    smooth = pd.Series(speed).rolling(3, center=True, min_periods=1).median().to_numpy()
    moving = (smooth >= 1.0) & np.r_[smooth[1:] >= 1.0, False]
    starts = np.flatnonzero(moving)
    if not len(starts):
        return {}
    start = int(starts[0])
    rel_t = t[start:] - t[start]
    dx = float(xy[-1, 0] - xy[start, 0])
    if abs(dx) < 0.25:
        return {}
    # Raw traces show a shared major course axis aligned with x. Use signed x
    # progress; estimate principal-component yaw only as a data-driven alignment check.
    sign = float(np.sign(dx))
    forward = (xy[start:, 0] - xy[start, 0]) * sign
    # The straight sprint axis is expected to progress monotonically. Enforce that
    # tiny coordinate jitter cannot make the distance interpolation nonmonotone.
    forward = np.maximum.accumulate(forward)
    trajectory = xy[start:] - xy[start]
    covariance = np.cov(trajectory.T)
    pca_yaw = 0.5 * np.degrees(np.arctan2(2 * covariance[0, 1], covariance[0, 0] - covariance[1, 1]))
    out: dict[str, float] = {"pca_yaw_deg_mod_180": float(pca_yaw)}
    s_at: dict[int, float] = {}
    t_at: dict[int, float] = {}
    for yards in (10, 20, 40):
        t_at[yards] = interp_at(forward, rel_t, yards)
        s_at[yards] = interp_at(forward, smooth[start:], yards)
    for key, value in {"t10": t_at[10], "t20": t_at[20], "t40": t_at[40],
                       "s10": s_at[10], "s20": s_at[20], "s40": s_at[40]}.items():
        out[key] = value
    out["early_accel_0_10yd"] = (s_at[10] - smooth[start]) / t_at[10] if np.isfinite(t_at[10]) and t_at[10] > 0 and np.isfinite(s_at[10]) else np.nan
    out["late_accel_10_20yd"] = (s_at[20] - s_at[10]) / (t_at[20] - t_at[10]) if np.isfinite(t_at[20] - t_at[10]) and t_at[20] > t_at[10] and np.isfinite(s_at[20] + s_at[10]) else np.nan
    out["speed_retention_20_40yd"] = s_at[40] / s_at[20] if np.isfinite(s_at[40] + s_at[20]) and s_at[20] > 0 else np.nan
    return out


def path_features(group: pd.DataFrame) -> dict[str, float]:
    """Recompute the former cumulative-path yard-marker definition for comparison."""
    g = group.sort_values("time")
    t0 = parse_sensor_times(g.time)
    t = (t0 - t0.iloc[0]).dt.total_seconds().to_numpy(float)
    xy = g[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    speed = pd.to_numeric(g.s, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(xy).all(axis=1) & np.isfinite(speed)
    t, xy, speed = t[valid], xy[valid], speed[valid]
    if len(t) < 4:
        return {}
    smooth = pd.Series(speed).rolling(3, center=True, min_periods=1).median().to_numpy()
    moving = (smooth >= 1.0) & np.r_[smooth[1:] >= 1.0, False]
    starts = np.flatnonzero(moving)
    if not len(starts):
        return {}
    start = int(starts[0])
    rel_t = t[start:] - t[start]
    rel_speed = smooth[start:]
    path = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    rel_path = path[start:] - path[start]
    out = {}
    for yards in (10, 20, 40):
        out[f"t{yards}"] = interp_at(rel_path, rel_t, yards)
        out[f"s{yards}"] = interp_at(rel_path, rel_speed, yards)
    out["early_accel_0_10yd"] = (out["s10"] - rel_speed[0]) / out["t10"] if np.isfinite(out["t10"]) and out["t10"] > 0 and np.isfinite(out["s10"]) else np.nan
    out["late_accel_10_20yd"] = (out["s20"] - out["s10"]) / (out["t20"] - out["t10"]) if np.isfinite(out["t20"] - out["t10"]) and out["t20"] > out["t10"] and np.isfinite(out["s20"] + out["s10"]) else np.nan
    out["speed_retention_20_40yd"] = out["s40"] / out["s20"] if np.isfinite(out["s40"] + out["s20"]) and out["s20"] > 0 else np.nan
    return out


def load_attempts(archive: Path) -> pd.DataFrame:
    with open_data_source(archive) as zf:
        names = [n for n in zf.namelist() if n.endswith("combine_tracking.csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one combine_tracking.csv member; found {names}")
        with zf.open(names[0]) as stream:
            combine = pd.read_csv(stream)
    combine = combine.loc[combine.entity_type.eq("PLAYER") & combine.drill_name.eq("FORTY_YARD_DASH")].copy()
    keys = ["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt"]
    rows = []
    for key, group in combine.groupby(keys, sort=False, dropna=False):
        path = path_features(group)
        projected = forward_features(group)
        row = dict(zip(keys, key))
        path_names = {
            "time_to_10yd_s": "t10", "time_to_20yd_s": "t20", "time_to_40yd_s": "t40",
            "speed_at_10yd_ydps": "s10", "speed_at_20yd_ydps": "s20", "speed_at_40yd_ydps": "s40",
            "early_accel_0_10yd_ydps2": "early_accel_0_10yd",
            "late_accel_10_20yd_ydps2": "late_accel_10_20yd",
            "speed_retention_20_40yd_ratio": "speed_retention_20_40yd",
        }
        for metric, source in path_names.items():
            row[f"path_{metric}"] = path.get(source, np.nan)
        for metric, value in projected.items():
            row[f"forward_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    common = pd.concat([a, b], axis=1).dropna()
    return float(common.iloc[:, 0].rank().corr(common.iloc[:, 1].rank())) if len(common) > 2 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    attempts = load_attempts(args.archive)
    attempts.to_csv(OUT / "forty_forward_axis_attempt_sensitivity.csv", index=False)
    attempts["path_reaches_40"] = attempts.path_time_to_40yd_s.notna()
    attempts["projected_reaches_40"] = attempts.forward_t40.notna()
    attempts["path_only_40"] = attempts.path_reaches_40 & ~attempts.projected_reaches_40
    signatures_path = OUT / "player_speed_signatures.csv"
    if signatures_path.exists():
        positions = pd.read_csv(signatures_path, usecols=["nfl_id", "combine_position"]).drop_duplicates("nfl_id")
        attempts = attempts.merge(positions, on="nfl_id", how="left", validate="many_to_one")
    coverage_by_year = attempts.groupby("draft_year").agg(
        attempts=("nfl_id", "size"), path_40=("path_reaches_40", "sum"),
        projected_40=("projected_reaches_40", "sum"), path_only_40=("path_only_40", "sum")
    ).astype(int).to_dict(orient="index")
    coverage_by_position = {}
    if "combine_position" in attempts:
        coverage_by_position = attempts.groupby("combine_position").agg(
            attempts=("nfl_id", "size"), path_40=("path_reaches_40", "sum"),
            projected_40=("projected_reaches_40", "sum"), path_only_40=("path_only_40", "sum")
        ).astype(int).to_dict(orient="index")
    phase_pairs = {
        "early_0_10yd_accel": ("path_early_accel_0_10yd_ydps2", "forward_early_accel_0_10yd"),
        "late_10_20yd_accel": ("path_late_accel_10_20yd_ydps2", "forward_late_accel_10_20yd"),
        "speed_retention_20_40yd": ("path_speed_retention_20_40yd_ratio", "forward_speed_retention_20_40yd"),
    }
    metrics = {}
    player = attempts.groupby("nfl_id", as_index=False)[[x for pair in phase_pairs.values() for x in pair]].mean()
    for name, (path_col, forward_col) in phase_pairs.items():
        pair = attempts[[path_col, forward_col]].dropna()
        metrics[name] = {
            "attempts_with_both": int(len(pair)),
            "pearson_correlation": float(pair[path_col].corr(pair[forward_col])) if len(pair) > 2 else None,
            "spearman_correlation": rank_corr(pair[path_col], pair[forward_col]),
            "median_forward_minus_path": float((pair[forward_col] - pair[path_col]).median()) if len(pair) else None,
            "p90_absolute_difference": float((pair[forward_col] - pair[path_col]).abs().quantile(.9)) if len(pair) else None,
            "player_mean_rank_correlation": rank_corr(player[path_col], player[forward_col]),
        }
    out = {
        "n_attempts": int(len(attempts)),
        "n_prospects": int(attempts.nfl_id.nunique()),
        "forty_yard_crossing_coverage": {
            "path_based_attempts_reaching_40yd": int(attempts.path_reaches_40.sum()),
            "projected_attempts_reaching_40yd": int(attempts.projected_reaches_40.sum()),
            "path_only_crossings": int(attempts.path_only_40.sum()),
            "both_axes_reach_40yd": int((attempts.path_reaches_40 & attempts.projected_reaches_40).sum()),
            "by_draft_year": coverage_by_year,
            "by_combine_position": coverage_by_position,
        },
        "axis_estimation": "signed x-coordinate progress; sign inferred from the net x displacement. Full-trace PCA major-axis yaw checks alignment with the shared x-axis.",
        "reference": "existing cumulative path length calculated by calculate_attempt",
        "metrics": metrics,
        "pca_alignment_by_cohort": {},
        "caveat": "The x-axis is strongly supported as the shared major coordinate axis by raw trajectory PCA, but the competition schema does not provide an explicit course-axis calibration. This is a coordinate-axis sensitivity, not proof that x is the official measured dash axis. The primary early acceleration feature is time-indexed over the first second and is unchanged.",
    }
    yaw = attempts.groupby("draft_year").forward_pca_yaw_deg_mod_180.agg(
        n="count", median="median", p10=lambda x: float(x.quantile(.1)), p90=lambda x: float(x.quantile(.9))
    )
    out["pca_alignment_by_cohort"] = {str(int(k)): {name: (int(value) if name == "n" else float(value)) for name, value in row.items()} for k, row in yaw.to_dict(orient="index").items()}
    (OUT / "forty_forward_axis_sensitivity.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Distance-axis sensitivity for 40-yard phases", "",
        "Forward distance is signed x-coordinate progress; direction sign is inferred from net movement. Raw full-trace PCA is used to check whether the shared course direction aligns with x. The reference uses cumulative x/y path length, as in the main pipeline.",
        "", "| Feature | Attempts | Attempt Spearman ρ | Player-mean Spearman ρ | Median projected − path | 90th percentile absolute difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        lines.append(f"| {name} | {metric['attempts_with_both']} | {metric['spearman_correlation']:.3f} | {metric['player_mean_rank_correlation']:.3f} | {metric['median_forward_minus_path']:+.4f} | {metric['p90_absolute_difference']:.4f} |")
    pca = out["pca_alignment_by_cohort"]
    lines += ["", "Full-trace PCA yaw relative to the x-axis (degrees; median [p10, p90]): " + "; ".join(
        f"{year}: {vals['median']:+.2f} [{vals['p10']:+.2f}, {vals['p90']:+.2f}] (n={vals['n']})" for year, vals in pca.items()
    ) + "."]
    cov = out["forty_yard_crossing_coverage"]
    lines += ["", f"40-yard crossings: {cov['path_based_attempts_reaching_40yd']} path-based versus {cov['projected_attempts_reaching_40yd']} projected; {cov['path_only_crossings']} attempts reach 40 only under cumulative path length.", "", out["caveat"], ""]
    (OUT / "forty_forward_axis_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
