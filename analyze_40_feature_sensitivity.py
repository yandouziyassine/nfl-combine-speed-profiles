"""Stress-test 40-yard phase features under movement-onset and smoothing choices.

Recomputes the 40-yard traces at three onset thresholds and three median-filter
widths. Results are compared with the production definition (1 yd/s, 3 frames)
at player level. This evaluates measurement robustness, not NFL transfer.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_analysis as core

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
GRID = [(t, w) for t in [0.5, 1.0, 1.5] for w in [1, 3, 5]]
METRICS = {
    "early_acceleration": "early_accel_0_1s_ydps2",
    "late_acceleration": "late_accel_10_20yd_ydps2",
    "speed_retention": "speed_retention_20_40yd_ratio",
    "sustained_peak_speed": "peak_sustained_speed_ydps",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=core.DEFAULT_ARCHIVE)
    parser.add_argument("--reuse-player-profiles", action="store_true",
                        help="Reuse the sensitivity profiles already written to outputs/ after a run interrupted during summarization.")
    args = parser.parse_args()
    profile_path = OUT / "forty_feature_sensitivity_player_profiles.csv"
    if args.reuse_player_profiles and profile_path.exists():
        player_config = pd.read_csv(profile_path)
    else:
        source = core.DataDirectory(args.archive) if args.archive.is_dir() else zipfile.ZipFile(args.archive)
        attempt_keys = ["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt"]
        attempt_rows = []
        with source as archive:
            raw = core.read_csv(archive, "combine_tracking.csv")
            raw = raw.loc[raw.entity_type.eq("PLAYER") & raw.drill_name.eq("FORTY_YARD_DASH")].copy()
            raw["nfl_id"] = pd.to_numeric(raw.nfl_id, errors="coerce")
            grouped = raw.groupby(attempt_keys, sort=False, dropna=False)
            for key, trace in grouped:
                player = {k: v for k, v in zip(attempt_keys, key)}
                for threshold, width in GRID:
                    values = core.calculate_attempt(trace, onset_threshold=threshold, smoothing_frames=width)
                    row = {**player, "onset_threshold_ydps": threshold, "smoothing_frames": width}
                    row.update(values)
                    attempt_rows.append(row)
        attempts = pd.DataFrame(attempt_rows)
        profiles = pd.read_csv(OUT / "player_speed_signatures.csv", usecols=lambda c: c in {"nfl_id", "draft_year", "combine_position"})
        player_config = (attempts.groupby(["nfl_id", "draft_year", "onset_threshold_ydps", "smoothing_frames"])[list(METRICS.values())]
                         .mean().reset_index().merge(profiles, on=["nfl_id", "draft_year"], how="left", validate="many_to_one"))
        player_config.to_csv(profile_path, index=False)
    default_attempts = pd.read_csv(OUT / "combine_attempt_speed_profiles.csv", usecols=["event_id", "nfl_id", "drill_name"])
    attempt_count = int(default_attempts.loc[default_attempts.drill_name.eq("FORTY_YARD_DASH")].groupby(["event_id", "nfl_id"]).ngroups)
    baseline = player_config.loc[player_config.onset_threshold_ydps.eq(1.0) & player_config.smoothing_frames.eq(3)].set_index("nfl_id")
    rows = []
    for threshold, width in GRID:
        current = player_config.loc[player_config.onset_threshold_ydps.eq(threshold) & player_config.smoothing_frames.eq(width)].set_index("nfl_id")
        for label, feature in METRICS.items():
            common = baseline[[feature]].join(current[[feature]], how="inner", lsuffix="_base", rsuffix="_current").dropna()
            b, c = common[f"{feature}_base"], common[f"{feature}_current"]
            sd = float(b.std(ddof=1))
            # Rank then Pearson avoids requiring SciPy for Spearman correlation.
            rho = float(b.rank(method="average").corr(c.rank(method="average"))) if len(common) > 2 else np.nan
            quintile_base = pd.qcut(b.rank(method="first"), 5, labels=False, duplicates="drop")
            quintile_current = pd.qcut(c.rank(method="first"), 5, labels=False, duplicates="drop")
            top_base, top_current = quintile_base.eq(quintile_base.max()), quintile_current.eq(quintile_current.max())
            bottom_base, bottom_current = quintile_base.eq(quintile_base.min()), quintile_current.eq(quintile_current.min())
            rows.append({"metric": label, "onset_threshold_ydps": threshold, "smoothing_frames": width,
                         "players_with_measure": int(current[feature].notna().sum()),
                         "players_common_with_default": int(len(common)),
                         "spearman_rank_vs_default": rho,
                         "mean_abs_change_in_default_sd": float((b-c).abs().mean()/sd) if sd > 0 else np.nan,
                         "top_quintile_overlap": float((top_base & top_current).sum()/max(int(top_base.sum()),1)),
                         "bottom_quintile_overlap": float((bottom_base & bottom_current).sum()/max(int(bottom_base.sum()),1))})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "forty_feature_sensitivity_summary.csv", index=False)
    meta = {"production_definition": {"onset_threshold_ydps": 1.0, "smoothing_frames": 3},
            "grid": [{"onset_threshold_ydps": t, "smoothing_frames": w} for t,w in GRID],
            "attempts": attempt_count,
            "players": int(player_config.nfl_id.nunique()),
            "interpretation": "Rank and magnitude stability against the default feature definition; not an inferential performance test.",
            "results": rows}
    (OUT / "forty_feature_sensitivity_results.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Sensitivity of 40-yard phase features to signal processing", "",
             "This audit recomputes player-attempt features using onset thresholds of 0.5, 1.0, and 1.5 yd/s and centered median filters of 1, 3, and 5 frames. The production definition is 1 yd/s and 3 frames. Each setting is averaged to the player level and compared on athletes with a value under both settings. This measures rank/magnitude sensitivity to reasonable processing choices; it does not test NFL transfer.", "",
             f"Source coverage: {meta['attempts']} 40-yard attempts across {meta['players']} prospects. For the 20–40 yard retention feature, fewer players have complete distance crossings.", "",
             "| Metric | Threshold | Filter frames | Players common | Spearman rank vs default | Mean absolute shift (default SD) | Top quintile overlap | Bottom quintile overlap |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in results.iterrows():
        lines.append(f"| {r.metric} | {r.onset_threshold_ydps:.1f} yd/s | {int(r.smoothing_frames)} | {int(r.players_common_with_default)} | {r.spearman_rank_vs_default:.3f} | {r.mean_abs_change_in_default_sd:.3f} | {r.top_quintile_overlap:.2f} | {r.bottom_quintile_overlap:.2f} |")
    lines += ["", "An athlete can move between scouting quintiles when the feature definition changes, especially for noisy early-phase measures. High rank stability would support using the feature as a descriptive profile; it still would not establish predictive or causal value. The onset threshold marks detected movement, not reaction time, because cue timestamps are not present.", ""]
    (OUT / "forty_feature_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
