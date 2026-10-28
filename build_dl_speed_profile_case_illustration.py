"""Build an outcome-blind matched-pair illustration for the competition writeup."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CALIPERS = {"forty": 0.05, "ten_yd_split": 0.05, "combine_height": 1.0, "combine_weight": 10.0}
FEATURE = "forty_early_accel_0_1s_ydps2_mean"


def main() -> None:
    signatures = pd.read_csv(OUT / "player_speed_signatures.csv")
    oof = pd.read_csv(OUT / "dl_early_accel_predraft_oof_100plays.csv")
    oof = oof.loc[oof.alpha.eq(5.0)].drop_duplicates("nfl_id")
    oof_ids = set(oof.nfl_id.astype(int))
    oof_by_id = oof.set_index("nfl_id")
    # Keep outcomes and OOF predictions out of the pair-selection frame.
    d = signatures.loc[signatures.nfl_id.isin(oof_ids)].copy()
    d = d.loc[d.combine_position.eq("DL")].dropna(
        subset=["draft_year", "display_name", "forty", "ten_yd_split", "combine_height", "combine_weight", FEATURE,
               ]
    )

    # Choose the pair using only pre-draft traditional-test proximity and the
    # early-acceleration gap. Outcomes are joined for plotting after pair selection.
    pairs = []
    for _, cohort in d.groupby("draft_year"):
        records = cohort.to_dict("records")
        for a, b in itertools.combinations(records, 2):
            diffs = {column: abs(float(a[column]) - float(b[column])) for column in CALIPERS}
            if all(diffs[key] <= limit for key, limit in CALIPERS.items()):
                accel_gap = abs(float(a[FEATURE]) - float(b[FEATURE]))
                pairs.append((accel_gap, diffs, a, b))
    if not pairs:
        raise RuntimeError("No within-cohort DL pair meets the documented Combine-profile calipers.")
    gap, diffs, first, second = max(pairs, key=lambda item: item[0])
    if first[FEATURE] > second[FEATURE]:
        first, second = second, first

    pair_rows = []
    for player in (first, second):
        outcome = oof_by_id.loc[int(player["nfl_id"])]
        pair_rows.append({
            "nfl_id": int(player["nfl_id"]), "display_name": str(player["display_name"]),
            "draft_year": int(player["draft_year"]), "forty": float(player["forty"]),
            "ten_yd_split": float(player["ten_yd_split"]), "height_in": float(player["combine_height"]),
            "weight_lb": float(player["combine_weight"]), "early_acceleration_ydps2": float(player[FEATURE]),
            "observed_rookie_p90_full_play_peak_speed_ydps": float(outcome.actual),
            "oof_baseline_prediction_ydps": float(outcome.baseline),
            "oof_acceleration_model_prediction_ydps": float(outcome.candidate),
            "baseline_error_actual_minus_prediction": float(outcome.actual - outcome.baseline),
            "candidate_error_actual_minus_prediction": float(outcome.actual - outcome.candidate),
        })
    result = {
        "selection": "Among pairs from the same held-out draft cohort, require absolute differences <= 0.05 s in official 40, <= 0.05 s in 10-yard split, <= 1.0 in height, and <= 10 lb in weight; select the pair with the largest early-acceleration gap. NFL outcomes are not used in pair selection.",
        "model": f"100-play DL sample, alpha 5, leave-one-draft-cohort-out predictions; {int(first['draft_year'])} is the shared held-out cohort for the selected pair.",
        "acceleration_gap_ydps2": float(gap),
        "traditional_test_differences": diffs,
        "players": pair_rows,
        "interpretation": "Illustrative matched case only. The target is observed upper-tail full-play speed, not pass-rush success; selection is outcome-blind, but the pair is not independent validation and the broader model screen remains exploratory.",
    }
    (OUT / "dl_speed_profile_case_illustration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    fig = plt.figure(figsize=(12.2, 6.6), facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.6, 2.5], width_ratios=[1.08, 1.35], hspace=0.35, wspace=0.28)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.85, "Same stopwatch profile. Different early speed build.", fontsize=20, weight="bold", color="#10243a", va="top")
    title_ax.text(0.0, 0.22, f"Outcome-blind matched DL examples from the {int(first['draft_year'])} held-out cohort", fontsize=11, color="#516579", va="top")

    names = [p["display_name"] for p in (first, second)]
    colors = ["#4378a8", "#db8b2b"]
    feature_ax = fig.add_subplot(grid[1, 0])
    feature_ax.set_facecolor("white")
    feature_ax.axis("off")
    feature_ax.set_xlim(0, 1)
    feature_ax.set_ylim(1.35, 0)
    feature_ax.set_title("Combine profile", loc="left", pad=12, color="#10243a", weight="bold")
    feature_ax.axhspan(1.08, 1.30, color="#fff2dc", zorder=0)
    feature_ax.text(0.52, 0.02, names[0], ha="center", va="top", weight="bold", color="#10243a", fontsize=9.5)
    feature_ax.text(0.82, 0.02, names[1], ha="center", va="top", weight="bold", color="#10243a", fontsize=9.5)
    feature_rows = [
        ("Official 40", f"{first['forty']:.2f} s", f"{second['forty']:.2f} s"),
        ("10-yard split", f"{first['ten_yd_split']:.2f} s", f"{second['ten_yd_split']:.2f} s"),
        ("Height", f"{first['combine_height']:.2f} in", f"{second['combine_height']:.2f} in"),
        ("Weight", f"{first['combine_weight']:.0f} lb", f"{second['combine_weight']:.0f} lb"),
        ("Early acceleration", f"{first[FEATURE]:.2f} yd/s²", f"{second[FEATURE]:.2f} yd/s²"),
    ]
    row_ys = [0.27, 0.50, 0.73, 0.96, 1.19]
    for i, ((label, first_value, second_value), yy) in enumerate(zip(feature_rows, row_ys)):
        color = "#10243a" if i == 4 else "#334e68"
        weight = "bold" if i == 4 else "normal"
        feature_ax.text(0.02, yy, label, ha="left", va="center", color=color, weight=weight, fontsize=9.2)
        feature_ax.text(0.52, yy, first_value, ha="center", va="center", color=color, weight=weight, fontsize=9.2)
        feature_ax.text(0.82, yy, second_value, ha="center", va="center", color=color, weight=weight, fontsize=9.2)
        feature_ax.plot([0, 0.98], [yy + 0.115, yy + 0.115], color="#d9e2ec", lw=0.8)
    feature_ax.text(0.0, -0.02, f"Traditional-test gaps: 40 {diffs['forty']:.2f}s · split {diffs['ten_yd_split']:.2f}s · height {diffs['combine_height']:.2f}in · weight {diffs['combine_weight']:.0f}lb",
                    transform=feature_ax.transAxes, fontsize=8.7, color="#516579", va="top")

    outcome_ax = fig.add_subplot(grid[1, 1])
    outcome_ax.set_facecolor("white")
    outcome_ax.set_title("Held-out rookie full-play peak speed (yd/s)", loc="left", pad=12, color="#10243a", weight="bold")
    outcome_ax.set_xlim(5.45, 7.25)
    outcome_ax.set_ylim(1.5, -0.5)
    outcome_ax.set_yticks([0, 1], names)
    outcome_ax.set_xlabel("Player-level 90th percentile of per-play peak speed")
    outcome_ax.grid(axis="x", color="#e6edf3", linewidth=0.8)
    for y, player, color in zip([0, 1], (first, second), colors):
        outcome = oof_by_id.loc[int(player["nfl_id"])]
        base, candidate, actual = float(outcome.baseline), float(outcome.candidate), float(outcome.actual)
        outcome_ax.plot([min(base, candidate), max(base, candidate)], [y, y], color="#a8b6c5", linewidth=2, zorder=1)
        outcome_ax.scatter(base, y, s=70, color="#ffffff", edgecolor="#52677d", linewidth=1.6, marker="o", zorder=3, label="Baseline prediction" if y == 0 else None)
        outcome_ax.scatter(candidate, y, s=85, color=color, edgecolor="white", linewidth=0.8, marker="D", zorder=4, label="+ early-acceleration prediction" if y == 0 else None)
        outcome_ax.scatter(actual, y, s=105, color="#123f5a", edgecolor="white", linewidth=0.8, marker="*", zorder=5, label="Observed outcome" if y == 0 else None)
        model_y = y + 0.25 if y == 0 else y - 0.25
        actual_y = y - 0.25 if y == 0 else y + 0.25
        outcome_ax.text(candidate, model_y, f"Model {candidate:.2f}", color=color, fontsize=8.5, ha="center", va="center")
        outcome_ax.text(actual + 0.03, actual_y, f"Observed {actual:.2f}", color="#123f5a", fontsize=8.8, weight="bold", va="center")
    outcome_ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8.2)
    for spine in outcome_ax.spines.values():
        spine.set_visible(False)

    fig.text(0.04, 0.035,
             f"Pair chosen using only same-cohort Combine calipers and acceleration gap—not rookie outcomes. {int(first['draft_year'])} model fold held out. Illustrative n=2; target is speed capacity, not pass-rush impact.",
             fontsize=8.5, color="#516579")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.24)
    fig.savefig(OUT / "dl_speed_profile_case_illustration.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({"players": [p["display_name"] for p in (first, second)], "acceleration_gap": gap,
                      "figure": str(OUT / "dl_speed_profile_case_illustration.png")}, indent=2))


if __name__ == "__main__":
    main()
