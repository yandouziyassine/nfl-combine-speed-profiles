"""Visualize outer-fold choices in the nested position-by-phase audit."""
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
data = json.loads((OUT / "position_phase_nested_selection_100plays.json").read_text(encoding="utf-8"))
folds = data["folds"]
years = [int(f["outer_held_out_year"]) for f in folds]
nested = [float(f["nested_rmse_change"]) for f in folds]
locked = [float(f["locked_dl_early_rmse_change"]) for f in folds]
labels = [f"{f['outer_held_out_year']}\n{f['selected_candidate'].replace(':', ' · ').replace('early_0_1s', 'early 0–1s').replace('late_10_20yd', 'late 10–20yd').replace('retention_20_40yd', 'retention 20–40yd')}" for f in folds]
ci_nested = data["nested_selected_fixed_oof_player_bootstrap_95pct_interval"]
ci_locked = data["locked_dl_early_fixed_oof_player_bootstrap_95pct_interval_same_players"]
pooled = [data["nested_selected_pooled_oof_rmse_change"],
          data["locked_dl_early_pooled_oof_rmse_change_same_players"]]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
x = range(len(years))
ax[0].plot(x, nested, "o-", color="#bd6b45", lw=2, label="Nested 12-candidate procedure")
ax[0].plot(x, locked, "o-", color="#176b87", lw=2, label="Locked DL early feature")
ax[0].axhline(0, color="#5b6572", ls="--", lw=1.4)
ax[0].set_xticks(list(x), labels)
ax[0].set_xlim(-.1, 2.5)
ax[0].set(ylabel="Outer-fold RMSE change (yd/s)", title="A. Selected phase and held-out cohort")
ax[0].text(2.04, nested[-1], "Nested", color="#bd6b45", va="center", fontsize=9)
ax[0].text(2.04, locked[-1], "Locked DL early", color="#176b87", va="center", fontsize=9)
for i, (a, b) in enumerate(zip(nested, locked)):
    ax[0].annotate(f"{a:+.3f}", (i, a), xytext=(0, 9), textcoords="offset points", ha="center", color="#8f4c32", fontsize=8)
    ax[0].annotate(f"{b:+.3f}", (i, b), xytext=(0, -15), textcoords="offset points", ha="center", color="#155a72", fontsize=8)

est = [pooled[0], pooled[1]]
cis = [ci_nested, ci_locked]
positions = [1, 0]
ax[1].axvline(0, color="#5b6572", ls="--", lw=1.4)
for y, value, ci, color in zip(positions, est, cis, ["#bd6b45", "#176b87"]):
    ax[1].errorbar(value, y, xerr=[[value-ci[0]], [ci[1]-value]], fmt="o", color=color,
                   ecolor=color, capsize=5, lw=2, markersize=7)
ax[1].set_yticks(positions, ["Nested 12-candidate", "Locked DL early"])
ax[1].set(xlim=(-.022, .01), xlabel="Pooled OOF RMSE change (yd/s)", title="B. Same 161-player sample")
ax[1].text(.02, -.19, "Whiskers are fixed-OOF player-bootstrap intervals; they do not repeat selection.",
           transform=ax[1].transAxes, ha="left", va="top", fontsize=8, color="#506174")
fig.suptitle("Selection across position and 40-yard phase leaves little nested lift", fontsize=15, fontweight="bold")
fig.savefig(OUT / "fig_position_phase_nested_selection.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "fig_position_phase_nested_selection.svg", bbox_inches="tight")
print(f"Wrote {OUT / 'fig_position_phase_nested_selection.png'}")
