"""Plot disjoint-game repeatability and model sensitivity."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
rho = pd.read_csv(OUT / "dl_game_speed_split_half_repeatability_100plays.csv")["spearman_rho"]
model = pd.read_csv(OUT / "dl_game_speed_split_half_model_deltas_100plays.csv")
groups = [(h, a) for h in ("A", "B") for a in (5, 20)]
labels = [f"Half {h}\nalpha {a}" for h, a in groups]
values = [model.loc[(model.half == h) & (model.alpha == a), "rmse_change_ydps"].to_numpy() for h, a in groups]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
ax[0].hist(rho, bins=14, range=(.60, .95), color="#8bc5ce", edgecolor="#176b87")
ax[0].axvline(rho.median(), color="#176b87", lw=2.5, label=f"Median rho = {rho.median():.3f}")
ax[0].set(xlim=(.60, .95), xlabel="Spearman correlation between half-season player ranks", ylabel="Random game partitions", title="A. Speed rank repeatability")
ax[0].legend(frameon=False)
bp = ax[1].boxplot(values, tick_labels=labels, showfliers=False, whis=(2.5, 97.5), patch_artist=True)
for patch, color in zip(bp["boxes"], ["#176b87", "#176b87", "#bd6b45", "#bd6b45"]):
    patch.set_facecolor(color); patch.set_alpha(.22); patch.set_edgecolor(color)
ax[1].axhline(0, color="#5b6572", ls="--", lw=1.5)
ax[1].set(ylabel="LOCO RMSE change (yd/s; negative favors early acceleration)", title="B. RMSE delta across repeated game-half splits")
fig.suptitle("Within-season partitions: speed ranks correlate; model RMSE favors early gain", fontsize=16, fontweight="bold")
fig.text(.5, -.01, "46 DLs · 250 repeated partitions · each half has ≥2 games · whiskers show partition range, not confidence intervals or external validation", ha="center", fontsize=9)
fig.savefig(OUT / "fig_dl_game_speed_split_half.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "fig_dl_game_speed_split_half.svg", bbox_inches="tight")
print(f"Wrote {OUT / 'fig_dl_game_speed_split_half.png'}")
