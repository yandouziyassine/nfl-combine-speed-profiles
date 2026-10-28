"""Build reproducible SVG and PNG summaries of the DL finding."""
from __future__ import annotations
import html, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
W, H = 1260, 750
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="#fbfcfe"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#19283b}.title{font-size:27px;font-weight:700}.sub{font-size:14px;fill:#506174}.panel{font-size:18px;font-weight:700}.axis{font-size:12px;fill:#506174}.label{font-size:13px}.small{font-size:11px;fill:#506174}.grid{stroke:#dce3eb;stroke-width:1}.zero{stroke:#5b6572;stroke-width:1.5;stroke-dasharray:4 4}.ci{stroke:#176b87;stroke-width:3}.point{fill:#176b87}.warn{fill:#a84b37}</style>',
         '<text x="54" y="48" class="title">Combine early acceleration: narrow signal, limited transfer</text>',
         '<text x="55" y="75" class="sub">The full-play peak association does not replicate consistently in fixed post-snap speed windows or pass-rush outcomes.</text>']

def txt(x, y, s, cls="label", anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{html.escape(str(s))}</text>')

def axis_panel(x, y, w, h, title, lo, hi, ticks, x_label):
    txt(x, y, title, "panel")
    left, right, top, bottom = x + 82, x + w - 18, y + 42, y + h - 58
    for t in ticks:
        xx = left + (t - lo) / (hi - lo) * (right-left)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" class="{"zero" if abs(t)<1e-9 else "grid"}"/>')
        txt(xx, bottom + 22, f'{t:g}', "axis", "middle")
    txt((left+right)/2, y+h-14, x_label, "axis", "middle")
    return left, right, top, bottom

# Panel 1: same DLs compared across full-play and fixed post-snap windows.
x0, x1, top, bot = axis_panel(48, 112, 360, 505, "A. Signal depends on speed window", -.10, .04,
                               [-.10, -.075, -.05, -.025, 0, .025], "Change in RMSE (yd/s)")
res = json.loads((OUT / "dl_early_accel_predraft_results_100plays.json").read_text(encoding="utf-8"))
e = next(v for v in res["comparisons"] if v["alpha"] == 5.0)
window = json.loads((OUT / "dl_fixed_window_game_speed_results.json").read_text(encoding="utf-8"))
points = [{"label": "Full-play p90 (n=46)", "value": e["rmse_change"], "ci": e["player_bootstrap_95pct_ci"]}]
for horizon in (1, 2, 3, 4):
    spec = next(x for x in window["results"] if x["horizon_seconds"] == horizon and x["minimum_complete_fixed_windows"] == 100)
    comp = next(x for x in spec["comparisons"] if x["alpha"] == 5.0)
    points.append({"label": f"First {horizon}s (n={spec['n_players']})", "value": comp["loco_rmse_change"],
                   "ci": comp["loco_player_bootstrap_95pct_ci"]})
for i, point in enumerate(points):
    yy = top + 47 + i*66
    val, (low, high) = point["value"], point["ci"]
    txt(x0 - 10, yy+5, point["label"], "label", "end")
    xa = x0 + (low+.10)/.14*(x1-x0); xb = x0 + (high+.10)/.14*(x1-x0)
    xx = x0 + (val+.10)/.14*(x1-x0)
    parts.append(f'<line x1="{xa:.1f}" y1="{yy}" x2="{xb:.1f}" y2="{yy}" class="ci"/><circle cx="{xx:.1f}" cy="{yy}" r="6" class="point"/>')
forward_full = next(f["rmse_change"] for f in e["folds"] if f["held_out_draft_year"] == 2025)
forward_windows = []
for horizon in (1, 2, 3, 4):
    spec = next(x for x in window["results"] if x["horizon_seconds"] == horizon and x["minimum_complete_fixed_windows"] == 100)
    comp = next(x for x in spec["comparisons"] if x["alpha"] == 5.0)
    forward_windows.append(comp["forward_2025"]["rmse_change"])
txt(65, 640, f"Forward 2025 (full n=14): full-play {forward_full:+.3f} yd/s.", "small")
txt(65, 657, "Fixed 1–4s (n=14,14,13,13): " + ", ".join(f"{v:+.3f}" for v in forward_windows) + " yd/s.", "small")

# Panel 2: matched complete-case ablation.
px0, px1, ptop, pbot = axis_panel(455, 112, 370, 505, "B. Which 40 phase adds signal?", -.10, .06,
                                  [-.10, -.05, 0, .05], "Change in RMSE (yd/s)")
ab = json.loads((OUT / "dl_game_speed_ablation_results_100plays.json").read_text(encoding="utf-8"))
specs = [("All phases", "all_three_phases"), ("Early 0–1 s", "early_acceleration_common_sample"),
         ("Late 10–20 yd", "late_acceleration_common_sample"), ("Speed retention", "speed_retention_common_sample")]
for i, (label, key) in enumerate(specs):
    yy = ptop + 70 + i*82
    valspec = next(v for v in ab["specifications"][key] if v["alpha"] == 5.0)
    d = valspec["rmse_change"]; low, high = valspec["player_bootstrap_95pct_ci"]
    a = px0 + (low+.10)/.16*(px1-px0); b = px0 + (high+.10)/.16*(px1-px0); cx = px0 + (d+.10)/.16*(px1-px0)
    label = f"{label} (n={ab['samples'][key]['n_players']})"
    txt(px0-10, yy+5, label, "label", "end")
    parts.append(f'<line x1="{a:.1f}" y1="{yy}" x2="{b:.1f}" y2="{yy}" class="ci"/><circle cx="{cx:.1f}" cy="{yy}" r="6" class="point"/>')
txt(470, 640, "34 common players; baseline includes draft capital; α=5.", "small")
txt(470, 657, "Dots are RMSE change; bars are 95% player-bootstrap intervals.", "small")
nested = json.loads((OUT / "dl_nested_phase_selection_100plays.json").read_text(encoding="utf-8"))
nested_delta = nested["nested_selected_pooled_oof_rmse_change"]
nested_low, nested_high = nested["nested_selected_player_bootstrap_95pct_interval"]
txt(470, 674, f"Nested phase choice (n=37): {nested_delta:+.3f} [{nested_low:+.3f}, {nested_high:+.3f}].", "small")

# Panel 3: within-session ICC for early acceleration.
ix0, ix1, itop, ibot = axis_panel(865, 112, 350, 505, "C. Attempt repeatability", -.4, 1,
                                  [-.4, 0, .25, .50, .75, 1], "ICC(A,1), early acceleration")
rep = pd.read_csv(OUT / "forty_repeatability_by_position.csv")
rep = rep.loc[rep.measure.eq("early_acceleration")]
for i, row in enumerate(rep.itertuples(index=False)):
    yy = itop + 62 + i*60
    low, high = json.loads(row.icc_bootstrap_95pct_ci)
    a = ix0 + (low+.4)/1.4*(ix1-ix0); b = ix0 + (high+.4)/1.4*(ix1-ix0)
    cx = ix0 + (row.icc_absolute_agreement_single_attempt+.4)/1.4*(ix1-ix0)
    txt(ix0-9, yy+5, f"{row.combine_position} (n={row.players_with_both_attempts})", "label", "end")
    parts.append(f'<line x1="{a:.1f}" y1="{yy}" x2="{b:.1f}" y2="{yy}" class="ci"/><circle cx="{cx:.1f}" cy="{yy}" r="6" class="point"/>')
txt(878, 640, "Within-session, not between-day reliability; player", "small")
txt(878, 657, "bootstrap intervals. Feature ranks also depend on onset rule.", "small")

parts += ['<line x1="55" y1="690" x2="1205" y2="690" stroke="#dce3eb"/>',
          '<text x="55" y="714" class="small">Interpretation: full-play peak association only; fixed-window/context-adjusted lift is near zero, and pressure transfer, causality, and a draft cutoff are unestablished.</text>',
          '</svg>']
(OUT / "dl_early_acceleration_summary.svg").write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {OUT / 'dl_early_acceleration_summary.svg'}")

# Produce the same evidence as a PNG for Kaggle's Writeup media uploader.
# Matplotlib is already a dependency of the other notebook figure builders.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

fig, axes = plt.subplots(1, 3, figsize=(12.6, 7.5), gridspec_kw={"width_ratios": [1.03, 1.08, 1.0]})
fig.patch.set_facecolor("#fbfcfe")
fig.suptitle("Combine early acceleration: narrow signal, limited transfer", x=0.05, y=0.99,
             ha="left", fontsize=18, fontweight="bold", color="#19283b")
fig.text(0.05, 0.945,
         "The full-play peak association does not replicate consistently in fixed post-snap speed windows or pass-rush outcomes.",
         ha="left", fontsize=10.5, color="#506174")

def forest_panel(ax, title, rows, limits, ticks, xlabel):
    ax.set_facecolor("#fbfcfe")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#19283b", pad=16)
    y = list(range(len(rows) - 1, -1, -1))
    for yi, row in zip(y, rows):
        value, low, high = row[1], row[2], row[3]
        ax.errorbar(value, yi, xerr=[[value - low], [high - value]], fmt="o", color="#176b87",
                    ecolor="#176b87", elinewidth=2.2, markersize=7, capsize=0, zorder=3)
    ax.axvline(0, color="#5b6572", linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlabel(xlabel, color="#506174", fontsize=9)
    ax.tick_params(axis="both", colors="#506174", labelsize=8.5, length=0, pad=7)
    ax.grid(axis="x", color="#dce3eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9aa6b2")

rows_a = [(p["label"].replace(" (n=", " ("),
           p["value"], p["ci"][0], p["ci"][1]) for p in points]
forest_panel(axes[0], "A. Signal depends on speed window", rows_a, (-0.10, 0.04),
             [-0.10, -0.075, -0.05, -0.025, 0, 0.025], "Change in RMSE (yd/s)")
axes[0].text(0, -0.27,
             f"Forward 2025 (full n=14): {forward_full:+.3f} yd/s.\n"
             "Fixed 1–4 s (n=14,14,13,13): " + ", ".join(f"{v:+.3f}" for v in forward_windows) + " yd/s.",
             transform=axes[0].transAxes, va="top", fontsize=8, color="#506174", clip_on=False)

rows_b = []
for label, key in specs:
    valspec = next(v for v in ab["specifications"][key] if v["alpha"] == 5.0)
    low, high = valspec["player_bootstrap_95pct_ci"]
    rows_b.append((f"{label} (n={ab['samples'][key]['n_players']})", valspec["rmse_change"], low, high))
forest_panel(axes[1], "B. Which 40 phase adds signal?", rows_b, (-0.10, 0.06),
             [-0.10, -0.05, 0, 0.05], "Change in RMSE (yd/s)")
axes[1].text(0, -0.27,
             f"34 common players; baseline includes draft capital; α=5.\n"
             f"Nested phase choice (n=37): {nested_delta:+.3f} [{nested_low:+.3f}, {nested_high:+.3f}].",
             transform=axes[1].transAxes, va="top", fontsize=8, color="#506174", clip_on=False)

rows_c = []
for row in rep.itertuples(index=False):
    low, high = json.loads(row.icc_bootstrap_95pct_ci)
    rows_c.append((f"{row.combine_position} (n={row.players_with_both_attempts})",
                   row.icc_absolute_agreement_single_attempt, low, high))
forest_panel(axes[2], "C. Attempt repeatability", rows_c, (-0.4, 1.0),
             [-0.4, 0, 0.25, 0.50, 0.75, 1.0], "ICC(A,1), early acceleration")
axes[2].text(0, -0.27,
             "Same-session; not day-to-day reliability.\n"
             "Player-bootstrap CI; onset affects ranks.",
             transform=axes[2].transAxes, va="top", fontsize=8, color="#506174", clip_on=False)

fig.text(0.05, 0.02,
         "Interpretation: full-play peak association only; fixed-window/context-adjusted lift is near zero, and pressure transfer, causality, and a draft cutoff are unestablished.",
         fontsize=8, color="#506174", ha="left")
fig.subplots_adjust(left=0.13, right=0.985, top=0.85, bottom=0.34, wspace=0.62)
fig.savefig(OUT / "dl_early_acceleration_summary.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Wrote {OUT / 'dl_early_acceleration_summary.png'}")
