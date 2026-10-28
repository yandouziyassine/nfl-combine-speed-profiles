"""Context-adjusted play-level model for the OL mirror candidate.

Pressure labels are modeled at the player-play level. Entire draft cohorts are
held out; performance is scored by player, with a player-stratified bootstrap
for the paired Brier-score difference. This is a sensitivity analysis, not a
replacement for the simpler player-level screen.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from run_analysis import DEFAULT_ARCHIVE, open_data_source

MEMBER = "nfl-big-data-bowl-2027/"
NUMERIC_BASE = ["forty", "ten_yd_split", "combine_height", "combine_weight", "down", "yards_to_go", "pass_rushers_encountered"]
CATEGORICAL_BASE = ["lined_up_position", "offense_formation", "team_abbr"]
TRACKING_FEATURES = ["mirror_abs_directional_speed_gap_ydps", "mirror_mean_bilateral_speed_ydps"]
MIN_OPPORTUNITIES = [50, 100]


def read_member(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(zf.open(MEMBER + name), **kwargs)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_design(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], categorical: list[str]):
    tr_num, te_num, numeric_names = [], [], []
    for name in numeric:
        tr = pd.to_numeric(train[name], errors="coerce")
        te = pd.to_numeric(test[name], errors="coerce")
        med = float(tr.median()) if tr.notna().any() else 0.0
        vals_tr, vals_te = tr.fillna(med), te.fillna(med)
        mu = float(vals_tr.mean())
        sd = float(vals_tr.std())
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        tr_num.append(((vals_tr - mu) / sd).to_numpy(float))
        te_num.append(((vals_te - mu) / sd).to_numpy(float))
        numeric_names.append(name)
        if tr.isna().any():
            tr_num.append(tr.isna().to_numpy(float))
            te_num.append(te.isna().to_numpy(float))
            numeric_names.append(f"{name}__missing")
    Xn = np.column_stack(tr_num) if tr_num else np.empty((len(train), 0))
    Zn = np.column_stack(te_num) if te_num else np.empty((len(test), 0))
    tr_cat = pd.get_dummies(train[categorical].astype("string").fillna("__MISSING__"), dtype=float)
    te_cat = pd.get_dummies(test[categorical].astype("string").fillna("__MISSING__"), dtype=float).reindex(columns=tr_cat.columns, fill_value=0.0)
    Xt = np.column_stack([np.ones(len(train)), Xn, tr_cat.to_numpy(float)])
    Zt = np.column_stack([np.ones(len(test)), Zn, te_cat.to_numpy(float)])
    return Xt, Zt


def fit_logistic_ridge(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    beta = np.zeros(X.shape[1], dtype=float)
    penalty = np.eye(X.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    for _ in range(80):
        p = np.clip(sigmoid(X @ beta), 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        grad = X.T @ (sample_weight * (y - p)) - penalty @ beta
        hess = X.T @ ((sample_weight * w)[:, None] * X) + penalty
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def bootstrap_brier_delta(preds: pd.DataFrame, draws: int = 3000, seed: int = 912) -> list[float] | None:
    if preds.empty:
        return None
    by_year = [g.index.to_numpy() for _, g in preds.groupby("draft_year")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ids = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_year])
        b = preds.loc[ids]
        values.append(float((b.brier_candidate - b.brier_baseline).mean()))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    args = parser.parse_args()

    players = pd.read_csv(out / "all_player_speed_profiles.csv")
    players = players.loc[players.nfl_position.isin(["T", "G", "C"]),
                          ["nfl_id", "draft_year", "nfl_position", "forty", "ten_yd_split", "combine_height", "combine_weight"]]
    drill = pd.read_csv(out / "player_drill_speed_profiles.csv")
    mirror = drill.loc[drill.drill_name.isin(["PASS_PRO-MIRROR_DRILL", "PASS_PRO_MIRROR_DRILL"])].copy()
    mirror[TRACKING_FEATURES[0]] = mirror.left_right_lateral_speed_difference_by_drill_ydps.abs()
    mirror[TRACKING_FEATURES[1]] = (mirror.left_lateral_speed_mean_ydps_mean + mirror.right_lateral_speed_mean_ydps_mean) / 2
    mirror = mirror.groupby("nfl_id", as_index=False)[TRACKING_FEATURES].mean()
    player_features = players.merge(mirror, on="nfl_id", how="left", validate="one_to_one")

    with open_data_source(args.archive) as zf:
        roster = read_member(zf, "players.csv", usecols=["nfl_id", "draft_year", "nfl_position"])
        games = read_member(zf, "games.csv", usecols=["game_id", "season", "season_type"])
        games = games.loc[games.season_type.eq("REG"), ["game_id", "season"]]
        columns = ["game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position", "pressure_allowed",
                   "down", "yards_to_go", "offense_formation", "pass_rushers_encountered"]
        play = read_member(zf, "player_play.csv", usecols=columns)
    play = play.merge(games, on="game_id", how="inner", validate="many_to_one")
    play = play.merge(roster, on="nfl_id", how="inner", validate="many_to_one")
    play = play.loc[
        play.season.eq(play.draft_year)
        & play.lined_up_position.isin(["T", "G", "C"])
        & play.nfl_position.isin(["T", "G", "C"])
        & play.pressure_allowed.notna()
    ].copy()
    play["pressure_allowed"] = play.pressure_allowed.astype(int)
    for c in ["down", "yards_to_go", "pass_rushers_encountered"]:
        play[c] = pd.to_numeric(play[c], errors="coerce")
    play = play.merge(player_features, on=["nfl_id", "draft_year", "nfl_position"], how="inner", validate="many_to_one")
    label_counts = play.groupby("nfl_id").play_id.size()
    play = play.loc[play.nfl_id.map(label_counts).ge(min(MIN_OPPORTUNITIES))].copy()
    eligible_summary = {str(t): int(label_counts.ge(t).sum()) for t in MIN_OPPORTUNITIES}

    results = {"analysis_unit": "Player for validation and uncertainty; play rows fit the contextual binary outcome model.",
               "outcome": "Supplied pressure_allowed label on rookie REG player-play rows for T/G/C when nonmissing.",
               "baseline_numeric": NUMERIC_BASE, "baseline_categorical": CATEGORICAL_BASE,
               "tracking_features": TRACKING_FEATURES,
               "context": "Line position, offense team, offensive formation, down, yards to go, and pass rushers encountered. No dropback duration is used because it may occur after pressure begins.",
               "validation": "Leave one draft cohort out. All plays from each held-out player stay in test. Each training player's total likelihood weight is one; the primary metric is each test player's mean Brier score, averaged equally across players.",
               "minimum_opportunity_players": eligible_summary, "comparisons": []}
    all_predictions = []
    for threshold in MIN_OPPORTUNITIES:
        threshold_rows = play.loc[play.nfl_id.map(label_counts).ge(threshold)].copy()
        for feature in TRACKING_FEATURES:
            subset = threshold_rows.loc[threshold_rows[feature].notna()].copy()
            player_oof = []
            fold_stats = []
            for cohort in sorted(subset.draft_year.unique()):
                tr = subset.loc[subset.draft_year.ne(cohort)].copy()
                te = subset.loc[subset.draft_year.eq(cohort)].copy()
                train_players, test_players = tr.nfl_id.nunique(), te.nfl_id.nunique()
                if train_players < 15 or test_players < 5:
                    fold_stats.append({"held_out_draft_year": int(cohort), "n_train_players": int(train_players), "n_test_players": int(test_players), "status": "too_few_players"})
                    continue
                ytr = tr.pressure_allowed.to_numpy(float)
                yte = te.pressure_allowed.to_numpy(float)
                X0, Z0 = make_design(tr, te, NUMERIC_BASE, CATEGORICAL_BASE)
                train_weights = 1.0 / tr.groupby("nfl_id").nfl_id.transform("size").to_numpy(float)
                b0 = fit_logistic_ridge(X0, ytr, train_weights, args.ridge_alpha)
                p0 = sigmoid(Z0 @ b0)
                Xt, Zt = make_design(tr, te, NUMERIC_BASE + [feature], CATEGORICAL_BASE)
                bt = fit_logistic_ridge(Xt, ytr, train_weights, args.ridge_alpha)
                p1 = sigmoid(Zt @ bt)
                test_rows = te[["nfl_id", "draft_year"]].copy()
                test_rows["actual"] = yte
                test_rows["pred_baseline"] = p0
                test_rows["pred_candidate"] = p1
                test_rows["brier_baseline"] = (yte - p0) ** 2
                test_rows["brier_candidate"] = (yte - p1) ** 2
                per_player = (test_rows.groupby(["nfl_id", "draft_year"], as_index=False)
                              .agg(actual_pressure_rate=("actual", "mean"),
                                   predicted_pressure_rate_baseline=("pred_baseline", "mean"),
                                   predicted_pressure_rate_candidate=("pred_candidate", "mean"),
                                   brier_baseline=("brier_baseline", "mean"),
                                   brier_candidate=("brier_candidate", "mean"), n_plays=("actual", "size")))
                player_oof.extend(per_player.to_dict("records"))
                fold_stats.append({"held_out_draft_year": int(cohort), "n_train_players": int(train_players),
                                   "n_test_players": int(test_players), "train_plays": int(len(tr)), "test_plays": int(len(te)),
                                   "mean_player_brier_baseline": float(per_player.brier_baseline.mean()),
                                   "mean_player_brier_candidate": float(per_player.brier_candidate.mean()),
                                   "brier_change": float((per_player.brier_candidate - per_player.brier_baseline).mean()),
                                   "status": "scored"})
            pred = pd.DataFrame(player_oof)
            if not pred.empty:
                ci = bootstrap_brier_delta(pred, draws=args.bootstrap_draws, seed=912 + threshold + len(pred))
                delta = float((pred.brier_candidate - pred.brier_baseline).mean())
            else:
                ci, delta = None, None
            results["comparisons"].append({"minimum_labeled_plays": threshold, "feature": feature,
                                           "n_players_oof": int(len(pred)),
                                           "players_by_cohort": {str(int(k)): int(v) for k, v in pred.draft_year.value_counts().sort_index().items()} if not pred.empty else {},
                                           "equal_player_weighted_brier_change": delta,
                                           "player_stratified_bootstrap_95pct_ci": ci,
                                           "folds": fold_stats})
            if not pred.empty:
                pred["minimum_labeled_plays"] = threshold
                pred["tracking_feature"] = feature
                all_predictions.append(pred)

    (out / "ol_mirror_playlevel_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(out / "ol_mirror_playlevel_oof_predictions.csv", index=False)
    lines = ["# Context-adjusted O-line play-level sensitivity analysis", "",
             "This model tests whether the Combine mirror-tracking traits add information about rookie pressure allowed after controlling play context. Logistic ridge models are fit on labeled player-play rows, but complete draft cohorts are held out. Each training player's total likelihood weight is one; every test player's plays remain together, and Brier scores are averaged within player before players are weighted equally. Bootstrap intervals resample players within draft cohort.", "",
             "Baseline includes official 40/split, body size, line position, team, formation, down, yards to go, and number of pass rushers encountered. Training-fold medians and missingness flags handle absent Combine measures. Dropback duration is excluded because it may be downstream of pressure.", "",
             "| Minimum labeled opportunities | Combine feature | Players scored | Equal-player Brier Δ (95% player-bootstrap CI) |", "|---:|---|---:|---:|"]
    for x in results["comparisons"]:
        ci = x["player_stratified_bootstrap_95pct_ci"]
        value = "NA" if x["equal_player_weighted_brier_change"] is None else f"{x['equal_player_weighted_brier_change']:+.5f} ({ci[0]:+.5f}, {ci[1]:+.5f})"
        lines.append(f"| {x['minimum_labeled_plays']} | {x['feature']} | {x['n_players_oof']} | {value} |")
    lines += ["", "Positive Brier-score change means the candidate model's probability errors are worse. This is still observational, small-sample, and exploratory; it does not isolate an individual blocker from teammate and opponent effects.", ""]
    (out / "ol_mirror_playlevel_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"labeled_player_play_rows": int(len(play)), "players_at_50_labels": eligible_summary["50"],
                      "players_at_100_labels": eligible_summary["100"], "comparisons": len(results["comparisons"]),
                      "report": str(out / "ol_mirror_playlevel_report.md")}, indent=2))


if __name__ == "__main__":
    main()
