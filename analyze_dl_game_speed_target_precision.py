"""Estimate game-cluster uncertainty in the DL rookie game-speed outcome.

Uses the same rookie regular-season player-play matching convention as
run_analysis.py, restricted to the locked >=100-play DL model cohort.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from run_analysis import open_data_source

from run_analysis import DEFAULT_ARCHIVE, MEMBER, read_csv
from model_dl_early_accel_predraft import BASE, EARLY

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CHUNK_ROWS = 300_000


def predict_with_target(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                        target: str, alpha: float) -> np.ndarray:
    """Ridge prediction with training-only preprocessing and an explicit target."""
    med = train[features].median()
    a = train[features].fillna(med)
    b = test[features].fillna(med)
    mu = a.mean()
    sd = a.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(a)), ((a - mu) / sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(b)), ((b - mu) / sd).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return z @ np.linalg.solve(x.T @ x + penalty, x.T @ train[target].to_numpy(float))


def split_half_analysis(by_player_game: dict[int, dict[int, list[float]]],
                        cohort: pd.DataFrame, players: pd.DataFrame,
                        repeats: int, seed: int) -> dict:
    """Measure p90 repeatability across balanced, disjoint sets of rookie games.

    For each player, candidate partitions are compared only on play counts; the
    observed speeds never influence the selected partition. Each half must have
    at least two distinct games. The model check refits leave-one-cohort-out
    baselines on each half-specific target, with the predictor set fixed.
    """
    rng = np.random.default_rng(seed)
    min_games_per_half = 2
    eligible = [int(pid) for pid in cohort.nfl_id.astype(int)
                if len(by_player_game.get(int(pid), {})) >= 2 * min_games_per_half]
    if len(eligible) < 20:
        raise ValueError(f"Only {len(eligible)} players have enough games for disjoint halves.")
    meta = players.loc[players.nfl_id.isin(eligible),
                       ["nfl_id", "draft_year", *BASE, EARLY]].drop_duplicates("nfl_id")
    meta = meta.set_index("nfl_id").loc[eligible].reset_index()
    years = sorted(meta.draft_year.unique())
    halves_rows = []
    model_rows = []
    max_candidates = 40

    for rep in range(repeats):
        targets: dict[int, list[float]] = {0: [], 1: []}
        ids_for_targets: list[int] = []
        for pid in eligible:
            game_map = by_player_game[pid]
            game_ids = np.asarray(sorted(game_map), dtype=int)
            n_left = len(game_ids) // 2
            best_left = None
            best_imbalance = np.inf
            play_counts = np.asarray([len(game_map[g]) for g in game_ids], dtype=int)
            for _ in range(max_candidates):
                order = rng.permutation(len(game_ids))
                left_ix = order[:n_left]
                right_ix = order[n_left:]
                imbalance = abs(int(play_counts[left_ix].sum()) - int(play_counts[right_ix].sum()))
                if imbalance < best_imbalance:
                    best_imbalance = imbalance
                    best_left = set(game_ids[left_ix].tolist())
            left_values = np.concatenate([game_map[g] for g in game_ids if g in best_left])
            right_values = np.concatenate([game_map[g] for g in game_ids if g not in best_left])
            y_left, y_right = float(np.quantile(left_values, .90)), float(np.quantile(right_values, .90))
            ids_for_targets.append(pid)
            targets[0].append(y_left)
            targets[1].append(y_right)
            halves_rows.append({"repeat": rep, "nfl_id": pid, "draft_year": int(
                meta.loc[meta.nfl_id.eq(pid), "draft_year"].iloc[0]),
                "half_a_p90_ydps": y_left, "half_b_p90_ydps": y_right,
                "n_games": len(game_map), "half_a_plays": int(len(left_values)),
                "half_b_plays": int(len(right_values))})

        split_df = meta.set_index("nfl_id").loc[ids_for_targets].reset_index().copy()
        for half, target_values in targets.items():
            split_df["split_target"] = target_values
            for alpha in (5.0, 20.0):
                actuals, base_preds, candidate_preds = [], [], []
                for year in years:
                    train = split_df.loc[split_df.draft_year.ne(year)]
                    test = split_df.loc[split_df.draft_year.eq(year)]
                    base_preds.extend(predict_with_target(train, test, BASE, "split_target", alpha))
                    candidate_preds.extend(predict_with_target(train, test, BASE + [EARLY], "split_target", alpha))
                    actuals.extend(test.split_target.to_numpy(float))
                actuals = np.asarray(actuals, dtype=float)
                base_preds = np.asarray(base_preds, dtype=float)
                candidate_preds = np.asarray(candidate_preds, dtype=float)
                d_rmse = float(np.sqrt(np.mean((actuals - candidate_preds) ** 2)) -
                               np.sqrt(np.mean((actuals - base_preds) ** 2)))
                model_rows.append({"repeat": rep, "half": "A" if half == 0 else "B",
                                   "alpha": alpha, "n_players": len(split_df),
                                   "rmse_change_ydps": d_rmse})

    half_table = pd.DataFrame(halves_rows)
    repeat_metrics = []
    for rep, frame in half_table.groupby("repeat", sort=True):
        repeat_metrics.append({
            "repeat": int(rep), "n_players": int(frame.nfl_id.nunique()),
            "pearson_r": float(frame.half_a_p90_ydps.corr(frame.half_b_p90_ydps)),
            "spearman_rho": float(frame.half_a_p90_ydps.corr(frame.half_b_p90_ydps, method="spearman")),
            "median_absolute_difference_ydps": float(
                np.median(np.abs(frame.half_a_p90_ydps - frame.half_b_p90_ydps))),
        })
    metrics = pd.DataFrame(repeat_metrics)
    model_table = pd.DataFrame(model_rows)
    model_summary = []
    for (half, alpha), frame in model_table.groupby(["half", "alpha"], sort=True):
        model_summary.append({
            "half": half, "alpha": float(alpha), "repeats": int(len(frame)),
            "median_loco_rmse_change_ydps": float(frame.rmse_change_ydps.median()),
            "partition_2_5pct": float(frame.rmse_change_ydps.quantile(.025)),
            "partition_97_5pct": float(frame.rmse_change_ydps.quantile(.975)),
            "fraction_repeats_improved": float((frame.rmse_change_ydps < 0).mean()),
        })
    summary = {
        "design": "Repeatedly partition each player's rookie regular-season games into disjoint halves with at least two games per half. Among 40 random partitions, choose the one with the closest play counts; selection uses counts only, never speeds. Compute each half's p90 of per-play peak speed.",
        "n_repeats": int(repeats), "players": int(len(eligible)),
        "players_excluded_fewer_than_four_games": int(len(cohort) - len(eligible)),
        "minimum_games_per_half": min_games_per_half,
        "repeatability_across_random_partitions": {
            "median_pearson_r": float(metrics.pearson_r.median()),
            "partition_2_5pct": float(metrics.pearson_r.quantile(.025)),
            "partition_97_5pct": float(metrics.pearson_r.quantile(.975)),
            "median_spearman_rho": float(metrics.spearman_rho.median()),
            "spearman_partition_2_5pct": float(metrics.spearman_rho.quantile(.025)),
            "spearman_partition_97_5pct": float(metrics.spearman_rho.quantile(.975)),
            "median_absolute_half_difference_ydps": float(metrics.median_absolute_difference_ydps.median()),
        },
        "same_locked_combine_model_refit_on_each_half_target": model_summary,
        "interpretation": "Intervals across repeated partitions describe sensitivity to which observed games land in each half; they are not confidence intervals and do not represent an independent cohort. The two half-targets are noisier than the full-season target. The model comparison keeps the original player cohort, pre-draft baseline, predictor, and cohort-held-out folds fixed.",
    }
    half_table.to_csv(OUT / "dl_game_speed_split_half_player_targets_100plays.csv", index=False)
    metrics.to_csv(OUT / "dl_game_speed_split_half_repeatability_100plays.csv", index=False)
    model_table.to_csv(OUT / "dl_game_speed_split_half_model_deltas_100plays.csv", index=False)
    (OUT / "dl_game_speed_split_half_results_100plays.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--split-repeats", type=int, default=250)
    args = ap.parse_args()
    oof = pd.read_csv(OUT / "dl_early_accel_predraft_oof_100plays.csv")
    players = pd.read_csv(OUT / "player_speed_signatures.csv")
    cohort = oof[["nfl_id", "draft_year"]].drop_duplicates()
    cohort = cohort.merge(players[["nfl_id", "combine_position"]], on="nfl_id", how="left")
    cohort = cohort.loc[cohort.combine_position.eq("DL")].copy()
    ids = set(cohort.nfl_id.astype(int))
    draft_year = dict(zip(cohort.nfl_id.astype(int), cohort.draft_year.astype(int)))

    with open_data_source(args.archive) as archive:
        games = read_csv(archive, "games.csv", usecols=["game_id", "season", "season_type"])
        games["season"] = pd.to_numeric(games.season, errors="coerce")
        games["game_id"] = pd.to_numeric(games.game_id, errors="coerce")
        games = games.loc[games.season_type.eq("REG")]
        games = games.loc[games.season.isin(set(draft_year.values())) & games.game_id.notna()]
        regular_games = set(games.game_id.astype(int))
        game_year = dict(zip(games.game_id.astype(int), games.season.astype(int)))
        metadata = read_csv(archive, "player_play.csv", usecols=["game_id", "play_id", "nfl_id"])
        for col in ["game_id", "play_id", "nfl_id"]:
            metadata[col] = pd.to_numeric(metadata[col], errors="coerce")
        metadata = metadata.loc[metadata.game_id.isin(regular_games) & metadata.nfl_id.isin(ids)].dropna()
        metadata["game_id"] = metadata.game_id.astype(int)
        metadata["play_id"] = metadata.play_id.astype(int)
        metadata["nfl_id"] = metadata.nfl_id.astype(int)
        metadata["game_season"] = metadata.game_id.map(game_year)
        metadata["player_class"] = metadata.nfl_id.map(draft_year)
        metadata = metadata.loc[metadata.game_season.eq(metadata.player_class)]
        metadata_keys = metadata[["game_id", "play_id", "nfl_id"]].drop_duplicates()
        peaks: dict[tuple[int, int, int], float] = {}
        for year in (2023, 2024, 2025):
            member = MEMBER + f"game_tracking_{year}.csv"
            for chunk in pd.read_csv(archive.open(member), usecols=["game_id", "play_id", "nfl_id", "s"],
                                     chunksize=CHUNK_ROWS, low_memory=False):
                for col in ["game_id", "play_id", "nfl_id", "s"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                chunk = chunk.loc[chunk.game_id.isin(regular_games) & chunk.nfl_id.isin(ids)].dropna()
                if chunk.empty:
                    continue
                chunk["game_id"] = chunk.game_id.astype(int)
                chunk["play_id"] = chunk.play_id.astype(int)
                chunk["nfl_id"] = chunk.nfl_id.astype(int)
                chunk["game_season"] = chunk.game_id.map(game_year)
                chunk["player_class"] = chunk.nfl_id.map(draft_year)
                chunk = chunk.loc[chunk.game_season.eq(chunk.player_class)]
                chunk_peaks = chunk.groupby(["game_id", "play_id", "nfl_id"], as_index=False).s.max()
                chunk_peaks = chunk_peaks.merge(metadata_keys, on=["game_id", "play_id", "nfl_id"], how="inner")
                for row in chunk_peaks.itertuples(index=False):
                    key = (int(row.game_id), int(row.play_id), int(row.nfl_id))
                    peaks[key] = max(peaks.get(key, -np.inf), float(row.s))
            print(f"Scanned {year} tracking; matched play keys so far: {len(peaks):,}", flush=True)

    by_player_game: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (game_id, _play_id, nfl_id), speed in peaks.items():
        by_player_game[nfl_id][game_id].append(speed)

    rng = np.random.default_rng(20271026)
    records = []
    draw_records = []
    for pid, year in zip(cohort.nfl_id.astype(int), cohort.draft_year.astype(int)):
        game_map = by_player_game.get(pid, {})
        game_ids = np.array(sorted(game_map), dtype=int)
        groups = [np.asarray(game_map[g], dtype=float) for g in game_ids]
        flat = np.concatenate(groups) if groups else np.array([], dtype=float)
        if len(flat) == 0:
            print(f"No matched rookie tracking plays for {pid} ({year})", flush=True)
            continue
        point = float(np.quantile(flat, .90))
        boot = np.empty(args.draws)
        for b in range(args.draws):
            sampled = rng.integers(0, len(groups), size=len(groups))
            values = np.concatenate([groups[j] for j in sampled])
            boot[b] = np.quantile(values, .90)
            draw_records.append({"nfl_id": int(pid), "draft_year": int(year),
                                 "draw": int(b), "p90_draw_ydps": float(boot[b])})
        records.append({
            "nfl_id": int(pid), "draft_year": int(year), "n_games": int(len(groups)),
            "n_matched_plays": int(len(flat)), "p90_game_peak_speed_ydps": point,
            "cluster_bootstrap_ci_low": float(np.quantile(boot, .025)),
            "cluster_bootstrap_ci_high": float(np.quantile(boot, .975)),
            "cluster_bootstrap_se": float(np.std(boot, ddof=1)),
            "cluster_bootstrap_draws": int(args.draws),
        })
    result = pd.DataFrame(records).sort_values(["draft_year", "nfl_id"])
    result.to_csv(OUT / "dl_game_speed_target_precision_100plays.csv", index=False)
    pd.DataFrame(draw_records).to_csv(OUT / "dl_game_speed_target_bootstrap_draws_100plays.csv", index=False)
    summary = {
        "players": int(len(result)),
        "minimum_plays": int(result.n_matched_plays.min()),
        "median_plays": float(result.n_matched_plays.median()),
        "median_games": float(result.n_games.median()),
        "median_cluster_bootstrap_se_ydps": float(result.cluster_bootstrap_se.median()),
        "median_cluster_bootstrap_ci_width_ydps": float(
            (result.cluster_bootstrap_ci_high - result.cluster_bootstrap_ci_low).median()
        ),
        "p90_se_overall_sd": float(result.cluster_bootstrap_se.median() / result.p90_game_peak_speed_ydps.std(ddof=1)),
        "method": "Within each player, resample observed rookie regular-season games with replacement, retaining all matched tracked player-play peaks from each sampled game; recompute p90. This estimates sampling uncertainty conditional on observed game schedule/role and available tracking.",
        "warning": "Players with one tracked game have no meaningful game-cluster bootstrap uncertainty estimate; intervals can be degenerate or optimistic with few games.",
    }
    (OUT / "dl_game_speed_target_precision_100plays.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    split_summary = split_half_analysis(by_player_game, cohort, players,
                                        repeats=args.split_repeats, seed=20271027)
    summary["disjoint_game_split_half_audit"] = split_summary
    (OUT / "dl_game_speed_target_precision_100plays.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
