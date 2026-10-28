"""Test Combine acceleration against player-play speed with pre-snap context controls.

The outcome is maximum DL speed during a complete fixed four-second window after
ball_snap. It controls play observation duration and evaluates whether the
Combine feature adds information beyond pre-draft tests/size and observed play
context. Each training player receives total weight 1; held-out losses are
computed per player and averaged equally.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from run_analysis import open_data_source, parse_sensor_times

from run_analysis import DEFAULT_ARCHIVE, MEMBER

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
YEARS = (2023, 2024, 2025)
CHUNK_ROWS = 300_000
WINDOW_SECONDS = 4
LAST_FRAME = WINDOW_SECONDS * 10 - 1
BASE_COMBINE = ["forty", "ten_yd_split", "combine_height", "combine_weight"]
EARLY = "forty_early_accel_0_1s_ydps2_mean"
TARGET = "speed_peak_first4s_ydps"
NUMERIC_CONTEXT = ["speed_at_snap_ydps", "pre_snap_max_speed_0_5s_ydps"]
CATEGORICAL_CONTEXT = ["lined_up_position", "down_category", "distance_category", "blitzing_category"]


def load_metadata(archive: zipfile.ZipFile) -> tuple[pd.DataFrame, dict]:
    sig = pd.read_csv(OUT / "player_speed_signatures.csv")
    dl = sig.loc[sig.combine_position.eq("DL")].copy()
    ids = set(dl.nfl_id.astype(int))
    draft = dict(zip(dl.nfl_id.astype(int), dl.draft_year.astype(int)))
    games = pd.read_csv(archive.open(MEMBER + "games.csv"), usecols=["game_id", "season", "season_type"])
    games["game_id"] = pd.to_numeric(games.game_id, errors="coerce")
    games["season"] = pd.to_numeric(games.season, errors="coerce")
    games = games.loc[games.season_type.eq("REG") & games.season.isin(YEARS)].dropna(subset=["game_id", "season"])
    season_by_game = dict(zip(games.game_id.astype(int), games.season.astype(int)))
    cols = ["game_id", "play_id", "nfl_id", "lined_up_position", "down", "yards_to_go", "blitzing"]
    meta = pd.read_csv(archive.open(MEMBER + "player_play.csv"), usecols=cols)
    for col in ["game_id", "play_id", "nfl_id", "down", "yards_to_go"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.dropna(subset=["game_id", "play_id", "nfl_id"])
    meta = meta.loc[meta.nfl_id.isin(ids)].copy()
    meta[["game_id", "play_id", "nfl_id"]] = meta[["game_id", "play_id", "nfl_id"]].astype(int)
    meta["draft_year"] = meta.nfl_id.map(draft)
    meta["game_season"] = meta.game_id.map(season_by_game)
    meta = meta.loc[meta.game_season.eq(meta.draft_year)].copy()
    meta = meta.drop_duplicates(["game_id", "play_id", "nfl_id"])
    meta["distance_category"] = pd.cut(meta.yards_to_go, [-np.inf, 2, 5, 8, 12, 20, np.inf],
                                        labels=["1-2", "3-5", "6-8", "9-12", "13-20", "21+"])
    meta["down_category"] = meta.down.fillna(0).astype(int).astype(str).replace({"0": "unknown"})
    meta["blitzing_category"] = meta.blitzing.astype("string").fillna("false").str.lower()
    return meta, draft


def extract_play_table(archive_path: Path) -> tuple[pd.DataFrame, dict]:
    with open_data_source(archive_path) as archive:
        meta, draft = load_metadata(archive)
        gameplays = set(zip(meta.game_id, meta.play_id))
        playerplay_keys = set(zip(meta.game_id, meta.play_id, meta.nfl_id))
        dl_ids = set(meta.nfl_id.unique())

        snap: dict[tuple[int, int], pd.Timestamp] = {}
        for year in YEARS:
            member = MEMBER + f"game_tracking_{year}.csv"
            for chunk in pd.read_csv(archive.open(member), usecols=["game_id", "play_id", "time", "event"],
                                     chunksize=CHUNK_ROWS, low_memory=False):
                is_snap = chunk.event.fillna("").astype(str).str.strip().str.lower().eq("ball_snap")
                chunk = chunk.loc[is_snap].copy()
                if chunk.empty:
                    continue
                chunk["game_id"] = pd.to_numeric(chunk.game_id, errors="coerce")
                chunk["play_id"] = pd.to_numeric(chunk.play_id, errors="coerce")
                chunk = chunk.dropna(subset=["game_id", "play_id", "time"])
                chunk[["game_id", "play_id"]] = chunk[["game_id", "play_id"]].astype(int)
                keep = [k in gameplays for k in zip(chunk.game_id, chunk.play_id)]
                chunk = chunk.loc[keep]
                if chunk.empty:
                    continue
                times = parse_sensor_times(chunk.time, utc=True)
                chunk = chunk.assign(_snap=times).dropna(subset=["_snap"])
                for (gid, pid), rows in chunk.groupby(["game_id", "play_id"], sort=False):
                    key = (int(gid), int(pid))
                    val = rows._snap.min()
                    if key not in snap or val < snap[key]:
                        snap[key] = val
            print(f"Snap-event scan {year}: {len(snap):,} eligible game-plays found.", flush=True)

        frames: dict[tuple[int, int, int], dict[int, float]] = defaultdict(dict)
        snap_speed: dict[tuple[int, int, int], float] = {}
        pre_snap_max: dict[tuple[int, int, int], float] = {}
        for year in YEARS:
            member = MEMBER + f"game_tracking_{year}.csv"
            for chunk in pd.read_csv(archive.open(member), usecols=["game_id", "play_id", "nfl_id", "time", "s"],
                                     chunksize=CHUNK_ROWS, low_memory=False):
                for col in ["game_id", "play_id", "nfl_id", "s"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                chunk = chunk.dropna(subset=["game_id", "play_id", "nfl_id", "time", "s"])
                chunk[["game_id", "play_id", "nfl_id"]] = chunk[["game_id", "play_id", "nfl_id"]].astype(int)
                chunk = chunk.loc[chunk.nfl_id.isin(dl_ids)]
                if chunk.empty:
                    continue
                keep = [k in playerplay_keys for k in zip(chunk.game_id, chunk.play_id, chunk.nfl_id)]
                chunk = chunk.loc[keep].copy()
                if chunk.empty:
                    continue
                chunk["_snap"] = [snap.get((g, p), pd.NaT) for g, p in zip(chunk.game_id, chunk.play_id)]
                chunk = chunk.dropna(subset=["_snap"])
                if chunk.empty:
                    continue
                parsed = parse_sensor_times(chunk.time, utc=True)
                chunk["_rel"] = (parsed - chunk._snap).dt.total_seconds()
                chunk = chunk.loc[chunk._rel.between(-0.55, LAST_FRAME / 10 + 0.05)]
                for gid, pid, nfl_id, rel, speed in chunk[["game_id", "play_id", "nfl_id", "_rel", "s"]].itertuples(index=False, name=None):
                    key = (int(gid), int(pid), int(nfl_id))
                    frame = int(round(float(rel) * 10))
                    value = float(speed)
                    if frame < 0:
                        pre_snap_max[key] = max(pre_snap_max.get(key, -np.inf), value)
                    else:
                        frames[key][frame] = max(frames[key].get(frame, -np.inf), value)
        print(f"Fixed-window pass: collected {len(frames):,} DL player-plays.", flush=True)

    rows = []
    for r in meta.itertuples(index=False):
        key = (int(r.game_id), int(r.play_id), int(r.nfl_id))
        f = frames.get(key, {})
        idx = sorted(i for i in f if 0 <= i <= LAST_FRAME)
        if len(idx) < 36 or not idx or idx[0] > 1 or idx[-1] < LAST_FRAME - 1:
            continue
        if max(np.diff(idx), default=1) > 2:
            continue
        peak = max(f[i] for i in idx)
        snap_v = f.get(0, f.get(1, np.nan))
        pre_max = pre_snap_max.get(key, np.nan)
        rows.append({"game_id": int(r.game_id), "play_id": int(r.play_id), "nfl_id": int(r.nfl_id),
                     "draft_year": int(r.draft_year), "game_season": int(r.game_season),
                     "lined_up_position": str(r.lined_up_position) if pd.notna(r.lined_up_position) else "unknown",
                     "down_category": str(r.down_category), "distance_category": str(r.distance_category) if pd.notna(r.distance_category) else "unknown",
                     "blitzing_category": str(r.blitzing_category), "speed_at_snap_ydps": float(snap_v),
                     "pre_snap_max_speed_0_5s_ydps": float(pre_max), "speed_peak_first4s_ydps": float(peak),
                     "complete_4s_window": 1})
    play = pd.DataFrame(rows)
    sig = pd.read_csv(OUT / "player_speed_signatures.csv")
    sig = sig.loc[sig.combine_position.eq("DL"), ["nfl_id", "forty", "ten_yd_split", "combine_height", "combine_weight", EARLY]]
    play = play.merge(sig, on="nfl_id", how="left", validate="many_to_one")
    summary = {"eligible_player_play_keys": int(len(playerplay_keys)),
               "eligible_gameplay_keys": int(len(gameplays)), "gameplays_with_ball_snap": int(len(snap)),
               "eligible_keys_missing_snap_event": int(len(gameplays - set(snap))),
               "eligible_playerplays_with_complete_4s_window": int(len(play)),
               "players_with_complete_4s_windows": int(play.nfl_id.nunique()),
               "window": "0.0–3.9 sec after ball_snap; >=36 distinct 10 Hz frames; both endpoints within 0.1 sec; no gaps >0.2 sec",
               "context_fields": ["lined_up_position", "down_category", "distance_category", "blitzing_category", *NUMERIC_CONTEXT]}
    return play, summary


def player_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("nfl_id").size()
    return frame.nfl_id.map(1.0 / counts).to_numpy(float)


def weighted_median(values: pd.Series, weights: np.ndarray) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    order = np.argsort(x[valid])
    xv, wv = x[valid][order], weights[valid][order]
    return float(xv[np.searchsorted(np.cumsum(wv), wv.sum() / 2, side="left")])


def design_matrix(train: pd.DataFrame, test: pd.DataFrame, include_early: bool) -> tuple[np.ndarray, np.ndarray]:
    numeric = BASE_COMBINE + NUMERIC_CONTEXT + ([EARLY] if include_early else [])
    train_num, test_num = train[numeric].copy(), test[numeric].copy()
    train_weights = player_weights(train)
    med = pd.Series({c: weighted_median(train_num[c], train_weights) for c in numeric})
    train_num, test_num = train_num.fillna(med), test_num.fillna(med)
    mu = pd.Series({c: float(np.average(train_num[c].to_numpy(float), weights=train_weights)) for c in numeric})
    sd = pd.Series({c: float(np.sqrt(np.average((train_num[c].to_numpy(float)-mu[c])**2, weights=train_weights))) for c in numeric})
    sd = sd.replace(0, 1).fillna(1)
    train_num = (train_num - mu) / sd
    test_num = (test_num - mu) / sd
    train_cat = pd.get_dummies(train[CATEGORICAL_CONTEXT].fillna("unknown").astype(str), dtype=float)
    test_cat = pd.get_dummies(test[CATEGORICAL_CONTEXT].fillna("unknown").astype(str), dtype=float)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0.0)
    train_raw = np.column_stack([train_num.to_numpy(float), train_cat.to_numpy(float)])
    test_raw = np.column_stack([test_num.to_numpy(float), test_cat.to_numpy(float)])
    # Equal-player weighted scaling keeps high-volume players from setting units
    # for the ridge penalty; train-fold categories only define the design columns.
    cat_mu = np.average(train_cat.to_numpy(float), axis=0, weights=train_weights) if train_cat.shape[1] else np.array([])
    cat_sd = np.sqrt(np.average((train_cat.to_numpy(float)-cat_mu)**2, axis=0, weights=train_weights)) if train_cat.shape[1] else np.array([])
    all_mu = np.r_[mu.to_numpy(float), cat_mu]
    all_sd = np.r_[sd.to_numpy(float), np.where(cat_sd > 0, cat_sd, 1.0)]
    train_raw = (train_raw - all_mu) / all_sd
    test_raw = (test_raw - all_mu) / all_sd
    x = np.column_stack([np.ones(len(train)), train_raw])
    z = np.column_stack([np.ones(len(test)), test_raw])
    return x, z


def predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float, include_early: bool) -> np.ndarray:
    x, z = design_matrix(train, test, include_early)
    w = player_weights(train)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w * train[TARGET].to_numpy(float)))
    return z @ beta


def player_losses(test: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    d = test[["nfl_id", "draft_year"]].copy()
    d["base_sqerr"] = (test[TARGET].to_numpy(float) - baseline) ** 2
    d["candidate_sqerr"] = (test[TARGET].to_numpy(float) - candidate) ** 2
    return d.groupby(["nfl_id", "draft_year"], as_index=False).agg(
        base_mse=("base_sqerr", "mean"), candidate_mse=("candidate_sqerr", "mean"), plays=("base_sqerr", "size"))


def rmse_delta(loss: pd.DataFrame) -> float:
    return float(np.sqrt(loss.candidate_mse.mean()) - np.sqrt(loss.base_mse.mean()))


def bootstrap_delta(loss: pd.DataFrame, seed: int, draws: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [g[["base_mse", "candidate_mse"]].to_numpy(float) for _, g in loss.groupby("draft_year")]
    draws_by_group = []
    for g in groups:
        ix = rng.integers(0, len(g), size=(draws, len(g)))
        draws_by_group.append(g[ix].mean(axis=1))
    b = np.concatenate([x[:, :1] for x in draws_by_group], axis=1)
    c = np.concatenate([x[:, 1:] for x in draws_by_group], axis=1)
    delta = np.sqrt(c.mean(axis=1)) - np.sqrt(b.mean(axis=1))
    return [float(np.quantile(delta, .025)), float(np.quantile(delta, .975))]


def evaluate(plays: pd.DataFrame, min_plays: int) -> tuple[dict, pd.DataFrame]:
    eligible = plays.groupby("nfl_id").size()
    selected = set(eligible[eligible.ge(min_plays)].index.astype(int))
    d = plays.loc[plays.nfl_id.isin(selected)].dropna(subset=[TARGET, *BASE_COMBINE, EARLY]).copy()
    result = {"minimum_complete_4s_windows": min_plays, "n_players": int(d.nfl_id.nunique()),
              "player_play_rows": int(len(d)),
              "by_cohort": {str(int(k)): int(v) for k, v in d.drop_duplicates("nfl_id").draft_year.value_counts().sort_index().items()},
              "comparisons": []}
    all_losses = []
    for ai, alpha in enumerate([5.0, 20.0]):
        oof = []
        for cohort in sorted(d.draft_year.unique()):
            train, test = d.loc[d.draft_year.ne(cohort)], d.loc[d.draft_year.eq(cohort)]
            p0 = predict(train, test, alpha, False)
            p1 = predict(train, test, alpha, True)
            loss = player_losses(test, p0, p1)
            oof.append(loss)
        loss = pd.concat(oof, ignore_index=True)
        result["comparisons"].append({"alpha": alpha, "loco_rmse_change": rmse_delta(loss),
                                      "loco_player_bootstrap_95pct_ci": bootstrap_delta(loss, 12000 + min_plays + ai),
                                      "loco_players": int(len(loss))})
        all_losses.append(loss.assign(alpha=alpha))

        train, test = d.loc[d.draft_year.lt(2025)], d.loc[d.draft_year.eq(2025)]
        p0 = predict(train, test, alpha, False)
        p1 = predict(train, test, alpha, True)
        forward_loss = player_losses(test, p0, p1)
        result["comparisons"][-1]["forward_2025"] = {
            "n_train_players": int(train.nfl_id.nunique()), "n_test_players": int(test.nfl_id.nunique()),
            "rmse_change": rmse_delta(forward_loss),
            "player_bootstrap_95pct_ci": bootstrap_delta(forward_loss, 14000 + min_plays + ai),
        }
        all_losses.append(forward_loss.assign(alpha=alpha, evaluation="forward_2025"))
    return result, pd.concat(all_losses, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--min-plays", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--reuse-plays", action="store_true")
    args = ap.parse_args()
    play_path = OUT / "dl_contextual_game_speed_plays.csv"
    coverage_path = OUT / "dl_contextual_game_speed_coverage.json"
    if args.reuse_plays and play_path.is_file():
        plays = pd.read_csv(play_path)
        coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    else:
        plays, coverage = extract_play_table(args.archive)
        OUT.mkdir(parents=True, exist_ok=True)
        plays.to_csv(play_path, index=False)
        coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    results, losses = [], []
    for min_plays in args.min_plays:
        res, loss = evaluate(plays, min_plays)
        results.append(res)
        losses.append(loss.assign(minimum_complete_4s_windows=min_plays))
    output = {"coverage": coverage, "results": results,
              "target": "per-player-play maximum speed from 0.0 through 3.9 sec after ball_snap",
              "baseline": "official 40, 10-yard split, height, weight, plus pre-snap speed and play context",
              "candidate": "baseline plus Combine 0-1 second 40-yard acceleration",
              "training_player_weighting": "Each player's total training sample weight equals 1.",
              "evaluation_player_weighting": "Calculate each player's mean squared prediction error across observed plays, then take the square root of the mean across players."}
    (OUT / "dl_contextual_game_speed_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.concat(losses, ignore_index=True).to_csv(OUT / "dl_contextual_game_speed_player_scores.csv", index=False)
    lines = ["# Context-adjusted fixed-window DL game speed", "",
             "Target: player-play maximum speed during the first 4 seconds after `ball_snap`. Only near-complete 10 Hz windows are retained. Models control for lineup position, down, yards-to-go band, blitz role, speed at snap, and pre-snap maximum speed. The baseline also contains official 40 time, 10-yard split, height, and weight; the candidate adds Combine first-second acceleration.", "",
             "Training rows are weighted so each player contributes total weight 1. Test error is computed within each player and averaged equally across players. Imputation, scaling, and categorical encodings are fit using training players only. The 2025 fold trains only on 2023–2024. Fixed-prediction player-bootstrap intervals do not refit models and this remains an exploratory sensitivity.", "",
             f"Coverage: {coverage.get('eligible_player_play_keys', 0):,} eligible rookie DL player-play keys; {coverage.get('eligible_playerplays_with_complete_4s_window', 0):,} complete 4-second windows across {coverage.get('players_with_complete_4s_windows', 0)} players.", ""]
    for res in results:
        lines += [f"## At least {res['minimum_complete_4s_windows']} complete windows", "",
                  f"Sample: {res['n_players']} players and {res['player_play_rows']:,} player-play rows; by cohort {res['by_cohort']}.", "",
                  "| Alpha | Player-balanced LOCO RMSE Δ (95% CI) | Forward 2025 players | Forward 2025 RMSE Δ (95% CI) |",
                  "|---:|---:|---:|---:|"]
        for x in res["comparisons"]:
            ci, fwd = x["loco_player_bootstrap_95pct_ci"], x["forward_2025"]
            fci = fwd["player_bootstrap_95pct_ci"]
            lines.append(f"| {x['alpha']:g} | {x['loco_rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {fwd['n_test_players']} | {fwd['rmse_change']:+.4f} ({fci[0]:+.4f}, {fci[1]:+.4f}) |")
        lines.append("")
    lines += ["This model assesses incremental prediction within tracked play contexts; it does not estimate a causal effect, individual pass-rush win probability, or a pre-draft performance outcome. Feature search and cohort size limit inference.", ""]
    (OUT / "dl_contextual_game_speed_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
