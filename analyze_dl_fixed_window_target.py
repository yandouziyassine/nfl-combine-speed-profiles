"""Audit Combine-to-game-speed transfer with a fixed post-snap observation window.

The existing outcome uses each play's maximum tracked speed over the whole play.
This sensitivity measures maximum speed in fixed 1, 2, 3, and 4 second windows
after ball_snap and requires near-complete 10 Hz coverage of each full window,
reducing bias from different play durations. It reports leave-one-cohort-out and
forward 2025 tests.
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

from model_dl_early_accel_predraft import BASE, EARLY, bootstrap
from run_analysis import DEFAULT_ARCHIVE, MEMBER

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
YEARS = (2023, 2024, 2025)
HORIZONS_SECONDS = (1, 2, 3, 4)
CHUNK_ROWS = 300_000
LAST_FRAME_SECONDS = max(HORIZONS_SECONDS) - 0.1


def predict_fixed_target(train: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float) -> np.ndarray:
    """Ridge prediction with training-only imputation/scaling for this target."""
    med = train[features].median()
    xraw, zraw = train[features].fillna(med), test[features].fillna(med)
    mu, sd = xraw.mean(), xraw.std().replace(0, 1).fillna(1)
    x = np.column_stack([np.ones(len(xraw)), ((xraw-mu)/sd).to_numpy(float)])
    z = np.column_stack([np.ones(len(zraw)), ((zraw-mu)/sd).to_numpy(float)])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    return z @ np.linalg.solve(x.T @ x + penalty, x.T @ train["rookie_reg_p90_max_speed_first_2s_post_snap_ydps"].to_numpy(float))


def to_frame_seconds(series: pd.Series, snap: pd.Timestamp) -> np.ndarray:
    times = parse_sensor_times(series, utc=True)
    return (times - snap).dt.total_seconds().to_numpy(dtype=float)


def make_outcome(archive_path: Path, min_plays: int) -> tuple[pd.DataFrame, dict]:
    signatures = pd.read_csv(OUT / "player_speed_signatures.csv")
    dl = signatures.loc[signatures.combine_position.eq("DL")].copy()
    ids = set(pd.to_numeric(dl.nfl_id, errors="coerce").dropna().astype(int))
    draft_year = dict(zip(dl.nfl_id.astype(int), dl.draft_year.astype(int)))
    id_to_class = {str(pid): year for pid, year in draft_year.items()}

    with open_data_source(archive_path) as archive:
        games = pd.read_csv(archive.open(MEMBER + "games.csv"), usecols=["game_id", "season", "season_type"])
        games["game_id"] = pd.to_numeric(games.game_id, errors="coerce")
        games["season"] = pd.to_numeric(games.season, errors="coerce")
        games = games.loc[games.season_type.eq("REG") & games.season.isin(YEARS)].dropna(subset=["game_id", "season"])
        game_season = dict(zip(games.game_id.astype(int), games.season.astype(int)))
        target_game_ids = set(game_season)

        metadata = pd.read_csv(archive.open(MEMBER + "player_play.csv"), usecols=["game_id", "play_id", "nfl_id"])
        for col in ("game_id", "play_id", "nfl_id"):
            metadata[col] = pd.to_numeric(metadata[col], errors="coerce")
        metadata = metadata.dropna(subset=["game_id", "play_id", "nfl_id"])
        metadata = metadata.loc[metadata.game_id.isin(target_game_ids) & metadata.nfl_id.isin(ids)].copy()
        metadata[["game_id", "play_id", "nfl_id"]] = metadata[["game_id", "play_id", "nfl_id"]].astype(int)
        metadata["game_season"] = metadata.game_id.map(game_season)
        metadata["player_class"] = metadata.nfl_id.map(draft_year)
        metadata = metadata.loc[metadata.game_season.eq(metadata.player_class)]
        metadata = metadata.drop_duplicates(["game_id", "play_id", "nfl_id"])
        valid_playerplays = set(zip(metadata.game_id, metadata.play_id, metadata.nfl_id))
        valid_gameplays = set(zip(metadata.game_id, metadata.play_id))

        # Pass 1: get the ball_snap timestamp for each rookie regular-season play.
        snap_time: dict[tuple[int, int], pd.Timestamp] = {}
        for year in YEARS:
            member = MEMBER + f"game_tracking_{year}.csv"
            for chunk in pd.read_csv(archive.open(member), usecols=["game_id", "play_id", "time", "event"],
                                     chunksize=CHUNK_ROWS, low_memory=False):
                event = chunk.event.fillna("").astype(str).str.strip().str.lower()
                chunk = chunk.loc[event.eq("ball_snap")].copy()
                if chunk.empty:
                    continue
                chunk["game_id"] = pd.to_numeric(chunk.game_id, errors="coerce")
                chunk["play_id"] = pd.to_numeric(chunk.play_id, errors="coerce")
                chunk = chunk.dropna(subset=["game_id", "play_id", "time"])
                chunk["game_id"] = chunk.game_id.astype(int)
                chunk["play_id"] = chunk.play_id.astype(int)
                chunk = chunk.loc[[key in valid_gameplays for key in zip(chunk.game_id, chunk.play_id)]]
                if chunk.empty:
                    continue
                parsed = parse_sensor_times(chunk.time, utc=True)
                for (game_id, play_id), group in chunk.assign(_snap=parsed).dropna(subset=["_snap"]).groupby(["game_id", "play_id"]):
                    key = (int(game_id), int(play_id))
                    value = group._snap.min()
                    if key not in snap_time or value < snap_time[key]:
                        snap_time[key] = value
            print(f"Found snap times for {len(snap_time):,} valid player-plays after scanning {year} event rows.", flush=True)

        without_snap = len(valid_gameplays - set(snap_time))
        observations: dict[tuple[int, int, int], dict[int, float]] = defaultdict(dict)
        for year in YEARS:
            member = MEMBER + f"game_tracking_{year}.csv"
            for chunk in pd.read_csv(archive.open(member), usecols=["game_id", "play_id", "nfl_id", "time", "s"],
                                     chunksize=CHUNK_ROWS, low_memory=False):
                for col in ("game_id", "play_id", "nfl_id", "s"):
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                chunk = chunk.dropna(subset=["game_id", "play_id", "nfl_id", "s", "time"])
                chunk[["game_id", "play_id", "nfl_id"]] = chunk[["game_id", "play_id", "nfl_id"]].astype(int)
                chunk = chunk.loc[chunk.nfl_id.isin(ids)]
                if chunk.empty:
                    continue
                keep = [key in valid_playerplays for key in zip(chunk.game_id, chunk.play_id, chunk.nfl_id)]
                chunk = chunk.loc[keep].copy()
                if chunk.empty:
                    continue
                chunk["_snap"] = [snap_time.get((g, p), pd.NaT) for g, p in zip(chunk.game_id, chunk.play_id)]
                chunk = chunk.dropna(subset=["_snap"])
                if chunk.empty:
                    continue
                parsed = parse_sensor_times(chunk.time, utc=True)
                chunk["_rel"] = (parsed - chunk._snap).dt.total_seconds()
                chunk = chunk.loc[chunk._rel.between(-0.05, LAST_FRAME_SECONDS + 0.05)]
                for row in chunk[["game_id", "play_id", "nfl_id", "_rel", "s"]].itertuples(index=False, name=None):
                    game_id, play_id, nfl_id, rel, speed = row
                    if rel < -0.05 or rel > LAST_FRAME_SECONDS + 0.05:
                        continue
                    # Timestamps are rounded to 0.1 s in this tracking source. The
                    # integer frame key deduplicates repeated rows at a timestamp.
                    frame = int(round((float(rel) / 0.1)))
                    observations[(int(game_id), int(play_id), int(nfl_id))][frame] = max(
                        observations[(int(game_id), int(play_id), int(nfl_id))].get(frame, -np.inf), float(speed))
            print(f"Collected fixed-window tracking through {year}; candidate player-play keys: {len(observations):,}.", flush=True)

    valid_window = {h: {} for h in HORIZONS_SECONDS}
    for key, frames in observations.items():
        vals = np.array(sorted(frames.items()), dtype=float)
        frame_ids = vals[:, 0].astype(int)
        for horizon in HORIZONS_SECONDS:
            last_frame = horizon * 10 - 1
            in_window = vals[frame_ids <= last_frame]
            observed = in_window[:, 0].astype(int)
            minimum_frames = int(np.ceil(horizon * 10 * 0.9))
            # Require near-complete 10 Hz coverage, both window endpoints within
            # one frame, and no gaps longer than 0.2s. This avoids treating
            # truncated plays as if they were observed for the full window.
            if (len(observed) < minimum_frames or observed.min() > 1 or
                    observed.max() < last_frame - 1 or np.diff(observed).max(initial=1) > 2):
                continue
            valid_window[horizon][key] = float(in_window[:, 1].max())

    by_player: dict[int, dict[int, list[float]]] = {h: defaultdict(list) for h in HORIZONS_SECONDS}
    for horizon in HORIZONS_SECONDS:
        for (game_id, play_id, nfl_id), peak in valid_window[horizon].items():
            by_player[horizon][nfl_id].append(peak)
    rows = []
    for pid, year in draft_year.items():
        row = {"nfl_id": pid, "draft_year": year}
        for horizon in HORIZONS_SECONDS:
            values = np.asarray(by_player[horizon].get(pid, []), dtype=float)
            row[f"rookie_reg_complete_{horizon}s_post_snap_plays"] = int(len(values))
            row[f"rookie_reg_p90_max_speed_first_{horizon}s_post_snap_ydps"] = float(np.quantile(values, .90)) if len(values) else np.nan
        rows.append(row)
    target = pd.DataFrame(rows)
    sample = signatures.loc[signatures.combine_position.eq("DL"), ["nfl_id", "draft_year", "forty", "ten_yd_split", "combine_height", "combine_weight", "forty_early_accel_0_1s_ydps2_mean"]]
    target = target.merge(sample, on=["nfl_id", "draft_year"], how="left", validate="one_to_one")
    coverage = {"valid_rookie_reg_playerplays": len(valid_playerplays),
                "valid_gameplays_with_snap": len(snap_time), "gameplays_missing_snap": without_snap,
                "playerplays_with_tracking_in_window": len(observations),
                "complete_window_playerplays_by_horizon": {str(h): len(valid_window[h]) for h in HORIZONS_SECONDS},
                "players_with_complete_window_by_horizon": {str(h): int(target.loc[target[f"rookie_reg_complete_{h}s_post_snap_plays"].gt(0), "nfl_id"].nunique()) for h in HORIZONS_SECONDS},
                "window_definition": "For each horizon, maximum recorded speed from 0.0 through horizon-0.1 seconds after the game_tracking event ball_snap; requires at least 90% of expected distinct 10 Hz timestamps, both endpoints within 0.1 s, and no gaps over 0.2 s."}
    return target, coverage


def evaluate(data: pd.DataFrame, min_plays: int, horizon: int) -> dict:
    target = f"rookie_reg_p90_max_speed_first_{horizon}s_post_snap_ydps"
    exposure = f"rookie_reg_complete_{horizon}s_post_snap_plays"
    required = ["nfl_id", "draft_year", exposure, target, *BASE, EARLY]
    sample = data.loc[data[exposure].ge(min_plays)].dropna(subset=required).copy()
    result = {"horizon_seconds": horizon, "minimum_complete_fixed_windows": min_plays, "n_players": int(len(sample)),
              "by_cohort": {str(int(k)): int(v) for k, v in sample.draft_year.value_counts().sort_index().items()},
              "comparisons": []}
    for ai, alpha in enumerate([5.0, 20.0]):
        rows, folds = [], []
        for cohort in sorted(sample.draft_year.unique()):
            tr, te = sample.loc[sample.draft_year.ne(cohort)], sample.loc[sample.draft_year.eq(cohort)]
            p0, p1 = predict_fixed_target(tr, te, BASE, alpha), predict_fixed_target(tr, te, BASE + [EARLY], alpha)
            y = te[target].to_numpy(float)
            for pid, year, actual, b, c in zip(te.nfl_id, te.draft_year, y, p0, p1):
                rows.append({"nfl_id": int(pid), "draft_year": int(year), "actual": float(actual), "baseline": float(b), "candidate": float(c)})
            folds.append({"held_out_draft_year": int(cohort), "n_players": int(len(te)),
                          "baseline_rmse": float(np.sqrt(np.mean((y-p0)**2))),
                          "candidate_rmse": float(np.sqrt(np.mean((y-p1)**2)))})
        oof = pd.DataFrame(rows)
        if len(oof):
            delta = float(np.sqrt(np.mean((oof.actual-oof.candidate)**2))-np.sqrt(np.mean((oof.actual-oof.baseline)**2)))
            ci = bootstrap(oof, 8400 + ai)
        else:
            delta, ci = np.nan, [np.nan, np.nan]
        forward = {"held_out_draft_year": 2025}
        if 2025 in sample.draft_year.values:
            tr, te = sample.loc[sample.draft_year.lt(2025)], sample.loc[sample.draft_year.eq(2025)]
            if len(tr) and len(te):
                p0, p1 = predict_fixed_target(tr, te, BASE, alpha), predict_fixed_target(tr, te, BASE + [EARLY], alpha)
                y = te[target].to_numpy(float)
                fwd_oof = pd.DataFrame({"nfl_id": te.nfl_id.to_numpy(), "draft_year": te.draft_year.to_numpy(),
                                        "actual": y, "baseline": p0, "candidate": p1})
                forward.update({"n_train_players": int(len(tr)), "n_test_players": int(len(te)),
                                "baseline_rmse": float(np.sqrt(np.mean((y-p0)**2))),
                                "candidate_rmse": float(np.sqrt(np.mean((y-p1)**2))),
                                "rmse_change": float(np.sqrt(np.mean((y-p1)**2))-np.sqrt(np.mean((y-p0)**2))),
                                "player_bootstrap_95pct_ci": bootstrap(fwd_oof, 91000 + horizon*100 + min_plays + ai)})
        result["comparisons"].append({"alpha": alpha, "loco_rmse_change": delta,
                                      "loco_player_bootstrap_95pct_ci": ci, "loco_folds": folds,
                                      "forward_2025": forward})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--min-plays", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--reuse-target", action="store_true", help="Reuse the fixed-window target table generated by a prior run.")
    args = ap.parse_args()
    target_path = OUT / "dl_fixed_window_game_speed_targets.csv"
    if args.reuse_target and target_path.is_file():
        target = pd.read_csv(target_path)
        coverage_path = OUT / "dl_fixed_window_game_speed_coverage.json"
        coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    else:
        target, coverage = make_outcome(args.archive, min(args.min_plays))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target.to_csv(target_path, index=False)
        (OUT / "dl_fixed_window_game_speed_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    results = {"coverage": coverage,
               "results": [evaluate(target, n, h) for h in HORIZONS_SECONDS for n in args.min_plays]}
    (OUT / "dl_fixed_window_game_speed_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Does the Combine signal depend on when game speed is measured?", "",
             "For each DL, the target is the 90th percentile of the per-play maximum speed observed during fixed 1, 2, 3, or 4 second windows after `ball_snap`. A player-play qualifies only when at least 90% of expected 10 Hz frames are present, the observed start and end are each within 0.1 seconds, and no timestamp gap exceeds 0.2 seconds. Fixed windows reduce bias from different full-play durations. They remain observed-play movement outcomes, not pass-rush-success outcomes.", "",
             f"Across the source data, eligible rookie DL player-play keys: {coverage.get('valid_rookie_reg_playerplays','not retained'):,}; keys with tracking in the window: {coverage.get('playerplays_with_tracking_in_window','not retained'):,}; keys missing a `ball_snap` event: {coverage.get('gameplays_missing_snap','not retained'):,}.", "",
             "A negative RMSE change favors adding Combine early acceleration. Leave-one-cohort-out (LOCO) intervals resample fixed player predictions. The forward 2025 test trains only on 2023–2024.", ""]
    for result in results["results"]:
        lines += [f"## First {result['horizon_seconds']} second(s), at least {result['minimum_complete_fixed_windows']} complete windows", "",
                  f"Sample: {result['n_players']} players; by cohort {result['by_cohort']}.", "",
                  "| Alpha | LOCO RMSE change (95% player-bootstrap CI) | Forward 2025 players | Forward 2025 RMSE change (95% player-bootstrap CI) |",
                  "|---:|---:|---:|---:|"]
        for comp in result["comparisons"]:
            fwd = comp["forward_2025"]
            ci = comp["loco_player_bootstrap_95pct_ci"]
            fci = fwd.get("player_bootstrap_95pct_ci", [float("nan"), float("nan")])
            lines.append(f"| {comp['alpha']:g} | {comp['loco_rmse_change']:+.4f} ({ci[0]:+.4f}, {ci[1]:+.4f}) | {fwd.get('n_test_players', 0)} | {fwd.get('rmse_change', float('nan')):+.4f} ({fci[0]:+.4f}, {fci[1]:+.4f}) |")
        lines.append("")
    lines += ["Interpret these as measurement sensitivities, not independent confirmations: the same prospects and screening history are reused. The forward 2025 test has a small test cohort and is the only strictly forward-in-time comparison available in this dataset.", ""]
    (OUT / "dl_fixed_window_game_speed_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
