from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import zipfile

import pandas as pd

from run_analysis import DEFAULT_ARCHIVE, open_data_source, read_csv


ROOT = Path(__file__).resolve().parent
ARCHIVE = DEFAULT_ARCHIVE
OUT = ROOT / "outputs"
CHUNK_ROWS = 1_000_000


def main(archive_path: Path) -> None:
    global ARCHIVE
    ARCHIVE = archive_path
    archive = open_data_source(ARCHIVE)
    players = read_csv(archive, "players.csv", usecols=["nfl_id", "draft_year", "nfl_position"])
    players = players.dropna(subset=["nfl_id", "draft_year"]).copy()
    players["nfl_id"] = pd.to_numeric(players.nfl_id, errors="coerce")
    players["draft_year"] = pd.to_numeric(players.draft_year, errors="coerce")
    players = players.dropna(subset=["nfl_id", "draft_year"])
    players["nfl_id"] = players.nfl_id.astype("int64")
    players["draft_year"] = players.draft_year.astype("int64")

    games = read_csv(archive, "games.csv", usecols=["game_id", "season", "season_type"])
    games = games.loc[games.season_type.eq("REG")].copy()
    games["game_id"] = pd.to_numeric(games.game_id, errors="coerce")
    games["season"] = pd.to_numeric(games.season, errors="coerce")
    games = games.dropna(subset=["game_id", "season"])
    games["game_id"] = games.game_id.astype("int64")
    games["season"] = games.season.astype("int64")
    season_by_game = games.set_index("game_id").season.to_dict()
    player_year = players.set_index("nfl_id").draft_year.to_dict()
    player_pos = players.set_index("nfl_id").nfl_position.to_dict()

    pp = read_csv(archive, "player_play.csv", usecols=["game_id", "play_id", "nfl_id"])
    for col in ("game_id", "play_id", "nfl_id"):
        pp[col] = pd.to_numeric(pp[col], errors="coerce")
    pp = pp.dropna(subset=["game_id", "play_id", "nfl_id"])
    all_pp = set(zip(pp.game_id.astype("int64"), pp.play_id.astype("int64"), pp.nfl_id.astype("int64")))
    player_play_game_keys = set(zip(pp.game_id.astype("int64"), pp.play_id.astype("int64")))
    rookie_pp = set()
    for game_id, play_id, nfl_id in all_pp:
        season = season_by_game.get(game_id)
        if season is not None and player_year.get(nfl_id) == season:
            rookie_pp.add((game_id, play_id, nfl_id))

    counts = Counter()
    by_year = Counter()
    by_position = Counter()
    total_by_year = Counter()
    total_by_position = Counter()
    by_reason = Counter()
    has_other_player_play_row = 0
    examples: dict[str, list[tuple[int, int, int]]] = {"no_player_play_row": [], "nonrookie_or_nonreg": []}
    seen_tracking_keys: set[tuple[int, int, int]] = set()

    with archive as zf:
        for year in (2023, 2024, 2025):
            member = next(n for n in zf.namelist() if n.endswith(f"game_tracking_{year}.csv"))
            for chunk in pd.read_csv(zf.open(member), usecols=["game_id", "play_id", "nfl_id"], chunksize=CHUNK_ROWS, low_memory=False):
                for col in ("game_id", "play_id", "nfl_id"):
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                chunk = chunk.dropna(subset=["game_id", "play_id", "nfl_id"])
                chunk["game_id"] = chunk.game_id.astype("int64")
                chunk["play_id"] = chunk.play_id.astype("int64")
                chunk["nfl_id"] = chunk.nfl_id.astype("int64")
                chunk = chunk.loc[chunk.game_id.isin(season_by_game) & chunk.nfl_id.isin(player_year)]
                chunk = chunk.loc[chunk.game_id.map(season_by_game).eq(chunk.nfl_id.map(player_year))]
                if chunk.empty:
                    continue
                # Track keys at play grain to avoid counting every 10 Hz frame as an unmatched play.
                keys = chunk[["game_id", "play_id", "nfl_id"]].drop_duplicates()
                for game_id, play_id, nfl_id in keys.itertuples(index=False, name=None):
                    key = (int(game_id), int(play_id), int(nfl_id))
                    if key in seen_tracking_keys:
                        continue
                    seen_tracking_keys.add(key)
                    counts["rookie_reg_tracking_keys"] += 1
                    season = season_by_game[key[0]]
                    position = player_pos.get(key[2], "UNKNOWN")
                    total_by_year[season] += 1
                    total_by_position[position] += 1
                    if key in rookie_pp:
                        counts["matched_rookie_reg_keys"] += 1
                        continue
                    by_year[season] += 1
                    by_position[position] += 1
                    if (key[0], key[1]) in player_play_game_keys:
                        has_other_player_play_row += 1
                    if key in all_pp:
                        reason = "player_play row exists, but not classified as rookie regular season"
                        by_reason["player_play row exists outside rookie REG filter"] += 1
                        bucket = "nonrookie_or_nonreg"
                    else:
                        reason = "no exact player_play key in source table"
                        by_reason["no exact player_play key"] += 1
                        bucket = "no_player_play_row"
                    if len(examples[bucket]) < 20:
                        examples[bucket].append(key)

    counts["unmatched_rookie_reg_keys"] = sum(by_year.values())
    missing_tracking_keys = rookie_pp - seen_tracking_keys
    missing_tracking_by_year = Counter(season_by_game[game_id] for game_id, _, _ in missing_tracking_keys)
    missing_tracking_by_position = Counter(player_pos.get(nfl_id, "UNKNOWN") for _, _, nfl_id in missing_tracking_keys)
    missing_tracking_by_player = Counter(nfl_id for _, _, nfl_id in missing_tracking_keys)
    model_sensitivity = []
    oof_path = OUT / "dl_early_accel_predraft_oof_100plays.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        affected_ids = set(missing_tracking_by_player).intersection(set(pd.to_numeric(oof.nfl_id, errors="coerce").dropna().astype(int)))
        for alpha, group in oof.groupby("alpha"):
            valid = group.dropna(subset=["actual", "baseline", "candidate"]).copy()
            affected = set(missing_tracking_by_player).intersection(set(valid.nfl_id.astype(int)))
            unaffected = valid.loc[~valid.nfl_id.isin(affected)]
            def rmse_delta(frame: pd.DataFrame) -> float | None:
                if frame.empty:
                    return None
                base = float(((frame.baseline - frame.actual) ** 2).mean() ** 0.5)
                model = float(((frame.candidate - frame.actual) ** 2).mean() ** 0.5)
                return model - base
            model_sensitivity.append({
                "alpha": float(alpha),
                "all_players": int(valid.nfl_id.nunique()),
                "affected_players": int(len(affected)),
                "rmse_delta_all_ydps": rmse_delta(valid),
                "rmse_delta_after_excluding_affected_ydps": rmse_delta(unaffected),
            })
    report = {
        "archive": ARCHIVE.name,
        "definition": "Unique (game_id, play_id, nfl_id) keys for rostered prospects' own rookie-year regular-season tracking, classified against exact player_play keys.",
        "counts": dict(counts),
        "unmatched_by_season": dict(sorted(by_year.items())),
        "unmatched_by_position": dict(sorted(by_position.items())),
        "tracking_keys_by_season": dict(sorted(total_by_year.items())),
        "tracking_keys_by_position": dict(sorted(total_by_position.items())),
        "unmatched_rate_by_season": {str(year): by_year[year] / total_by_year[year] for year in sorted(total_by_year)},
        "unmatched_rate_by_position": {position: by_position[position] / total_by_position[position] for position in sorted(total_by_position)},
        "unmatched_keys_on_plays_with_some_player_play_rows": has_other_player_play_row,
        "player_play_keys_without_tracking": len(missing_tracking_keys),
        "player_play_keys_without_tracking_by_season": dict(sorted(missing_tracking_by_year.items())),
        "player_play_keys_without_tracking_by_position": dict(sorted(missing_tracking_by_position.items())),
        "players_with_player_play_keys_without_tracking": [
            {"nfl_id": int(nfl_id), "draft_year": int(player_year[nfl_id]),
             "position": player_pos.get(nfl_id, "UNKNOWN"), "missing_keys": int(n)}
            for nfl_id, n in sorted(missing_tracking_by_player.items(), key=lambda item: (-item[1], item[0]))
        ],
        "early_acceleration_oof_exclusion_sensitivity": model_sensitivity,
        "unmatched_reason": dict(by_reason),
        "examples": {k: [list(v) for v in vals] for k, vals in examples.items()},
        "caveat": "An exact player_play miss identifies a source-table linkage gap; it does not establish which source table is wrong or imply bad sensor data."
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "game_tracking_linkage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Game-tracking linkage audit",
        "",
        f"Rookie regular-season tracking play keys: {counts['rookie_reg_tracking_keys']:,}",
        f"Matched exact player_play keys: {counts['matched_rookie_reg_keys']:,}",
        f"Unmatched exact keys: {counts['unmatched_rookie_reg_keys']:,}",
        f"Rookie player_play keys without tracking: {len(missing_tracking_keys):,}",
        "",
        "## Unmatched by season",
        "",
        "| Season | Keys |",
        "|---:|---:|",
        *[f"| {year} | {n:,} |" for year, n in sorted(by_year.items())],
        "",
        "## Match rate by position",
        "",
        "| Position | Tracking keys | Unmatched keys | Unmatched rate |",
        "|---|---:|---:|---:|",
        *[f"| {position} | {total_by_position[position]:,} | {by_position[position]:,} | {by_position[position] / total_by_position[position]:.2%} |" for position in sorted(total_by_position)],
        "",
        "## Classification",
        "",
        *[f"- {reason}: {n:,}" for reason, n in by_reason.items()],
        f"- Unmatched keys on plays with some player_play rows: {has_other_player_play_row:,}",
        "",
        "## Player-play keys without tracking",
        "",
        "| Position | Keys |",
        "|---|---:|",
        *[f"| {position} | {n:,} |" for position, n in sorted(missing_tracking_by_position.items())],
        "",
        "## Players with the most missing tracking keys",
        "",
        "| NFL ID | Draft year | Position | Keys |",
        "|---:|---:|---|---:|",
        *[
            f"| {int(nfl_id)} | {int(player_year[nfl_id])} | {player_pos.get(nfl_id, 'UNKNOWN')} | {n:,} |"
            for nfl_id, n in sorted(missing_tracking_by_player.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
        "",
        "## Primary DL model exclusion sensitivity",
        "",
        "Players with missing tracking keys are excluded here as a conservative check; this is not an imputation of their missing play speeds.",
        "",
        "| Alpha | Players | Affected | RMSE change, all | RMSE change, excluding affected |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {r['alpha']:g} | {r['all_players']} | {r['affected_players']} | {r['rmse_delta_all_ydps']:.4f} | {r['rmse_delta_after_excluding_affected_ydps']:.4f} |"
            for r in model_sensitivity
        ],
        "",
        "Exact-key misses indicate a source-table linkage gap; they do not establish which table is wrong or imply incorrect sensor data.",
        ""
    ]
    (OUT / "game_tracking_linkage_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit rookie game-tracking linkage to player_play metadata.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Competition ZIP archive")
    args = parser.parse_args()
    main(args.archive)
