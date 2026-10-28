"""Integrity and lineage audit for ZIP or extracted NFL Big Data Bowl inputs.

This audit checks source-container integrity when applicable, core competition
schemas, source IDs, tracking row counts, and derived-table grain. It cannot
certify byte identity with Kaggle's server copy because no official checksum is
provided in the local project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from run_analysis import DEFAULT_ARCHIVE, open_data_source, parse_sensor_times

MEMBER = "nfl-big-data-bowl-2027/"
REQUIRED = {
    "combine_results.csv", "combine_tracking.csv", "game_tracking_2023.csv",
    "game_tracking_2024.csv", "game_tracking_2025.csv", "games.csv",
    "player_career_successes.csv", "player_play.csv", "players.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(zf: zipfile.ZipFile, member: str, usecols: list[str] | None = None) -> tuple[int, int]:
    rows = null_ids = 0
    for chunk in pd.read_csv(zf.open(MEMBER + member), usecols=usecols,
                             chunksize=500_000, low_memory=False):
        rows += len(chunk)
        if usecols and "nfl_id" in usecols:
            null_ids += int(chunk.nfl_id.isna().sum())
    return rows, null_ids


def check_unique(frame: pd.DataFrame, keys: list[str]) -> dict:
    missing = int(frame[keys].isna().any(axis=1).sum())
    duplicates = int(frame.duplicated(keys).sum())
    return {"rows": int(len(frame)), "key_columns": keys,
            "rows_with_missing_key": missing, "duplicate_keys": duplicates,
            "passes": missing == 0 and duplicates == 0}


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--outputs", type=Path, default=project / "outputs")
    parser.add_argument("--report", type=Path, default=project / "outputs" / "project_data_audit.json")
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    is_directory = archive_path.is_dir()
    report = {"audit_version": 1, "archive_name": archive_path.name,
              "source_mode": "extracted_directory" if is_directory else "zip_archive",
              "archive_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
              "sha256": sha256(archive_path) if archive_path.is_file() else None,
              "checks": {}, "warnings": [], "overall_pass": False}

    with open_data_source(archive_path) as zf:
        names = {Path(n).name for n in zf.namelist()}
        missing = sorted(REQUIRED - names)
        report["archive_members"] = len(zf.namelist())
        if is_directory:
            report["uncompressed_bytes"] = sum((archive_path / name).stat().st_size for name in zf.namelist())
            report["checks"]["zip_crc"] = {"applicable": False, "corrupt_member": None, "passes": None,
                                             "note": "Input is already extracted; ZIP CRC is not applicable."}
        else:
            report["uncompressed_bytes"] = sum(info.file_size for info in zf.infolist())
            corrupt = zf.testzip()
            report["checks"]["zip_crc"] = {"applicable": True, "corrupt_member": corrupt,
                                             "passes": corrupt is None}
        report["checks"]["required_members"] = {"missing": missing, "passes": not missing}
        if missing:
            report["warnings"].append("Required competition files are missing; downstream checks were skipped.")
        else:
            players = pd.read_csv(zf.open(MEMBER + "players.csv"))
            player_ids = set(pd.to_numeric(players.nfl_id, errors="coerce").dropna().astype("int64"))
            report["source_counts"] = {"players": int(len(players)),
                                       "players_unique_ids": int(players.nfl_id.nunique()),
                                       "draft_years": sorted(int(x) for x in players.draft_year.dropna().unique())}
            report["checks"]["player_roster"] = {
                "rows": int(len(players)), "unique_nfl_ids": int(players.nfl_id.nunique()),
                "null_nfl_ids": int(players.nfl_id.isna().sum()),
                "passes": bool(len(players) == 510 and players.nfl_id.nunique() == len(players) and players.nfl_id.notna().all()),
            }

            combine_results = pd.read_csv(zf.open(MEMBER + "combine_results.csv"))
            report["checks"]["combine_results"] = check_unique(combine_results, ["draft_year", "nfl_id"])
            unknown_results = int((~pd.to_numeric(combine_results.nfl_id, errors="coerce").isin(player_ids)).sum())
            report["checks"]["combine_result_roster_link"] = {"unknown_player_rows": unknown_results, "passes": unknown_results == 0}

            combine_rows, combine_null_ids = count_rows(zf, "combine_tracking.csv", ["nfl_id"])
            combine = pd.read_csv(zf.open(MEMBER + "combine_tracking.csv"),
                                  usecols=["draft_year", "event_id", "nfl_id", "entity_type", "drill_type", "drill_name", "attempt", "time"])
            athlete = combine.loc[combine.entity_type.eq("PLAYER")].copy()
            attempt_keys = ["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt"]
            # Tracking rows repeat by frame. Check raw frame identity/timing as
            # well as the derived player-attempt grain; dropping duplicate
            # attempt keys alone cannot detect duplicated or malformed frames.
            attempt_keys_frame = athlete[attempt_keys].drop_duplicates().reset_index(drop=True)
            attempt_check = check_unique(attempt_keys_frame, attempt_keys)
            missing_attempt_keys = int(athlete[attempt_keys].isna().any(axis=1).sum())
            parsed_time = parse_sensor_times(athlete.time, utc=True)
            duplicate_frame_rows = int(athlete.duplicated(attempt_keys + ["time"]).sum())
            intervals = parsed_time.groupby([athlete[k] for k in attempt_keys], sort=False).diff().dt.total_seconds()
            bad_time_rows = int(parsed_time.isna().sum())
            nonincreasing_intervals = int((intervals <= 0).sum())
            out_of_cadence_intervals = int((intervals.notna() & ((intervals - 0.1).abs() > 0.005)).sum())
            attempt_check.update({"source_combine_rows": combine_rows,
                                  "player_tracking_rows": int(len(athlete)),
                                  "player_tracking_null_ids": int(athlete.nfl_id.isna().sum()),
                                  "unique_player_attempts": int(len(attempt_keys_frame)),
                                  "raw_rows_with_missing_attempt_keys": missing_attempt_keys,
                                  "duplicate_attempt_timestamp_rows": duplicate_frame_rows,
                                  "unparseable_timestamps": bad_time_rows,
                                  "nonincreasing_within_attempt_intervals": nonincreasing_intervals,
                                  "intervals_outside_0_100s_plusminus_5ms": out_of_cadence_intervals,
                                  "within_attempt_intervals": int(intervals.notna().sum())})
            attempt_check["passes"] = bool(
                attempt_check["passes"]
                and athlete.nfl_id.isin(player_ids).all()
                and missing_attempt_keys == 0
                and duplicate_frame_rows == 0
                and bad_time_rows == 0
                and nonincreasing_intervals == 0
                and out_of_cadence_intervals == 0
            )
            report["checks"]["combine_attempt_keys"] = attempt_check

            games = pd.read_csv(zf.open(MEMBER + "games.csv"))
            report["checks"]["games_keys"] = check_unique(games, ["game_id"])
            tracking_counts = {}
            tracking_passes = True
            for year in (2023, 2024, 2025):
                member = f"game_tracking_{year}.csv"
                n_rows, n_null = count_rows(zf, member, ["nfl_id"])
                tracking_counts[str(year)] = {"rows": n_rows, "null_nfl_ids": n_null}
                tracking_passes &= n_rows > 0 and n_null == 0
            report["source_counts"]["game_tracking_rows_by_year"] = tracking_counts
            report["checks"]["game_tracking_files"] = {"passes": bool(tracking_passes)}

            player_play = pd.read_csv(zf.open(MEMBER + "player_play.csv"),
                                      usecols=["game_id", "play_id", "nfl_id"])
            play_key_check = check_unique(player_play, ["game_id", "play_id", "nfl_id"])
            play_key_check["players_in_roster"] = bool(player_play.nfl_id.isin(player_ids).all())
            play_key_check["passes"] = play_key_check["passes"] and play_key_check["players_in_roster"]
            report["checks"]["player_play_keys"] = play_key_check

            career = pd.read_csv(zf.open(MEMBER + "player_career_successes.csv"))
            career_key_check = check_unique(career, ["nfl_id"])
            snap_cols = [c for c in career.columns if c.endswith("_snaps")]
            negative = {c: int((pd.to_numeric(career[c], errors="coerce") < 0).sum()) for c in snap_cols}
            career_key_check["negative_snap_counts"] = negative
            career_key_check["players_in_roster"] = bool(career.nfl_id.isin(player_ids).all())
            career_key_check["passes"] = bool(career_key_check["passes"] and career_key_check["players_in_roster"] and not any(negative.values()))
            report["checks"]["career_outcomes"] = career_key_check

    # These checks confirm the local derived tables maintain their stated grain.
    checks = report["checks"]
    output_checks = {}
    tables = {
        "all_player_speed_profiles.csv": (["nfl_id"], 510),
        "player_speed_signatures.csv": (["nfl_id"], 510),
        "player_rookie_game_speed_profiles.csv": (["nfl_id"], 510),
        "combine_attempt_speed_profiles.csv": (["draft_year", "event_id", "nfl_id", "drill_type", "drill_name", "attempt"], 6310),
        "player_drill_speed_profiles.csv": (["nfl_id", "draft_year", "drill_type", "drill_name"], 4986),
        "player_40_speed_profiles.csv": (["nfl_id"], 418),
    }
    for filename, (keys, expected_rows) in tables.items():
        path = args.outputs / filename
        if not path.exists():
            output_checks[filename] = {"exists": False, "passes": False}
            continue
        frame = pd.read_csv(path)
        result = check_unique(frame, keys)
        result["exists"] = True
        result["expected_rows"] = expected_rows
        result["row_count_matches"] = len(frame) == expected_rows
        result["passes"] = bool(result["passes"] and result["row_count_matches"] and frame.nfl_id.isin(player_ids).all())
        output_checks[filename] = result
    checks["derived_core_tables"] = {"tables": output_checks,
                                     "passes": all(x.get("passes", False) for x in output_checks.values())}

    # No interpolation or synthetic player rows are expected in these artifacts.
    # Missing drill measurements remain null by design.
    report["method_note"] = "Bootstrap draws are used only for player-level uncertainty intervals; they are not synthetic player observations."
    report["overall_pass"] = all(c.get("passes") is True for c in checks.values()
                                  if c.get("applicable", True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"overall_pass": report["overall_pass"], "sha256": report["sha256"],
                      "checks": {k: ("not applicable" if not v.get("applicable", True)
                                     else bool(v.get("passes", False)))
                                 for k, v in checks.items()},
                      "report": str(args.report.resolve())}, indent=2))


if __name__ == "__main__":
    main()
