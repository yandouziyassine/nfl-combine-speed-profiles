"""Validate and inventory the completed core profile pipeline outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import DEFAULT_ARCHIVE
from run_analysis import OutputDirectoryLock, game_profile_cache_identity, write_json_atomic


REQUIRED_OUTPUTS = (
    "combine_attempt_speed_profiles.csv",
    "player_drill_speed_profiles.csv",
    "player_40_speed_profiles.csv",
    "player_rookie_game_speed_profiles.csv",
    "all_player_speed_profiles.csv",
    "model_results.json",
    "coverage_and_results.json",
    "speed_analysis_report.md",
    "game_speed_profiles_cache_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    outdir = args.output.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with OutputDirectoryLock(outdir):
        write_core_manifest(args, outdir)


def write_core_manifest(args, outdir: Path) -> None:
    if not args.archive.exists():
        raise FileNotFoundError(f"Competition input path not found: {args.archive}")

    identity = game_profile_cache_identity(args.archive)
    manifest_path = outdir / "core_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Core run start marker is missing; run run_analysis.py first.")
    started = json.loads(manifest_path.read_text(encoding="utf-8"))
    if started.get("status") != "promoted":
        raise RuntimeError("Core outputs have not completed promotion; refusing to certify incomplete or potentially stale outputs.")
    if started.get("core_profile_cache_key") != identity["cache_key"]:
        raise RuntimeError("Core run start marker does not match the current input identity.")
    cache_path = outdir / "game_speed_profiles_cache_manifest.json"
    if not cache_path.is_file():
        raise FileNotFoundError("Core pipeline cache manifest is missing; run run_analysis.py first.")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("cache_key") != identity["cache_key"]:
        raise RuntimeError("Core outputs were generated from a different input identity or producer version.")

    coverage_path = outdir / "coverage_and_results.json"
    if not coverage_path.is_file():
        raise FileNotFoundError("Core coverage report is missing; run run_analysis.py first.")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("game_profile_cache_key") != identity["cache_key"]:
        raise RuntimeError("Coverage report does not match the game-profile cache identity.")
    run_id = started.get("run_id")
    if not run_id or coverage.get("core_run_id") != run_id or cache.get("run_id") != run_id:
        raise RuntimeError("Core artifacts do not share the active run ID; refusing to certify mixed or stale outputs.")

    names = list(REQUIRED_OUTPUTS)
    chart = coverage.get("output_chart")
    if chart:
        names.append(chart)
    expected_inventory = started.get("staged_outputs")
    if not isinstance(expected_inventory, dict) or set(expected_inventory) != set(names):
        raise RuntimeError("Promoted output inventory does not match the required core output set.")
    missing = [name for name in names if not (outdir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Core pipeline outputs are incomplete: {missing}")
    inventory = {}
    for name in sorted(set(names)):
        path = outdir / name
        observed = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed != expected_inventory[name]:
            raise RuntimeError(f"Promoted output {name} does not match its staged size/SHA-256 inventory.")
        inventory[name] = observed

    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "manifest_schema_version": 2,
        "pipeline": "combine-to-rookie-game-speed-core",
        "status": "complete",
        "completed_at_utc": completed_at,
        "started_at_utc": started.get("started_at_utc"),
        "run_id": run_id,
        "core_profile_cache_key": identity["cache_key"],
        "input": identity["input"],
        "producer": identity["producer"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": inventory,
        "interpretation": "This manifest certifies that the listed core files exist and records their hashes; it does not certify scientific validity or Kaggle-source authenticity.",
    }
    write_json_atomic(manifest_path, payload)
    print(json.dumps({"status": payload["status"], "run_id": payload["run_id"],
                      "core_profile_cache_key": payload["core_profile_cache_key"],
                      "input_fingerprint_method": identity["input"]["fingerprint_method"],
                      "outputs": len(inventory), "manifest": manifest_path.name}, indent=2))


if __name__ == "__main__":
    main()
