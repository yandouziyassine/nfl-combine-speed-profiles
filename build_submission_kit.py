"""Refresh the self-contained, locally reviewable competition handoff."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KIT = ROOT / "submission_kit"
LOCK = KIT / ".submission_kit.lock"
FIGURES = (
    "dl_early_acceleration_summary.png",
    "dl_speed_profile_case_illustration.png",
    "fig_dl_game_speed_split_half.png",
    "fig_position_phase_nested_selection.png",
)
OTHER_FILES = (
    "nfl_combine_speed_analysis.ipynb",
    "nfl_speed_project_code.zip",
    "nfl_speed_project_code.zip.sha256",
    "KAGGLE_SUBMISSION_GUIDE.md",
    "DEMO_README.md",
)
AUDIT_DOCS = (
    "ANALYSIS_SEARCH_REGISTRY.md",
    "PRIMARY_OUTCOME_CHARTER.md",
    "MEASUREMENT_UNCERTAINTY.md",
)

KIT_README = """# NFL Big Data Bowl 2027: Combine speed profiles

This handoff contains an interactive prospect profile explorer, the competition writeup, a Kaggle notebook, and a reproducible source bundle.

## Start here

1. Open [`speed_profile_explorer.html`](speed_profile_explorer.html), select Prospect A, choose a same-position Prospect B, then use **Print comparison brief** for a descriptive side-by-side report.
2. Read [`COMPETITION_SUBMISSION.md`](COMPETITION_SUBMISSION.md) for methods, figures, and evidence limits.
3. Read [`ANALYSIS_SEARCH_REGISTRY.md`](ANALYSIS_SEARCH_REGISTRY.md), [`PRIMARY_OUTCOME_CHARTER.md`](PRIMARY_OUTCOME_CHARTER.md), and [`MEASUREMENT_UNCERTAINTY.md`](MEASUREMENT_UNCERTAINTY.md) for the search audit, locked next-study endpoint, and uncertainty treatment.
4. Use [`KAGGLE_SUBMISSION_GUIDE.md`](KAGGLE_SUBMISSION_GUIDE.md) to run [`nfl_combine_speed_analysis.ipynb`](nfl_combine_speed_analysis.ipynb) with the official competition data.
5. Verify the source archive with [`nfl_speed_project_code.zip.sha256`](nfl_speed_project_code.zip.sha256).

## Main finding and evidence boundary

The selection-aware 12-candidate audit found near-zero pooled held-out error change (-0.0033 yd/s; its interval includes zero) and did not select DL early acceleration. The larger -0.0497 yd/s estimate comes from a post-screen discovery analysis of 46 exposed DLs. Treat that estimate as hypothesis-generating, not confirmatory or an individual player projection. See the writeup for the fixed-window, context-adjusted, and pressure results that limit interpretation.

The HTML preview embeds player-level values derived from the competition data. It is included for local review; check Kaggle's data-use terms before redistributing or publishing it. The HTML is separate from the source ZIP, which contains reproducible code without embedded player profiles.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_files() -> dict[str, Path]:
    files = {
        "README.md": None,
        "COMPETITION_SUBMISSION.md": ROOT / "docs" / "COMPETITION_SUBMISSION.md",
        "speed_profile_explorer.html": ROOT / "outputs" / "speed_profile_explorer.html",
    }
    for name in AUDIT_DOCS:
        files[name] = ROOT / "docs" / name
    for name in OTHER_FILES:
        files[name] = (KIT / name) if name in {"KAGGLE_SUBMISSION_GUIDE.md", "DEMO_README.md"} else ROOT / name
    for name in FIGURES:
        files[name] = ROOT / "outputs" / name
    return files


def main() -> None:
    KIT.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"A submission-kit build is active, or a stale lock remains at {LOCK}.") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token)

    staging = ROOT / f".submission_kit.stage-{token}"
    try:
        previous = json.loads((KIT / "MANIFEST.json").read_text(encoding="utf-8")) if (KIT / "MANIFEST.json").is_file() else {}
        atomic_json(KIT / "MANIFEST.json", {
            "schema_version": 1,
            "status": "in_progress",
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "previous_manifest": previous.get("generated_at_utc"),
        })
        staging.mkdir()
        files = source_files()
        code_zip = files["nfl_speed_project_code.zip"]
        checksum = files["nfl_speed_project_code.zip.sha256"]
        expected = checksum.read_text(encoding="ascii").split()[0].lower()
        if sha256_file(code_zip) != expected:
            raise RuntimeError("Code-bundle SHA-256 does not match its sidecar; rebuild the code bundle first.")

        writeup = files["COMPETITION_SUBMISSION.md"].read_text(encoding="utf-8")
        for name in FIGURES:
            writeup = writeup.replace(f"../outputs/{name}", name)
        (staging / "COMPETITION_SUBMISSION.md").write_text(writeup, encoding="utf-8")
        (staging / "README.md").write_text(KIT_README, encoding="utf-8")
        for name, source in files.items():
            if name in {"README.md", "COMPETITION_SUBMISSION.md"}:
                continue
            if source is None or not source.is_file():
                raise FileNotFoundError(f"Required handoff input is missing: {source}")
            if name == "KAGGLE_SUBMISSION_GUIDE.md":
                guide = source.read_text(encoding="utf-8")
                guide = guide.replace("docs/COMPETITION_SUBMISSION.md", "COMPETITION_SUBMISSION.md")
                for figure in FIGURES:
                    guide = guide.replace(f"outputs/{figure}", figure)
                (staging / name).write_text(guide, encoding="utf-8")
            else:
                shutil.copy2(source, staging / name)

        inventory = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "generated_at_utc": generated_at,
            "artifact_count": len(inventory),
            "artifacts": inventory,
            "note": "The explorer embeds competition-derived player profiles and is for locally authorized review; check data-use terms before redistribution.",
        }
        for source in sorted(staging.iterdir(), key=lambda item: item.name):
            source.replace(KIT / source.name)
        staging.rmdir()
        for name, expected in inventory.items():
            target = KIT / name
            observed = {"size_bytes": target.stat().st_size, "sha256": sha256_file(target)}
            if observed != expected:
                raise RuntimeError(f"Promoted handoff artifact {name} failed size/SHA-256 verification.")
        atomic_json(KIT / "MANIFEST.json", manifest)
        print(json.dumps({"status": "complete", "artifacts": len(inventory), "manifest": "submission_kit/MANIFEST.json"}, indent=2))
    finally:
        try:
            if LOCK.is_file() and LOCK.read_text(encoding="utf-8") == token:
                LOCK.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
