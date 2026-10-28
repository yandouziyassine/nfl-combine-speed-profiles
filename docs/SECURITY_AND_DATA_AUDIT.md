# Security, software, and data audit

**Audit date:** 2026-10-26  
**Scope:** local source archive, core derived tables, notebook packaging path, profile explorer, cache provenance, and selected metric transformations.  
**Status:** integrity and schema checks pass for the complete ZIP currently used by this project. This is not a certification that the archive is byte-for-byte identical to Kaggle's hosted copy or that every analysis is scientifically validated.

## Executive summary

The current source ZIP passes its ZIP CRC check and the project data audit. Required competition files are present; the main source and derived player-level tables have the expected keys, unique grains, and row counts. Two security issues found in the local presentation/packaging code have been corrected: unsafe extraction of a code bundle in the Kaggle notebook and unsafe interpolation of dataset-derived text in the HTML explorer.

The audit supports claims about file integrity and table structure. It does not prove that every raw sensor value is physically accurate, that the ZIP matches Kaggle's server copy, or that exploratory models generalize to future draft cohorts. Current model evidence is weak or negative for added career prediction value from the 40-yard phases.

## Dataset provenance and integrity

| Check | Result |
|---|---|
| Audited archive | `nfl-big-data-bowl-2027.zip` |
| Archive size | 512,718,566 bytes |
| SHA-256 | `27d52f594d6d31d4f617f9c94b0b37d11e280397ad8293c0987da5f91a8d0ced` |
| ZIP members | 9 |
| Uncompressed size | 2,306,804,135 bytes (~2.31 GB) |
| ZIP CRC test | Passed; no corrupt member reported |
| Automated project audit | Passed; see [`../outputs/project_data_audit.json`](../outputs/project_data_audit.json) |

The archive contains 510 prospects across draft cohorts 2023–2025. The source audit found 463,189 Combine tracking rows, of which 431,094 are player rows; these resolve to 6,310 unique player/drill/attempt keys. An independent timestamp check using the full attempt key found no duplicate or non-increasing times; the 424,784 within-attempt intervals are all within 5 ms of the expected 0.1-second cadence. All 793 40-yard attempts meet this cadence check. `player_play.csv` has 314,197 unique `(game_id, play_id, nfl_id)` rows and its player IDs map to the roster. Regular-season game-tracking files contain 22,654,290 rows in total (2023: 3,665,443; 2024: 7,827,944; 2025: 11,160,903); no null `nfl_id` values were found. Career outcomes have 510 unique roster players and no negative snap totals.

The `.crdownload` path cited earlier was not present during this audit. There was no trusted Kaggle-provided checksum available for comparison. Therefore, the SHA-256 above identifies the local archive audited here; it does not establish remote-source authenticity or prove equivalence to the partial download.

## Derived-table integrity

The audit validates key uniqueness, key completeness, roster linkage where applicable, and expected row counts for the primary outputs:

| Output | Grain | Rows |
|---|---|---:|
| `all_player_speed_profiles.csv` | player | 510 |
| `player_speed_signatures.csv` | player | 510 |
| `player_rookie_game_speed_profiles.csv` | player, including missing outcomes | 510 |
| `combine_attempt_speed_profiles.csv` | player × drill attempt | 6,310 |
| `player_drill_speed_profiles.csv` | player × drill | 4,986 |
| `player_40_speed_profiles.csv` | player with tracked 40-yard profile | 418 |

The full machine-readable check report records the keys tested and per-check results. Missing measurements remain missing; the profile does not turn an unobserved drill or game outcome into zero or an average. Source row counts and key checks cannot detect every unit, coordinate, timestamp, or sensor-calibration error, so metric definitions and outcome caveats are documented separately in [`METHODS.md`](METHODS.md).

### Game-tracking to player-play linkage

A separate exact-key audit compared rookie regular-season tracking keys with `player_play.csv` in both directions. Of 123,524 tracked player-play keys for the 510 Combine prospects, 122,335 (99.04%) had an exact `(game_id, play_id, nfl_id)` metadata match; 1,189 did not. The unmatched rate was 1.09% in 2023, 0.98% in 2024, and 0.82% in 2025. Those unmatched tracking keys were concentrated in WR (868/17,674; 4.91%) and TE (306/4,400; 6.95%), with 15/44 fullback keys (34.09%). 1,028 of the 1,189 unmatched player keys occurred on plays that had at least one `player_play` row for another player; 161 occurred on plays with no `player_play` row.

In the reverse direction, 195 of 122,530 rookie `player_play` keys had no tracking trace. Ten were defensive-line positions (8 DE, 2 DT). Three of the 46 players in the primary 100-play DL model sample had seven such keys in total. As a conservative point-estimate sensitivity, excluding those three players changed the early-feature RMSE difference from -0.0497 to -0.0638 yd/s at alpha 5, and from -0.0410 to -0.0504 yd/s at alpha 20. This leave-out check is not a correction for missing-not-at-random tracking and does not recover unobserved speeds. These are key-level coverage gaps, not evidence that either source is erroneous. See [`../outputs/game_tracking_linkage_audit_report.md`](../outputs/game_tracking_linkage_audit_report.md) and its JSON source.

The current profile builder excludes tracking keys that do not have an exact `player_play` match. This omission is small overall but can affect position-specific analyses for WRs, TEs, and especially FBs; those analyses should report linkage coverage and should not use unmatched plays for metadata-dependent outcomes.

## Software and security review

- **Unsafe archive extraction — fixed.** The Kaggle notebook no longer calls `ZipFile.extractall` for its code bundle. Its checked extraction rejects absolute paths, parent traversal, Windows drive/path forms, backslashes, symbolic links, duplicate normalized members, and file/directory conflicts. It caps the bundle at 500 members and 128 MiB uncompressed, with a 32 MiB per-file ceiling, and refuses ambiguous inputs instead of silently selecting the first match.
- **HTML injection — fixed.** Dataset-derived player/drill strings are escaped before insertion in the profile explorer. Embedded JSON escapes characters that could terminate the inline script context. The explorer was regenerated after this change.
- **Portable input configuration — fixed.** Analysis scripts now use the `NFL_BDB_INPUT` environment variable or a conventional local ZIP name instead of a developer-specific Windows path; `--archive` remains the explicit override.
- **Stale game-profile cache — fixed in source.** `--reuse-game-profiles` now requires a matching source-member identity, producing-script digest, cache schema, linkage metadata, and cached-CSV checksum. ZIP inputs use member CRCs and sizes; extracted files use relative paths, sizes, and modification times. This is a practical cache identity check, not cryptographic provenance for the full dataset.
- **Core output inventory — added.** `write_core_run_manifest.py` verifies the game-profile cache key and required outputs before writing a run manifest with artifact hashes and environment versions. It distinguishes a completed local output set from an incomplete or stale one; it does not validate scientific conclusions.
- **Bundle provenance — improved.** The source bundle now uses sorted entries and fixed ZIP timestamps, includes pinned direct dependencies, writes a SHA-256 sidecar, and packages the current packaging audit after updating it.
- **Static dangerous-pattern scan.** No use of `eval`, `exec`, `os.system`, or `shell=True`, and no hard-coded passwords, API keys, or tokens were found in the reviewed project sources. This is a source scan, not a full secrets scanner or dependency vulnerability assessment.
- **Reproducibility.** Bootstrap uncertainty intervals use seeded pseudo-random resampling of observed players. This creates resampled inference draws, not synthetic player records; it does not fabricate source tracking or outcome data. Tests use small controlled fixtures to verify calculations and are explicitly not research observations.
- **Automated checks.** `audit_project_data.py` validates archive CRC, required members, source keys, roster joins, tracking row counts, and primary derived-table grains. Four focused unit tests cover interpolation, absolute-agreement ICC behavior on a controlled fixture, Combine movement metrics, and 10 Hz snap-relative timing.

## Scientific reliability and current evidence

The local archive passed CRC and source-key/grain checks. Models use player-level inference, draft-cohort holdouts, and training-fold preprocessing. The full-play p90 early-acceleration result is positive in a small DL sample, but fixed 1–4 second post-snap targets show no consistent gain; it does not establish improved pressure production, player quality, or a causal training effect. The career-role, WR route-recovery, DB tackle-rate, and OL mirror screens remain null, negative, or exploratory. Detailed results and limitations are in [RESEARCH_DESIGN.md](../RESEARCH_DESIGN.md) and [MODEL_AUDIT.md](MODEL_AUDIT.md).

The code bundle now contains the notebook, all analysis modules used by it, and project documentation. Its archive layout is checked against the notebook's extraction path. A clean cloud execution on Kaggle has not yet been performed from this workspace.

## Remaining validation work

1. Compare the local archive SHA-256 with a trusted Kaggle checksum or a fresh authenticated download; the current hash identifies the local file but does not prove remote-source authenticity.
2. Run the full notebook from a clean Kaggle session and inspect all outputs before publishing. The local package check verifies archive integrity, referenced scripts, and notebook syntax, but cannot substitute for Kaggle execution.
3. Review raw DL 40-yard traces and sensor ranges for outliers, coordinate anomalies, and feature-boundary behavior; current reliability/sensitivity checks do not constitute a blinded visual review of every trace.
4. Perform a dependency vulnerability review before treating the code as production software.
5. Validate on a future draft/season cohort when matching NFL tracking becomes available.

No claim in this report should be read as external validation or as a guarantee that every raw sensor value is error-free.

