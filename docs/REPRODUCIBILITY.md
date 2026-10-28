# Reproducibility runbook

## Environment

Python 3.10 or newer. Install the pinned direct dependencies with `python -m pip install -r requirements.txt`; the locally replayed environment used Python 3.10.11, NumPy 1.26.4, pandas 2.3.3, and Matplotlib 3.10.8. The pipeline accepts the official competition ZIP or an extracted competition directory and does not require network access after dependencies and data are present. The full Kaggle notebook and code bundle are supplied together.

## Local reproduction

From the project root, pass the archive path explicitly:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:NFL_BDB_INPUT = "C:\path\to\nfl-big-data-bowl-2027.zip"
python audit_project_data.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip"
if ($LASTEXITCODE -ne 0) { throw "Data audit failed; stop before analysis." }
python run_analysis.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip" --output outputs
if ($LASTEXITCODE -ne 0) { throw "Core analysis failed; outputs remain uncertified." }
python write_core_run_manifest.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip" --output outputs
if ($LASTEXITCODE -ne 0) { throw "Core output inventory failed; outputs remain uncertified." }
python build_speed_signatures.py
python analyze_40_repeatability.py
python analyze_40_repeatability_by_position.py
python analyze_40_feature_sensitivity.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip"
python audit_dl_early_accel_measurement_transfer.py
python model_dl_game_speed_ablation.py --min-game-plays 50
python model_dl_game_speed_ablation.py --min-game-plays 100
python model_dl_game_speed_role_sensitivity.py --min-game-plays 100
python model_dl_early_accel_transfer.py --min-game-plays 50
python model_dl_early_accel_transfer.py --min-game-plays 100
python model_dl_early_accel_predraft.py --min-game-plays 25
python model_dl_early_accel_predraft.py --min-game-plays 50
python model_dl_early_accel_predraft.py --min-game-plays 100
python audit_dl_attempt_specific_transfer.py
python model_dl_early_accel_predraft.py --min-game-plays 200
python audit_game_tracking_linkage.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip"
python build_dl_speed_profile_case_illustration.py
python analyze_dl_hoop.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip"
python analyze_dl_game_burst.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip"
python model_dl_transfer.py
python model_dl_pressure_context.py
python model_dl_40_accel_to_football_outcomes.py --min-burst-plays 50
python model_dl_40_accel_to_football_outcomes.py --min-burst-plays 100
python model_game_speed_by_position.py --min-game-plays 100
python model_position_specific_game_speed.py --min-game-plays 100
python audit_position_phase_nested_selection.py
python build_position_phase_nested_figure.py
python analyze_dl_game_speed_target_precision.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip" --draws 1000 --split-repeats 250
python build_dl_split_half_figure.py
python analyze_dl_fixed_window_target.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip" --min-plays 50 100
python audit_dl_nested_phase_selection.py --min-game-plays 100
python audit_40_forward_axis_sensitivity.py --archive "C:\path\to\nfl-big-data-bowl-2027.zip"
python model_dl_target_uncertainty.py --draws 1000
python build_dl_transfer_figure.py
python build_profile_explorer.py
```

Core analysis scripts write machine-readable JSON, readable Markdown, and out-of-fold player prediction tables into `outputs/`. Run upstream profiling before downstream models. The DL early-acceleration and exposure-sensitivity scripts require `outputs/player_speed_signatures.csv`. The target-precision script makes an additional full pass over game-tracking members to retain per-player, per-game play peaks; this supports player-specific game-resampling draws (independent across players) and 250 within-season split-half reliability/model-sensitivity partitions.

`NFL_BDB_INPUT` supplies a portable default for command-line scripts when `--archive` is omitted; an explicit `--archive` always takes precedence. The optional `--reuse-game-profiles` flag only reuses `player_rookie_game_speed_profiles.csv` when its provenance manifest matches the relevant source members, producing script, cache schema, and cached output checksum. ZIP inputs are identified from member CRCs and sizes; extracted-directory inputs use relative paths, sizes, and modification times. This is a cache identity check, not a cryptographic certification of the entire source dataset. A cache without a matching manifest is rejected and must be rebuilt.

`run_analysis.py` acquires an exclusive `.core_run.lock` in the output directory and atomically writes `core_run_manifest.json` with `status=in_progress` and a unique run ID. A second core run or manifest certification targeting that directory fails instead of interleaving. It builds core files in a sibling staging directory, records size and SHA-256 for every staged output, then promotes the inventory while the manifest remains uncertified. If a run stops while building, prior core files remain intact; if promotion stops partway through, the manifest cannot be certified. A failed staging directory is retained for diagnosis; after preserving any files needed to diagnose the failure, remove that inactive sibling directory to reclaim space. If the process is forcibly terminated and leaves a lock behind, first confirm no run is active, then remove `.core_run.lock` before retrying. With `--reuse-game-profiles`, the existing cached game profiles are copied into staging and revalidated before use. The cache manifest and coverage report carry the same run ID. `write_core_run_manifest.py` only writes `status=complete` when the promoted marker, run IDs, output names, sizes, and hashes all agree. The manifest confirms artifact integrity and run provenance, not model validity or source authenticity. Run downstream analysis after the core certification step and never while a core run is active; those other analysis scripts are sequenced by the supplied runbook but do not take this lock themselves.

## Kaggle notebook

Attach both the official NFL Big Data Bowl 2027 dataset and `nfl_speed_project_code.zip` as notebook inputs. Run `nfl_combine_speed_analysis.ipynb` from a clean Kaggle session. If Kaggle discovers multiple datasets or bundles, set `NFL_BDB_INPUT` to the exact official ZIP or extracted directory; the notebook exports its resolved data path so subprocess analysis scripts use the same input.

The notebook:
1. validates that an unambiguous input is available and safely unpacks the code bundle;
2. rebuilds profile and signature tables from source data;
3. runs 40-yard reliability and signal-processing sensitivity analyses;
4. performs an outcome-blind raw-trace review on ten DL profiles (all available attempts) and reports cadence/speed consistency checks;
5. evaluates DL early acceleration at 50- and 100-play exposure, compares phase components, tests sensitivity to selecting attempt 1 versus 2, runs a post-draft role sensitivity, and benchmarks other position groups;
6. evaluates fixed 1–4 second post-snap outcomes and the context-adjusted four-second diagnostic;
7. prints the generated reports and lists the generated figures.

The focused notebook's burst, fixed-window, and target-uncertainty steps add additional scans of the game-tracking files; the target-uncertainty step draws 1,000 outcomes per player. Expect the full notebook to take appreciably longer than the core feature pipeline.

The archive reader processes a large tracking dataset. Expect the core pass to take appreciable time and disk space. This local workspace cannot certify a clean Kaggle cloud execution; the notebook owner must run and inspect it on Kaggle before publication.

## Clean local replay

On October 25, 2026, the staged notebook completed all 43 analysis calls against the Kaggle-style extracted competition directory using the notebook and analysis scripts packaged at replay time. The final report cell passed, and the notebook generated 146 output files, including all four submission PNGs. The extracted-directory source audit passed. The 56 CSVs from the previous analysis replay matched byte-for-byte; the updated notebook adds a paired-attempt OOF CSV sensitivity output. Fixed-format timestamp parsing changed runtime behavior without changing derived results; on 10,000 rows from each of the four tracking CSVs it matched pandas' default parser, with all 40,000 timestamps parsed and none invalid. The attempt-1-versus-2 sensitivity uses 27 DLs with both 40-yard attempts and is explicitly non-independent. The profile explorer's lateral metric label and note describe a path-relative cross-axis speed gap; its HTML was regenerated from verified profile tables. After the replay, the contextual model's nullable-string conversion replaced a deprecated fillna downcast; rerunning that stage emitted no warning, and its coverage, JSON, Markdown, and player-score outputs matched the replay byte-for-byte. In a later presentation-only update, the split-half figure title was clarified to say its partitions are within-season; the PNG/SVG were regenerated from verified results, with no analysis values changed. The final bundle includes these updates, the current notebook, writeup, and sensitivity analysis. An earlier ZIP-mode staged replay completed before the timestamp optimization. Kaggle-hosted execution and public attachment checks remain outstanding.

## Scoring design

- Hold out complete draft cohorts (2023, 2024, or 2025), keeping each player within one fold.
- Fit imputation and scaling on training cohorts only.
- Compare baseline and augmented models on identical complete players.
- Score player-level RMSE in yards/second.
- Bootstrap players within cohort from fixed out-of-fold predictions; this does not refit models or account for all feature/position/outcome selection.
- Independently resample each player's observed games, retaining that player's matched play peaks per game, then refit cohort-held-out models. This propagates player-specific finite-game p90 uncertainty but does not preserve shared game, opponent, or weather shocks, or correct for discovery selection.
- Report fold changes individually. Three cohorts limit temporal generalization.

## Input integrity and package review

The local ZIP was CRC-checked and audited for required files, unique keys, joins, and primary derived-table grains; see [SECURITY_AND_DATA_AUDIT.md](SECURITY_AND_DATA_AUDIT.md). The SHA-256 identifies the local file and does not verify it against a trusted Kaggle checksum. The code bundle includes the analysis scripts used by the notebook, project documentation, and notebook. Inspect its file list with Python's `zipfile` module if it is refreshed.


