# Project guide

**Project:** Combine speed profile to rookie NFL game-speed capacity  
**Competition:** 2027 NFL Big Data Bowl  
**Analysis status:** Discovery-stage primary lead; external validation unavailable in current data.

## Purpose

The project derives interpretable movement features from 10 Hz Combine traces, measures their reliability, and evaluates whether they add information about later NFL tracking outcomes beyond conventional testing. The current narrow lead is 0–1 second 40-yard acceleration for defensive linemen and rookie-season upper-tail game speed.

The lead is deliberately framed as a candidate signal. It is not a player rating, draft recommendation, draft-bust predictor, causal claim, or validated coaching intervention.

## Data flow

- `players.csv` and `combine_results.csv`: roster, positions, official testing and anthropometrics.
- `combine_tracking.csv`: 10 Hz drill movement traces; source of attempt-level features.
- `games.csv`, `player_play.csv`, and `game_tracking_{2023,2024,2025}.csv`: rookie regular-season tracked plays and outcomes.
- `player_career_successes.csv`: snap/role outcomes used only in secondary analyses.

The archive is read directly. Derived outputs are written to `outputs/`. The source ZIP is not extracted or modified by the local analysis.

## Main code map

| Component | Purpose |
|---|---|
| `run_analysis.py` | Core data processing, Combine movement features, rookie game-speed profiles, coverage report. |
| `build_speed_signatures.py` | Prospect-level movement profiles and neutral within-position ranks. |
| `analyze_40_repeatability.py` | Same-session 40-yard attempt reliability. |
| `analyze_40_repeatability_by_position.py` | Position-specific repeatability; key measurement check for the DL lead. |
| `analyze_40_feature_sensitivity.py` | Onset/filter sensitivity for 40-yard features. |
| `model_dl_game_speed_ablation.py` | DL baseline versus early, late, retained-speed, and combined phase features. |
| `model_dl_early_accel_transfer.py` | Early-only model and phase-block sensitivity with observed NFL subposition controls. |
| `model_dl_early_accel_predraft.py` | Primary early-acceleration model using only pre-draft Combine tests and size. |
| `model_dl_40_accel_to_football_outcomes.py` | Tests whether 40-yard early acceleration transfers to DL post-snap burst, pressure rate, or get-off. |
| `model_dl_pressure_context.py` | Player-balanced play-level pressure model with down, distance, and lineup-position controls. |
| `analyze_dl_game_speed_target_precision.py` | Resamples matched rookie plays by game and repeatedly splits games into disjoint halves to quantify p90 sampling uncertainty, rank repeatability, and model sensitivity. |
| `build_dl_split_half_figure.py` | Visualizes player p90 rank repeatability and held-out model deltas across the disjoint game partitions. |
| `analyze_dl_fixed_window_target.py` | Compares Combine transfer across fixed 1–4 second post-snap speed windows and a forward 2025 holdout. |
| `model_dl_target_uncertainty.py` | Refits cohort-held-out models across 1,000 game-cluster outcome draws. |
| `build_dl_transfer_figure.py` | Creates the compact three-panel SVG summary for a competition writeup. |
| `model_dl_game_speed_role_sensitivity.py` | Compares pre-draft baseline with post-draft subposition-adjusted baseline. |
| `model_game_speed_by_position.py` | Position-stratified discovery screen for 40-yard phase transfer. |
| `model_position_specific_game_speed.py` | Position-by-phase slope interaction screen. |
| `audit_position_phase_nested_selection.py` | Outer cohort-holdout audit that selects one of 12 position-by-phase 40-yard candidates inside training cohorts. |
| `build_position_phase_nested_figure.py` | Plots outer-fold choices and pooled uncertainty for the position-by-phase selection audit. |
| `build_profile_explorer.py` | Self-contained coach-facing prospect explorer with same-position A/B speed-profile comparison, position percentiles, raw values, per-player 40-yard attempt spread, and a print-ready comparison brief. |
| `write_core_run_manifest.py` | Validates core outputs against their source/cache identity and writes a checksummed environment and output inventory. |
| `audit_project_data.py` | Archive integrity, key/linkage, and derived-table grain audit. |
| `nfl_combine_speed_analysis.ipynb` | Kaggle-oriented reproduction of the focused analysis and supporting checks. |

The remaining WR, DB, DL hoop, OL mirror, and career scripts implement exploratory or negative screens. Their results are summarized in [RESEARCH_DESIGN.md](../RESEARCH_DESIGN.md), not elevated as primary claims.

The core pipeline writes `game_speed_profiles_cache_manifest.json` for the reusable game-speed table. After the run, `write_core_run_manifest.py` confirms source/cache identity, required outputs, checksums, and environment versions in `outputs/core_run_manifest.json`.

## Main output grains

| Output | Grain | Current rows |
|---|---|---:|
| `all_player_speed_profiles.csv` | Prospect | 510 |
| `player_speed_signatures.csv` | Prospect | 510 |
| `combine_attempt_speed_profiles.csv` | Player × drill attempt | 6,310 |
| `player_drill_speed_profiles.csv` | Player × drill | 4,986 |
| `player_40_speed_profiles.csv` | Prospect with tracked 40 | 418 |
| `player_rookie_game_speed_profiles.csv` | Prospect, missing outcomes retained | 510 |

## Current headline result

A minimum of 100 matched rookie tracked plays leaves 46 DLs for the primary pre-draft early-only model (13/19/14 by cohort); 42 remain in the additional draft-capital/subposition sensitivity; and 34 have complete data for all phase comparisons. Using only official 40, 10-yard split, height, and weight, adding early 0–1 second acceleration reduces pooled OOF RMSE for full-play p90 speed by 0.0497 yd/s (95% fixed-prediction player-bootstrap interval -0.0901 to -0.0092) at ridge alpha 5. The forward 2025 split improves by 0.0629 yd/s on 14 players, but feature selection included all cohorts, so it is exploratory. Fixed 1–4 second post-snap p90 targets show no consistent improvement; the feature should not be interpreted as first-step burst.

The play-context diagnostic uses 15,067 complete four-second windows from 45 DLs with at least 100 windows. After controlling alignment, situation, blitz role, and speed at/just before the snap, adding early acceleration changes player-balanced RMSE by -0.00016 yd/s in leave-one-cohort-out validation and -0.00113 in the 2025 forward fold; both 95% intervals include zero. This suggests the full-play association is not a useful incremental predictor of fixed-window peak speed once play context is represented. The model is diagnostic, not draft-time deployable.

A direct DL football-outcome screen finds no incremental value for one-second post-snap speed gain or blitz pressure rate, and worse error for post-snap displacement and get-off time. The narrow conclusion is that early Combine acceleration is associated with the full-play peak-speed target in this selected sample; it is not robust across fixed game windows and has not shown first-step or pressure-production transfer. In 1,000 game-cluster outcome simulations refitting the primary pre-draft model, all replicates favor early acceleration (median RMSE change -0.0402 yd/s; 95% interval -0.0648 to -0.0182), conditional on the same selected target and sample. Player outcomes were drawn independently, so shared game, opponent, and weather shocks are not preserved.

See [the submission draft](COMPETITION_SUBMISSION.md), [model audit](MODEL_AUDIT.md), and [measurement audit](MEASUREMENT_AUDIT.md) for detail.

