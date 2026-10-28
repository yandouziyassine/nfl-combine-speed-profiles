# NFL Combine Speed Profiles and Rookie Game Movement

A reproducible analytics project for the **NFL Big Data Bowl 2027**. It studies whether movement hidden inside a Combine 40-yard dash—especially how speed builds during the first second—adds information about later NFL movement, beyond the official 40-yard time, 10-yard split, and body measurements.

> **Status:** Complete local research and demo package; the competition notebook and writeup have not yet been published on Kaggle. Findings are exploratory and should not be used as a player grade, draft cutoff, or training prescription.

## What was built

- A 10 Hz Combine tracking pipeline that derives early acceleration, later acceleration, speed retention, and position-aware drill profiles.
- Data-quality and linkage audits covering trace cadence, movement onset, course-axis assumptions, repeated attempts, and Combine-to-game player matching.
- Cohort-held-out player-level analyses comparing Combine movement features with rookie regular-season tracking outcomes.
- A nested feature-selection audit, measurement-reliability analyses, outcome and role sensitivities, and a registry of the analysis search.
- A reproducible Kaggle notebook, pinned Python requirements, a source bundle, competition writeup, and a local interactive prospect explorer.

The headline is deliberately cautious: the post-screen early-acceleration association with full-play peak speed is not confirmatory, the broader selection-aware phase search showed little average lift, and the direct first-step blitz-speed analysis did not establish added predictive value. The project documents these negative and uncertain results rather than turning them into a scouting score.

## Research question and method

The exploratory lead question asks whether a defensive lineman's tracked first-second 40-yard acceleration is associated with the 90th percentile of per-play peak speed during the player's rookie regular season. The pre-draft baseline uses official 40 time, 10-yard split, height, and weight. Models use ridge regression and leave-one-draft-cohort-out evaluation; preprocessing is fitted on the training cohorts. Players are the independent analysis units.

The next-study outcome is locked to **first-second post-snap speed gain on stationary-snap blitz opportunities**. The existing analysis of this endpoint is exploratory and does not show that adding Combine early acceleration improves prediction. Details, eligibility, and inference boundaries are in [`docs/PRIMARY_OUTCOME_CHARTER.md`](docs/PRIMARY_OUTCOME_CHARTER.md).

This is a small, selected population of rookies with enough tracked NFL opportunities, not a sample that can estimate draft-bust risk for every prospect. Combine movement onset is a speed threshold, not a reaction-time measurement; the source data do not contain a stimulus timestamp. Association does not establish causation or a training effect.

## Data and access

The analysis uses the competition's Combine tracking and regular-season NFL tracking data. **The competition data and derived player-level files are not included in this repository.** Obtain the official data through Kaggle, accept the competition rules, and follow the current data-use terms. Do not add the downloaded archive, extracted data, profile explorer, player-level CSVs, or generated result tables to a public Git repository.

Competition page: [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)

The Kaggle submission is intended to include a public notebook and a readable Writeup. Publicly sharing competition code also needs to follow the competition's code-sharing rules; this repository should be published alongside the required Kaggle share, not as a substitute for it.

## Reproduce locally

Requirements: Python 3.10 or newer and the official competition data downloaded through your Kaggle account.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:NFL_BDB_INPUT = "C:\path\to\nfl-big-data-bowl-2027.zip"
python audit_project_data.py
if ($LASTEXITCODE -ne 0) { throw "Data audit failed; stop before analysis." }
python run_analysis.py --output outputs
if ($LASTEXITCODE -ne 0) { throw "Core analysis failed; outputs remain uncertified." }
python write_core_run_manifest.py
if ($LASTEXITCODE -ne 0) { throw "Core output inventory failed; outputs remain uncertified." }
python build_speed_signatures.py
python build_profile_explorer.py
```

`NFL_BDB_INPUT` may point to the competition ZIP or an extracted competition-data directory. For the complete analysis and figures, run all cells in [`nfl_combine_speed_analysis.ipynb`](nfl_combine_speed_analysis.ipynb) after attaching the official competition dataset and the source bundle as Kaggle notebook inputs. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full runbook.

## Repository map

| Path | Purpose |
|---|---|
| `nfl_combine_speed_analysis.ipynb` | Competition notebook and end-to-end orchestration |
| `run_analysis.py` | Core feature extraction and movement-profile pipeline |
| `model_*.py`, `analyze_*.py`, `audit_*.py` | Outcome models, measurement analyses, and data audits |
| `docs/` | Research design, methods, uncertainty, search registry, and reproducibility notes |
| `requirements.txt` | Pinned direct Python dependencies |
| `nfl_speed_project_code.zip` | Flat source bundle used by the Kaggle notebook |
| `outputs/`, `submission_kit/` | Generated locally; intentionally excluded from the public source repository |

## Limits and honest interpretation

- The feature search covered multiple positions, drills, outcomes, exposure thresholds, and processing choices. The nested selection audit covers only a narrow declared candidate family; it is not a global correction.
- Only three draft cohorts are available, and 2025 was involved in feature screening. It is not an untouched confirmation set.
- Rookie tracking exposure is affected by playing time, team role, health, assignment, and tracking availability.
- Pressure labels do not identify every blocker assignment or opponent effect.
- Repeat-attempt reliability is same-session evidence, not between-day or between-device validation.
- The prospect explorer contains embedded player-level profiles and stays out of this public source release unless redistribution is confirmed as permitted.

Read [`docs/ANALYSIS_SEARCH_REGISTRY.md`](docs/ANALYSIS_SEARCH_REGISTRY.md), [`docs/MEASUREMENT_UNCERTAINTY.md`](docs/MEASUREMENT_UNCERTAINTY.md), and [`docs/MODEL_AUDIT.md`](docs/MODEL_AUDIT.md) for the full evidence boundaries.

## Citation

Lopez, M., Blake, A., Bliss, T., Howard, A., & Mooney, P. (2026). *NFL Big Data Bowl 2027*. Kaggle. [Competition page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027).
