# Kaggle submission handoff

## Judging priorities

The official rubric weights Football and Data Science at 30% each, then Writeup and Data Visualization at 20% each. The analysis therefore needs a football-useful interpretation and a credible, reproducible validation story alongside clear figures and concise writing. The submission gate also requires a readable Writeup and an attached Public Kaggle Notebook.

## Files to upload

- `nfl_combine_speed_analysis.ipynb` — public analysis notebook.
- `nfl_speed_project_code.zip` — source bundle; attach it as a Kaggle input so the notebook can unpack and run the project scripts.
- `docs/COMPETITION_SUBMISSION.md` — writeup text to paste into the competition Writeup.
- `outputs/dl_early_acceleration_summary.png`, `outputs/dl_speed_profile_case_illustration.png`, `outputs/fig_dl_game_speed_split_half.png`, and `outputs/fig_position_phase_nested_selection.png` — visual assets to upload through Kaggle's media/image control and embed in the Writeup.

The current writeup contains four figures plus one table. It has fewer than 1,500 whitespace-separated words and about 1,650 alphanumeric tokens when hyphenated compounds are split, leaving room below Kaggle's 2,000-word limit. Kaggle's Writeup editor does not resolve local relative image paths: upload all four PNGs and insert the resulting Kaggle-hosted images before publication.

## Notebook run

1. Create a Kaggle Notebook under the NFL Big Data Bowl 2027 competition.
2. Attach the competition data and the public code-bundle dataset as notebook inputs.
3. Import `nfl_combine_speed_analysis.ipynb` and run all cells in a fresh session. The notebook searches attached inputs for one competition archive and one code bundle. If multiple matches exist, set `NFL_BDB_INPUT` in the notebook to the exact archive or extracted data directory.
   All notebook-invoked readers support either a ZIP archive or Kaggle's extracted competition-data directory.
4. Inspect the generated source and two-way linkage audits, axis and signal-processing transfer sensitivity, attempt-1-versus-2 robustness, nested 12-candidate position-phase and three-phase reports, disjoint-game p90 reliability check, 25–200-play exposure thresholds, OOF score audit, model reports, and summary figures. The notebook scans the large game-tracking inputs several times and may take a while.
5. Save the completed notebook as **Public** so judges can inspect and rerun it. Confirm it has both the official data input and the source bundle attached.

## Writeup and final checks

1. Paste `docs/COMPETITION_SUBMISSION.md` into the competition Writeup editor.
2. Replace all four local figure references with the uploaded PNGs and confirm the embedded figures render.
3. Attach the public Kaggle Notebook to the Writeup using Kaggle's notebook-attachment control.
4. Confirm the final Writeup remains under 2,000 words and below 10 combined figures/tables; verify that the public notebook opens and its output cells contain the refreshed signed-x phase results and case illustration.
5. Publish the Writeup through the competition page.

No Kaggle upload, publication, or competition submission has been performed from this workspace. A Kaggle-hosted execution and the final public attachment checks remain outstanding.



