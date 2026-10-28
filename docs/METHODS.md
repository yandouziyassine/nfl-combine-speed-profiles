# Methods and interpretation

## Data and analysis grain

The supplied competition archive contains 510 prospects across the 2023–2025 draft cohorts. Combine tracking is sampled at 10 Hz; regular-season tracking is linked to player-play metadata. The independent unit is the player. Frames and plays contribute to player-level measurements but are not treated as independent prospects.

Missing tests and outcomes remain missing. Players without sufficient game exposure or required Combine features are excluded from the relevant complete-case comparison; they are not assigned zeros or peer averages.

## 40-yard trace processing

Speed is smoothed with a centered three-frame median filter. Movement onset is the first sampled speed at or above 1 yd/s for two consecutive frames. Early acceleration is the change in smoothed speed during the first second after detected movement onset. This is a movement threshold, not reaction time: cue timestamps are not in the data.

For straight-line 40-yard phase features, forward progress is signed x-coordinate displacement from detected movement onset; the sign is set by the attempt's net x direction and tiny backsteps are clipped from the interpolation axis. This coordinate choice is supported by full-trace PCA: the major movement axis is nearly parallel to x in each cohort. It remains an inferred course axis because the supplied schema does not identify the official dash direction. These yard markers are onset-relative, not official start-line splits. Late acceleration is the speed difference over 10–20 yards divided by elapsed time; speed retention summarizes the 20–40-yard segment. These phases remain separate measures. True x/y path length is retained for path-distance summaries.

Directional components are path-relative and do not identify anatomical left/right, preferred foot, or body orientation. Braking is derived from reductions in smoothed speed rather than the supplied acceleration field.

## Rookie game-speed outcome

For each matched rookie regular-season player-play, calculate maximum observed game speed. The player target is the 90th percentile of those per-play peaks, in yd/s. It summarizes upper-tail speed among observed plays; it is not maximum standardized ability, pass-rush success, pressure rate, or assignment-adjusted performance. Position, role, play type, coaching, tracking coverage, health, and opportunity can affect it.

As a duration sensitivity, a separate pass anchors frames to the `ball_snap` event and calculates the player p90 of per-play maximum speed in fixed 1-, 2-, 3-, and 4-second windows. A player-play is retained only with at least 90% of expected 10 Hz frames, observed start and end within 0.1 seconds, and no gap exceeding 0.2 seconds. This reduces unequal observation-window bias but is not a context or role adjustment; interpret it as a different movement outcome.

A contextual diagnostic models per-play maximum speed in the complete first four seconds after `ball_snap`, with fixed effects for lineup position, down category, yards-to-go band, blitz role, speed at snap, and recent pre-snap maximum speed. It compares the same pre-draft Combine baseline with and without early acceleration. Each player receives equal total training weight; prediction error is computed within each player and averaged equally across players. This estimates incremental association conditional on observed play context, not a draft-time forecast or causal effect. At the 100-window threshold, it includes 45 players and 15,067 player-plays.

Minimum exposure thresholds of 25, 50, and 100 matched tracked player-plays are used. At 100, 63 DLs meet exposure. Complete-case modeling samples are 46 for the pre-draft early-only specification, 42 when adding draft capital and NFL subposition sensitivity controls, and 34 when requiring all three phases.

## Candidate models

The draft-time baseline uses official 40 time, 10-yard split, height, and weight. It excludes draft capital and post-draft NFL position, which are unavailable at scouting time. The candidate adds only early 0–1 second acceleration.

Sensitivity models add log draft pick and, separately, observed NFL subposition indicators (DE, DT, NT, OLB). These test whether the feature adds signal after accounting for draft selection and observed role; they are not the primary deployable model.

A separate ablation compares early acceleration, late acceleration, 20–40-yard retention, and all three phases. Feature-specific estimates use all complete players for their feature block. For a direct comparison among phases, each candidate is also evaluated on the common subset with all phases available. Those feature comparisons use a conventional-test/draft-capital baseline and are discovery analyses.

Ridge models leave out one full draft cohort in turn. Median imputation and feature scaling are fit on training cohorts only. Baseline and candidate are compared on identical players within each specification. Negative candidate-minus-baseline RMSE change favors the candidate. The 2025 test is the only forward-in-time cohort split; the 2023 and 2024 leave-one-cohort-out folds include later cohorts in training. Results show pooled and cohort-specific RMSE in yd/s.

To quantify phase/position selection, a separate nested audit uses the 161-player common complete-case sample from DB, DL, OL, and WR. The baseline has position-specific slopes for official 40, 10-yard split, height, and weight. Its 12 candidates add one phase (early acceleration, late acceleration, or speed retention) within one position. In each outer draft-cohort fold, candidate choice minimizes pooled inner leave-one-cohort-out RMSE over the two training cohorts; the selected feature is refit on those cohorts and scored on the outer cohort. The penalty is fixed at alpha 5. This audits only the stated candidate family, not the broader search across outcomes, drills, exposure cutoffs, or processing definitions.

## Uncertainty and reliability

Player bootstrap intervals resample players within held-out cohort from fixed OOF predictions. They do not refit models and do not correct for outcome, position, or feature selection. A separate outcome-sampling analysis resamples each player's observed regular-season games, retains all tracked player-play peaks within each sampled game, recalculates the p90, and refits the cohort-held-out models. This propagates finite-game uncertainty conditional on the observed season; it does not represent a future cohort.

A complementary split-half analysis partitions each player's tracked rookie games into disjoint halves, requiring at least two games per half. For each player and repeat, 40 candidate partitions are evaluated using play counts only; the partition with the closest play counts is retained without inspecting speed outcomes. Each half gets its own p90 target. We summarize the distribution of half-to-half rank correlations and refit the same cohort-held-out Combine model separately on each half target. Quantiles across 250 partitions describe sensitivity to game assignment, not sampling confidence intervals. The predictor, cohort, and model family remain fixed; this is not independent validation.

Combine attempts 1 and 2 are paired within player and position. ICC(A,1) estimates absolute agreement for one attempt; ICC(A,2) estimates reliability of the two-attempt mean. They are same-session estimates, not between-day reliability. Typical error is reported in feature units.

The signal-processing grid recomputes 793 40-yard attempts from 418 prospects using 0.5/1.0/1.5 yd/s onset thresholds and 1/3/5-frame median filters. Rank correlation and top/bottom quintile overlap are compared with the default 1 yd/s, 3-frame definition. Processing robustness and repeated-attempt reliability answer different questions.

## Position-specific screens

Secondary analyses test WR curl recovery against rookie route separation, DL hoop/burst features against rookie post-snap movement and pressure labels, OL mirror movement against pressure labels, DB speed phases against observed tackle rate, and 40-yard phases against career playing-time proxies. These outcomes differ in sample size, opportunity, alignment, and context; they are not combined with the primary model.

The football-specific DL screen did not show meaningful transfer from early acceleration to one-second post-snap speed gain or pressure production. The pressure context model's Brier improvement is less than 0.1% relative and is too small and exploratory for a practical claim.

## Limitations

The early-acceleration lead followed a broad screen across positions, drills, and NFL outcomes. It is not multiplicity-adjusted and can be optimistic. Only three draft cohorts are available. The 100-play primary sample has 46 DLs, while some subpositions are represented by few players. All results are observational; they do not identify a causal training intervention or establish a player-quality cutoff. External future-cohort validation and visual review of raw traces remain necessary.

