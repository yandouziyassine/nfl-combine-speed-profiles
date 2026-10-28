# Measurement uncertainty: reporting and propagation plan

## What is reported now

The core prospect profile already records each player's tracked 40 attempt count, mean early acceleration, and within-player attempt standard deviation when repeats are available. The explorer displays attempt support and observed spread. This spread is descriptive: many players have only one or two attempts, so it is not a stable standard error or a population confidence interval.

For defensive linemen, attempt 1/2 early acceleration has same-session ICC(A,1)=0.727 (95% player-bootstrap interval 0.570–0.834), typical error 0.296 yd/s², and estimated two-attempt mean reliability ICC(A,2)=0.842 (n=63 paired players). These statistics describe repeatability in this session/sample; they do not establish between-day, facility, or device reliability. The primary 46-player model sample includes 27 players with both attempts.

The point estimate uses the mean of all available attempts. The attempt-specific sensitivity on the 27 paired primary-sample players shows similar directional improvements for attempt 1 and attempt 2 in the post-screen full-play p90 analysis. It is an overlapping sensitivity sample, not validation. We retain the simple mean and avoid an unstable reliability correction.

## Separate the uncertainty sources

1. **Attempt measurement variation:** summarize attempt count and within-player SD/range. A one-attempt player has no observed within-player spread; do not imply certainty from the missing SD.
2. **Feature-definition sensitivity:** report rank correlations and top/bottom overlap across onset and smoothing choices. Early acceleration is less robust than late acceleration/retention.
3. **Football target sampling:** the game-cluster bootstrap estimates uncertainty from which observed games enter each player's season p90, conditional on the observed cohort and role. It does not represent a new season or draft class.
4. **Model estimation and selection:** fixed-prediction OOF player bootstrap intervals do not refit the model or repeat feature selection. The nested 12-feature audit handles only its declared family.
5. **Population transport:** no method in the current analysis creates new-cohort validation; only three cohorts are available and 2025 was included in screening.

These sources answer different questions and should not be collapsed into a single confidence interval.

## Fold-safe propagation plan for a future study

If a larger paired-attempt sample becomes available, estimate a random-intercept measurement model separately within each outer training fold:

`observed attempt value = latent player value + fixed attempt-order effect + measurement error`.

Estimate order and variance components on training players only. For held-out players, estimate latent Combine values from their attempts without using NFL outcomes. Draw plausible latent values and player football targets, refit the complete predictor inside each draw, and report the distribution of paired held-out score changes. Repeat candidate selection inside each outer training set if selection is part of the estimand. Keep the raw all-attempt mean as the benchmark.

This is a future sensitivity plan, not a result from the current 27 paired DLs. The small paired sample is insufficient to justify replacing the current predictor with a shrinkage estimate.
