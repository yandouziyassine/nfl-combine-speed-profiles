# Primary outcome charter: DL post-snap speed gain

**Status:** Locked now for the next confirmatory analysis. This charter was written after exploratory analyses had already examined several outcomes; it is not preregistration and does not convert existing results into confirmatory evidence.

## Question

Among defensive-line prospects with sufficient tracked rookie-season blitz opportunities, does tracked Combine 40-yard early acceleration add predictive information for first-second post-snap speed gain beyond the official 40-yard time, 10-yard split, height, and weight?

## Target and analysis unit

- **Football outcome:** for each eligible player-play, speed at 1.0 second after `ball_snap` minus speed at `ball_snap`, measured in yd/s; aggregate to the player median across eligible opportunities.
- **Eligible opportunity:** a rookie regular-season blitz player-play with usable tracking, a stationary pre-snap state (no speed ≥1 yd/s during the preceding 0.5 seconds), and complete measurements at both endpoints.
- **Exposure:** at least 100 eligible opportunities per player. The existing 100-opportunity screen leaves 32 players across 2023–2025. The 50-opportunity screen is sensitivity only.
- **Independent unit:** player. Plays are exposures used to estimate that player's target, not independent model samples.

## Predictor and comparator

- **Baseline:** official Combine 40 time, official 10-yard split, height, and weight; all available before the draft.
- **Single candidate:** player's mean tracked 0–1 second 40-yard speed gain/acceleration, averaged over available attempts using the existing pipeline.
- **Metric:** paired change in player-level RMSE; negative favors adding the candidate. Keep the identical complete-case players for both models.
- **Model/split:** ridge regression with the existing fixed alpha grid; leave one complete draft cohort out. Report every cohort separately. 2025 is the only forward-in-time cohort split in the current data.

## Current evidence (exploratory)

At the 100-opportunity threshold, 32 players qualify. At alpha 5, adding Combine early acceleration **worsens** pooled held-out RMSE by +0.0099 yd/s (fixed-prediction player-bootstrap 95% interval -0.0097 to +0.0339); the forward 2025 fold also worsens (+0.0106). At alpha 20, pooled change is +0.0067 and 2025 is +0.0207. This result does not support incremental prediction of first-second blitz speed gain in this sample.

The separate all-exposure analysis reports 48 players at the 50-opportunity threshold and shows a near-zero improvement for pressure rate. Pressure is not the primary endpoint because blitz pressure does not identify blocker assignment or opponent quality. Displacement, pressure, get-off time, and full-play p90 remain secondary outcomes, not alternate chances to declare success.

## Measurement and inference boundaries

- The Combine 0–1 s feature starts at a speed threshold. It is **not reaction time** because no stimulus/cue timestamp is available.
- “Stationary snap” reduces pre-snap movement contamination, but does not control for assignment, alignment, opponent, scheme, coaching, health, or play context.
- The 100-opportunity rule selects for rookie usage and tracking availability; conclusions apply only to this observed population.
- Report predictor attempt count and within-player attempt SD wherever player-level profiles are shown. Do not turn the SD into a confidence interval.
- Preserve the existing all-available-attempt mean. Do not reliability-correct it unless a future, fold-specific measurement model can be fit and independently validated.
- Existing bootstrap intervals condition on fixed predictions and do not correct the full search; three historical cohorts are not an independent future validation set.

## Decision rule for future work

Do not claim utility from a favorable point estimate alone. Before a new cohort is analyzed, freeze the player eligibility, outcome extraction, baseline, candidate, missing-data policy, metric, and minimum practically useful RMSE change. Require the direction and useful magnitude to hold in a genuinely untouched future cohort, with uncertainty that includes model fitting and target estimation. If the lift is not practically useful, report the null result and do not replace the outcome with a more favorable secondary target.
