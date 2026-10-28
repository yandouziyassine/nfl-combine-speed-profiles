# Research design: Combine speed profile to rookie game-speed capacity

## Decision question

Among defensive-line prospects, does acceleration during the first second of the tracked 40-yard dash add information about rookie regular-season high-speed capacity beyond official Combine tests and body size? Draft capital and post-draft position are included only as sensitivity controls because they are not available at the draft-time scouting decision.

This narrow question was selected from a broader discovery program that tested all-position speed phases, career role, defensive-back tackle rate, receiver route speed recovery, defensive-line drill movement, and offensive-line mirror movement. The broad screens are retained as negative or exploratory results. They are not pooled into a composite scouting grade.

The saved full-play p90 analysis is the original discovery estimand, not the locked football endpoint for future confirmation. After those exploratory screens, we locked DL first-second post-snap speed gain on stationary-snap blitz opportunities as the next-study primary outcome. The endpoint, 100-opportunity population rule, pre-draft baseline, one candidate, metric, and limits are recorded in [`docs/PRIMARY_OUTCOME_CHARTER.md`](docs/PRIMARY_OUTCOME_CHARTER.md); the current 32-player result is already exploratory and does not support incremental prediction. The complete search/provenance map is in [`docs/ANALYSIS_SEARCH_REGISTRY.md`](docs/ANALYSIS_SEARCH_REGISTRY.md).

## Competition fit and practical use

The NFL Big Data Bowl asks participants to connect Combine player tracking with later regular-season movement and provide useful, reproducible analysis. This project contributes a complete descriptive speed profile for 510 prospects, then evaluates whether one interpretable 40-yard feature adds signal for a position-relevant population.

The possible coaching use is only as a descriptive movement-profile feature. The full-play peak-speed association has not replicated across fixed post-snap windows, so early acceleration should not be treated as an NFL first-step cue. Do not use it as a cutoff, player-quality rank, causal claim, draft-bust detector, or training prescription; any development protocol needs controlled field validation.

## Primary measurement and outcome

**Combine predictor:** mean player-level 0–1 second change in smoothed speed during the tracked 40-yard dash. The trace uses the supplied speed field, a centered three-frame median filter, and movement onset at the first speed of at least 1 yd/s on two consecutive frames. The onset rule is a signal threshold, not reaction time; there is no cue timestamp.

**NFL outcome:** the player-level 90th percentile of per-play peak speed from 10 Hz tracking during the rookie regular season, in yards/second. Per-play peak is the maximum observed speed for a matched player-play. This represents upper-tail speed in observed tracked plays. It is not pass-rush win rate, pressure, assignment-adjusted performance, or a causal measure of talent.

**Analysis unit:** player. Plays are used to estimate a player's outcome and enforce minimum exposure; they are not treated as independent players.

## Sample and current model

At least 100 matched tracked rookie player-plays leaves 63 exposure-eligible Combine DLs. The primary early-only model with pre-draft features has 46 complete cases (13 from 2023, 19 from 2024, 14 from 2025). The post-draft role-adjusted sensitivity has 42 (13, 17, 12; NFL roles DE=13, DT=18, NT=3, OLB=8). Requiring all three 40-yard phases reduces the common phase-ablation sample to 34 (10, 12, 12 by cohort). Early-only and all-phase results use different complete-case samples; the ablation on the 34-player subset supports a fairer direct phase comparison. Small-role counts limit role-specific interpretation.

The primary pre-draft candidate baseline uses official 40 time, official 10-yard split, height, and weight. A robustness sensitivity adds log draft capital and observed NFL subposition indicators (DE, DT, NT, OLB); draft capital and post-draft roles are unavailable before draft selection and are not part of the primary baseline. The candidate adds only first-second acceleration. A three-phase comparison adds early acceleration, 10–20-yard acceleration, and 20–40-yard retention.

Models use ridge regression. Each draft cohort is held out once. Missing-value medians and feature scaling are fit on training cohorts only. Baseline and candidate are scored on the same complete players. A player bootstrap stratified within held-out cohort is applied to fixed out-of-fold predictions; it does not refit the models or account for all model-selection uncertainty. Results report each cohort separately.

## Discovery result

At the 100-play threshold, the primary pre-draft early-only model on 46 DLs reduces pooled OOF RMSE by 0.0497 yd/s at alpha 5 (95% fixed-prediction player-bootstrap interval -0.0901 to -0.0092). All three held-out cohorts improve: -0.0498, -0.0397, and -0.0629 yd/s for 2023, 2024, and 2025. At alpha 20, the change is -0.0410 yd/s (-0.0684 to -0.0117). The post-draft role-adjusted sensitivity on 42 DLs also improves (-0.0481; interval -0.0818 to -0.0141). A feature-specific ablation with a draft-capital baseline gives early acceleration -0.0422 yd/s on 42 players (interval -0.0793 to -0.0069), late acceleration +0.0099 on 42 (-0.0025 to +0.0222), speed retention -0.0029 on 34 (-0.0193 to +0.0124), and all three phases -0.0253 on 34 (-0.0725 to +0.0129). For direct phase comparison on the same 34 complete players, early-only changes RMSE by -0.0539 (-0.0845 to -0.0243), late-only by +0.0143 (-0.0087 to +0.0365), retention by -0.0029 (-0.0190 to +0.0131), and all phases by -0.0253 (-0.0725 to +0.0129). These ablations are exploratory and not multiplicity-adjusted. At the 50-play exposure threshold, the pre-draft early-only result on 57 DLs also favors adding the feature (change -0.0535; interval -0.0857 to -0.0186).

The forward 2025 test trains only on 2023–2024 and has 14 DLs at the 100-play threshold; adding the feature lowers full-play target RMSE by 0.0629 yd/s (95% fixed-prediction interval -0.1089 to -0.0093). This is still an exploratory holdout because the feature was selected after screening analyses that included all three cohorts. The 2023 and 2024 leave-one-cohort-out folds are not forward tests because their training data include later cohorts.

## Nested phase-by-position selection audit

To estimate selection optimism within the 40-yard phase family, the audit uses a 161-player common sample from DB, DL, OL, and WR, each with complete early, late, and retention features. The position-specific pre-draft baseline is augmented by one of 12 candidates (four positions × three phases). For each outer held-out cohort, an inner leave-one-cohort-out comparison on the two remaining cohorts selects the candidate with lowest pooled RMSE. The procedure selects WR late acceleration for the 2023 outer fold and DL late acceleration for 2024 and 2025. The nested pooled RMSE change is -0.0033 yd/s (95% fixed-OOF player-bootstrap interval -0.0099 to +0.0027); a locked DL early term on the same players gives -0.0102 (-0.0149 to -0.0055). This adjustment does not include the broader search across drills, outcomes, exposure thresholds, or processing. The earlier 3-phase DL-only nested audit is retained as a separate, narrower comparison.

## Fixed-window outcome sensitivity

To determine whether the full-play peak association reflects early on-field speed, we calculated each prospect's p90 of per-play peak speed in fixed 1-, 2-, 3-, and 4-second windows after the tracking `ball_snap` event. Player-plays require at least 90% of expected 10 Hz frames, both window endpoints within 0.1 seconds, and no gap longer than 0.2 seconds. This is a measurement sensitivity, not a second discovery sample.

At a 100-complete-window threshold, the 1- and 2-second targets have the same 46 players as the primary sample; the 3- and 4-second targets have 45. At alpha 5, LOCO RMSE changes are -0.0054 yd/s (95% interval -0.0198 to +0.0093), +0.0025 (-0.0085 to +0.0152), +0.0059 (-0.0032 to +0.0147), and +0.0070 (-0.0019 to +0.0162), respectively. Forward 2025 changes are +0.0185, -0.0001, -0.0157, and -0.0187. At alpha 20, the forward 2025 3- and 4-second changes reverse and worsen error. No fixed window shows a stable improvement across both pooled cohort-held-out and forward 2025 results and the regularization grid. The full-play target's apparent gain therefore does not demonstrate a transferable first-step or fixed-horizon speed trait; play duration, later movement, and role/context remain unresolved explanations.

The post-draft subposition-adjusted early-only sensitivity is the reported 42-player model above. This is useful as a role-confounding check, but NFL subposition is not a predictor available to teams at draft time. These results are small-sample discoveries selected after broader screens, not an unbiased estimate of future performance. The player bootstrap conditions on fixed predictions and does not correct for model/feature selection.

## Does the signal reach football-specific DL outcomes?

A follow-up tests the same 40-yard early-acceleration measure against stationary-snap post-snap speed gain and displacement over one second, rookie blitz pressure rate, and get-off time. The baseline uses only pre-draft 40, split, height, and weight. With at least 50 eligible stationary-snap blitz bursts and alpha 5, adding early acceleration changes RMSE by +0.0028 yd/s for post-snap speed gain (n=48; 95% player-bootstrap interval -0.0140 to +0.0195) and +0.0113 yd for post-snap displacement (n=48; +0.0030 to +0.0209). For pressure rate (at least 50 blitz opportunities; n=48), RMSE changes by -0.0003 (-0.0011 to +0.0006); for get-off time (at least 20 events; n=57), it worsens by +0.0037 s (+0.0006 to +0.0070). At the stricter 100-burst threshold, the post-snap burst sample has 32 players and remains non-improving; pressure-rate coverage is unchanged because its separate minimum is 50 blitz opportunities. These results do not establish transfer to first-step execution or pressure production; the fixed 1–4 second speed-window analysis also shows no stable transfer. This is a boundary on the claim, not evidence that Combine acceleration has no value in other roles or outcomes.

A player-balanced play-level logistic sensitivity adds down, yards-to-go, distance band, and lined-up position to the same pre-draft baseline. Across 48 DLs and 8,371 rookie blitz rows, early acceleration changes equal-player-weighted held-out Brier by -0.000052 at alpha 5 (baseline 0.072714; fixed-prediction player-bootstrap interval -0.000102 to -0.000002). Fold changes are negative in all three classes at that penalty, but the relative reduction is below 0.1%, the result is sensitive to regularization, and pressure opportunities do not identify individual blocker assignment or opponent quality. Treat it as a near-null exploratory result, not evidence of meaningful pressure lift.

Because the outcome is itself an estimated player-level p90, a second sensitivity resampled each player's observed rookie regular-season games, retaining all matched tracked player-play peaks within each sampled game. For the 46-player primary sample, the median player had 268.5 matched plays over 14 games; the median p90 standard error was 0.149 yd/s. In 1,000 simulations, each player's outcome was drawn from its game-cluster bootstrap distribution and the entire pre-draft cohort-held-out model was refit. At alpha 5, the median RMSE change was -0.0402 yd/s (95% simulation interval -0.0648 to -0.0182); all 1,000 simulations favored the early-acceleration model. At alpha 20, the median change was -0.0333 (-0.0517 to -0.0160). This addresses finite-game sampling error in the outcome more directly than fixed-prediction resampling, but not selection across candidate screens or future-class validity. The cluster bootstrap is conditional on each player's observed season, role, available tracking, and games.

Across 250 repeated balanced partitions into disjoint game halves, the half-specific p90 rankings had median Spearman rho 0.813 (2.5–97.5% partition range 0.740–0.872). The locked Combine model reduced cohort-held-out RMSE in both halves for every partition. This tests whether the association depends on a particular subset of observed games, but does not correct discovery selection or substitute for a future class.

## Measurement evidence

The feature's same-session ICC(A,1) for defensive linemen is 0.727 (63 paired players, 95% player-bootstrap interval 0.570–0.833); the estimated reliability of the average of two attempts is 0.842. This is stronger than the corresponding estimate for receivers and defensive backs, but still does not establish between-day or multi-site reliability. In the full cohort, early-acceleration ranks show more sensitivity to movement-onset thresholds and smoothing choices than late acceleration and speed retention. Top/bottom-quintile overlap varies about 0.79–0.88 under the tested alternatives. This limits fine ranking and calls for trace-level review.

## Broader screens and negative findings

- The all-position game-speed study showed a small aggregate improvement at some regularization settings but did not identify stable position-wide value. At 100 plays, only DL showed a plausible improvement at stronger regularization; DB and OL were not improved and WR was mixed. Position-wise estimates are exploratory.
- Career role models did not show consistent improvement from the phase block.
- DB tackle-rate, WR curl-recovery, and OL mirror-drill screens did not establish incremental value over their baselines. Tackle rate is not tackle success; route separation and pressure labels have context and opportunity limitations.
- The hoop-drill DL analyses have limited repeat-attempt reliability and do not replace the narrower 40-yard acceleration question.

The broad search increases the chance that the chosen lead overstates its true effect. All screens and their limitations remain available in the output reports.

## Interpretation, uncertainty, and next validation

The result is an association in three historical rookie cohorts. It does not show that early acceleration causes higher game speed or that training can improve the outcome. Draft capital, body size, available snaps, subposition, coaching, and play assignment can confound the relationship. The role-adjusted model is a sensitivity analysis because NFL subposition is determined after the draft.

Before an operational scouting recommendation, require:
1. independent reproduction from the source tracking and outcome files;
2. sensitivity to alternative game-speed summaries, opportunity/context adjustments, and player exposure;
3. direct visual review of raw 40-yard traces and feature boundaries;
4. future-cohort validation when new season data become available.

The saved OOF arithmetic and fold scores have been independently recomputed from the prediction table (74/74 score and integrity checks pass). This does not independently refit the models or re-extract sensor features.

Until those checks pass, frame the finding as a candidate signal: early acceleration may add information for defensive-line high-speed capacity, while evidence does not establish NFL success or an actionable threshold.

