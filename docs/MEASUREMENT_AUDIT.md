# Measurement audit: 40-yard early acceleration

## Audit question

Can first-second acceleration from Combine tracking be measured consistently enough to compare defensive-line prospects, and how much does the estimate depend on processing choices?

## Operational definition

Combine tracking is sampled at 10 Hz. The pipeline applies a centered three-frame median filter to speed. Movement onset is the first sample at or above 1 yd/s for two consecutive frames. Early acceleration is the change in smoothed speed over the first second after onset. Player features are aggregated from available attempts.

The onset threshold identifies observed movement. It does not measure reaction time because the source data contain no stimulus/cue timestamp. The calculation does not describe stance, force, body angle, foot placement, or first-step mechanics.

## Repeat-attempt reliability by position

Paired attempt 1 and 2 estimates show meaningful position differences:

| Combine group | Paired players | ICC(A,1) | 95% player-bootstrap interval | Typical error |
|---|---:|---:|---:|---:|
| DL | 63 | 0.727 | 0.570–0.833 | 0.296 yd/s² |
| OL | 78 | 0.581 | 0.385–0.717 | 0.451 yd/s² |
| DB | 69 | 0.201 | -0.006–0.519 | 0.645 yd/s² |
| TE | 26 | 0.277 | -0.009–0.524 | 0.379 yd/s² |
| WR | 66 | 0.051 | -0.314–0.512 | 0.654 yd/s² |

For DLs, the estimated reliability of the mean of two attempts is 0.842. The mean attempt-2 minus attempt-1 difference is -0.004 yd/s², so there is little average session-order shift in this sample. These are same-session estimates; they do not establish day-to-day, facility-to-facility, or sensor-to-sensor repeatability. ICC is influenced by the spread of players in the sample, so the typical error is reported alongside it.

The result supports position-specific measurement work for DLs. It does not justify precise one-attempt ranking near a scouting threshold. The position estimates are based on modest samples, particularly TE and the speed-retention measures.

## Attempt-specific transfer sensitivity

In the primary 100-play DL sample, 27 of 46 players had both tracked 40-yard attempts (2023: 6; 2024: 13; 2025: 8). Attempt-level early-acceleration values correlated at Pearson r=0.792 and Spearman ρ=0.720. On that identical paired subset, the attempt-1-only and attempt-2-only features each lowered held-out RMSE at ridge alpha 5 (-0.064 and -0.066 yd/s; 95% fixed-OOF player-bootstrap intervals -0.112 to -0.017 and -0.106 to -0.027); each had negative point estimates in all three held-out cohorts. The subset is small and overlaps the discovery sample, so this is measurement robustness sensitivity, not independent validation. See the [report](../outputs/dl_attempt_specific_transfer_report_100plays.md), [results](../outputs/dl_attempt_specific_transfer_results_100plays.json), and [OOF predictions](../outputs/dl_attempt_specific_transfer_oof_100plays.csv).

## Processing sensitivity

The sensitivity analysis recomputed 793 attempts from 418 prospects under 0.5, 1.0, and 1.5 yd/s onset thresholds and 1-, 3-, and 5-frame median filters. Against the default 1 yd/s and 3-frame definition:

- Early-acceleration rank correlation was about 0.923–0.953 under alternative 0.5 or 1.5 yd/s onset thresholds. Top- and bottom-quintile overlap was generally about 0.79–0.88. At the 1 yd/s threshold, changing the filter alone also shifted some player ranks.
- Late-acceleration and speed-retention rank correlations were generally at least 0.986.
- The 20–40 yd retention feature has lower coverage because some attempts lack a complete distance crossing.

The broad rank ordering for early acceleration is related across reasonable definitions, but edge-of-rank decisions are less stable than for later phases. A percent-level scouting cutoff would therefore be especially vulnerable to processing choices. Signal-processing stability and repeated-attempt reliability answer different questions; both are required.

## Rookie game-speed target repeatability

The full-season p90 target is also sensitive to the games observed. In 250 repeated, play-count-balanced partitions of the 46 primary DLs' rookie seasons, each half contained at least two games. The p90 rank correlation between disjoint halves had median Spearman rho 0.813, with a 2.5–97.5% range of 0.740–0.872 across random partitions; the median absolute difference between half-specific p90s was 0.220 yd/s. Refitting the locked Combine model on each half-specific target favored early acceleration in all 250 partitions per half at both tested penalties. This is a within-sample game-sampling sensitivity, not external validation; the partition range is not a confidence interval. See [`dl_game_speed_split_half_results_100plays.json`](../outputs/dl_game_speed_split_half_results_100plays.json) and [`fig_dl_game_speed_split_half.png`](../outputs/fig_dl_game_speed_split_half.png).

## Coverage and missingness

At the 100 tracked-player-play threshold, 63 defensive linemen have sufficient rookie game exposure. 42 have complete data for the early-only draft-capital-adjusted model; 34 have complete late-phase and retention features for the three-phase comparison. The early-only analysis uses a feature-specific sample and no longer discards players solely because late-phase features are missing.

Missing official 40/split values, draft pick, or tracked 40 features reduce the complete-case model sample. They are not imputed before cohort splitting; medians are estimated within each training fold.

## Measurement conclusion

First-second acceleration is more repeatable within the DL sample than within DB or WR, making DL the defensible position for a focused study. The feature remains less processing-stable than later phases. Keep its output as a continuous descriptive measure, display the trace and quality/coverage context, avoid hard cutoffs, and seek independent repeated testing before operational scouting use.

## Audit limits

An outcome-blind trace sample selected ten defensive-line player profiles at evenly spaced ranks across the early-acceleration distribution and plotted all 19 available 40-yard attempts for those players. The raw-speed and smoothed-speed traces are shown in [the trace review figure](../outputs/forty_raw_trace_review.svg), with attempt-level checks in the [sample table](../outputs/forty_raw_trace_review_sample.csv). Visual review found smooth early speed build, close agreement between raw and smoothed traces, and no obvious isolated one-frame spikes in this sample. All 19 had 10 Hz cadence within 5 ms, no negative speed, and a detectable onset. Across all 793 tracked 40-yard attempts, there were no intervals outside 5 ms of expected 10 Hz cadence, nonpositive intervals, or negative speeds. Supplied speed versus coordinate-derived speed had median attempt-level MAE 0.108 yd/s (95th percentile 0.141; maximum 0.260). This checks time-series integrity and sensor-field consistency, not device calibration or biomechanical validity.

The broader audit verifies definitions, paired-attempt reliability, processing sensitivity, trace cadence/speed summaries, a visual review of 19 selected traces, and feature availability. Visual review of the other 774 attempts, between-day test-retest reliability, and device-to-device agreement remain outstanding.

## Distance-axis sensitivity

Distance-indexed 40-yard features are now computed from signed x-coordinate progress from movement onset. The sign is set by net x displacement; minor backsteps are clipped from the interpolation axis. This shared coordinate axis is supported by full-trace PCA: cohort median yaw is 0.11° in 2023, -0.05° in 2024, and 0.14° in 2025; the cohort p10–p90 ranges all fall within about one degree of the x-axis. This is an inferred course axis, not an official sensor calibration. In the raw sensitivity, x progress reaches 40 yd in 572/793 attempts versus 590/793 for cumulative x/y path, leaving 18 path-only crossings. Among common attempts, rank correlation was 0.999 for 10-to-20-yard acceleration (n=784) and 0.9997 for 20-to-40-yard retention (n=572). The broad player rankings are nearly unchanged, with modest coverage loss on the long segment. The primary 0-to-1-second acceleration feature is time-indexed and unaffected. See [`forty_forward_axis_sensitivity.json`](../outputs/forty_forward_axis_sensitivity.json) and [`forty_forward_axis_attempt_sensitivity.csv`](../outputs/forty_forward_axis_attempt_sensitivity.csv).

## Cross-axis speed gap in the prospect explorer

The explorer reports an absolute cross-axis speed gap for agility and skill drills. It is the difference between mean positive and negative velocity projected onto the axis perpendicular to the initial movement path, averaged across available attempts when both signs are observed. This is a path-relative trace description: the tracking data do not include body orientation or standardized drill-side labels, so it is not anatomical left/right ability and should not be read as a pure 5-10-5 shuttle-direction speed comparison. The current derived table has both signs for 134/170 short-shuttle profiles and 159/160 three-cone profiles. The equivalent within-attempt gap is present in only 1/793 tracked 40-yard attempts, so it is not used as a 40-yard asymmetry metric. This exploratory feature has not been validated against an NFL outcome.

The prospect explorer also displays the number of tracked 40-yard attempts contributing to the player's mean 0-to-1-second acceleration and, when at least two attempts are available, their within-session sample standard deviation. Of 510 prospects, 375 have two usable attempts, 43 have one, and 92 have no usable early-acceleration estimate. This spread describes variation among the supplied attempts; with at most two attempts, it is not a stable reliability estimate, confidence interval, or between-day test-retest measure. A one-attempt profile has no repeat-spread estimate.

## Selection-aware phase check

An outer leave-one-cohort-out check selected among early acceleration, 10-to-20-yard acceleration, and 20-to-40-yard retention using only inner held-out results from the two outer-training cohorts. On the 37 complete-case DLs at the 100-play threshold, the selected phase differed across all three outer folds; pooled selected-feature RMSE change was -0.0012 yd/s (95% fixed-OOF player-bootstrap interval -0.0294 to +0.0269). A locked early feature on the same 37 players gave -0.0475 yd/s (-0.0754 to -0.0203). This narrows selection to three already-screened phases and cannot correct the broader discovery search or provide independent future validation. The fixed-prediction bootstrap also does not repeat model fitting or phase selection. See [`dl_nested_phase_selection_report_100plays.md`](../outputs/dl_nested_phase_selection_report_100plays.md).

