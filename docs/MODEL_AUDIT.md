# Model audit

## Primary target and baseline

The outcome is player-level rookie regular-season p90 of per-play peak speed. Players need at least 100 matched player-play keys. The primary 46-player DL early-only complete-case model is compared with only official 40 time, 10-yard split, height, and weight. A separate role sensitivity adds draft capital and observed NFL subposition indicators; those are not features available before the draft.

The candidate adds only the mean 0–1 second acceleration feature. A separate ablation reports feature-specific complete-case samples, plus a direct 34-player comparison restricted to athletes with all three phases.

## Validation

Leave one entire draft cohort out in turn (2023, 2024, 2025). Each prospect is scored once. Median imputation and standardization are fit on the training folds. Ridge alpha is reported over a fixed grid; the held-out folds do not tune the penalty. The main metric is pooled player-level RMSE in yd/s. The baseline/candidate comparisons are paired on identical players.

The fixed-prediction player bootstrap resamples OOF players within each held-out cohort. It does not refit models and should not be read as a full training or model-selection interval. Separately, game-cluster resampling estimates uncertainty in each player's p90 outcome; each simulation independently redraws player outcomes and refits every held-out-fold model. Independent draws do not preserve shared game, opponent, or weather shocks. This captures one source of outcome uncertainty but remains conditional on the observed rookie games, role, available tracking, and selected analysis.

An additional nested phase-selection audit chooses among early, late, and retention features using inner leave-one-cohort-out performance within the outer training cohorts. On the 37-player common sample, the selected phase changes across folds and yields only -0.0012 yd/s pooled outer RMSE change (95% fixed-prediction player-bootstrap interval -0.0294 to +0.0269). A locked early feature on the same players yields -0.0475 (-0.0754 to -0.0203). This tests only the narrow three-phase choice; it cannot undo the wider search that motivated the project or supply an independent cohort. The bootstrap does not repeat selection. See [`dl_nested_phase_selection_report_100plays.md`](../outputs/dl_nested_phase_selection_report_100plays.md).

## Main early-only result

At 100 tracked plays, 46 DLs enter the primary early-only model (13/19/14 by class). At alpha 5:

| Held-out class | Players | Baseline RMSE | Candidate RMSE | Change |
|---:|---:|---:|---:|---:|
| 2023 | 13 | 0.4646 | 0.4148 | -0.0498 |
| 2024 | 19 | 0.4741 | 0.4344 | -0.0397 |
| 2025 | 14 | 0.4926 | 0.4297 | -0.0629 |
| Pooled | 46 | — | — | -0.0497 (95% fixed-prediction CI -0.0901, -0.0092) |

The target-uncertainty sensitivity independently draws each player's p90 from game-cluster bootstrap replicates and refits all three held-out-cohort models. At alpha 5, the median RMSE change is -0.0402 yd/s (95% simulation interval -0.0648 to -0.0182) across 1,000 refits; the candidate improves in all simulations. At alpha 20, the median change is -0.0333 (-0.0517 to -0.0160). Because draws are independent across players, shared game, opponent, and weather shocks are not preserved. A post-draft role-adjusted sensitivity on 42 players found a -0.0481 yd/s alpha 5 RMSE change (95% fixed-prediction interval -0.0818 to -0.0141).

As a complementary check, 250 balanced disjoint-game partitions gave median half-season p90 Spearman rho 0.813 (2.5–97.5% partition range 0.740–0.872). Refitting the same locked model on each half-specific target favored early acceleration in every partition, with median RMSE deltas from -0.030 to -0.043 yd/s across the two halves and ridge settings. This indicates the association is not driven by a particular subset of rookie games, but remains conditional on the same three historical cohorts and prior feature screen. The partition range is not a confidence interval. See [`dl_game_speed_split_half_results_100plays.json`](../outputs/dl_game_speed_split_half_results_100plays.json).

The forward 2025 fold (training only on 2023–2024) reduces RMSE by 0.0629 yd/s on 14 players at alpha 5 (95% fixed-prediction player-bootstrap interval -0.1089 to -0.0093). It is the sole forward-time cohort split and remains exploratory because the broader feature search included 2025.

## Phase ablation: feature-specific samples

The primary ablation baseline includes official 40, 10-yard split, height, weight, and log draft pick. Because late distance crossings are less available, each phase-specific model uses all players complete for that feature; the samples therefore differ.

| Added feature | Players | RMSE change (yd/s) | 95% fixed-prediction player-bootstrap interval |
|---|---:|---:|---:|
| Early 0–1 s acceleration | 42 | -0.0422 | -0.0793 to -0.0069 |
| Late 10–20 yd acceleration | 42 | +0.0099 | -0.0025 to +0.0222 |
| Speed retention, 20–40 yd | 34 | -0.0029 | -0.0193 to +0.0124 |
| All three phases | 34 | -0.0253 | -0.0725 to +0.0129 |

## Direct phase comparison on common players

To compare the added features on identical athletes, the analysis also restricts every candidate to the 34 players complete for all phases:

| Added feature | RMSE change (yd/s) | 95% fixed-prediction player-bootstrap interval |
|---|---:|---:|
| Early 0–1 s acceleration | -0.0539 | -0.0845 to -0.0243 |
| Late 10–20 yd acceleration | +0.0143 | -0.0087 to +0.0365 |
| Speed retention, 20–40 yd | -0.0029 | -0.0190 to +0.0131 |
| All three phases | -0.0253 | -0.0725 to +0.0129 |

All phase comparisons use alpha 5 and the draft-capital baseline. They are exploratory, unadjusted for the broad screening process, and do not identify causal mechanisms. Correlated movement phases can substitute for one another.

**Distance-axis sensitivity:** the revised phase features use signed x-coordinate progress, supported by full-trace PCA alignment within about one degree across cohorts. This yields 572/793 attempts reaching 40 yd versus 590 using cumulative x/y path; on common attempts, late-acceleration and retention rank correlations are 0.999 and 0.9997. This is a coordinate-axis sensitivity because the official course direction is not explicitly supplied. The ablations below were regenerated from the revised features; the primary early-acceleration feature is time-based and unaffected.

## Nested phase-choice sensitivity

To estimate optimism from choosing among the three 40-yard phases, an outer cohort holdout selects a phase using inner leave-one-cohort-out performance among only the two training cohorts. On 37 complete players (10/13/14 by cohort), the selected phase changes in every outer fold. Pooled RMSE change is -0.0012 yd/s (95% fixed-OOF player-bootstrap interval -0.0294 to +0.0269); a locked early feature on the same players gives -0.0475 (-0.0754 to -0.0203). This selection set contains only three previously screened phases, the bootstrap does not repeat selection, and no genuinely independent cohort is available. See [nested audit](../outputs/dl_nested_phase_selection_report_100plays.md).

An expanded nested audit selects among 12 position-phase candidates (DB, DL, OL, WR × three 40-yard phases) on 161 players with complete phase and pre-draft data at 100 matched plays. The pre-draft baseline allows separate official-test and size slopes by position. Inner cohort holdouts selected WR late acceleration for the 2023 outer fold and DL late acceleration for the 2024 and 2025 folds. The nested procedure's pooled OOF RMSE change was -0.0033 yd/s (95% fixed-OOF player-bootstrap interval -0.0099 to +0.0027). A locked DL early term on the same players yielded -0.0102 (-0.0149 to -0.0055), but it was selected after a wider screen. This audits only the 12-candidate 40-phase-by-position family, not other drills, outcomes, or processing choices. See [expanded nested audit](../outputs/position_phase_nested_selection_report_100plays.md).

## OOF score audit

The audit script independently recomputes pooled and held-out-cohort RMSE from the saved player-level predictions, checks prediction uniqueness, sample counts, and score agreement with the model JSON. **All 74 checks pass.** The audit verifies arithmetic and prediction-table integrity; it does not re-fit the models or independently derive the Combine features from raw tracking. See [the audit report](../outputs/dl_early_accel_score_audit.md).

## Minimum-play exposure sensitivity

The pre-draft model was repeated at 25, 50, 100, and 200 matched rookie tracked player-plays. At alpha 5, pooled OOF RMSE changes were -0.0362 (n=67; 95% fixed-OOF player-bootstrap interval -0.0636 to -0.0077), -0.0535 (n=57; -0.0857 to -0.0186), -0.0497 (n=46; -0.0901 to -0.0092), and -0.0640 yd/s (n=34; -0.1006 to -0.0228). The direction is consistent, with a smaller estimate at 25 plays and wider uncertainty at higher cutoffs. These are nested, overlapping player samples, not independent replications. At 200 plays the forward 2025 test has 10 players and the alpha-5 interval crosses zero. All cutoffs condition on post-draft tracked opportunity, so these results do not estimate performance for every draft prospect. See the corresponding `dl_early_accel_predraft_results_*plays.json` files.

## Signal-processing transfer sensitivity

The early feature was recomputed at each combination of three movement-onset thresholds (0.5, 1.0, 1.5 yd/s) and three centered median-filter widths (1, 3, 5 frames). On the same 46-player, 100-play sample, all 18 combinations (nine definitions × two fixed alpha values) reduced pooled OOF RMSE versus the pre-draft baseline. At alpha 5, the RMSE change ranged from -0.0358 to -0.0497 yd/s; at alpha 20, from -0.0306 to -0.0440. All 54 feature-setting-by-cohort fold comparisons were also negative. This supports robustness to these processing choices, not independent confirmation: the same selected outcome and cohort sample are reused, and the grid does not account for broader discovery selection. See [the measurement-transfer sensitivity report](../outputs/dl_early_accel_measurement_transfer_sensitivity_report.md).

## Fixed post-snap speed windows

The full-play p90 result is sensitive to how the game-speed outcome is observed. A separate tracking pass uses only the maximum speed during fixed 1-, 2-, 3-, and 4-second intervals after `ball_snap`; a play must have at least 90% of expected 10 Hz frames, start and end coverage within 0.1 seconds, and no gap above 0.2 seconds. At the 100-window threshold, 1 and 2 seconds retain the same 46 players, while 3 and 4 seconds retain 45.

| Window | Players | LOCO RMSE Δ at α=5 (95% fixed-prediction CI) | Forward 2025 Δ at α=5 (n; 95% CI) |
|---|---:|---:|---:|
| 1 s | 46 | -0.0054 (-0.0198, +0.0093) | +0.0185 (14; +0.0117, +0.0256) |
| 2 s | 46 | +0.0025 (-0.0085, +0.0152) | -0.0000 (14; -0.0160, +0.0146) |
| 3 s | 45 | +0.0059 (-0.0032, +0.0147) | -0.0157 (13; -0.0278, -0.0036) |
| 4 s | 45 | +0.0070 (-0.0019, +0.0162) | -0.0187 (13; -0.0281, -0.0114) |

The pooled fixed-window intervals cross zero, while forward 2025 effects vary by horizon: alpha-5 error worsens at 1 second and improves at 3–4 seconds; the 3–4 second benefits reverse at alpha 20. The 2025 fold trains only on 2023–2024 and is the sole forward-time split, but feature selection included all cohorts and the holdout is exploratory. Full-play peak speed remains sensitive to opportunity, role, and tracking duration; do not call the primary association first-step or burst transfer.

## Football-specific transfer screen

Using only pre-draft baseline features, an additional early-acceleration screen tests one-second post-snap speed gain and displacement on stationary-snap blitz plays, blitz pressure rate, and get-off time. With 50 eligible stationary-snap burst plays and alpha 5, RMSE changes are +0.0028 yd/s (n=48; 95% CI -0.0140 to +0.0195) and +0.0113 yd (n=48; +0.0030 to +0.0209). Pressure rate (50 blitz opportunities) changes by -0.0003 rate units (n=48; -0.0011 to +0.0006); get-off time (20 events) worsens by +0.0037 s (n=57; +0.0006 to +0.0070). These results do not support a claim that Combine early acceleration predicts post-snap burst or pressure production. They are context-sensitive and exploratory.

A separate player-balanced logistic screen models pressure per rookie blitz play with down, yards-to-go, distance band, and lined-up position. It includes 48 players and 8,371 player-play rows. At alpha 5, adding early acceleration changes OOF Brier from 0.072714 to 0.072662 (-0.000052; 95% fixed-prediction player-bootstrap interval -0.000102 to -0.000002), a reduction under 0.1% relative. Alpha 0.1 and 1 intervals include zero. Although alpha 5 changes are negative in all three cohorts, its magnitude is negligible for practical use and it follows multiple candidate screens. The pressure event does not identify individual blocker responsibility or opponent quality.

## Known risks

- The signal followed a broad discovery screen across positions, outcomes, and drill features. No multiplicity adjustment or nested feature-selection procedure removes this selection optimism.
- Only three draft cohorts are available. The 2025-only forward split trains on 2023–2024, but it is exploratory because the feature was discovered using all cohorts; the 2023 and 2024 leave-one-cohort-out folds include future classes in training.
- The full-play p90 target represents speed peaks in observed tracked plays and is sensitive to play duration/opportunity. It is not a route-adjusted or assignment-adjusted performance outcome. Fixed 1–4 second post-snap targets show no consistent early-acceleration lift.
- Draft capital, body size, team, coaching, health, position usage, and play assignment may confound associations.
- Subposition is observed post-draft and is only a role sensitivity check.
- The number of eligible players differs by model (46 pre-draft early-only; 42 role-adjusted early-only; 34 all-phase). Direct phase comparisons use the common 34-player subset.
- Bootstrap intervals condition on the observed players and cohorts. Outcome simulations model game-sampling uncertainty but do not simulate a new league or new draft class.

All fold-level and player-level predictions remain in `outputs/` for audit. The model is a research lead, not a prospect-ranking system.

