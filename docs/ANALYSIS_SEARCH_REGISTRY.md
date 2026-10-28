# Analysis search registry

**Purpose.** Record the analyses that informed the current narrative so readers can distinguish hypothesis generation, robustness checks, and outcome boundaries. This registry is a transparent audit, not a multiplicity correction. Analyses with different populations, football endpoints, and scoring rules are separate estimands and must not be pooled as if they were replications.

## Candidate and estimand map

| ID | Domain / population | Combine signal family | NFL target / unit | Search dimensions represented in saved artifacts | Status and interpretation |
|---|---|---|---|---|---|
| S01 | All tracked prospects, position-specific subsets | 40-yard early, late, retention; traditional 40/split/size baseline | Rookie game-speed p90; player | Position groups, 25/50/100-play thresholds, regularization grid | Exploratory broad screen. Reports: `game_speed_by_position_*`, `position_specific_game_speed_*`. |
| S02 | Defensive line | Early 40 acceleration | Rookie full-play p90; player | 25/50/100/200-play thresholds; pre-draft vs draft-capital/role sensitivity; fixed ridge grid | Post-screen discovery estimate, not confirmation. Reports: `dl_early_accel_predraft_*`, `dl_early_accel_*`, `dl_game_speed_role_*`. |
| S03 | DL with complete phase measurements | Early, 10–20 yd acceleration, 20–40 yd speed retention; phase block | Rookie full-play p90; player | Feature-specific versus common-case samples; phase/regularization choices | Exploratory ablation. Common-case comparisons use 34 or 37 players. Reports: `dl_game_speed_ablation_*`, `dl_nested_phase_selection_*`. |
| S04 | DB, DL, OL, WR common complete cases | Four positions × three 40-yard phases | Position-specific rookie p90; player | 12 candidate terms; inner LOCO selection within each outer held-out cohort | Narrow nested selection audit; pooled delta -0.0033 yd/s (interval crosses zero). It does not correct the wider search. `position_phase_nested_selection_*`. |
| S05 | DL | Early 40 acceleration | Fixed 1-, 2-, 3-, 4-second post-snap peak speed; player | Four horizons, cohort folds, ridge penalties | Target-definition sensitivity; no stable benefit across settings. `dl_fixed_window_game_speed_*`, `dl_fixed_2s_post_snap_*`. |
| S06 | DL | Early 40 acceleration + context | Four-second peak speed after snap | Down, distance, alignment, blitz role, snap speed, pre-snap motion | In-game diagnostic, not a pre-draft predictor. `dl_contextual_game_speed_*`. |
| S07 | DL | Early 40 acceleration | Stationary-snap blitz speed gain and displacement; player | 50- and 100-burst exposure thresholds, ridge penalties, draft cohorts | Direct football movement screen; at 100 plays n=32, speed-gain RMSE worsens by +0.0099 yd/s (alpha 5). `dl_40_accel_football_*`. |
| S08 | DL | Early 40 acceleration | Rookie blitz pressure rate and get-off time; player | Separate minimum exposures (50 blitz opportunities; 20 get-off events), penalties | Context-sensitive outcome screens. Pressure is near-null; get-off error worsens. Not blocker-assignment-adjusted. `dl_40_accel_football_*`. |
| S09 | DL | Early 40 acceleration plus Combine baseline and play context | Pressure event; player-balanced player-play Brier | Three regularization settings and cohort folds | Near-null incremental pressure result (<0.1% relative Brier change at alpha 5). `dl_pressure_context_*`. |
| S10 | DL | Combine hoop-drill movement features | Burst movement and pressure-related outcomes | Drill attempts, minimum opportunity thresholds, processing choices | Current notebook screen. Exploratory; measurement repeatability and eligible samples constrain use. `dl_hoop_*`, `dl_game_burst_*`, `dl_transfer_*`. |
| S11 | WR | Curl-drill direction-change/recovery features | Rookie route separation; player/route aggregation | Drill features and outcome definitions | Legacy exploratory output; producer is not called by the current notebook. Do not cite as part of the packaged notebook run. `wr_curl_*`. |
| S12 | DB | Backpedal drill movement features | Rookie tackle rate; player | Drill features and 50/100-play thresholds | Legacy exploratory output; backpedal producer is not called by the current notebook. Tackle rate is opportunity- and role-sensitive, not tackle success. `db_backpedal_*`, `db_tackle_*`. |
| S13 | OL | Mirror-drill direction/recovery features | Pressure / peak pressure probability; player and player-play | Aggregate and play-level formulations; context variants | Legacy exploratory output; mirror-drill producer is not called by the current notebook. Do not treat pressure as an individually assigned blocking result. `ol_mirror_*`. |
| S14 | Multiple positions | Combine phase profiles / traditional tests | Rookie career role/speed summaries | Position-specific and pooled career targets | Exploratory; no stable career-role lift. `career_speed_*`, `position_specific_career_*`. |
| S15 | Tracked 40-yard traces | Early acceleration, late acceleration, speed retention | Measurement-only, no NFL endpoint | 0.5/1.0/1.5 yd/s onset × 1/3/5-frame filters; signed course axis; repeat attempts | Processing and repeatability robustness; not outcome replication. `forty_feature_sensitivity_*`, `forty_forward_axis_*`, `forty_repeatability_by_position_*`. |

## What has and has not been selection-adjusted

- The 12-candidate nested audit adjusts selection only within four positions × three 40-yard phases, on a common 161-player sample and one game-speed endpoint.
- A separate three-phase DL nested audit covers only a 37-player common sample.
- Neither audit includes the complete search over drill families, endpoints, thresholds, inclusion rules, processing alternatives, or narrative choices.
- The saved player bootstrap intervals generally condition on fixed OOF predictions. The game-cluster target simulation refits models but conditions on the screened feature and observed players/cohorts.
- Results across 25/50/100/200-play cutoffs overlap heavily; they are sensitivity analyses, not independent replications.
- Therefore the post-screen -0.0497 yd/s DL full-play p90 estimate must remain labeled a discovery estimate. No global adjusted estimate or valid “winner” across all families has been computed.

## Reproducible inventory

The notebook contains 44 explicit runner calls: 43 analytical calls plus the core-run manifest writer. It invokes 38 distinct scripts including that manifest. The source archive also includes other root-level scripts and historical reports, so **presence in the archive or `outputs/` directory does not mean an analysis ran in the current notebook replay**. In particular, the WR curl, DB backpedal, and OL mirror reports are historical exploratory artifacts whose producer scripts are not invoked by the current notebook. The current notebook and its ordered calls are the reproducible release path; do not present those legacy results as part of the current run.

The core model families use complete draft-cohort holdouts (2023/24/25), training-fold preprocessing, fixed regularization grids, and player-level scoring unless the report says otherwise. Most bootstrap intervals resample fixed OOF player losses and do not refit models. The underlying source files retain feature construction, population filters, model formulas, split logic, and output paths. This registry groups analyses by scientific question rather than treating every script, threshold, alpha, or fold as a separate discovery.

## Next audit needed for a formal global claim

1. Freeze this registry before running any new candidate analysis.
2. Predefine one estimand, sample rule, baseline, candidate feature, score, and temporal split.
3. For any feature-selection claim, repeat the full selection procedure inside training folds, not just phase choice.
4. If a family-wide p-value is desired, define a common statistic and exchangeability scheme first; unrelated football outcomes cannot be combined by simply taking the smallest p-value.
5. Obtain a genuinely future draft cohort before calling a result confirmatory.
