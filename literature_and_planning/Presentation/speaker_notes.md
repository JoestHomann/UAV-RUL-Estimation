# Speaker notes — UAV remaining useful life estimation

Deck: `UAV_RUL_Project_20min.pptx` · Evidence cut-off: 8 September 2026

The narration below is embedded in the .pptx notes pane, slide for slide. The
"speaker reference" block under each slide is **not spoken**: it carries the
planned duration, the claim IDs from `evidence_ledger.csv`, and the artifact
the numbers came from.

Delivery notes:

- Do not read every number off a chart. The spoken text names the numbers that
  carry the argument; the rest are there for questions.
- Slides 4, 5, 7 and 10 carry the scope caveats. If time runs short, cut
  narration inside slides 3 and 6 rather than dropping either of those.
- Pause after slide 10's second panel; that is the point most audiences want a
  moment on.
- Backup slides B1-B6 follow the divider and are not part of the 20 minutes.


## Slide 1 — How we reached a public R² of 0.87652

*Estimating remaining useful life from a partial UAV telemetry history*

A fleet of inspection UAVs records twenty-eight anonymous telemetry channels once per flight cycle. We are given a UAV we have never seen, observed up to some cutoff, and asked how many cycles it has left.

This is a real training UAV: two hundred and twenty-nine cycles to failure. Cut it at cycle one hundred and forty-eight, the median test length, and everything right of the line is unknown. Those eighty-one cycles are what we predict.

Our best submitted model scores zero point eight seven six five two. This talk answers one question: which decisions produced that number. Most were made before any model was trained.

> **Speaker reference (not spoken).** Planned 0.75 min · 106 words · about 0.85 min at 125 wpm.  
> Figure: generated for this talk from data/train.csv · claims C01-C04

## Slide 2 — Five submissions, five decisions

*Each point is a submitted model; the label names the decision that run introduced*

Here is the whole answer on one axis.

Run three: a tuned XGBoost on the raw remaining-life target under our original scenarios, zero point five four. Run four changed the validation scenarios and capped the fitting target at a hundred and twenty-five: zero point eight four five. Run five brought the drift-pruned features and a calibrated tree blend. Run six added conditional calibration, run seven a cross-fitted residual correction: zero point eight seven six five.

Two qualifications. Each point is a submission, not an isolated ablation, so a run can carry more than one change. And the deltas shrink by two orders of magnitude: one decision is worth three tenths of R squared, everything after it about three hundredths. That first decision is about the data and the evaluation, not the model. So we start there.

> **Speaker reference (not spoken).** Planned 1.00 min · 136 words · about 1.09 min at 125 wpm.  
> Figure: generated from kaggle_scores.csv and pipeline_experiments.md · claims C05-C12

## Slide 3 — Six of twenty-eight channels carry no information

*Red: unique-value count and numeric range at the effectively-constant threshold*

Phase 0 is a reproducible audit of the data before any model exists. Its first job is subtraction.

Two channels, twenty and twenty-seven, take exactly one value in the whole training set. Four more — three, eight, fourteen and seventeen — vary only at machine precision: between ninety-three and ninety-eight percent of their rows sit inside a flatline run of at least five identical cycles. All six fail the effectively-constant test.

Twenty-eight channels become twenty-two, and the six never enter any feature set. That decision comes from the data, not from a model score.

> **Speaker reference (not spoken).** Planned 0.75 min · 92 words · about 0.74 min at 125 wpm.  
> Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_constant_features.py · claims C13-C18

## Slide 4 — Ten channels move with age inside every UAV

*Twelve of the twenty-two channels: the ten degradation candidates, plus telemetry_18 and 26*

The second job is to find which of the twenty-two remaining channels track degradation. Four views, one per panel.

The left panel pools every row — the weakest evidence, because it mixes UAVs of different ages. The second controls for flight cycle. The third computes the correlation inside each UAV separately, and the fourth gives the share of UAVs trending the same way.

The declared threshold is a within-UAV correlation of at least zero point three, with the same direction in at least seventy percent of UAVs. Ten channels pass. Seven, thirteen, nineteen, twenty-one and twenty-two increase in one hundred percent of UAVs; sixteen, twenty-five and twenty-eight decrease in one hundred percent.

Now telemetry eighteen. Visible age-controlled association in the second panel; essentially nothing in the third, at minus zero point zero one nine. A pooled analysis would have promoted it as a degradation signal. Inside a single UAV it does not move with remaining life at all.

A correlation is still not predictive value. These ten are candidates; whether they help is settled later by grouped validation.

> **Speaker reference (not spoken).** Planned 1.50 min · 177 words · about 1.42 min at 125 wpm.  
> Figure: repository script re-run on a channel subset · 0_data_analysis/core_data_analysis/temporal_rul_analysis.py · claims C19-C27

## Slide 5 — Variance decomposition assigns each channel its role

*Dark: variation inside one UAV over time. Light: persistent differences between UAVs.*

The previous slide asked whether a channel moves with age. This one asks where its variation lives, and the answer decides what kind of feature it should become.

Each bar splits a channel's total variance into the part that happens inside one UAV over time, dark blue, and the part that is a persistent difference between UAVs, light blue.

Seventeen channels are within-UAV dominated, including all ten degradation candidates, at seventy-four to eighty-seven percent. Those get temporal features: slope, rolling statistics, recent change, deviation from the UAV's own baseline.

Four are between-UAV dominated. Telemetry eighteen is the extreme case: ninety-six percent of its variance is simply that different UAVs sit at different levels. With the previous slide that is a coherent story — eighteen describes the UAV, not its degradation — so it is kept only as a baseline and level description.

The six removed channels appear as constant, with no bar at all: the same conclusion by a different route.

> **Speaker reference (not spoken).** Planned 1.25 min · 159 words · about 1.27 min at 125 wpm.  
> Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_within_between_variance.py · claims C28-C34

## Slide 6 — Train and test differ in age, not in distribution

*Left two panels: raw shift. Right two: the same comparison at matched flight cycle.*

Before trusting any of it, one question: are the test UAVs the same kind of object as the training UAVs?

The two left panels compare raw distributions and look alarming. The two right panels compare each test endpoint only against training UAVs observed at the same flight cycle. Almost all of the apparent shift disappears: the median age-matched shift is below zero point two seven training interquartile ranges for every channel.

So test UAVs are younger, not different. That is a statement about validation design, and it is the bridge into Phase 1.

Two channels stay on a watch list: telemetry five, with sixteen percent of its endpoints outside the same-age training range, and twenty-two with nine. Neither was removed on that evidence.

> **Speaker reference (not spoken).** Planned 1.00 min · 123 words · about 0.98 min at 125 wpm.  
> Figure: repository script re-run at slide size · 0_data_analysis/core_data_analysis/train_test_drift.py · claims C35-C41

## Slide 7 — The screening matrix that Phase 1 inherited

*A blue cell means the channel meets that documented threshold. Roles are not exclusive.*

Everything Phase 0 found, in one matrix. Each row is a channel, each column a documented threshold, each blue cell a channel that meets it. The roles are deliberately not exclusive.

Six removal candidates, ten degradation candidates, four context channels, one state channel. On top of that, ten channels flagged for anomaly review, fourteen for redundancy, fifteen for drift.

The important point is what we did not do with the last three columns. Nothing was removed for being correlated, for containing extreme readings, or for drifting. Telemetry nineteen and twenty-one correlate almost perfectly at UAV level and both were kept. A screening matrix summarises statistical evidence; it does not measure predictive value. Every warning became an experiment later, and several were answered with no.

Telemetry seven is the interesting row: it meets five criteria at once, so it is represented as a discrete operating state rather than a continuous sensor.

> **Speaker reference (not spoken).** Planned 1.25 min · 150 words · about 1.20 min at 125 wpm.  
> Figure: repository script re-run at slide size · 0_data_analysis/core_data_analysis/channel_classification.py · claims C42-C50

## Slide 8 — Phase 1 fixed what every later number is measured against

*Whole-UAV nested cross-validation, and ten automated assertions that must pass*

Phase 1 turns a hundred complete run-to-failure histories into something we can honestly measure a model on.

The split is by UAV, never by row. A hundred training UAVs become five outer folds of twenty, balanced by terminal lifetime. Inside each outer round the eighty training UAVs split again into four inner folds. Features, hyperparameters, early stopping and blend weights are all selected on the inner folds; the twenty outer UAVs only ever evaluate the selected procedure. Twenty inner rounds in total, and every UAV held out exactly once.

Why nest rather than take one eighty-twenty split? With a hundred UAVs and large UAV-to-UAV differences, one split is not a measurement.

None of this is trusted on faith. Ten assertions are re-checked automatically. The one to point at is prefix causality: the checker recomputes features for ten prefixes after replacing every post-cutoff telemetry value with extreme artificial numbers, and requires the features to come out unchanged. The status is passed.

> **Speaker reference (not spoken).** Planned 1.25 min · 160 words · about 1.28 min at 125 wpm.  
> Sources: 1_dataset_construction/2_UAV_grouped_validation_folds and 10_automated_leakage_checks/artifacts/verification_report.json · claims C51-C59

## Slide 9 — Cutoffs are drawn from the observed test history lengths

*Test UAVs are younger than training UAVs, so training samples must be truncated the same way*

This is the single most consequential slide in Phase 1.

Training histories run from a hundred and forty-five to five hundred and twenty-five cycles, median two hundred and twenty. Test histories run from thirty-eight to four hundred and seventy-five, median a hundred and forty-eight. Read the empirical CDF at a hundred and forty-five: about forty-seven percent of test UAVs stop before the shortest complete training lifetime begins.

So a model trained on complete run-to-failure sequences is asked to do something it has never seen. The fix is to truncate: every training UAV gets twenty distinct cutoffs, drawn from the actual test history lengths and shorter than its own life.

Each of a UAV's twenty prefixes carries weight one twentieth, so a five-hundred-cycle UAV and a hundred-and-fifty-cycle one influence training equally; otherwise long-lived UAVs would dominate simply by having more rows.

One honest weakness: the cutoffs come from the test length distribution, which is information from the test inputs. No label is used, and the audit checks that, but it is worth naming.

> **Speaker reference (not spoken).** Planned 1.25 min · 172 words · about 1.38 min at 125 wpm.  
> Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_history_length_distributions.py · claims C60-C68

## Slide 10 — 606 prefix features, and four sets to compare them with

*Every feature is computed from cycles at or before the cutoff, and scaled inside each fold*

A prefix is a variable-length sequence, and the models we intend to use need a fixed-width row. So each prefix becomes twenty-seven summaries per channel.

Five groups: where the channel is now, where it started, how far it has moved from its own baseline, how the whole history behaves, and what happened over the last five, twenty and fifty cycles. Comparing those windows lets a tree see acceleration, not only level.

Twenty-two channels times twenty-seven, plus ten state features, plus flight cycle and its logarithm, is six hundred and six.

Rather than pick one set, four nested sets were declared in advance, each answering a different question — is age alone predictive, does the current snapshot add anything, do the Phase 0 roles help, did the screening discard something useful.

Scaling is median and interquartile range, refitted for every fold and feature set on the training UAVs alone. Twelve of the four thousand seven hundred and ten fitted scales fell back to unit scale, and they are recorded rather than silently dropped.

> **Speaker reference (not spoken).** Planned 1.25 min · 171 words · about 1.37 min at 125 wpm.  
> Sources: 1_dataset_construction steps 5-7 · feature_catalog.csv and preprocessing_config.json · claims C69-C77

## Slide 11 — The benchmark: age alone predicts nothing

*A weighted linear fit on the cutoff cycle, evaluated on the same 20 locked scenarios*

The last thing Phase 1 builds is the simplest model that could work: predicted remaining life is a hundred and eighty-seven point three minus zero point five four two times the cutoff cycle, floored at zero. No telemetry at all.

On the left is why it fails. At any cutoff the true remaining life spans hundreds of cycles, because UAVs do not all live the same length. Age tells you almost nothing about which one you hold.

R squared of minus zero point zero zero five — slightly worse than the mean predictor — RMSE sixty-four and a half cycles, bias plus seventeen point seven, bootstrap interval minus zero point one five to plus zero point one three.

This is the reference for everything after it. An RMSE of eleven cycles means nothing alone; eleven against sixty-four and a half on the same locked scenarios means the engineered features carry real information. The interval resamples whole UAVs, because each UAV appears in twenty scenarios and resampling rows would be falsely narrow.

> **Speaker reference (not spoken).** Planned 1.25 min · 169 words · about 1.35 min at 125 wpm.  
> Figure: generated from 1_dataset_construction/9_cycle_only_baseline artifacts · claims C78-C86

## Slide 12 — Decision 1: bounded scenarios and a fitting cap at 125

*PE_2 paired RMSE improvement, mean-fold, blue ExtraTrees and orange XGBoost. Public: +0.30377*

With the protocol fixed, the first experiment crossed two choices: which validation scenarios to score on, and whether to cap the fitting target.

Capping means fitting against remaining life truncated at a hundred and twenty-five cycles, while evaluation still uses the raw label. A UAV with four hundred cycles left and one with two hundred look identical, so forcing the model to separate them wastes capacity on a region nobody scores. Bounding the scenarios means concentrating validation cutoffs on the bands the test set actually contains — the Phase 1 history-length finding, applied.

Both help, and they help together. The four cells are not on a common R squared scale — changing the evaluation target changes the denominator — so this is paired RMSE improvement within each fold.

Run four carried both changes and moved the public score from zero point five four to zero point eight four five. Three tenths of R squared from deciding what to fit and what to measure. No new model, no new feature.

> **Speaker reference (not spoken).** Planned 1.25 min · 166 words · about 1.33 min at 125 wpm.  
> Figure: lower panel of the generated PE_2 figure · experiments/PE_2/runs/run_1 · claims C87-C94

## Slide 13 — Decision 2: pruned features and a calibrated tree blend

*Locked architecture comparison, architecture study run 5, five held-out UAV folds. Public: +0.02012*

Two things changed between run four and run five, and both trace back to Phase 0.

The feature set became the drift-pruned screened set: two hundred and ninety-eight features, taking the Phase 0 screened definition and removing features whose train-test behaviour did not hold up. That is the drift-warning column being acted on at last — by experiment, not assumption.

The model family was chosen here, under one locked protocol, with every architecture given the same folds, the same cutoffs and the same features. The two tree families come out ahead and, more usefully, they make different mistakes, so a fifty-fifty blend beats either alone. The blend weight is selected on training-side data only.

Public score: zero point eight six five two five. Two hundredths. Notice the ratio — the model family, what people usually call the modelling decision, was worth about a fifteenth of the target and scenario decision.

> **Speaker reference (not spoken).** Planned 1.25 min · 148 words · about 1.18 min at 125 wpm.  
> Figure: repository figure, unchanged · 2_model_architecture_study/runs/run_5 · claims C95-C103

## Slide 14 — Decision 3: conditional conservative calibration

*PE_4: a quantile shift applied only where the model is likely to overpredict. Public: +0.00216*

Run seven's predecessor overpredicts remaining life more often than it underpredicts, and in maintenance an overprediction is the expensive direction.

PE_4 swept a conservative quantile shift and measured the cost. Shifting everything hurts accuracy. Applying the shift only conditionally, at quantile zero point five five, and only where the model's own signals say overprediction is likely, keeps almost all of the accuracy and removes part of the bias.

It is worth two thousandths of public R squared. I show it because it is the one decision in the chain taken partly for a reason other than the score.

> **Speaker reference (not spoken).** Planned 0.75 min · 99 words · about 0.79 min at 125 wpm.  
> Figure: repository figure, unchanged · experiments/PE_4/runs/run_1 · claims C104-C108

## Slide 15 — Decision 4: a cross-fitted residual correction

*PE_11: a small model that predicts the ensemble's own error from its disagreement. Public: +0.00911*

The last retained decision. Six tree members — three XGBoost and three ExtraTrees at seeds thirteen, thirty-seven and seventy-three — are averaged within family and then blended.

A small histogram gradient model, seven leaves and a hundred iterations, then predicts that base prediction's residual. Its inputs are the base prediction, the observed history length, how much the six members disagree, and a few baseline deltas and slopes. Member disagreement is a usable signal about where the ensemble is unreliable.

The leakage question matters here, so: the out-of-fold predictions that fit the residual model come from models which excluded that UAV, and calibration endpoints are restricted to training UAVs. True future remaining life is never an input.

Development effect: eleven point five one down to eleven point zero two cycles, four point two percent better, winning in all five folds. Public score zero point eight seven six five two.

> **Speaker reference (not spoken).** Planned 1.25 min · 146 words · about 1.17 min at 125 wpm.  
> Figure: repository figure, unchanged · experiments/PE_11/runs/run_1 · claims C109-C116

## Slide 16 — Why the chain stops at 0.87652

*Run 7 development out-of-fold predictions, and every candidate evaluated since*

Where the remaining error is, and why we stopped.

On the left are the out-of-fold development predictions. The structure is systematic, not random: the model compresses towards the middle, so long remaining lives are underpredicted and short ones overpredicted. That is what the fitting cap buys us, and it is also the ceiling it imposes.

On the right, six candidates tried after run seven, each with a promotion gate declared beforehand. A full nested refit of the most promising screen came back one point six percent worse. Extra seeds, a regime gate and a short-history specialist all failed. One TabPFN blend is promising and unconfirmed.

So nothing was promoted, nothing was submitted, and the public score is still zero point eight seven six five two. I would rather report that than a number obtained by submitting until something stuck.

> **Speaker reference (not spoken).** Planned 1.25 min · 139 words · about 1.11 min at 125 wpm.  
> Figures and tables: run_7/7_post_run_reporting and experiments/PE_15-PE_28 · claims C117-C127

## Slide 17 — How we reached 0.87652

*Ordered by the submission that carried each decision*

The answer to the question in the title.

Three tenths of R squared came from deciding what to fit and what to measure it on — and both halves of that decision were read off Phase 0 and Phase 1 evidence: the history-length distributions and the age-matched drift comparison. Two hundredths came from the feature set and the model family. Roughly one hundredth came from calibration and the residual correction together.

If there is one thing to take away, it is that the audit and the dataset construction were not preliminaries to the modelling. They were the modelling.

> **Speaker reference (not spoken).** Planned 0.75 min · 97 words · about 0.78 min at 125 wpm.  
> Sources: kaggle_scores.csv, pipeline_experiments.md, phase_0 and phase_1 artifacts · claims C128-C132

---

## Backup slides (not timed)

**B1 — Backup: the complete Phase 0 broad review**  
Nine analyses, and the decision each one contributed  
Source: literature_and_planning/development_documentation/phase_0_dataAnalysis.md

**B2 — Backup: redundancy, anomalies, and what was not removed**  
The strongest correlated pairs and the anomaly rates; every warning became an experiment  
Source: 0_data_analysis/core_data_analysis/figures/feature_redundancy and figures/anomalies

**B3 — Backup: the 27 features derived from every channel**  
All computed from cycles at or before the cutoff; verified by the prefix-causality assertion  
Source: 1_dataset_construction/5_prefix_feature_engineering and 6_feature_sets/artifacts/feature_catalog.csv

**B4 — Backup: the cycle-only baseline, group by group**  
The same held-out predictions, reported by cutoff band, outer fold and terminal-lifetime quintile  
Source: 1_dataset_construction/9_cycle_only_baseline/artifacts/metrics/

**B5 — Backup: full locked architecture comparison**  
Architecture study run 5, five held-out UAV folds, identical folds and cutoffs for every entry  
Source: 2_model_architecture_study/runs/run_5/7_architecture_comparison/architecture_comparison.csv

**B6 — Backup: sequence and hybrid models, in full**  
Architecture study run 8: hybrid combinations of a tree ensemble with a sequence encoder  
Source: 2_model_architecture_study/runs/run_8

**B7 — Backup: the complete cap and scenario matrix**  
PE_2, all four cells, with the target policy of each cell stated explicitly  
Source: experiments/PE_2/runs/run_1/reporting

**B8 — Backup: residual correction, calibration and the safety table**  
PE_11 and PE_4 in detail, with the nested-calibration guarantee stated in full  
Source: experiments/PE_11 and PE_4, runs/run_1/reporting

**B9 — Backup: representation experiments that were rejected**  
Every row is a matched development comparison whose control was retained  
Source: experiments/PE_5, PE_6, PE_7, PE_9, PE_13, PE_22, PE_28

**B10 — Backup: uncertainty, endpoint duplication and score provenance**  
How the intervals are built, what the repeated endpoints do, and where each score comes from  
Source: r2_research_2026_09_07/ and kaggle_scores.csv

---

## Timing check

| Slide | Topic | Planned min | Words | Spoken min at 125 wpm |
| ---: | --- | ---: | ---: | ---: |
| 1 | How we reached a public R² of 0.87652 | 0.75 | 106 | 0.85 |
| 2 | Five submissions, five decisions | 1.00 | 136 | 1.09 |
| 3 | Six of twenty-eight channels carry no information | 0.75 | 92 | 0.74 |
| 4 | Ten channels move with age inside every UAV | 1.50 | 177 | 1.42 |
| 5 | Variance decomposition assigns each channel its role | 1.25 | 159 | 1.27 |
| 6 | Train and test differ in age, not in distribution | 1.00 | 123 | 0.98 |
| 7 | The screening matrix that Phase 1 inherited | 1.25 | 150 | 1.20 |
| 8 | Phase 1 fixed what every later number is measured against | 1.25 | 160 | 1.28 |
| 9 | Cutoffs are drawn from the observed test history lengths | 1.25 | 172 | 1.38 |
| 10 | 606 prefix features, and four sets to compare them with | 1.25 | 171 | 1.37 |
| 11 | The benchmark: age alone predicts nothing | 1.25 | 169 | 1.35 |
| 12 | Decision 1: bounded scenarios and a fitting cap at 125 | 1.25 | 166 | 1.33 |
| 13 | Decision 2: pruned features and a calibrated tree blend | 1.25 | 148 | 1.18 |
| 14 | Decision 3: conditional conservative calibration | 0.75 | 99 | 0.79 |
| 15 | Decision 4: a cross-fitted residual correction | 1.25 | 146 | 1.17 |
| 16 | Why the chain stops at 0.87652 | 1.25 | 139 | 1.11 |
| 17 | How we reached 0.87652 | 0.75 | 97 | 0.78 |
| | **Total** | **19.00** | **2410** | **19.28** |

Transitions and pauses: 1.00 min. Planned content 19.00 min plus transitions = 20.00 min.

The spoken-minute column is an estimate at 125 words per minute; it is not a substitute for a rehearsal. Slides whose spoken estimate exceeds the planned duration need either faster delivery or a sentence cut.
