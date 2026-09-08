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

This is a real training UAV: two hundred and twenty-nine cycles before failure. Cut it at cycle one hundred and forty-eight, the median observed test length, and everything right of the line is unknown. Those eighty-one cycles are the quantity to predict.

Our best submitted model scores zero point eight seven six five two on the public leaderboard. The rest of this talk is the answer to one question: which decisions produced that number.

> **Speaker reference (not spoken).** Planned 0.75 min · 110 words · about 0.88 min at 125 wpm.  
> Figure: generated for this talk from data/train.csv · claims C01-C04

## Slide 2 — Five submissions, five decisions

*Each point is a submitted model; the label names the decision that run introduced*

Here is the whole answer on one axis, and then we will walk it.

Run three was a tuned XGBoost on the raw remaining-life target under our original validation scenarios. Zero point five four. Run four changed the validation scenarios and capped the fitting target at a hundred and twenty-five, and jumped to zero point eight four five. Run five brought the drift-pruned feature set and a calibrated fifty-fifty tree blend: zero point eight six five. Run six added conditional conservative calibration: zero point eight six seven. Run seven added a cross-fitted residual correction on six seeded tree members: zero point eight seven six five.

Two honest qualifications. Each point is a submission, not an isolated ablation, so a run can carry more than one change; the development experiment that motivated each one is named under it. And the deltas shrink by two orders of magnitude across the chain. One decision is worth three tenths of R squared. Everything after it is worth about three hundredths in total.

> **Speaker reference (not spoken).** Planned 1.25 min · 168 words · about 1.34 min at 125 wpm.  
> Figure: generated from kaggle_scores.csv and pipeline_experiments.md · claims C05-C12

## Slide 3 — Phase 0 decided which channels may carry information

*Consistency inside a UAV, not pooled correlation, is what generalises to an unseen UAV*

The channels are anonymous, so we can only ask how they behave.

The two right-hand panels are the ones that decided things. Channel twenty-one and channel nineteen have a strong median within-UAV correlation with remaining life and move in the same direction in one hundred per cent of UAVs. Channel twenty-five and channel sixteen do the same in the opposite direction. Those are the degradation candidates.

Channel eighteen is the interesting counter-example. It has a clear pooled and age-controlled association, and almost no within-UAV relationship: ninety-six per cent of its variance sits between UAVs. That is operating context, not degradation, so we use it as a baseline rather than as a trend. Channel twenty is flat everywhere; six such channels were removed, leaving twenty-two.

The decision from this phase was a taxonomy, not a model: ten degradation candidates, four context channels, one state channel, six removals.

> **Speaker reference (not spoken).** Planned 1.25 min · 146 words · about 1.17 min at 125 wpm.  
> Figure: repository Phase 0 script re-run for the eight channels shown · claims C13-C19

## Slide 4 — Phase 1 decided what every later number is measured against

*One hundred UAVs, not 24,720 rows: the UAV is the independent unit*

This slide is why the later numbers mean anything, so it is worth two minutes.

We have twenty-four thousand seven hundred and twenty cycle rows, but only one hundred independent machines. Splitting rows at random would put cycles from the same UAV on both sides and answer the wrong question. So every fold is a whole-UAV fold: five outer folds of twenty UAVs, and inside each outer-training partition four inner folds that do all feature, model and hyperparameter selection. An automated check re-verifies this, including recomputing prefix features after corrupting post-cutoff telemetry.

The figure decided the cutoffs. Test histories are systematically shorter than complete training lifetimes: the empirical CDF shows forty-seven of a hundred test UAVs stopping before cycle one hundred and forty-five, which is the shortest complete training life we have. So we never validate on complete run-to-failure histories. We sample cutoffs from the observed test length distribution, twenty per training UAV, and build every feature from cycles at or before the cutoff.

Two more rules follow. Each UAV carries a total training weight of one, so a five-hundred-cycle UAV does not outvote a hundred-and-fifty-cycle one. And uncertainty resamples whole UAVs: extra prefixes and extra seeds do not create extra independent machines.

> **Speaker reference (not spoken).** Planned 2.00 min · 203 words · about 1.62 min at 125 wpm.  
> Figure: empirical-CDF panel of the repository Phase 0 history-length figure · claims C20-C27

## Slide 5 — Decision 1: bounded scenarios and a fitting cap at 125

*Run 3 → Run 4, +0.30377 public. Blue = ExtraTrees, orange = XGBoost, paired against the control*

This one decision is worth more than everything after it combined, so it deserves the most scrutiny.

Offline scores and the leaderboard disagreed, and the hypothesis was that we were asking for exact remaining life very early in a UAV's life, where it is barely identifiable. We tested it as a two-by-two: validation scenario profile crossed with fitting target, everything else fixed. Each bar is the paired fold improvement over the unchanged control, for both model families.

Capping alone, under the original scenarios, made things clearly worse: about five cycles of RMSE lost. Bounded scenarios with a raw target gained about six. Bounded scenarios plus the cap gained about eighteen to twenty cycles, and both model families agree.

Now the caveat that this slide exists to carry. The two right-hand cells restrict true validation remaining life to one through one hundred and twenty-five. That is not the same evaluation: it changes the difficulty and it changes the denominator of R squared. So the development gain is not a clean measurement of model quality.

What we can say is narrower and is what the leaderboard confirmed. We froze the joint policy, bounded scenarios together with the cap, not capping in isolation; and the public score moved from zero point five four to zero point eight five. Every number after this slide uses that policy, with metrics still computed against raw labels.

> **Speaker reference (not spoken).** Planned 2.00 min · 230 words · about 1.84 min at 125 wpm.  
> Figure: lower panel of the generated PE_2 paired comparison · claims C28-C35

## Slide 6 — Decision 2: which engineered features earn their place

*Blue = ExtraTrees, orange = XGBoost, each treatment paired against the same control*

Every channel becomes twenty-seven numbers: the latest value, a baseline from the first ten cycles, the deviation between them, the slope over the whole prefix, and level, spread and slope over the last five, twenty and fifty cycles. All computed from cycles at or before the cutoff.

The question was whether those summaries actually add anything. The control here is cycle age plus the latest value of every nonconstant channel. Each treatment adds one degradation-signal family, with folds, models and search budget fixed.

The inverse pair fifteen and twenty-three is the strongest single family, worth about six cycles of RMSE and about zero point one two R squared to both models. All four families together are worth about ten cycles and zero point one seven, improving in five folds out of five for both families. The discrete state channel, oh seven, is the one treatment that did not survive.

This is what the Run 5 feature set was built from. Three further representation experiments were rejected on the same folds: compressed health indices, denser training prefixes and per-UAV normalisation. Those are in the backup.

> **Speaker reference (not spoken).** Planned 1.50 min · 184 words · about 1.47 min at 125 wpm.  
> Figure: lower panel of the generated PE_2 paired comparison · claims C36-C42

## Slide 7 — Decision 3: the model family, chosen under one locked protocol

*Twenty locked scenarios, 100 unseen UAVs, three seeds; no result was seen before the protocol froze*

We did not assume trees. Sixteen model families were declared, and the sequence models got the same folds, the same UAV weighting and their own tuning budget.

This is the locked comparison on the original task. XGBoost at zero point eight zero four, Random Forest at zero point seven eight, Extra Trees at zero point seven six. Then a wide gap: trajectory retrieval at zero point five one, sensor-graph TCN at zero point four three, multi-scale CNN at zero point three six. The mean baseline is negative because it predicts a constant, and the TCN at minus two point four was unstable across seeds rather than merely weak.

We ran the fairest rematch we could later, under the current protocol, where hybrid models get the raw telemetry window and all two hundred and ninety-eight engineered features as well. The tree blend was at eleven point five one cycles, hybrid GRU at fourteen point six, hybrid CNN at seventeen point seven, and neither hybrid won a single fold out of five. A separate dedicated temporal study reached mean R squared zero point six five with an LSTM.

Read that narrowly. It says the architectures, representations and budgets we tested here did not beat engineered-feature trees on one hundred UAVs. It does not say deep learning cannot do this task.

> **Speaker reference (not spoken).** Planned 2.00 min · 218 words · about 1.74 min at 125 wpm.  
> Figure: repository architecture study run 5 comparison, unchanged · claims C43-C53

## Slide 8 — Decision 4: a cross-fitted residual correction

*Run 6 → Run 7, +0.00911 public. Averaging six members did not help; learning their residual did*

This is the last decision in the chain and the one that produced zero point eight seven six five two.

Three XGBoost models and three ExtraTrees models, seeds thirteen, thirty-seven and seventy-three. Each family is averaged, and the weight between the families is chosen on the training side only. That gives a base prediction. A deliberately small second model then predicts the residual of that base prediction from the base prediction itself, the observed history length, the spread and range across the six members, the disagreement between the two families, and eight sensor features. Seven leaves, one hundred iterations. The correction is subtracted and the estimate is clipped at zero.

The calibration step is where a stack like this normally leaks. The out-of-fold predictions used to fit the correction come from models that excluded the corresponding UAV, and calibration endpoints are filtered to training UAVs. The true future remaining life is never an input.

The chart is why we attribute the gain to the correction rather than to bagging. Averaging the six members, taking their median, or trimming them all land near twelve cycles, worse than the frozen control at eleven point five one. The residual correction reached eleven point zero two, four point two per cent better, winning all five folds. On the leaderboard that became plus zero point zero zero nine one one.

> **Speaker reference (not spoken).** Planned 2.00 min · 225 words · about 1.80 min at 125 wpm.  
> Figure: repository PE_11 comparison, unchanged · claims C54-C60

## Slide 9 — Decision 5: conditional conservative calibration

*Run 5 → Run 6, +0.00216 public. A monotone trade-off: safety is bought with accuracy*

Predicting more life than a UAV has is a different kind of error from predicting less. We tested whether that asymmetry can be reduced without paying for it.

The correction is a function of predicted remaining life, fitted on four folds and applied to the fifth, and it can only lower a prediction. The control is top right: R squared zero point eight nine four, root-mean-square overprediction seven point two five cycles. Moving left buys safety and sells accuracy, monotonically. We selected quantile zero point five five, the safest policy still within zero point zero zero five R squared of the best.

On the leaderboard this was the smallest retained step, plus zero point zero zero two. And it is an error property measured on development data, not evidence about flight safety or certification.

> **Speaker reference (not spoken).** Planned 1.00 min · 134 words · about 1.07 min at 125 wpm.  
> Figure: repository PE_4 trade-off, unchanged · claims C61-C65

## Slide 10 — What Run 7 still gets wrong

*Development out-of-fold predictions: 500 rows, 100 UAVs, 444 distinct endpoints*

These are the retained model's own saved development predictions. Read the left panel; the right one is a diagnostic offset that we do not apply.

The scatter is tight below about forty cycles and fans out above it. The band from fifty-one to one hundred and twenty-five holds ninety and a half per cent of the squared error, and the bias flips sign inside it — plus five cycles between seventy-six and one hundred, minus four and a half above that, the signature of the cap.

By observed history it is sharper: the rows with at most a hundred cycles are twenty-one per cent of rows and thirty-two and a half per cent of the error.

That suggested a specialist for short histories. A hundred and fifty model fits later, the combined predictor gained zero point three nine per cent and won six folds of fifteen. Diagnosing a subgroup did not establish that the obvious remedy fixes it.

> **Speaker reference (not spoken).** Planned 1.25 min · 157 words · about 1.26 min at 125 wpm.  
> Figure: repository Phase 3 Run 7 report, unchanged · claims C66-C73

## Slide 11 — Why the chain stops at 0.87652

*The strongest post-Run-7 candidate lost its entire gain under a complete nested refit*

Six things were tried after Run 7. None replaced it, and the reason is the most useful thing in this talk.

On the left, features built from the model's own earlier forecasts: what did we predict two, five, ten and twenty cycles ago, and how has it changed. Screened on saved predictions it cut mean RMSE from eleven point zero to nine point six, thirteen per cent, winning all five folds, with a bootstrap interval below zero. The strongest single result anyone produced here.

It was labelled screening-only, because the base predictions came from a global out-of-fold table. So we rebuilt it: for every evaluation fold, refit the six tree members, select the blend weight, and fit both residual heads entirely inside the remaining UAVs. On the right is the result. One and a half per cent worse than its control, two folds of five. The thirteen per cent was a property of the evaluation procedure, not of the features.

The other five are variations on the same lesson. A pretrained tabular blend gained one point three per cent on five folds, then one point one on two new seeds, then one point nine on three more with eleven of fifteen fold wins against a declared bar of twelve, with a bootstrap interval still spanning zero. A short-history specialist gained zero point four. None was promoted, so no submission was made, so the score is still zero point eight seven six five two.

> **Speaker reference (not spoken).** Planned 1.75 min · 243 words · about 1.94 min at 125 wpm.  
> Figure: repository PE_15 comparison, unchanged; later results as numbers · claims C74-C84

## Slide 12 — Development performance and public performance

*Development R² above 0.9; the recorded public score is 0.87652*

Two panels, deliberately not one curve.

Left, development performance on held-out UAV folds: run six at mean-fold R squared zero point eight nine two seven, run seven at zero point nine zero zero four. The third bar pools the same five hundred predictions instead of averaging by fold. A different aggregation, not a better result.

Right, the recorded public score for those submissions. Run seven is zero point eight seven six five two. Encouragingly, the public gain from run six corresponds to three and a half per cent lower RMSE, close to the three and a half per cent we measured in development, so the two are moving together.

The honest headline is the gap: reaching zero point nine on that scored set needs about ten per cent lower RMSE. The objective is met in development and is not demonstrated publicly.

> **Speaker reference (not spoken).** Planned 1.00 min · 141 words · about 1.13 min at 125 wpm.  
> Figure: generated from run_6/run_7 report_summary.json and kaggle_scores.csv · claims C85-C93

## Slide 13 — What the next comparison can resolve

*PE_28: ten feature representations × two model recipes — training started, no winner yet*

One open question is whether the feature representation is the limiting factor. An alternative script outside our pipeline was reported to score zero point eight seven eight eight publicly. That is user-reported, for a different script, and not a matched experiment. What it gives us is a specific representation to test: two hundred and sixty-six features from twenty-two sensors against our two hundred and ninety-eight.

So we cross ten representations, from fifty to three hundred and ten features, with two model recipes, varying one factor at a time.

Status today: training started, no complete winner manifest.

> **Speaker reference (not spoken).** Planned 0.75 min · 96 words · about 0.77 min at 125 wpm.  
> Source: PE_28 README.md and run_1 input_verification.json, inspected 8 September 2026 · claims C94-C99

## Slide 14 — How we reached 0.87652

*One decision did most of the work; four smaller ones did the rest*

Ranked by what they were worth publicly: defining the validation scenarios and the fitting target together, zero point three. The feature set and tree blend, zero point zero two. The residual correction, zero point zero zero nine. Conservative calibration, zero point zero zero two.

Evaluation and target design dominated model choice, and after Run 7 nothing passed a nested confirmation. Development R squared is above zero point nine; the public score is not.

Happy to take questions.

> **Speaker reference (not spoken).** Planned 0.50 min · 77 words · about 0.62 min at 125 wpm.  
> Source: evidence_ledger.csv · claims C100

---

## Backup slides (not timed)

**B1 — Backup: full locked architecture comparison**  
Architecture study run 5, settings version 11; 20 locked scenarios × 100 UAVs  
Source: runs/run_5/7_architecture_comparison/architecture_comparison.csv

**B2 — Backup: sequence and hybrid models, in full**  
The matched rematch under the current protocol, and the dedicated temporal study  
Figure: repository architecture study run 8, unchanged; run 7 temporal summary

**B3 — Backup: the complete cap and scenario matrix**  
Why the two scenario profiles cannot share one R² axis  
Source: pipeline_experiments.md; _internal/shared_settings.toml

**B4 — Backup: residual correction, nested calibration and the safety table**  
How the Run 7 adapter avoids leaking, and what the calibration bought  
Source: residual_ensemble_contract.json; PE_4 calibration_summary.csv; r2 research report

**B5 — Backup: representation experiments that were rejected**  
Seven matched comparisons, all development-only, all with their control retained  
Source: pipeline_experiments.md experiment register

**B6 — Backup: uncertainty, endpoint duplication and score provenance**  
What the bootstrap intervals do and do not cover  
Source: diagnostic_summary.json; kaggle_scores.csv; PE_14 notes in the r2 report

---

## Timing check

| Slide | Topic | Planned min | Words | Spoken min at 125 wpm |
| ---: | --- | ---: | ---: | ---: |
| 1 | How we reached a public R² of 0.87652 | 0.75 | 110 | 0.88 |
| 2 | Five submissions, five decisions | 1.25 | 168 | 1.34 |
| 3 | Phase 0 decided which channels may carry information | 1.25 | 146 | 1.17 |
| 4 | Phase 1 decided what every later number is measured against | 2.00 | 203 | 1.62 |
| 5 | Decision 1: bounded scenarios and a fitting cap at 125 | 2.00 | 230 | 1.84 |
| 6 | Decision 2: which engineered features earn their place | 1.50 | 184 | 1.47 |
| 7 | Decision 3: the model family, chosen under one locked protocol | 2.00 | 218 | 1.74 |
| 8 | Decision 4: a cross-fitted residual correction | 2.00 | 225 | 1.80 |
| 9 | Decision 5: conditional conservative calibration | 1.00 | 134 | 1.07 |
| 10 | What Run 7 still gets wrong | 1.25 | 157 | 1.26 |
| 11 | Why the chain stops at 0.87652 | 1.75 | 243 | 1.94 |
| 12 | Development performance and public performance | 1.00 | 141 | 1.13 |
| 13 | What the next comparison can resolve | 0.75 | 96 | 0.77 |
| 14 | How we reached 0.87652 | 0.50 | 77 | 0.62 |
| | **Total** | **19.00** | **2332** | **18.66** |

Transitions and pauses: 1.00 min. Planned content 19.00 min plus transitions = 20.00 min.

The spoken-minute column is an estimate at 125 words per minute; it is not a substitute for a rehearsal. Slides whose spoken estimate exceeds the planned duration need either faster delivery or a sentence cut.
