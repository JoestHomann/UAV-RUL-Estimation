"""Slide text, speaker notes and source labels for UAV_RUL_Project_20min.pptx.

The deck answers one question: how did this project reach a recorded Kaggle
public R2 of 0.87652, and which decision at each step produced it?

After the 9 September 2026 review the front half of the talk is the Phase 0
data audit and the Phase 1 dataset construction, because that is where the
decisions that the public score is most sensitive to were actually made.

Kept separate from ``build_deck.py`` so the narration can be reviewed and
re-exported to ``speaker_notes.md`` without touching the layout code.
"""

MAIN = [
    dict(
        n=1, minutes=0.75, layout="title",
        title="How we reached a public R² of 0.87652",
        subtitle="Estimating remaining useful life from a partial UAV telemetry history",
        figure="prediction_timeline.png",
        source="Figure: generated for this talk from data/train.csv · claims C01-C04",
        narration=(
            "A fleet of inspection UAVs records twenty-eight anonymous telemetry "
            "channels once per flight cycle. We are given a UAV we have never "
            "seen, observed up to some cutoff, and asked how many cycles it has "
            "left.\n\n"
            "This is a real training UAV: two hundred and twenty-nine cycles to "
            "failure. Cut it at cycle one hundred and forty-eight, the median "
            "test length, and everything right of the line is unknown. Those "
            "eighty-one cycles are what we predict.\n\n"
            "Our best submitted model scores zero point eight seven six five "
            "two. This talk answers one question: which decisions produced that "
            "number. Most were made before any model was trained."
        ),
    ),
    dict(
        n=2, minutes=1.00, layout="figure",
        title="Five submissions, five decisions",
        subtitle="Each point is a submitted model; the label names the decision that run introduced",
        figure="public_score_chain.png",
        source="Figure: generated from kaggle_scores.csv and pipeline_experiments.md · claims C05-C12",
        narration=(
            "Here is the whole answer on one axis.\n\n"
            "Run three: a tuned XGBoost on the raw remaining-life target under "
            "our original scenarios, zero point five four. Run four changed the "
            "validation scenarios and capped the fitting target at a hundred and "
            "twenty-five: zero point eight four five. Run five brought the "
            "drift-pruned features and a calibrated tree blend. Run six added "
            "conditional calibration, run seven a cross-fitted residual "
            "correction: zero point eight seven six five.\n\n"
            "Two qualifications. Each point is a submission, not an isolated "
            "ablation, so a run can carry more than one change. And the deltas "
            "shrink by two orders of magnitude: one decision is worth three "
            "tenths of R squared, everything after it about three hundredths. "
            "That first decision is about the data and the evaluation, not the "
            "model. So we start there."
        ),
    ),

    # ------------------------------------------------- Phase 0: data audit --
    dict(
        n=3, minutes=0.75, layout="figure",
        title="Six of twenty-eight channels carry no information",
        subtitle="Red: unique-value count and numeric range at the effectively-constant threshold",
        figure="constant_features.png",
        source="Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_constant_features.py · claims C13-C18",
        narration=(
            "Phase 0 is a reproducible audit of the data before any model "
            "exists. Its first job is subtraction.\n\n"
            "Two channels, twenty and twenty-seven, take exactly one value in "
            "the whole training set. Four more — three, eight, fourteen and "
            "seventeen — vary only at machine precision: between ninety-three "
            "and ninety-eight percent of their rows sit inside a flatline run of "
            "at least five identical cycles. All six fail the effectively-"
            "constant test.\n\n"
            "Twenty-eight channels become twenty-two, and the six never enter "
            "any feature set. That decision comes from the data, not from a "
            "model score."
        ),
    ),
    dict(
        n=4, minutes=1.50, layout="figure",
        title="Ten channels move with age inside every UAV",
        subtitle="Twelve of the twenty-two channels: the ten degradation candidates, plus telemetry_18 and 26",
        figure="temporal_rul_summary_subset.png",
        source="Figure: repository script re-run on a channel subset · 0_data_analysis/core_data_analysis/temporal_rul_analysis.py · claims C19-C27",
        narration=(
            "The second job is to find which of the twenty-two remaining "
            "channels track degradation. Four views, one per panel.\n\n"
            "The left panel pools every row — the weakest evidence, because it "
            "mixes UAVs of different ages. The second controls for flight cycle. "
            "The third computes the correlation inside each UAV separately, and "
            "the fourth gives the share of UAVs trending the same way.\n\n"
            "The declared threshold is a within-UAV correlation of at least zero "
            "point three, with the same direction in at least seventy percent of "
            "UAVs. Ten channels pass. Seven, thirteen, nineteen, twenty-one and "
            "twenty-two increase in one hundred percent of UAVs; sixteen, "
            "twenty-five and twenty-eight decrease in one hundred percent.\n\n"
            "Now telemetry eighteen. Visible age-controlled association in the "
            "second panel; essentially nothing in the third, at minus zero point "
            "zero one nine. A pooled analysis would have promoted it as a "
            "degradation signal. Inside a single UAV it does not move with "
            "remaining life at all.\n\n"
            "A correlation is still not predictive value. These ten are "
            "candidates; whether they help is settled later by grouped "
            "validation."
        ),
    ),
    dict(
        n=5, minutes=1.25, layout="split_figure",
        title="Variance decomposition assigns each channel its role",
        subtitle="Dark: variation inside one UAV over time. Light: persistent differences between UAVs.",
        figure="within_between_variance.png",
        figure_width=5.42,
        panels=[
            ("Within-UAV dominated → sequence features",
             ["Seventeen channels, including all ten degradation candidates "
              "(74-87% within).",
              "Get slope, rolling statistics, recent change and deviation from "
              "the UAV's own early-life baseline."]),
            ("Between-UAV dominated → context features",
             ["telemetry_18 (95.7% between), 26 (85.3%), 01 (79.8%), 06 (50.2%).",
              "Kept as level and baseline descriptions only. Their relationship "
              "must be shown to generalise to unseen UAVs."]),
            ("What the plot does not settle",
             ["Within-UAV variation may be degradation, operating state or "
              "noise; between-UAV variation may be useful context or UAV "
              "identity. Only grouped validation separates them."]),
        ],
        source="Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_within_between_variance.py · claims C28-C34",
        narration=(
            "The previous slide asked whether a channel moves with age. This one "
            "asks where its variation lives, and the answer decides what kind of "
            "feature it should become.\n\n"
            "Each bar splits a channel's total variance into the part that "
            "happens inside one UAV over time, dark blue, and the part that is a "
            "persistent difference between UAVs, light blue.\n\n"
            "Seventeen channels are within-UAV dominated, including all ten "
            "degradation candidates, at seventy-four to eighty-seven percent. "
            "Those get temporal features: slope, rolling statistics, recent "
            "change, deviation from the UAV's own baseline.\n\n"
            "Four are between-UAV dominated. Telemetry eighteen is the extreme "
            "case: ninety-six percent of its variance is simply that different "
            "UAVs sit at different levels. With the previous slide that is a "
            "coherent story — eighteen describes the UAV, not its degradation — "
            "so it is kept only as a baseline and level description.\n\n"
            "The six removed channels appear as constant, with no bar at all: "
            "the same conclusion by a different route."
        ),
    ),
    dict(
        n=6, minutes=1.00, layout="figure",
        title="Train and test differ in age, not in distribution",
        subtitle="Left two panels: raw shift. Right two: the same comparison at matched flight cycle.",
        figure="train_test_drift.png",
        source="Figure: repository script re-run at slide size · 0_data_analysis/core_data_analysis/train_test_drift.py · claims C35-C41",
        narration=(
            "Before trusting any of it, one question: are the test UAVs the same "
            "kind of object as the training UAVs?\n\n"
            "The two left panels compare raw distributions and look alarming. "
            "The two right panels compare each test endpoint only against "
            "training UAVs observed at the same flight cycle. Almost all of the "
            "apparent shift disappears: the median age-matched shift is below "
            "zero point two seven training interquartile ranges for every "
            "channel.\n\n"
            "So test UAVs are younger, not different. That is a statement about "
            "validation design, and it is the bridge into Phase 1.\n\n"
            "Two channels stay on a watch list: telemetry five, with sixteen "
            "percent of its endpoints outside the same-age training range, and "
            "twenty-two with nine. Neither was removed on that evidence."
        ),
    ),
    dict(
        n=7, minutes=1.25, layout="split_figure",
        title="The screening matrix that Phase 1 inherited",
        subtitle="A blue cell means the channel meets that documented threshold. Roles are not exclusive.",
        figure="channel_classification.png",
        figure_width=5.42,
        panels=[
            ("What Phase 0 handed over",
             ["6 removal · 10 degradation · 4 context · 1 state/regime",
              "10 flagged for anomaly review, 14 for redundancy, 15 for a "
              "train/test drift warning."]),
            ("Warnings, not removal rules",
             ["Four redundancy groups: 19/21, 15/23, 13/16/22/25/28 and "
              "06/07/11/12/24. telemetry_19 and 21 correlate at r ≈ 1.00 "
              "between UAVs.",
              "No channel was dropped for redundancy, anomalies or drift here. "
              "Those are questions for grouped validation."]),
            ("telemetry_07 meets five criteria at once",
             ["Degradation, state, anomaly, redundancy and drift. It is treated "
              "as an operating state with transition and dwell features, not as "
              "a smooth sensor."]),
        ],
        source="Figure: repository script re-run at slide size · 0_data_analysis/core_data_analysis/channel_classification.py · claims C42-C50",
        narration=(
            "Everything Phase 0 found, in one matrix. Each row is a channel, "
            "each column a documented threshold, each blue cell a channel that "
            "meets it. The roles are deliberately not exclusive.\n\n"
            "Six removal candidates, ten degradation candidates, four context "
            "channels, one state channel. On top of that, ten channels flagged "
            "for anomaly review, fourteen for redundancy, fifteen for drift.\n\n"
            "The important point is what we did not do with the last three "
            "columns. Nothing was removed for being correlated, for containing "
            "extreme readings, or for drifting. Telemetry nineteen and "
            "twenty-one correlate almost perfectly at UAV level and both were "
            "kept. A screening matrix summarises statistical evidence; it does "
            "not measure predictive value. Every warning became an experiment "
            "later, and several were answered with no.\n\n"
            "Telemetry seven is the interesting row: it meets five criteria at "
            "once, so it is represented as a discrete operating state rather "
            "than a continuous sensor."
        ),
    ),

    # ------------------------------------------ Phase 1: dataset construction
    dict(
        n=8, minutes=1.25, layout="phase1_design",
        title="Phase 1 fixed what every later number is measured against",
        subtitle="Whole-UAV nested cross-validation, and ten automated assertions that must pass",
        figure=None,
        source="Sources: 1_dataset_construction/2_UAV_grouped_validation_folds and 10_automated_leakage_checks/artifacts/verification_report.json · claims C51-C59",
        narration=(
            "Phase 1 turns a hundred complete run-to-failure histories into "
            "something we can honestly measure a model on.\n\n"
            "The split is by UAV, never by row. A hundred training UAVs become "
            "five outer folds of twenty, balanced by terminal lifetime. Inside "
            "each outer round the eighty training UAVs split again into four "
            "inner folds. Features, hyperparameters, early stopping and blend "
            "weights are all selected on the inner folds; the twenty outer UAVs "
            "only ever evaluate the selected procedure. Twenty inner rounds in "
            "total, and every UAV held out exactly once.\n\n"
            "Why nest rather than take one eighty-twenty split? With a hundred "
            "UAVs and large UAV-to-UAV differences, one split is not a "
            "measurement.\n\n"
            "None of this is trusted on faith. Ten assertions are re-checked "
            "automatically. The one to point at is prefix causality: the checker "
            "recomputes features for ten prefixes after replacing every "
            "post-cutoff telemetry value with extreme artificial numbers, and "
            "requires the features to come out unchanged. The status is passed."
        ),
    ),
    dict(
        n=9, minutes=1.25, layout="figure_notes",
        title="Cutoffs are drawn from the observed test history lengths",
        subtitle="Test UAVs are younger than training UAVs, so training samples must be truncated the same way",
        figure="history_length_distributions.png",
        figure_height=2.44,
        panels=[
            ("What the distributions say",
             ["Train 145-525 cycles, median 220.5.",
              "Test 38-475, median 148.0.",
              "About 47% of test UAVs end before cycle 145, the shortest "
              "complete training lifetime."]),
            ("How a training sample is made",
             ["20 distinct cutoffs per UAV, sampled without replacement from "
              "the empirical test lengths, eligible ones only.",
              "Each prefix uses cycles 1..cutoff and carries weight 1/20, so "
              "every UAV has total training weight 1."]),
            ("What that produces",
             ["2,000 training rows, 5 development scenarios and 20 locked "
              "scenarios of 100 UAVs each.",
              "Every scenario reproduces the exact multiset of test history "
              "lengths; the check is one of the ten assertions."]),
        ],
        source="Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_history_length_distributions.py · claims C60-C68",
        narration=(
            "This is the single most consequential slide in Phase 1.\n\n"
            "Training histories run from a hundred and forty-five to five "
            "hundred and twenty-five cycles, median two hundred and twenty. Test "
            "histories run from thirty-eight to four hundred and seventy-five, "
            "median a hundred and forty-eight. Read the empirical CDF at a "
            "hundred and forty-five: about forty-seven percent of test UAVs stop "
            "before the shortest complete training lifetime begins.\n\n"
            "So a model trained on complete run-to-failure sequences is asked "
            "to do something it has never seen. The fix is to truncate: every "
            "training UAV gets twenty distinct cutoffs, drawn from the actual "
            "test history lengths and shorter than its own life.\n\n"
            "Each of a UAV's twenty prefixes carries weight one twentieth, so a "
            "five-hundred-cycle UAV and a hundred-and-fifty-cycle one influence "
            "training equally; otherwise long-lived UAVs would dominate simply "
            "by having more rows.\n\n"
            "One honest weakness: the cutoffs come from the test length "
            "distribution, which is information from the test inputs. No label "
            "is used, and the audit checks that, but it is worth naming."
        ),
    ),
    dict(
        n=10, minutes=1.25, layout="features",
        title="606 prefix features, and four sets to compare them with",
        subtitle="Every feature is computed from cycles at or before the cutoff, and scaled inside each fold",
        figure=None,
        source="Sources: 1_dataset_construction steps 5-7 · feature_catalog.csv and preprocessing_config.json · claims C69-C77",
        narration=(
            "A prefix is a variable-length sequence, and the models we intend to "
            "use need a fixed-width row. So each prefix becomes twenty-seven "
            "summaries per channel.\n\n"
            "Five groups: where the channel is now, where it started, how far "
            "it has moved from its own baseline, how the whole history behaves, "
            "and what happened over the last five, twenty and fifty cycles. "
            "Comparing those windows lets a tree see acceleration, not only "
            "level.\n\n"
            "Twenty-two channels times twenty-seven, plus ten state features, "
            "plus flight cycle and its logarithm, is six hundred and six.\n\n"
            "Rather than pick one set, four nested sets were declared in "
            "advance, each answering a different question — is age alone "
            "predictive, does the current snapshot add anything, do the Phase 0 "
            "roles help, did the screening discard something useful.\n\n"
            "Scaling is median and interquartile range, refitted for every fold "
            "and feature set on the training UAVs alone. Twelve of the four "
            "thousand seven hundred and ten fitted scales fell back to unit "
            "scale, and they are recorded rather than silently dropped."
        ),
    ),
    dict(
        n=11, minutes=1.25, layout="figure_notes",
        title="The benchmark: age alone predicts nothing",
        subtitle="A weighted linear fit on the cutoff cycle, evaluated on the same 20 locked scenarios",
        figure="cycle_only_baseline.png",
        figure_height=2.44,
        panels=[
            ("Locked-scenario result",
             ["R² −0.005 · RMSE 64.554 · MAE 51.912 cycles",
              "Bias +17.712 cycles: age alone systematically overpredicts.",
              "95% UAV-bootstrap interval for R²: [−0.153, +0.129]."]),
            ("Why it is worth the compute",
             ["It is the only number that makes a later RMSE interpretable.",
              "A complex model failing to beat it points at leakage control, "
              "weak features or an implementation fault, not at the data."]),
            ("How the interval is built",
             ["1,000 resamples of whole UAVs, never of rows, because each UAV "
              "appears in 20 scenarios and those rows are not independent.",
              "Seed 20260814, as for the folds and cutoffs."]),
        ],
        source="Figure: generated from 1_dataset_construction/9_cycle_only_baseline artifacts · claims C78-C86",
        narration=(
            "The last thing Phase 1 builds is the simplest model that could "
            "work: predicted remaining life is a hundred and eighty-seven point "
            "three minus zero point five four two times the cutoff cycle, "
            "floored at zero. No telemetry at all.\n\n"
            "On the left is why it fails. At any cutoff the true remaining life "
            "spans hundreds of cycles, because UAVs do not all live the same "
            "length. Age tells you almost nothing about which one you hold.\n\n"
            "R squared of minus zero point zero zero five — slightly worse than "
            "the mean predictor — RMSE sixty-four and a half cycles, bias plus "
            "seventeen point seven, bootstrap interval minus zero point one "
            "five to plus zero point one three.\n\n"
            "This is the reference for everything after it. An RMSE of eleven "
            "cycles means nothing alone; eleven against sixty-four and a half on "
            "the same locked scenarios means the engineered features carry real "
            "information. The interval resamples whole UAVs, because each UAV "
            "appears in twenty scenarios and resampling rows would be falsely "
            "narrow."
        ),
    ),

    # ------------------------------------------- Phase 2 and 3: the chain ---
    dict(
        n=12, minutes=1.25, layout="figure",
        title="Decision 1: bounded scenarios and a fitting cap at 125",
        subtitle="PE_2 paired RMSE improvement, mean-fold, blue ExtraTrees and orange XGBoost. Public: +0.30377",
        figure="pe2_target_scenario_2x2_rmse_panel.png",
        source="Figure: lower panel of the generated PE_2 figure · experiments/PE_2/runs/run_1 · claims C87-C94",
        narration=(
            "With the protocol fixed, the first experiment crossed two choices: "
            "which validation scenarios to score on, and whether to cap the "
            "fitting target.\n\n"
            "Capping means fitting against remaining life truncated at a "
            "hundred and twenty-five cycles, while evaluation still uses the raw "
            "label. A UAV with four hundred cycles left and one with two hundred "
            "look identical, so forcing the model to separate them wastes "
            "capacity on a region nobody scores. Bounding the scenarios means "
            "concentrating validation cutoffs on the bands the test set actually "
            "contains — the Phase 1 history-length finding, applied.\n\n"
            "Both help, and they help together. The four cells are not on a "
            "common R squared scale — changing the evaluation target changes the "
            "denominator — so this is paired RMSE improvement within each "
            "fold.\n\n"
            "Run four carried both changes and moved the public score from zero "
            "point five four to zero point eight four five. Three tenths of R "
            "squared from deciding what to fit and what to measure. No new "
            "model, no new feature."
        ),
    ),
    dict(
        n=13, minutes=1.25, layout="figure",
        title="Decision 2: pruned features and a calibrated tree blend",
        subtitle="Locked architecture comparison, architecture study run 5, five held-out UAV folds. Public: +0.02012",
        figure="r2_comparison.png",
        source="Figure: repository figure, unchanged · 2_model_architecture_study/runs/run_5 · claims C95-C103",
        narration=(
            "Two things changed between run four and run five, and both trace "
            "back to Phase 0.\n\n"
            "The feature set became the drift-pruned screened set: two hundred "
            "and ninety-eight features, taking the Phase 0 screened definition "
            "and removing features whose train-test behaviour did not hold up. "
            "That is the drift-warning column being acted on at last — by "
            "experiment, not assumption.\n\n"
            "The model family was chosen here, under one locked protocol, with "
            "every architecture given the same folds, the same cutoffs and the "
            "same features. The two tree families come out ahead and, more "
            "usefully, they make different mistakes, so a fifty-fifty blend "
            "beats either alone. The blend weight is selected on training-side "
            "data only.\n\n"
            "Public score: zero point eight six five two five. Two hundredths. "
            "Notice the ratio — the model family, what people usually call the "
            "modelling decision, was worth about a fifteenth of the target and "
            "scenario decision."
        ),
    ),
    dict(
        n=14, minutes=0.75, layout="figure",
        title="Decision 3: conditional conservative calibration",
        subtitle="PE_4: a quantile shift applied only where the model is likely to overpredict. Public: +0.00216",
        figure="calibration_tradeoff.png",
        source="Figure: repository figure, unchanged · experiments/PE_4/runs/run_1 · claims C104-C108",
        narration=(
            "Run seven's predecessor overpredicts remaining life more often than "
            "it underpredicts, and in maintenance an overprediction is the "
            "expensive direction.\n\n"
            "PE_4 swept a conservative quantile shift and measured the cost. "
            "Shifting everything hurts accuracy. Applying the shift only "
            "conditionally, at quantile zero point five five, and only where the "
            "model's own signals say overprediction is likely, keeps almost all "
            "of the accuracy and removes part of the bias.\n\n"
            "It is worth two thousandths of public R squared. I show it because "
            "it is the one decision in the chain taken partly for a reason other "
            "than the score."
        ),
    ),
    dict(
        n=15, minutes=1.25, layout="ensemble",
        title="Decision 4: a cross-fitted residual correction",
        subtitle="PE_11: a small model that predicts the ensemble's own error from its disagreement. Public: +0.00911",
        figure="bagging_residual_comparison.png",
        source="Figure: repository figure, unchanged · experiments/PE_11/runs/run_1 · claims C109-C116",
        narration=(
            "The last retained decision. Six tree members — three XGBoost and "
            "three ExtraTrees at seeds thirteen, thirty-seven and seventy-three "
            "— are averaged within family and then blended.\n\n"
            "A small histogram gradient model, seven leaves and a hundred "
            "iterations, then predicts that base prediction's residual. Its "
            "inputs are the base prediction, the observed history length, how "
            "much the six members disagree, and a few baseline deltas and "
            "slopes. Member disagreement is a usable signal about where the "
            "ensemble is unreliable.\n\n"
            "The leakage question matters here, so: the out-of-fold predictions "
            "that fit the residual model come from models which excluded that "
            "UAV, and calibration endpoints are restricted to training UAVs. "
            "True future remaining life is never an input.\n\n"
            "Development effect: eleven point five one down to eleven point "
            "zero two cycles, four point two percent better, winning in all five "
            "folds. Public score zero point eight seven six five two."
        ),
    ),
    dict(
        n=16, minutes=1.25, layout="screen",
        title="Why the chain stops at 0.87652",
        subtitle="Run 7 development out-of-fold predictions, and every candidate evaluated since",
        figure="development_prediction_scatter.png",
        source="Figures and tables: run_7/7_post_run_reporting and experiments/PE_15-PE_28 · claims C117-C127",
        narration=(
            "Where the remaining error is, and why we stopped.\n\n"
            "On the left are the out-of-fold development predictions. The "
            "structure is systematic, not random: the model compresses towards "
            "the middle, so long remaining lives are underpredicted and short "
            "ones overpredicted. That is what the fitting cap buys us, and it is "
            "also the ceiling it imposes.\n\n"
            "On the right, six candidates tried after run seven, each with a "
            "promotion gate declared beforehand. A full nested refit of the most "
            "promising screen came back one point six percent worse. Extra "
            "seeds, a regime gate and a short-history specialist all failed. One "
            "TabPFN blend is promising and unconfirmed.\n\n"
            "So nothing was promoted, nothing was submitted, and the public "
            "score is still zero point eight seven six five two. I would rather "
            "report that than a number obtained by submitting until something "
            "stuck."
        ),
    ),
    dict(
        n=17, minutes=0.75, layout="conclusions",
        title="How we reached 0.87652",
        subtitle="Ordered by the submission that carried each decision",
        figure=None,
        source="Sources: kaggle_scores.csv, pipeline_experiments.md, phase_0 and phase_1 artifacts · claims C128-C132",
        narration=(
            "The answer to the question in the title.\n\n"
            "Three tenths of R squared came from deciding what to fit and what "
            "to measure it on — and both halves of that decision were read off "
            "Phase 0 and Phase 1 evidence: the history-length distributions and "
            "the age-matched drift comparison. Two hundredths came from the "
            "feature set and the model family. Roughly one hundredth came from "
            "calibration and the residual correction together.\n\n"
            "If there is one thing to take away, it is that the audit and the "
            "dataset construction were not preliminaries to the modelling. They "
            "were the modelling."
        ),
    ),
]


BACKUP = [
    dict(
        n="B1",
        title="Backup: the complete Phase 0 broad review",
        subtitle="Nine analyses, and the decision each one contributed",
        source="Source: literature_and_planning/development_documentation/phase_0_dataAnalysis.md",
    ),
    dict(
        n="B2",
        title="Backup: redundancy, anomalies, and what was not removed",
        subtitle="The strongest correlated pairs and the anomaly rates; every warning became an experiment",
        source="Source: 0_data_analysis/core_data_analysis/figures/feature_redundancy and figures/anomalies",
    ),
    dict(
        n="B3",
        title="Backup: the 27 features derived from every channel",
        subtitle="All computed from cycles at or before the cutoff; verified by the prefix-causality assertion",
        source="Source: 1_dataset_construction/5_prefix_feature_engineering and 6_feature_sets/artifacts/feature_catalog.csv",
    ),
    dict(
        n="B4",
        title="Backup: the cycle-only baseline, group by group",
        subtitle="The same held-out predictions, reported by cutoff band, outer fold and terminal-lifetime quintile",
        source="Source: 1_dataset_construction/9_cycle_only_baseline/artifacts/metrics/",
    ),
    dict(
        n="B5",
        title="Backup: full locked architecture comparison",
        subtitle="Architecture study run 5, five held-out UAV folds, identical folds and cutoffs for every entry",
        note="Sequence models were given the same protocol, the same prefixes and a tuned budget.",
        source="Source: 2_model_architecture_study/runs/run_5/7_architecture_comparison/architecture_comparison.csv",
    ),
    dict(
        n="B6",
        title="Backup: sequence and hybrid models, in full",
        subtitle="Architecture study run 8: hybrid combinations of a tree ensemble with a sequence encoder",
        figure="hybrid_architecture_comparison.png",
        source="Source: 2_model_architecture_study/runs/run_8",
    ),
    dict(
        n="B7",
        title="Backup: the complete cap and scenario matrix",
        subtitle="PE_2, all four cells, with the target policy of each cell stated explicitly",
        note="Cells with different evaluation targets are not on a common R² scale. "
             "Fitting cap, evaluation support and prediction clipping are three different things.",
        source="Source: experiments/PE_2/runs/run_1/reporting",
    ),
    dict(
        n="B8",
        title="Backup: residual correction, calibration and the safety table",
        subtitle="PE_11 and PE_4 in detail, with the nested-calibration guarantee stated in full",
        source="Source: experiments/PE_11 and PE_4, runs/run_1/reporting",
    ),
    dict(
        n="B9",
        title="Backup: representation experiments that were rejected",
        subtitle="Every row is a matched development comparison whose control was retained",
        source="Source: experiments/PE_5, PE_6, PE_7, PE_9, PE_13, PE_22, PE_28",
    ),
    dict(
        n="B10",
        title="Backup: uncertainty, endpoint duplication and score provenance",
        subtitle="How the intervals are built, what the repeated endpoints do, and where each score comes from",
        source="Source: r2_research_2026_09_07/ and kaggle_scores.csv",
    ),
]
