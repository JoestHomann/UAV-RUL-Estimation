# Storyboard — How we reached a public R² of 0.87652

Deck: `UAV_RUL_Project_20min.pptx` · Template: University of Stuttgart 16:9 (10 × 5.625 in) · 17 main slides + divider + 10 backup slides
Evidence cut-off for every number in the deck: **9 September 2026**.

The deck answers one question: **how did this project reach a recorded Kaggle public R² of 0.87652, and which decision at each step produced it?**

After the 9 September review the front half of the talk is Phase 0 and Phase 1. That is not padding: the single largest public-score movement in the project, +0.30377, came from a decision about the evaluation target and the validation cutoffs, and both halves of that decision were read directly off Phase 0 and Phase 1 evidence. Slides 3–11 are the audit and the dataset construction; slides 12–16 are the modelling chain they made possible.

Claim IDs (`C##`) link every number to `evidence_ledger.csv` (192 claims).

---

## The chain the deck follows

| Submission | Decision it introduced | Development experiment | Recorded public R² | Change |
| --- | --- | --- | ---: | ---: |
| Run 3 | XGBoost on raw RUL, current scenarios | Phase 2 run 3 | 0.54136 | — |
| Run 4 | Bounded validation scenarios + fitting cap 125 | PE_2 target/scenario 2×2 | 0.84513 | +0.30377 |
| Run 5 | Drift-pruned 298 features + calibrated 50/50 tree blend | PE_3 | 0.86525 | +0.02012 |
| Run 6 | Conditional conservative calibration, q = 0.55 | PE_4, PE_5 | 0.86741 | +0.00216 |
| Run 7 | Cross-fitted residual correction on six seeded members | PE_11 | 0.87652 | +0.00911 |

Each row is a submission-to-submission difference, not an isolated ablation: a run can carry more than one change. The deck says so on slides 2 and 17.

---

## Time budget

| Slide | Title | Layout | Minutes | Cumulative |
| ---: | --- | --- | ---: | ---: |
| 1 | How we reached a public R² of 0.87652 | `title` | 0.75 | 0.75 |
| 2 | Five submissions, five decisions | `figure` | 1.00 | 1.75 |
| 3 | Six of twenty-eight channels carry no information | `figure` | 0.75 | 2.50 |
| 4 | Ten channels move with age inside every UAV | `figure` | 1.50 | 4.00 |
| 5 | Variance decomposition assigns each channel its role | `split_figure` | 1.25 | 5.25 |
| 6 | Train and test differ in age, not in distribution | `figure` | 1.00 | 6.25 |
| 7 | The screening matrix that Phase 1 inherited | `split_figure` | 1.25 | 7.50 |
| 8 | Phase 1 fixed what every later number is measured against | `phase1_design` | 1.25 | 8.75 |
| 9 | Cutoffs are drawn from the observed test history lengths | `figure_notes` | 1.25 | 10.00 |
| 10 | 606 prefix features, and four sets to compare them with | `features` | 1.25 | 11.25 |
| 11 | The benchmark: age alone predicts nothing | `figure_notes` | 1.25 | 12.50 |
| 12 | Decision 1: bounded scenarios and a fitting cap at 125 | `figure` | 1.25 | 13.75 |
| 13 | Decision 2: pruned features and a calibrated tree blend | `figure` | 1.25 | 15.00 |
| 14 | Decision 3: conditional conservative calibration | `figure` | 0.75 | 15.75 |
| 15 | Decision 4: a cross-fitted residual correction | `ensemble` | 1.25 | 17.00 |
| 16 | Why the chain stops at 0.87652 | `screen` | 1.25 | 18.25 |
| 17 | How we reached 0.87652 | `conclusions` | 0.75 | 19.00 |
| | **Total** | | **19.00** | |

Sections: slides 1–2 orientation (1.75 min) · **Phase 0, slides 3–7 (5.75 min)** · **Phase 1, slides 8–11 (5.00 min)** · Phase 2/3 decision chain, slides 12–16 (5.75 min) · close, slide 17 (0.75 min).

---

## Figure policy

Charts come from the repository's own plotting code wherever that code exists. `sources/figure_manifest.csv` records provenance, source path and placement for every figure.

| Provenance | Count | Meaning |
| --- | ---: | --- |
| `repository figure, unchanged` | 13 | Byte-identical copy |
| `repository script re-run at slide size` | 6 | Unmodified script, presentation canvas |
| `repository figure, cropped …` | 2 | Same image, cropped to one panel |
| `generated for the talk` | 4 | No repository figure or script covers it |

### Why the Phase 0 figures are re-run rather than copied

The Phase 0 figures are drawn on canvases between 11 × 9 in and 19 × 10 in, sized for a monitor. Scaled onto this template's 10 × 5.625 in canvas their axis labels arrive at three to four points, which is not readable from a lecture room.

`sources/run_repo_figure.py` imports the repository script **unmodified** and overrides exactly four presentation quantities before calling its `main()`: the figure canvas size; the tick-label size used by the shared `style_axis` helper, which hard-codes `labelsize=8`; the matplotlib base font size; and the figure-title size, which a few scripts hard-code at 14 pt. No repository file is copied with edits. Every statistic, colour, ordering and label still comes from the repository code.

`build_figures.py` then re-reads the CSV each re-run wrote and compares it with the repository's own CSV for the same analysis, reporting the largest numeric difference. On the build shipped with this deck all six comparisons return zero, except the temporal/RUL summary at 8.4 × 10⁻¹⁵ (float ordering; that run is restricted to twelve channels, so only the shared rows are compared). The check line is stored in the manifest's `note` column.

### Cropping

Two figures are cropped rather than re-run, because the PE_2 experiment's plotting code was not available here: its two paired-comparison figures stack a paired-R² panel above a paired-RMSE panel on an 11.9 × 7.9 in canvas. The deck shows the lower panel, which carries the category labels. The legend lives in the upper panel, so the colour key (blue = ExtraTrees, orange = XGBoost) is stated in the slide subtitle instead.

### Figures generated here

| Figure | Why no repository figure exists |
| --- | --- |
| `prediction_timeline.png` | No repository figure shows the prediction cutoff itself |
| `public_score_chain.png` | Kaggle scores are stored as a CSV with no plotting script |
| `cycle_only_baseline.png` | Phase 1 records the baseline's coefficients and metrics but stores no figure |
| `development_versus_public.png` | The two scopes are never joined in any repository figure |

All four use the repository's Phase 0 palette (`#0072B2`, `#56B4E9`, `#D55E00`, `#7A7A7A`, `#D9D9D9`) and a local copy of `plotting_common.style_axis`.

### Figures collected but not placed

- `correlation_heatmaps.png` — Phase 0 core review: row- and UAV-level redundancy heatmaps
- `anomaly_summary.png` — Phase 0 core review: extreme readings, jumps and persistent shifts
- `grouped_permutation_importance.png` — FE_run_1 model-guided grouped permutation importance
- `representative_trajectories.png` — Phase 0 core review: six representative UAV trajectories
- `temporal_architecture_comparison.png` — Dedicated temporal study, architecture study run 7
- `pe20_comparison.png` — PE_20 complete nested refit of the forecast-history recipe
- `development_overprediction_diagnostics.png` — Phase 3 Run 7 development overprediction diagnostics
- `pe2_signal_family_ablation_rmse_panel.png` — Lower panel (paired RMSE improvement) of the generated PE_2 figure

---

## Slide-by-slide plan

### 1. How we reached a public R² of 0.87652

*Estimating remaining useful life from a partial UAV telemetry history*

- **Layout** `title` · **0.75 min** · claims **C01–C04** (4)
- **Figure** `prediction_timeline.png` — generated for the talk
- **Source label** Figure: generated for this talk from data/train.csv · claims C01-C04

### 2. Five submissions, five decisions

*Each point is a submitted model; the label names the decision that run introduced*

- **Layout** `figure` · **1.00 min** · claims **C05–C12** (8)
- **Figure** `public_score_chain.png` — generated for the talk
- **Source label** Figure: generated from kaggle_scores.csv and pipeline_experiments.md · claims C05-C12

### 3. Six of twenty-eight channels carry no information

*Red: unique-value count and numeric range at the effectively-constant threshold*

- **Layout** `figure` · **0.75 min** · claims **C13–C18** (6)
- **Figure** `constant_features.png` — repository script re-run at slide size
- **Source label** Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_constant_features.py · claims C13-C18

### 4. Ten channels move with age inside every UAV

*Twelve of the twenty-two channels: the ten degradation candidates, plus telemetry_18 and 26*

- **Layout** `figure` · **1.50 min** · claims **C19–C27** (9)
- **Figure** `temporal_rul_summary_subset.png` — repository script re-run at slide size
- **Source label** Figure: repository script re-run on a channel subset · 0_data_analysis/core_data_analysis/temporal_rul_analysis.py · claims C19-C27

### 5. Variance decomposition assigns each channel its role

*Dark: variation inside one UAV over time. Light: persistent differences between UAVs.*

- **Layout** `split_figure` · **1.25 min** · claims **C28–C34** (7)
- **Figure** `within_between_variance.png` — repository script re-run at slide size
- **Source label** Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_within_between_variance.py · claims C28-C34

### 6. Train and test differ in age, not in distribution

*Left two panels: raw shift. Right two: the same comparison at matched flight cycle.*

- **Layout** `figure` · **1.00 min** · claims **C35–C41** (7)
- **Figure** `train_test_drift.png` — repository script re-run at slide size
- **Source label** Figure: repository script re-run at slide size · 0_data_analysis/core_data_analysis/train_test_drift.py · claims C35-C41

### 7. The screening matrix that Phase 1 inherited

*A blue cell means the channel meets that documented threshold. Roles are not exclusive.*

- **Layout** `split_figure` · **1.25 min** · claims **C42–C50** (9)
- **Figure** `channel_classification.png` — repository script re-run at slide size
- **Source label** Figure: repository script re-run at slide size · 0_data_analysis/core_data_analysis/channel_classification.py · claims C42-C50

### 8. Phase 1 fixed what every later number is measured against

*Whole-UAV nested cross-validation, and ten automated assertions that must pass*

- **Layout** `phase1_design` · **1.25 min** · claims **C51–C59** (9)
- **Figure** none; native PowerPoint drawing and tables
- **Source label** Sources: 1_dataset_construction/2_UAV_grouped_validation_folds and 10_automated_leakage_checks/artifacts/verification_report.json · claims C51-C59

### 9. Cutoffs are drawn from the observed test history lengths

*Test UAVs are younger than training UAVs, so training samples must be truncated the same way*

- **Layout** `figure_notes` · **1.25 min** · claims **C60–C68** (9)
- **Figure** `history_length_distributions.png` — repository script re-run at slide size
- **Source label** Figure: repository script re-run at slide size · 0_data_analysis/broad_data_review/plot_history_length_distributions.py · claims C60-C68

### 10. 606 prefix features, and four sets to compare them with

*Every feature is computed from cycles at or before the cutoff, and scaled inside each fold*

- **Layout** `features` · **1.25 min** · claims **C69–C77** (9)
- **Figure** none; native PowerPoint drawing and tables
- **Source label** Sources: 1_dataset_construction steps 5-7 · feature_catalog.csv and preprocessing_config.json · claims C69-C77

### 11. The benchmark: age alone predicts nothing

*A weighted linear fit on the cutoff cycle, evaluated on the same 20 locked scenarios*

- **Layout** `figure_notes` · **1.25 min** · claims **C78–C86** (9)
- **Figure** `cycle_only_baseline.png` — generated for the talk
- **Source label** Figure: generated from 1_dataset_construction/9_cycle_only_baseline artifacts · claims C78-C86

### 12. Decision 1: bounded scenarios and a fitting cap at 125

*PE_2 paired RMSE improvement, mean-fold, blue ExtraTrees and orange XGBoost. Public: +0.30377*

- **Layout** `figure` · **1.25 min** · claims **C87–C94** (8)
- **Figure** `pe2_target_scenario_2x2_rmse_panel.png` — repository figure, cropped from y=0.462H
- **Source label** Figure: lower panel of the generated PE_2 figure · experiments/PE_2/runs/run_1 · claims C87-C94

### 13. Decision 2: pruned features and a calibrated tree blend

*Locked architecture comparison, architecture study run 5, five held-out UAV folds. Public: +0.02012*

- **Layout** `figure` · **1.25 min** · claims **C95–C103** (9)
- **Figure** `r2_comparison.png` — repository figure, unchanged
- **Source label** Figure: repository figure, unchanged · 2_model_architecture_study/runs/run_5 · claims C95-C103

### 14. Decision 3: conditional conservative calibration

*PE_4: a quantile shift applied only where the model is likely to overpredict. Public: +0.00216*

- **Layout** `figure` · **0.75 min** · claims **C104–C108** (5)
- **Figure** `calibration_tradeoff.png` — repository figure, unchanged
- **Source label** Figure: repository figure, unchanged · experiments/PE_4/runs/run_1 · claims C104-C108

### 15. Decision 4: a cross-fitted residual correction

*PE_11: a small model that predicts the ensemble's own error from its disagreement. Public: +0.00911*

- **Layout** `ensemble` · **1.25 min** · claims **C109–C116** (8)
- **Figure** `bagging_residual_comparison.png` — repository figure, unchanged
- **Source label** Figure: repository figure, unchanged · experiments/PE_11/runs/run_1 · claims C109-C116

### 16. Why the chain stops at 0.87652

*Run 7 development out-of-fold predictions, and every candidate evaluated since*

- **Layout** `screen` · **1.25 min** · claims **C117–C127** (11)
- **Figure** `development_prediction_scatter.png` — repository figure, unchanged
- **Source label** Figures and tables: run_7/7_post_run_reporting and experiments/PE_15-PE_28 · claims C117-C127

### 17. How we reached 0.87652

*Ordered by the submission that carried each decision*

- **Layout** `conclusions` · **0.75 min** · claims **C128–C132** (5)
- **Figure** none; native PowerPoint drawing and tables
- **Source label** Sources: kaggle_scores.csv, pipeline_experiments.md, phase_0 and phase_1 artifacts · claims C128-C132

---

## Backup slides

Ten backup slides follow a divider. The brief asked for four to six; the expanded Phase 0 and Phase 1 emphasis added four (B1–B4) on top of the six that already existed. They are held for questions and are not spoken.

- **B1** Backup: the complete Phase 0 broad review · claims C133–C140 (8)
- **B2** Backup: redundancy, anomalies, and what was not removed · claims C141–C148 (8)
- **B3** Backup: the 27 features derived from every channel · claims C149–C153 (5)
- **B4** Backup: the cycle-only baseline, group by group · claims C154–C159 (6)
- **B5** Backup: full locked architecture comparison · claims C160–C163 (4)
- **B6** Backup: sequence and hybrid models, in full · claims C164–C167 (4)
- **B7** Backup: the complete cap and scenario matrix · claims C168–C169 (2)
- **B8** Backup: residual correction, calibration and the safety table · claims C170–C172 (3)
- **B9** Backup: representation experiments that were rejected · claims C173–C186 (14)
- **B10** Backup: uncertainty, endpoint duplication and score provenance · claims C187–C192 (6)

---

## Scope rules held on-slide

- Phase 0 evidence is labelled by the view it comes from: pooled, age-controlled, within-UAV, or between-UAV. Slide 4 exists because those four views disagree about telemetry_18.
- Nothing in Phase 0 is presented as a model result. Screening outcomes are called candidates, and slides 4, 5 and 7 each say that grouped validation decides.
- Cutoff sampling from the test length distribution is named on slide 9 as a design choice that uses test *inputs*, never test labels.
- The 2×2 cells on slide 12 are not on a common R² scale, because changing the evaluation target changes the denominator; the panel shown is paired RMSE improvement within each fold.
- Architecture studies from run 5, run 7 and run 8 are kept on separate axes and never combined.
- Fitting cap 125, evaluation on raw endpoint labels, and the nonnegative prediction clip are stated as three different things.
- Development and public scores never share a curve. Slide 17 states the 0.90041 development figure and the 0.87652 public figure separately and quantifies the gap.
- Fresh split seeds (PE_21, PE_28 confirmation) are described as robustness checks on the same UAV population, not as fresh holdouts.
- The retained model is called "retained", never "deployed".

---

## Template adaptation

- The supplied file is untouched; `build_deck.py` opens a copy.
- Masters, layouts, theme colours, Arial and the university logo are reused as they are. The ten sample slides are removed from `sldIdLst` and their relationships dropped.
- The canvas is 10 × 5.625 in, not 13.33 in. A point on this canvas projects like 1.33 points on a 13.33 in deck, so the master's 18 pt title and 16 pt body already read like 24 pt and 21 pt. Titles are therefore left at the master's 18 pt bold rather than pushed to 28–36 pt, which would not fit the placeholder.
- Titles are kept under about 76 characters, the one-line limit at 18 pt across the 9.02 in content width.
- Date, footer and slide-number placeholders are cloned onto each slide, because python-pptx does not inherit them automatically.
- The title slide keeps the layout's own design: dark title panel on the right, presenter circle, logo. The prediction-timeline figure occupies the free left half in place of the layout's photograph placeholder.
- No stock photography, decorative icons or gradients were added.

---

## Rehearsal estimate

- Main-deck narration: **2,425 words**
- At 125 words per minute: **19.40 min**, against 19.00 min of planned content and a 20-minute slot.
- At 135 words per minute: 17.96 min.
- The brief's band was 2,100–2,350 words; this draft sits about 60 words above it, which leaves little slack for questions inside the slot. The three longest slides to cut first are 9, 10 and 11.

---

## Verification performed

- Every re-run repository script's output CSV was compared with the repository's own CSV for the same analysis; largest numeric difference zero (temporal/RUL: 8.4 × 10⁻¹⁵ on shared rows).
- `build_ledger.py` validates field names, sequential claim IDs, non-decreasing slide order and status values: 192 claims.
- Every number printed as slide text was extracted from the built `.pptx` and matched against the ledger, `chart_data/` and the deck sources. Result: no unmatched numbers.
- Shape geometry was checked against the slide canvas, and every main slide was checked for speaker notes. Result: clean.
- All 28 slides were rendered to PDF with LibreOffice and inspected as images.

## Standing limitations

- The deck has **not** been opened in Microsoft PowerPoint. It was checked by rendering to PDF with LibreOffice 24.2 and inspecting every page. Liberation Sans substitutes metrically for Arial, so line breaks should hold, but that is not the same as opening it in PowerPoint.
- PE_27's screen numbers could not be read from their artifact: the file sits deeper than the file bridge's folder-depth limit. They are not asserted on any slide. The run's `winner_manifest.json` was reachable and records `status: no_promotion` with the control retained.
- The narration is about 60 words above the brief's upper word band.

