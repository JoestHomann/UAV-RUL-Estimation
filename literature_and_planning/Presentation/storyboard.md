# Storyboard — How we reached a public R² of 0.87652

Deck: `UAV_RUL_Project_20min.pptx` · Template: University of Stuttgart 16:9 (10 × 5.625 in)
Evidence cut-off for every number in the deck: **8 September 2026**.

The deck answers one question: **how did this project reach a recorded Kaggle
public R² of 0.87652, and which decision at each step produced it?** Every main
slide names a decision, shows the evidence that settled it, and — where the
decision produced a submission — states what it was worth on the public
leaderboard. Claim IDs (`C##`) link each number to `evidence_ledger.csv`.

---

## The chain the deck follows

| Submission | Decision it introduced | Development experiment | Recorded public R² | Change |
| --- | --- | --- | ---: | ---: |
| Run 3 | XGBoost on raw RUL, current scenarios | Phase 2 run 3 | 0.54136 | — |
| Run 4 | Bounded validation scenarios + fitting cap 125 | PE_2 target/scenario 2×2 | 0.84513 | +0.30377 |
| Run 5 | Drift-pruned 298 features + calibrated 50/50 tree blend | PE_3 | 0.86525 | +0.02012 |
| Run 6 | Conditional conservative calibration, q = 0.55 | PE_4, PE_5 | 0.86741 | +0.00216 |
| Run 7 | Cross-fitted residual correction on six seeded members | PE_11 | 0.87652 | +0.00911 |

Each row is a submission-to-submission difference, not an isolated ablation: a
run can carry more than one change. The deck says so on slides 2 and 14.

---

## Time budget

| Slide | Title | Minutes | Cumulative |
| ---: | --- | ---: | ---: |
| 1 | How we reached a public R² of 0.87652 | 0.75 | 0.75 |
| 2 | Five submissions, five decisions | 1.25 | 2.00 |
| 3 | Phase 0 decided which channels may carry information | 1.25 | 3.25 |
| 4 | Phase 1 decided what every later number is measured against | 2.00 | 5.25 |
| 5 | Decision 1: bounded scenarios and a fitting cap at 125 | 2.00 | 7.25 |
| 6 | Decision 2: which engineered features earn their place | 1.50 | 8.75 |
| 7 | Decision 3: the model family, chosen under one locked protocol | 2.00 | 10.75 |
| 8 | Decision 4: a cross-fitted residual correction | 2.00 | 12.75 |
| 9 | Decision 5: conditional conservative calibration | 1.00 | 13.75 |
| 10 | What Run 7 still gets wrong | 1.25 | 15.00 |
| 11 | Why the chain stops at 0.87652 | 1.75 | 16.75 |
| 12 | Development performance and public performance | 1.00 | 17.75 |
| 13 | What the next comparison can resolve | 0.75 | 18.50 |
| 14 | How we reached 0.87652 | 0.50 | 19.00 |
| — | Transitions and pauses | 1.00 | **20.00** |

Backup slides B1–B6 follow a divider and are not part of the 20 minutes.

---

## Figure policy

The user's instruction was to use the figures the repository has already
generated. Of the 17 figures in `sources/figures/`:

| Provenance | Count | What it means |
| --- | ---: | --- |
| Repository figure, unchanged | 10 | Byte-identical copy of a generated PNG |
| Repository figure, cropped to one panel | 3 | Same image, cropped; nothing re-plotted |
| Repository script re-run on a channel subset | 1 | `temporal_rul_analysis.py`, same code and colours |
| Generated for this talk | 3 | No repository figure exists for that evidence |

`sources/figure_manifest.csv` records each figure's source path, provenance and
the slide it is placed on.

**Why three figures were cropped.** The Phase 0 history-length figure is a
15 × 4.8 in three-panel strip and the two PE_2 paired-comparison figures are
11.9 × 7.9 in two-panel stacks. Scaled to fit this template's 10 × 5.625 in
canvas they render at roughly 4–6 pt. Each was cropped to the single panel the
slide argues from — the empirical CDF, and the paired-RMSE panel. The cropped
PE_2 panels lose the legend that sits in the upper panel, so the colour key
(blue = ExtraTrees, orange = XGBoost) is stated in each slide's subtitle.

**Why one script was re-run.** `temporal_rul_summary.png` is 19 × 10 in because
it plots all 28 channels. It was regenerated with the repository's own script,
same statistics and same colours, restricted with `--channels` to the eight
channels the slide discusses, with the fixed canvas scaled to the channel
count. The command is in `sources/README.md`.

**The three generated figures.** The prediction-cutoff timeline, the public
score chain and the development-versus-public panels have no repository
equivalent. They use the repository's own Phase 0 palette and axis styling from
`0_data_analysis/broad_data_review/plotting_common.py`.

**Native PowerPoint drawings, not charts.** The nested whole-UAV schematic
(slide 4), the retained-model schematic (slide 8) and every table are drawn as
editable PowerPoint shapes, because the repository has no figure for them and
the brief asks for editable diagrams.

Two repository figures were copied but not placed: `pe20_comparison.png` (a
two-bar chart whose difference is invisible at slide size — the PE_20 result is
given numerically on slide 11) and `development_overprediction_diagnostics.png`
(superseded by the prediction scatter on slide 10). Both are in
`sources/figures/` and are marked "not placed" in the manifest.

---

## Slide-by-slide plan

### 1 — How we reached a public R² of 0.87652 (0.75 min)
Sets the task and the target number. Figure: prediction-cutoff timeline built
from `UAV_0019` (complete life 229 cycles, cut at 148). Claims C01–C04.

### 2 — Five submissions, five decisions (1.25 min)
The whole answer on one axis, then the talk walks it. Figure: public score
chain, each point labelled with the decision that run introduced. Carries the
two qualifications: submissions are not isolated ablations, and the deltas
shrink by two orders of magnitude. Claims C05–C12.

### 3 — Phase 0 decided which channels may carry information (1.25 min)
- **Question:** what can 28 anonymous channels be trusted to carry?
- **Evidence:** within-UAV correlation and trend consistency, not pooled
  correlation. Channels 21 and 19 move the same way in 100/100 UAVs; 25 and 16
  the opposite way; 18 has almost no within-UAV relationship and 96% of its
  variance between UAVs; 20 is flat.
- **Decision:** a channel taxonomy — 10 degradation candidates, 4 context
  channels, 1 state channel, 6 removals. Claims C13–C19.

### 4 — Phase 1 decided what every later number is measured against (2.00 min)
- **Question:** what has to be held out for a result to mean "generalises to an
  unseen UAV"?
- **Evidence:** 100 UAVs but 24,720 rows; the empirical CDF showing 47 of 100
  test UAVs stopping before cycle 145.
- **Decision:** whole-UAV outer folds with training-side inner selection,
  test-like cutoffs, prefix-only features, equal UAV weight, UAV-level
  bootstrap. Claims C20–C27.

### 5 — Decision 1: bounded scenarios and a fitting cap at 125 (2.00 min)
- **Comparison:** 2 × 2 — scenario profile × fitting target, everything else
  fixed, paired against the unchanged control, both model families.
- **Evidence:** capping alone loses about 5 cycles; bounded scenarios with a raw
  target gain about 6; both together gain 18–20.
- **Caveat carried on the slide:** the two bounded cells restrict true
  validation RUL to 1–125, which changes the task and the R² denominator.
- **Decision and its public effect:** freeze the joint policy; Run 3 → Run 4,
  +0.30377. Claims C28–C35.

### 6 — Decision 2: which engineered features earn their place (1.50 min)
- **Comparison:** matched signal-family ablation against age + latest values.
- **Evidence:** the inverse pair 15/23 is the strongest single family; all four
  families together are worth about 10 cycles and ΔR² 0.17, 5/5 folds for both
  models; the channel-07 state family failed.
- **Decision:** the Run 5 feature set. Three further representation experiments
  were rejected on the same folds (backup B5). Claims C36–C42.

### 7 — Decision 3: the model family (2.00 min)
- **Comparison:** locked architecture study, 20 locked scenarios × 100 unseen
  UAVs, three seeds, protocol frozen before any result was seen.
- **Evidence:** XGBoost 0.804, Random Forest 0.783, Extra Trees 0.764, then a
  wide gap to trajectory retrieval 0.507, sensor-graph TCN 0.429, multi-scale
  CNN 0.361; TCN unstable at −2.423. The later matched hybrid rematch is
  summarised verbally and shown in backup B2.
- **Decision:** retain engineered-feature trees, stated for the architectures,
  representations and budgets tested here. Claims C43–C53.

### 8 — Decision 4: a cross-fitted residual correction (2.00 min)
- **Comparison:** six-member mean, median, trimmed mean and nonnegative blend
  against a cross-fitted residual correction, identical folds.
- **Evidence:** 11.5080 → 11.0243 cycles, 4.20%, 5/5 fold wins; simple averaging
  landed near 12.0.
- **Mechanism:** editable schematic of the retained contract, including that
  out-of-fold calibration predictions come from models that excluded that UAV.
- **Public effect:** Run 6 → Run 7, +0.00911. Claims C54–C60.

### 9 — Decision 5: conditional conservative calibration (1.00 min)
- **Evidence:** the pre-declared quantile sweep; a monotone trade-off between
  accuracy and RMS overprediction; q = 0.55 selected within 0.005 R² of the
  best.
- **Public effect:** Run 5 → Run 6, +0.00216, the smallest retained step.
  Claims C61–C65.

### 10 — What Run 7 still gets wrong (1.25 min)
Prediction alignment on the saved development out-of-fold set; the error is
concentrated in RUL 51–125 and in short histories; the short-history specialist
gained 0.389% and was rejected. Claims C66–C73.

### 11 — Why the chain stops at 0.87652 (1.75 min)
The forecast-history screen (−13.0%, 5/5) and its collapse under a complete
nested refit (+1.56% worse, 2/5), plus a table of the five other post-Run-7
candidates and their gates. Nothing passed, so nothing was submitted.
Claims C74–C84.

### 12 — Development performance and public performance (1.00 min)
Two panels, deliberately not one curve. Development mean-fold 0.89274 → 0.90041,
pooled 0.90446 as a different aggregation; recorded public 0.84513 → 0.87652;
the remaining 0.02348 gap needs about 10.0% lower RMSE. Claims C85–C93.

### 13 — What the next comparison can resolve (0.75 min)
PE_28's ten feature representations × two model recipes, its declared gates, and
its **pending** status on 8 September 2026. Claims C94–C99.

### 14 — How we reached 0.87652 (0.50 min)
The decision chain as a table with the phase that owned each decision, and what
is not claimed. Claim C100.

---

## Backup slides

| # | Topic | Why it is held back |
| --- | --- | --- |
| B1 | Full locked architecture comparison, all eight families, seed SD and bootstrap intervals | Slide 7 shows the chart; the numbers are question material |
| B2 | Sequence and hybrid models in full: the matched run-8 rematch and the dedicated temporal study | Needed only if someone asks whether neural models got a fair chance |
| B3 | Complete cap/scenario matrix and the target-support explanation | Needed only if someone questions the R² denominator argument on slide 5 |
| B4 | Residual correction, nested UAV calibration and the full PE_4 policy table | Implementation-level detail |
| B5 | Seven rejected representation experiments | Question material, not talk material |
| B6 | Uncertainty, endpoint duplication and public-score provenance | Methodological caveats questions usually reach |

---

## Experiments deliberately left out of the main deck

| Omitted | Reason |
| --- | --- |
| PE_5 target-tail variants | Confirms the cap decision already made on slide 5 |
| PE_6 sequence sampling, PE_7 stacking | Preparatory or blocked; PE_7 never ran |
| PE_8 onset targets, PE_9 drift pruning, PE_17 population features | Rejected formulations; collected in backup B5 |
| PE_10 multi-resolution hybrid | Subsumed by the hybrid result on slide 7 and backup B2 |
| PE_12 test-like weighting | Confirms the slide-8 decision under a reweighting |
| PE_16 residual-head variants | Missed its 1% gate; same lesson as slide 11 |
| PE_19 temporal ensemble weights, Run 9 censored/horizon targets | Same conclusion as slide 7 by a different mechanism |
| PE_25 restricted blending | Screening variant of the PE_24 result on slide 11 |
| Seed enumeration, Optuna plumbing, checkpoint formats, TensorBoard views | Reproducibility machinery, not evidence that changes a conclusion |

---

## Template adaptation notes

- Base file: the supplied `UAVRULEstimation_presentation.pptx` (University of
  Stuttgart, "UNI COLOUR" theme, Arial, 10 × 5.625 in). Work was done on a copy;
  the original is unmodified.
- Masters, layouts, theme colours, logo and footer/slide-number placeholders are
  the template's own. The ten sample slides were removed after inspection.
- Layouts used: `Titelfolie` (slide 1), `Titel und Inhalt` (content and backup
  slides), `Kapitel` (backup divider).
- Type scale is the template's own: 18 pt bold titles, 14 pt subtitle line,
  8–12 pt inside diagrams and tables. On this 10-inch-wide canvas an 18 pt title
  projects the same size as 24 pt on a 13.33-inch deck.
- Schematic colour roles follow the template theme: dark blue `#00519E` for
  model members and outputs, cyan `#00BEFF` for the selected step, yellow
  `#FFD500` for the element under discussion, grey `#9F9998` for intermediate
  state. The repository figures keep their own original colours; the deck does
  not restyle them.

---

## Rehearsal estimate

The main-deck narration is **2,332 spoken words**. At an unhurried 125 words per
minute that is **18.7 minutes**, against a planned 19.0 minutes of content plus
1.0 minute of transitions and pauses. The per-slide table is at the end of
`speaker_notes.md`.

This is an arithmetic estimate, not a rehearsal. Slides 4, 5, 7 and 11 carry the
scope caveats and should not be the ones cut if a live run goes long.

---

## Verification performed

- All 21 slides were exported to PDF with LibreOffice 24.2 and inspected as
  images at presentation size for clipping, overlap, contrast and label
  legibility.
- A geometry pass confirmed that no shape falls outside the slide and that every
  slide carries speaker notes.
- Every number printed as slide text was matched back to `evidence_ledger.csv`,
  to `sources/chart_data/`, or to the deck's own source files; the check reports
  no unmatched values.
- `build_ledger.py` validates the ledger's claim numbering, slide ordering and
  status vocabulary before writing it.

Not verified: the deck has **not** been opened in Microsoft PowerPoint. Font
substitution in the LibreOffice render is Liberation Sans for Arial, which is
metrically identical, so the layout checks hold; kerning and the template's
autofit behaviour should still be confirmed on the target machine.
