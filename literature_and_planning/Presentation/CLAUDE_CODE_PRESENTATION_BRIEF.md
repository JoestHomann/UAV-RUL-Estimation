# Claude Code brief: UAV remaining useful life estimation

This is an executable work brief for Claude Code. Read it together with the
PowerPoint template the user supplies. Create the presentation and supporting
files described below. Do not stop at an outline.

## 1. Assignment

Create a **20-minute scientific presentation** about the UAV remaining useful
life estimation project in this repository. Explain our approach, the purpose
of each development phase, the experiments that changed our understanding,
and the evidence behind the retained model and remaining research questions.

The audience should leave understanding:

1. What we predict from an incomplete UAV telemetry history.
2. Why generalization to unseen UAVs determines the validation design.
3. Which data representations, target assumptions and model choices mattered.
4. Which apparent improvements survived stronger evaluation and which did not.
5. What we can currently claim, and what still requires evidence.

Organize the presentation around **questions, evidence and decisions**. Do not
produce a chronological catalogue of pipeline runs. A rejected experiment can
be central if it changed a decision. A completed run can be omitted if it added
no distinct information.

Use English unless the user requests another language. Assume an audience
familiar with basic machine learning but unfamiliar with this project. Explain
UAV, RUL, OOF and the evaluation setup when first introduced. Treat 20 minutes
as presentation time, with questions afterward unless the user says otherwise.

## 2. Inputs and working boundaries

Repository root on the current machine:

```text
C:\Users\joest\UAV-RUL-Estimation
```

Resolve the paths below relative to that root. If working elsewhere, locate
the repository and adapt the root without changing source files.

The user will supply a `.pptx` or `.potx` template. If it is not attached or
locatable, ask for it once while continuing the source review and storyboard.
Do not invent a replacement visual identity or claim to have followed an
unseen template. The template's sample content is a design reference, not
evidence about this project.

Create outputs in `literature_and_planning/presentation/`. Keep temporary
rendering files in a `build/` subdirectory. Work on copies of the template.
Do not modify scientific code, settings, checkpoints, raw data, experiment
verdicts, or the original template. Do not rerun training, launch experiments,
submit to Kaggle, or interrupt active work. Read completed reports and artifacts.
Do not open new locked evaluations to obtain a presentation result.

## 3. Inspect evidence before writing slides

Read these navigation documents first:

| Source | Purpose |
| --- | --- |
| `README.md` | Four-phase project structure |
| `literature_and_planning/development_documentation/phase_0_dataAnalysis.md` | Data audit, channel roles and initial hypotheses |
| `literature_and_planning/development_documentation/phase_1_datasetConstruction.md` | Grouped validation, causal prefixes, features and weighting |
| `literature_and_planning/development_documentation/phase_2_modelArchitectureStudy.md` | Model families and architecture-study protocol |
| `literature_and_planning/development_documentation/phase_3_finalModelTrainingAndInference.md` | Final model selection, fitting and inference |
| `literature_and_planning/development_documentation/pipeline_experiments.md` | Experiment questions, decisions and result locations |
| `literature_and_planning/development_documentation/r2_research_2026_09_07/report.md` | Generalization gap and error diagnosis |
| `literature_and_planning/KAGGLE_CHALLENGE.md` | Dataset/task context; verify substantive claims against its sources |

These documents contain historical plans as well as results. Some describe an
earlier state as incomplete even when later artifacts exist. **Use them as a
map, then verify numbers and completion status in the relevant CSV/JSON files.**

Source priority:

1. Completed run artifacts and their exact settings, prediction scope and manifests.
2. Recorded public scores with their submission provenance.
3. Project documentation for interpretation and navigation.
4. This brief's snapshot, which may become outdated.

If sources disagree, investigate scope, run identity and aggregation first.
Do not silently choose the more impressive number. Record unresolved differences
in the evidence ledger and omit the disputed claim from the main presentation.

Create `evidence_ledger.csv` with these fields:

```text
claim_id,slide_number,claim,source_path,table_row_or_json_key,experiment_identity,
data_scope,target_policy,evaluation_support,aggregation,comparator,
status,limitation
```

Every numerical claim and substantive empirical conclusion needs an entry.
Include source paths and relevant keys in speaker notes. Use compact source
labels on slides rather than long filesystem paths.

### Sources to inspect selectively

- Existing Phase 0 plots and companion CSVs under
  `0_data_analysis/core_data_analysis/figures/` and
  `0_data_analysis/broad_data_review/figures/`.
- The Phase 1 feature manifest used by the retained pipeline:
  `2_architecture_experiments/1_pipeline_experiments/experiments/PE_3/runs/run_1/PE3_features_drift/phase2/2_tabular_data_adapter/artifacts/tabular_dataset_manifest.json`.
- Original architecture comparisons under
  `2_architecture_experiments/2_model_architecture_study/runs/run_4/7_architecture_comparison/`
  and `runs/run_5/7_architecture_comparison/`.
- Temporal study:
  `2_architecture_experiments/2_model_architecture_study/runs/run_7/7_architecture_comparison/temporal_architecture_summary.csv`.
- Hybrid study:
  `2_architecture_experiments/2_model_architecture_study/runs/run_8/7_architecture_comparison/hybrid_architecture_summary.csv`.
- Censored/horizon study, if needed for backup:
  `2_architecture_experiments/2_model_architecture_study/runs/run_9/7_architecture_comparison/censored_architecture_summary.csv`.
- Selected pipeline reports under
  `2_architecture_experiments/1_pipeline_experiments/experiments/PE_<number>/runs/run_1/reporting/`.
  Some experiments use additional stage or sub-experiment directories. Locate the
  actual manifest and matching summary rather than assuming every path is identical.
- Retained final-model reports:
  `3_final_model_training_and_inference/runs/run_6/7_post_run_reporting/report_summary.json`
  and the equivalent `run_7` path.
- Final Run 7 architecture contract:
  `3_final_model_training_and_inference/runs/run_7/1_winning_architecture_selection/artifacts/residual_ensemble_contract.json`.
- Recorded public scores and error diagnostics:
  `literature_and_planning/development_documentation/r2_research_2026_09_07/`.

Disambiguate run names: **architecture-study Run 7 is a temporal-model study;
Phase 3 Run 7 is the retained residual-corrected tree ensemble.** Bare run
numbers are unsuitable slide labels.

## 4. Select experiments by their contribution to understanding

For every candidate experiment, write four short statements in the storyboard:

1. **Question:** What uncertainty motivated it?
2. **Comparison:** What changed, and what stayed fixed?
3. **Evidence:** What did the matched evaluation show?
4. **Decision:** What did we retain, reject or investigate next?

Use “information gain” in this ordinary research sense. Do not invent an
information-theoretic score or confuse it with tree split importance.

An experiment earns main-deck space if it changes an assumption, changes the
retained approach, eliminates a plausible alternative, or exposes a limitation
that affects interpretation. Group experiments that answer the same question.
Keep identifiers in notes or small source labels. Give the audience the finding.

### Priority evidence clusters

| Cluster | Information worth explaining | Treatment |
| --- | --- | --- |
| Data audit and channel roles | Constant channels, UAV-specific baselines and degradation patterns require different treatment; pooled correlations alone are insufficient | Main deck, Phase 0 |
| UAV grouping and causal prefixes | Many cycle rows from one UAV do not constitute independent machines; features must use only observations available at the cutoff | Main deck, Phase 1 |
| Target cap by validation scenario, PE_2 matrix | The benefit of capping depends on the target support used for evaluation; narrowing evaluation changes the problem | Main deck, with the full 2x2 logic |
| Engineered features and matched ablations | History summaries can be useful, while larger feature unions, denser sampling and compressed representations did not consistently help | One selected comparison in the main deck; details in backup |
| Tree, temporal and hybrid studies | Tested temporal complexity did not displace engineered-feature trees under the examined protocols | Main deck, one matched chart and a limited interpretation |
| Residual correction, PE_11, then Phase 3 Run 7 | Learned correction provided a useful gain beyond simple seed averaging in its matched comparison | Main deck; explain the mechanism |
| Accuracy versus conservative prediction, PE_4/PE_13 | Reducing overprediction is a different objective from maximizing R² | One concise main-deck slide or integrated result panel |
| Forecast-history screen and complete refit, PE_15/PE_20 | A large screening improvement did not survive a complete nested refit | Main deck; strong example of learning from failed confirmation |
| Small complementary blends, PE_18/PE_21/PE_24 | Favorable average gains can be too small or inconsistent to justify replacement | Group into one evidence sequence |
| Error concentration and short-history specialist, PE_26 | A diagnosed difficult subgroup does not guarantee that a specialist will improve the combined predictor | Main deck error diagnosis; specialist result can be a brief annotation |
| UAV subsets, PE_27 | A mean improvement can still miss the required fold consistency | Brief latest-result example or backup |
| Feature/model comparison, PE_28 | Broader sensor coverage and simpler summaries are explicit hypotheses under test | Closing work-in-progress only until complete |

Do not give each row a dedicated slide automatically. Fit the evidence into the
storyboard below. For example, the architecture slide can combine temporal and
hybrid findings, and the confirmation slide can combine forecast history with
the small-blend evidence.

Routine reruns, seed enumeration, adapter inventories, Optuna plumbing,
checkpoint formats, TensorBoard screenshots, failed installations, and every
hyperparameter value belong outside the main talk. Mention reproducibility
mechanisms only when they explain why a result can be trusted.

## 5. Verified starting points, not permanent results

The following snapshot was checked on **8 September 2026**. Refresh relevant
artifacts before presenting. Never treat a gate threshold as an achieved result.

| Evidence | Snapshot | Correct interpretation |
| --- | --- | --- |
| Raw training data | 100 UAVs, 24,720 cycle rows, 28 telemetry channels | The independent unit is the UAV, not the cycle row |
| Current retained feature representation | 298 features; 2,000 training prefixes; 500 development scenario rows | These counts describe the current pipeline, not every historical study |
| Distinct development endpoints | 444 distinct UAV/cutoff pairs among 500 rows | Repeated endpoints affect weighting; bootstrap whole UAVs |
| PE_11 residual correction | Mean RMSE 11.5080 to 11.0243, 4.20% improvement, 5/5 wins | A matched result against that experiment's tree-blend control |
| Phase 3 Run 6 | Mean-fold R² 0.89274, mean-fold RMSE 10.6332; recorded public R² 0.86741 | Keep development and public metrics separate |
| Phase 3 Run 7 | Mean-fold R² 0.90041, mean-fold RMSE 10.2361; pooled development R² 0.90446 | Mean-fold and pooled R² are different aggregations |
| Phase 3 Run 7 public score | Recorded R² 0.87652 | The public R² >0.9 goal has not been demonstrated |
| PE_15 then PE_20 | Forecast-history screen improved mean RMSE about 13%; complete refit regressed it about 1.56% | Different evaluation procedures and controls; show as a screen/confirmation sequence, not a common-scale leaderboard |
| PE_18 then PE_24 | Initial TabPFN blend gain about 1.30%; later regime blend about 1.896%, 11/15 wins, bootstrap interval crossing zero | Initial practical screening success did not establish a reliable replacement |
| PE_26 | Mean RMSE gain about 0.389%, 6/15 wins, pooled R² 0.89924 | The full short-history combination did not pass its gate |
| PE_27 | Four-member screen: 3.003% mean RMSE gain, 3/5 wins; bootstrap RMSE-change interval approximately [-0.622, +0.085] cycles | Screen stopped after 25 complete model fits; eight-member expansion and confirmation were skipped |
| PE_28 | Training started; no complete winner manifest at inspection | Refresh status; partial cells are not a final comparison |

PE_27's latest sources are
`experiments/PE_27/runs/run_1/reporting/winner_manifest.json` and
`experiments/PE_27/runs/run_1/stages/screen_4/reporting/{summary,promotion_decisions}.csv`
under the pipeline-experiments directory. Historical planning text may still
call this run pending.

Two particularly useful architecture sources:

- In the matched hybrid study, mean RMSE was 11.508 for the tree control,
  14.618 for hybrid GRU, and 17.671 for hybrid CNN. Neither hybrid won a fold.
- The dedicated temporal study's best tested model was LSTM at mean RMSE
  19.570 and mean R² 0.6531. Do not place this next to the hybrid numbers as
  one controlled ranking unless you verify identical endpoints and protocol.

For error diagnosis, the saved Run 7 development analysis attributes about
32.5% of squared error to histories of at most 100 cycles and about 38.7% to
the ten highest-error UAVs. These are descriptive values for that saved set.
Do not mix them with percentages from later split seeds.

The user also reported a public score of **0.87877** for the alternative script
in `other_pipelines/uav_rul_pipeline_v4 (1).py`. Treat this as a user-reported
score unless a submission artifact independently establishes its provenance.
Do not call it Run 7's score or proof that the controlled PE_28 recipe wins.
The script constructs **266 features from 22 sensors**, not four features.
Its internal “RUL_raw” labels have already been capped, and its early-stopping
and blend-selection evaluation differs from ours. Its reported public score
is motivation for PE_28, not a completed matched experiment.

## 6. Suggested 20-minute storyboard

Use approximately **14 main slides** plus optional backup slides. The allocation
below is **19 minutes of content plus one minute for transitions and pauses**.
Adapt layouts to the template. Keep the total speaking time at 20 minutes.

| Slide | Topic and purpose | Preferred evidence or visual | Minutes |
| --- | --- | --- | ---: |
| 1 | RUL prediction from an incomplete UAV history | A simple timeline with observed prefix, prediction cutoff and unknown future; title and research question | 0.75 |
| 2 | Four development phases | Phase 0 data analysis, Phase 1 dataset construction, Phase 2 architecture studies, Phase 3 final model and inference; one output/decision per phase | 0.75 |
| 3 | What the telemetry contains | Two or three representative real trajectories illustrating degradation, context and constant channels; explain why age alone is insufficient only if baseline evidence supports it | 1.50 |
| 4 | Evaluation on unseen UAVs | Whole-UAV outer split and training-side inner selection diagram; prefix-only features and equal UAV weights | 2.00 |
| 5 | Target assumptions changed the result | The target-cap-by-scenario 2x2 experiment; label differing target support prominently | 2.00 |
| 6 | Turning histories into model inputs | One real prefix with latest level, baseline change, trend and recent variability; one informative matched feature ablation | 1.50 |
| 7 | What architecture studies taught us | One matched tree/temporal-or-hybrid comparison, plus the decision it supported; backup for wider family inventory | 2.00 |
| 8 | The retained residual-corrected ensemble | Editable architecture diagram and PE_11's matched improvement; explain training-side calibration and final refit | 1.75 |
| 9 | Accuracy and overprediction | A matched accuracy/conservative-prediction tradeoff, with bias or RMS overprediction; distinguish objective from operational safety | 1.00 |
| 10 | Promising screens and stronger evaluation | Forecast history screen versus complete refit; compact supporting sequence for small complementary blends | 2.00 |
| 11 | Where the remaining errors occur | RMSE and squared-error share by observed history or RUL band, with subgroup sizes; short-history specialist result as a restrained annotation | 1.25 |
| 12 | Development performance and public performance | Separate labelled panels for development and recorded public results; explain the remaining R² gap | 1.00 |
| 13 | What the next comparison can resolve | Group PE_28's ten sets into sensor coverage, summaries, baselines and windows; two model recipes; current status label | 1.00 |
| 14 | Conclusions supported by the evidence | Three concise findings and one remaining uncertainty, without a generic “thank you” slide replacing the conclusion | 0.50 |

If the template has a required agenda or closing slide, integrate it into this
time budget. Do not add several minutes of administrative slides.

For the retained architecture, verify the contract and depict:

- Three seeded XGBoost models and three seeded ExtraTrees models.
- Within-family averaging and a training-side selected family blend.
- Residual correction using the base prediction, observed history length,
  ensemble spread/disagreement and selected sensor features.
- A nonnegative final RUL estimate.

Explain that out-of-fold calibration predictions come from models that excluded
the corresponding UAV. Do not draw the future true RUL as an inference input.
Distinguish the training diagram from the inference diagram if combining them
would obscure that boundary.

## 7. Scientific interpretation rules

1. Label every performance comparison by data scope: development, previously
   reported locked evaluation, or Kaggle public evaluation. Never put unlike
   scopes on one improvement curve.
2. Distinguish fitting-target capping from evaluation-target capping and from
   prediction clipping. The retained fitting cap is 125; current metrics use raw
   endpoint labels, while nominal scenarios restrict the support to 1–125 RUL.
3. For the cap/scenario matrix, changing the evaluated target distribution changes
   both task difficulty and the R² denominator. Do not attribute the entire
   improvement across those cells to the model or claim it covers unrestricted RUL.
4. Distinguish mean-fold metrics, pooled metrics, seed standard deviation and
   bootstrap confidence intervals. Preserve the source's aggregation.
5. Resample UAVs for uncertainty. Additional prefixes and repeated random seeds
   do not create additional independent UAVs.
6. A positive mean improvement with an interval crossing zero is uncertain evidence,
   not proof of no effect and not a confirmed improvement. A failed practical gate
   is a decision outcome, not automatically a statistical significance result.
7. Fresh splits of the same previously examined UAVs are robustness checks. They
   do not erase earlier feature, model or leaderboard selection. Fixed-prediction
   bootstrap intervals omit some training and selection uncertainty.
8. State neural findings as results for the architectures, representations and
   budgets tested here. Do not claim that deep learning cannot solve RUL estimation.
9. A difficult subgroup motivates a hypothesis. PE_26 shows that recognizing a
   problem does not establish that the proposed specialist fixes it.
10. Explain observed feature associations without inventing physical sensor
    meanings, failure mechanisms or causal effects. “Causal features” here means
    prefix-only construction, not identification of causal physical mechanisms.
11. Use “retained final model” rather than implying deployment on operational UAVs.
    Reduced overprediction is a relevant error objective, not proof of flight
    safety, maintenance effectiveness or certification.
12. Report completed failures as well as successes when they change the conclusion.
    Do not turn the project into an uninterrupted progress story.

Optional calculation for the public-score slide: on the same scored target
set, improving R² from 0.87652 to 0.90 requires roughly 10.0% lower RMSE,
because the ratio is `sqrt((1 - 0.90) / (1 - 0.87652))`. This is a relative
requirement, not an observed gain or an estimate of absolute public RMSE.

## 8. Follow the supplied PowerPoint template

Inspect every template slide and available layout before constructing the deck.
Identify its slide dimensions, theme, fonts, title/body placeholders, colors,
margins, footer rules, slide numbering, logos and chart conventions.

Reuse or duplicate suitable template layouts and edit their placeholders. Keep
the actual master/theme structure where possible. Preserve required branding
and institutional elements. Do not assume a 16:9 size or replace the template
with a generic blue technology theme. Remove unused sample text and placeholders.

Use the tools available in the Claude Code environment that best preserve the
template. Do not assume Codex-only packages, tools or local runtime paths exist
there. Inspect the exported file to confirm that the chosen method preserved
masters, fonts, logos, notes and editable content. If a conversion loses a required
element, fix it or disclose the specific limitation rather than silently replacing it.

Prefer editable text, tables, diagrams and charts. Scientific plots can be
high-resolution exported figures with their source data and generation script.
Do not flatten an entire slide into a bitmap. Never generate an image of a chart
as a substitute for plotting the measured values.

## 9. Writing and visual style

Write plain scientific English. Prefer the actual subject or finding to a
slogan. A setup slide can have a direct topic title; a result slide can state
the supported finding. Titles do not all need the same grammatical formula.

Examples:

| Avoid | Prefer |
| --- | --- |
| “Unlocking the predictive potential of UAV telemetry” | “Predicting RUL from partial telemetry histories” |
| “A journey from raw data to actionable intelligence” | “Four development phases” |
| “Advanced architectures drive transformative performance” | “The tested hybrid models did not improve the tree control” |
| “Robust gains across diverse conditions” | “Mean RMSE improved 3.0%; the candidate won 3 of 5 folds” |
| “Promising opportunities remain on the horizon” | “PE_28 tests sensor coverage and feature summaries” |

Avoid “leverage,” “unlock,” “revolutionize,” “cutting-edge,” “seamless,”
“holistic,” “game-changing,” generic “key insights,” and empty claims of
robustness. Avoid rhetorical contrasts, dramatic fragments, repeated
three-part slogans and long strings of punctuation. Use active, concrete
sentences without forcing every sentence into the same style.

One main question or finding per slide. Usually use one substantial visual and
a small amount of explanation. Put derivations, qualifications and secondary
numbers in notes or backup slides. Do not shrink paragraphs to fit a slide.
Keep necessary scope labels visible rather than hiding important caveats in notes.

Follow the template rather than adding visual decoration. In particular:

- No stock drone photos as filler, decorative AI brains, circuit backgrounds,
  unexplained icons, gradients or dashboard-style cards unless the template
  explicitly requires them.
- Use consistent colors for the reference model, challenger and uncertainty.
- Label axes, units, subgroup counts and aggregation. Explain interval types.
- Use at most roughly four or five displayed methods in a main-deck comparison;
  put the complete table in backup. Select them by the question being answered,
  not solely by which numbers favor the narrative.
- Keep charts comparable only where the data and evaluation match. Do not
  connect scores from changing target distributions as a learning curve.
- Avoid tiny run identifiers, screenshots of CSV files, code dumps and unreadable
  feature-name heatmaps. Redraw selected evidence with presentation-sized labels.
- Preserve readable fonts at projection size. If the template permits, target
  roughly 28–36 pt titles and 20–24 pt body text. Cut content before shrinking it.

## 10. Speaker notes, backup and delivery

Write usable speaker notes for every main slide, not just a list of reminders.
They should explain the experiment's question, result, interpretation and the
transition to the next slide. Include the planned duration, claim IDs and source
references in a clearly separated notes section so citations are not spoken.

Aim initially for about **2,100–2,350 spoken words** across the main slides,
then check timing at a realistic speaking rate and allow pauses for figures.
The rehearsal estimate, not an arbitrary word quota, determines final length.
Avoid reading every number from a chart aloud.

Prepare four to six optional backup slides for questions, chosen from:

- Full architecture comparison within one fixed protocol.
- The complete cap/scenario matrix and target-support explanation.
- Residual correction and nested UAV calibration details.
- Feature definitions and PE_28's ten-by-two design.
- Overprediction metrics and their relationship to RMSE.
- Bootstrap limitations, endpoint duplication and public-score provenance.

Deliver:

1. `UAV_RUL_Project_20min.pptx`: editable presentation using the supplied template,
   with embedded speaker notes and clearly separated backup slides.
2. `UAV_RUL_Project_20min.pdf`: rendered slide export if a reliable renderer is available.
3. `speaker_notes.md`: the same narration, slide numbers, timing and source notes.
4. `evidence_ledger.csv`: traceable claims and numerical scope.
5. `storyboard.md`: the final slide plan, the important experiments selected and
   the reason major omitted experiments did not need main-deck time.
6. Reproducible chart data and the deck/figure build script when generated
   programmatically, in a compact `sources/` directory.

Render and inspect every slide at presentation size. Check clipping, overlaps,
font substitution, logo placement, text contrast, chart labels, notes and the
transition between main and backup slides. Compare representative slides with
the template side by side. Verify all numbers against the evidence ledger and
check that time allocations sum to the talk budget. A successful file export
does not establish visual quality.

Do not claim to have checked the deck in PowerPoint unless you actually opened
it there. If rendering or PDF export is unavailable, state exactly what could
not be checked. Do not deliver an unrelated rendered document as proof that the
PowerPoint is correct.

The final handoff should identify the deck, notes and evidence ledger, state
the estimated speaking time, and mention only limitations that affect use.
Do not describe the deck as polished, stunning, professional-grade or flawless.

## 11. Final editorial check

Before delivery, answer these questions in your own review:

- Does each main experiment earn its place through information gained?
- Can the audience explain what each of the four phases contributed?
- Are model-family findings separated from changes in features and evaluation?
- Are the strongest successful and unsuccessful findings both represented?
- Does the talk explain the retained ensemble without implementation clutter?
- Are development R² above 0.9 and the lower public score clearly distinguished?
- Is every pending experiment labelled pending, even if partial cells look good?
- Could any line appear unchanged in a presentation about an unrelated ML project?
  If so, replace it with a concrete project fact or remove it.
- Can the speaker present the main deck in 20 minutes without rushing?

Proceed from the evidence review to the storyboard, template adaptation,
presentation construction, rendering and final handoff. Ask the user only for
missing inputs that materially block the work; do not require approval of every slide.
