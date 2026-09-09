# PE_31: coordinated comparison and nested model selection

This campaign implements the plan after PE_30: traceable references, a common evaluation benchmark, a six-policy interaction study, and a bounded joint search. PE_28–PE_30 remain unchanged. No prior negative result is reclassified as a success.

## Four executable stages

1. **Reproduce the alternative submission.** `reproduce_simple_submission.py` copies the original script and input CSVs into this run's `reproduction/` directory, verifies their hashes, and executes the script unchanged. It preserves the script's actual feature order, unit row weights, five grouped CV fits per family, capped-label single-draw blend selection, and final 90% fitting/10% stopping split without refitting the stopped-on UAVs. It records Python/package versions, logs and the submission hash. This is an exact source replay in the current environment, not a guarantee of the historical score. Only a user upload can establish the new Kaggle score. The script's approximately 0.94 internal scores use capped labels and its internal selection procedure; they are not outer-validation results.
2. **Preflight and provenance.** Verify PE_28's registered source/environment, construct all feature matrices and endpoint suites, export the study matrix, enforce the fit budget, and hash the available Run 7 submission. The historical alternative CSV was not available; its score-to-file link remains explicitly missing. Reproduction and historical identity are different claims.
3. **Interaction benchmark.** Evaluate six fixed simple-model policies plus Run 7 and an outer-held reproduction of the alternative fitting protocol. These comparisons are diagnostic; they do not automatically select a production replacement or decide which models the nested search may see.
4. **Nested joint search.** For each outer training set independently, compare all six policies on three inner UAV folds. Select one policy by historical-endpoint inner RMSE. Under that policy evaluate twelve model/representation/capacity recipes and five blend weights against Run 7, entirely inside the same outer training set. Fit the selected complete procedure on the outer training UAVs and predict the outer-held UAVs once. Outer benchmark outcomes never choose the inner policy or recipe.

## Six distinct training policies

| ID | Sampling | Relative weighting | Total weight in each fit |
|---|---|---|---|
| `sparse_uav_mass` | 20 prefixes/UAV | Equal total per UAV | Number of fitting UAVs |
| `sparse_unit_mean` | 20 prefixes/UAV | Equal total per UAV | Number of fitting rows |
| `dense_uav_mass` | All cycles | Equal total per UAV | Number of fitting UAVs |
| `dense_uav_unit_mean` | All cycles | Equal total per UAV | Number of fitting rows |
| `dense_cycle_mass` | All cycles | Equal per cycle | Number of fitting UAVs |
| `dense_unit_rows` | All cycles | Equal per cycle | Number of fitting rows |

With 20 rows per UAV, equal-UAV and equal-cycle weighting coincide; duplicate conceptual settings are collapsed. Normalization is recalculated inside every actual fitting and early-stopping subset. This separates relative influence from overall loss scale without changing XGBoost parameters in the interaction study. The original simple recipe, 266 B features, cap 125 and inner calibration procedure are fixed. All-cycle training includes terminal RUL-zero rows.

The PE_28 and PE_30 sparse/dense settings are historical anchors. Their old predictions cannot cover the new endpoints, and only predictions—not reusable fitted models—were saved. The campaign therefore refits matched references; it does not relabel old scores as new evidence.

## Shared evaluation

- Two outer partition seeds, five UAV folds each; two candidate model seeds per fold. Run 7's internal recipe/seeds remain frozen, so its repeated model-seed rows are controls, not independent stochastic replicates.
- Historical 500 scenario rows are retained for continuity. Three new nominal endpoint draws use raw RUL within 1–125. Three unrestricted draws remove the upper bound and retain RUL ≥1.
- Each new draw samples from the empirical test-cutoff multiset feasible for that individual UAV. It approximately follows feasible test history lengths; it **does not** reproduce the marginal histogram exactly. Feasible-cutoff distributions differ with lifetime and label support. Assignment RNGs are per UAV, so changing another UAV's labels cannot alter its draws. These are explicit robustness profiles, not claims about the unknown hidden-test sampling mechanism.
- A model is fitted once and predicts all suites. Each suite is reported separately with raw labels. Selection uses historical endpoints as its primary objective, with new nominal/unrestricted checks. Only active training-UAV endpoints participate in stopping, blend selection or candidate selection.
- `variation.csv` retains partition, model-seed and endpoint-draw dimensions. These are descriptive sensitivities, not independent observations or a formal variance-component decomposition. `summary.csv` averages predictions over model seeds first, evaluating the defined two-seed ensemble. UAVs remain the bootstrap unit across repeated splits/endpoints.

## Outer reference protocol versus exact source replay

The benchmark's `reference_protocol` preserves the original feature order, fold-held early stopping inside its training set, capped internal blend objective, unit row weights and final 10% stopping holdout without refit. For smaller outer-training populations, the original global 100-cutoff assignment cannot be copied literally. This benchmark explicitly uses one seeded uniform feasible-cutoff draw per active training UAV instead. Its single-draw calibration and original five grouped CV folds stay entirely inside the outer training set. Both source model seeds are replaced by the declared model seed for the benchmark only. The separate reproduction stage executes the actual original script without these adaptations and generates the file for Kaggle.

## Twelve joint-search recipes

Cross two causal feature views (B: 266 features; E: 134 compact features), three families (XGBoost–CatBoost blend, CatBoost alone, ExtraTrees alone), and two capacity settings. The compact view's previous failure remains evidence; it is revisited here only as part of the declared capacity/policy interaction.

Original blend/CatBoost parameters come from PE_28. The regularized setting uses XGBoost depth 3, lambda 5, minimum child weight 2; CatBoost depth 4 and L2 10. ExtraTrees uses 400 trees, 0.8 feature fraction and one thread: unrestricted depth/minimum leaf 1 versus depth 12/minimum leaf 5. Boosting iteration counts and within-recipe blend weights are selected using four grouped training-side calibration folds and separate stopping UAVs. This additional inner layer prevents candidate-selection validation UAVs from influencing those choices.

For the final Run 7 blend, weights are fixed at 0, .25, .5, .75 and 1. Candidate/weight combinations must have inner nominal RMSE no more than 1% above Run 7 and unrestricted RMSE no more than 5% above Run 7. Among eligible combinations choose historical RMSE, then lower added weight, then lexical recipe ID. Zero weight guarantees an eligible Run 7 fallback. The standalone winner is also selected inside training folds and reported separately. Residual correlation is diagnostic only.

After the complete outer evaluation, the selected two-seed blend advances to final review only if historical mean-fold RMSE improves at least 2%, historical pooled R² is at least .90, the conditional paired-UAV bootstrap 95% RMSE-change interval is wholly below zero, and the nominal/unrestricted regression bounds pass. Fold wins are reported, not treated as a significance test. These new prospective rules do not change previous experiment verdicts. No production replacement or Kaggle upload happens automatically.

## Budget, execution and resume

This is substantially larger than PE_29/30: **160 benchmark recipe evaluations**, plus at most **1,200 nested-search recipe evaluations**, bounded by a 1,400-evaluation configuration limit. Reused inner/base/control cells reduce actual work. A recipe evaluation itself can include many estimator fits; the reported 40,800-estimator bound is intentionally conservative and excludes residual heads. Exact source reproduction adds 12 estimator fits separately. Two process workers run outer tasks concurrently; each worker's fits are sequential with numerical-library threads limited to four. Workers checkpoint individual recipe fits rather than waiting for an entire outer task.

From the repository root:

```powershell
# Complete implemented campaign (reproduction resumes if already complete).
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_31\run.py

# Preflight only: no scientific model training.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_31\run.py --only validate_inputs

# Run stages separately, with the same resume semantics.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_31\run.py --only interaction_benchmark
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_31\run.py --only nested_joint_search
```

The search can run independently of the benchmark because its selection is fully nested, rather than selected from benchmark outer scores. Inputs, source modules, recipes, package versions and settings are frozen before fitting. Individual checkpoint hashes include training, calibration and evaluation data and their identities. Changed/unregistered/corrupted cells are rejected. Changing settings after training begins requires a new `pipeline.run`; there is no force override that mixes contracts. Parallel dispatch is bounded; worker failure stops new dispatch and lets active tasks finish their current work before shutdown.

## Outputs and limits

- `reproduction/submission.csv`, source/data copies, contract and log: uploadable replay and traceability.
- `reporting/provenance.json`, `input_verification.json`, `endpoints.csv`, `outer_folds.csv`: reference status, full budget and evaluation design.
- `stages/benchmark/reporting/interaction_contrasts.csv`: paired conditional sampling/weight-scale contrasts.
- Each stage's `summary.csv`, `fold_metrics.csv`, `variation.csv`, `paired_comparisons.csv`, `regions.csv`, `predictions.csv.gz`, `winner_manifest.json`.
- `reporting/fit_costs.csv` and `selections.csv`: actual estimator counts/times and training-side policy/recipe/blend choices. Each outer-task directory also retains the complete selection record and its actual inner UAV partitions. The exported partition-builder inner table is provenance only; nested selection uses the independently declared shuffled three-fold KFold over sorted active UAV IDs recorded in `selection.json`.

All training UAVs have influenced previous project decisions. New grouped partitions are robustness checks, not untouched data. Bootstrap intervals condition on fitted predictions and exclude uncertainty from the project's adaptive history. Nominal/unrestricted R² denominators differ. An offline R² ≥.90 does not establish a public or private leaderboard score ≥.90. This campaign makes those limitations visible rather than promising the target.
