# Run 11 — the same six architectures under one equal tuning protocol

Run 10 compared six model families, but it did not compare them under equal
conditions. Its recipes were imported from four different selection runs, with
four different budgets and three different settings versions:

| Family | Recipe source | Settings version | Trials per outer fold | Best trial found at |
| --- | --- | ---: | ---: | --- |
| XGBoost | run_9 | 1 | 6 | 4–5 of 6 |
| LSTM | run_7 | 8 | 10 | 1–6 of 10 |
| Multi-scale CNN | run_7 | 8 | 10 | 2–5 of 10 |
| Transformer | run_3 | 7 | 25 | 6–25 of 25 |
| MLP | run_3 | 7 | 25 | 20–24 of 25 |
| Trajectory DTW-kNN | run_5 | 11 | **1** | — |

Three defects follow from that table.

1. **DTW-kNN was never tuned.** Every entry in its search table was
   `kind = "fixed"`, and `CandidateSpace.candidate_budget` returns one trial
   when nothing is tunable. Its `candidate_results.csv` has one row per outer
   fold with identical hyperparameters on all five, against 250 rows for every
   other family in run_5. Its Run 10 result is a default, not an optimum.
2. **MLP and Transformer were tuned against an older pipeline.** Both come from
   run_3 at settings version 7. The LSTM shows what that costs: tuned in run_3
   its best inner RMSE was 33.72; re-tuned in run_7 under the newer settings it
   reached 19.33, a 43% improvement from re-tuning alone. The MLP and
   Transformer never received that re-tune.
3. **The run_3 searches had not converged.** The MLP's winning trial sits at
   index 20, 20, 20, 23, 24 out of 25. A search whose winner is at the budget
   ceiling was still improving when it stopped.

Run 11 removes all three. Every family is tuned in **one** Step 5 run, from
**one** settings version, at **25 trials per outer fold**, with one search seed.

## What changed in the configuration

`1_architecture_study_settings/architecture_study_settings.toml`:

- `run_number` 6 → **11**.
- `study.enabled`: exactly the six Run 10 families are true
  (`xgboost`, `mlp`, `lstm`, `multiscale_cnn`, `transformer`,
  `trajectory_dtw_knn`); `extra_trees` and `hist_gradient_boosting` were
  switched off, everything else was already off.
- `architectures.trajectory_dtw_knn.search`: the five fixed values were replaced
  with real ranges, so the family now draws a full 25-trial budget like the
  others.

Deliberately **not** changed:

- `settings_version` stays **14**, so the already-built tabular, sequence and
  trajectory data adapters remain valid and do not have to be regenerated.
- `tuning.candidate_budget_per_architecture` was already 25.
- `representations.sequence_lookbacks` stays `[50, 100]`. All three sequence
  families already declare that same set in the TOML, so running them from one
  Step 5 gives them the same lookback choices for free. Adding lookback 20
  would require rebuilding the sequence tensors, which is why it was not done.

## The DTW-kNN search space

```toml
neighbors           categorical [1, 3, 5, 10]
reference_pool_size categorical [20, 40]
max_points          categorical [24, 48, 72]
warping_window      categorical [2, 4, 8, 16]
distance_power      uniform     0.25 – 2.0
```

The adapter rejects a reference pool smaller than `k`, and Optuna samples each
parameter independently, so the smallest pool (20) is kept at twice the largest
`neighbors` choice (10) — the constraint cannot be violated by construction.
The cost ceiling is bounded too: DTW work scales with
`max_points x (2 x warping_window + 1) x reference_pool_size`, so the worst
corner of this space costs about six times the old fixed point, not the 16x a
wider `max_points` range would have allowed.

## One consequence to expect

The LSTM and Multi-scale CNN used lookback 20 in Run 10, inherited from run_7
via PE_6. In Run 11 they search over 50 and 100 like the Transformer. Their
Run 11 cells are therefore **not** directly comparable with their Run 10 cells;
the comparison Run 11 makes valid is the one *across families within Run 11*,
which is the comparison the chart actually claims.

## Run it

From the repository root, in order.

**1. Validate the edited settings and rebuild the specification. Do not skip
this.** `run_phase_2.py` and Step 5 both read the generated
`1_architecture_study_settings/artifacts/experiment_specification.json`, never
the TOML. Until this command has run, the edits below are inert and the pipeline
silently continues to use whatever run the old specification names — it will
report that run's studies as already complete and do nothing. This step is also
the gate: schema errors or failed Phase 1 checks stop here without writing
anything.

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\2_model_architecture_study\1_architecture_study_settings\build_architecture_study_settings.py
```

Confirm it took before committing hours to the run:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\2_model_architecture_study\run_phase_2.py --status
```

It must print `Run: 11` and show Step 5 with **0 of 30** studies complete. Thirty
is six families times five outer folds. Any other run number, or a count of 15,
means the rebuild did not happen and Step 5 would run against the old
specification.

**2. Archive that specification where the outer evaluation looks for it.**
`read_recipe` reads `runs/run_<n>/1_architecture_study_settings/artifacts/experiment_specification.json`
to recover the neural training block and the XGBoost patience.

```powershell
New-Item -ItemType Directory -Force -Path 2_architecture_experiments\2_model_architecture_study\runs\run_11\1_architecture_study_settings\artifacts
Copy-Item 2_architecture_experiments\2_model_architecture_study\1_architecture_study_settings\artifacts\experiment_specification.json 2_architecture_experiments\2_model_architecture_study\runs\run_11\1_architecture_study_settings\artifacts\
```

**3. Step 5, the tuning.** This is the long part: 30 independent studies, one
per family and outer fold. Launch it through `run_phase_2.py`, **not** through
`run_inner_model_selection.py` directly — the parallel fan-out lives in
`run_phase_2.py`, which dispatches each family/outer-fold pair as its own
subprocess, up to `[execution].max_workers` (currently 6). Calling the Step 5
entry point directly runs all 30 studies one after another in a single process.

```powershell
.venv\Scripts\python.exe -u 2_architecture_experiments\2_model_architecture_study\run_phase_2.py --from-step 5 --through-step 5
```

Check the first line of output: it should say it is dispatching **30**
independent studies. Fifteen means the specification is still the old one.

`--max-workers N` overrides the setting for one run without editing the TOML.
Note that the five neural families share one GPU, so raising the worker count
past a handful buys progressively less; `--max-workers 1` runs sequentially.

The measured training seconds behind the estimate total roughly 33 hours across
the six families, dominated by the LSTM (~22 h) and the CNN (~19 h). Wall clock
with the fan-out will be lower than that sum, but not by the full worker count,
because those two families contend for the same GPU.

It can also be split and resumed at a finer granularity. To run one family, or
one outer fold, sequentially in the current shell:

```powershell
.venv\Scripts\python.exe -u 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py --family xgboost --family trajectory_dtw_knn
.venv\Scripts\python.exe -u 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py --family lstm --outer-fold 0
```

Cheapest first is a good order for a manual split: XGBoost (~1 h), MLP (~3 h),
DTW-kNN (~3 h), Transformer (~5 h), then the CNN and the LSTM. If something is
wrong with the setup, the first hour finds it.

**4. The outer evaluation.** Validate first; it fits nothing. This stage is
sequential by design — `engineered_run_11_settings.json` sets `max_workers: 1`
and `prepare()` refuses any other value, so the 90 cells run one at a time with
each fit limited to `cpu_threads: 4`. It is the short stage (~2 h in Run 10).

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\2_model_architecture_study\run_engineered_feature_study.py --settings 2_architecture_experiments\2_model_architecture_study\engineered_run_11_settings.json --check
.venv\Scripts\python.exe -u 2_architecture_experiments\2_model_architecture_study\run_engineered_feature_study.py --settings 2_architecture_experiments\2_model_architecture_study\engineered_run_11_settings.json
```

`engineered_run_11_settings.json` is byte-identical to Run 10's except for
`run_directory` and `selection_runs`, which now point every family at run 11.
Same feature script, same folds, same endpoint seeds, same model seeds — the
recipes are the only thing that moves.

**5. The charts.**

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\2_model_architecture_study\plot_run_10_cost_split.py --reporting 2_architecture_experiments\2_model_architecture_study\runs\run_11\reporting
```

## What this can and cannot settle

It settles whether the Run 10 ranking survives an equal-budget, single-version
protocol. It does not test the mechanistic explanations for *why* the families
rank as they do — that the 266 engineered features already carry the history,
so a sequence model re-learns what the columns supply. Those remain readings of
the recipes, and a separate ablation would be needed to test them.

The expected outcome is that XGBoost still wins. It was the least-searched
family in Run 10 (6 trials) and still beat the next model by 7.3 RMSE; a gap
that size does not usually close with tuning. What Run 11 should change is the
ordering among the five losers, and in particular the MLP's negative R², which
is much more likely a stale-recipe artefact than a property of MLPs on this
problem.
