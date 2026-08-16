# Reducing Phase 2 Runtime — Options and Recommendations

This note summarizes concrete ways to cut the wall-clock time needed to run
the Phase 2 architecture study (`2_model_architecture_study`), based on the
actual numbers in `1_experiment_contract/experiment_contract.toml` and the
model adapter code as of August 2026.

## Where the time is going

Step 5 runs 8 families × 5 outer folds × 4 inner folds, with the six tunable
families each getting a 25-candidate search — up to 3,040 individual model
fits before Step 6 even starts. The three neural families (`mlp`, `tcn`,
`lstm`) alone account for 1,500 of those fits, each up to 300 epochs
(patience 25), single-threaded, CPU-only. `lstm` is likely the slowest of the
bunch, since RNN recurrence can't vectorize across time steps the way TCN's
convolutions can. XGBoost is set to a 2,000-tree ceiling with a 50-round
early-stopping patience — generous enough that it can run long before
stopping. Everything is deliberately single-threaded (`n_jobs=1` on both
XGBoost and Random Forest, with a code comment explicitly noting that
candidate-level parallelism is left for "the later runner" — i.e., the
codebase was designed to be parallelized eventually, just not yet).

## Recommendations, ranked by impact for effort

1. **Run independent studies in parallel across CPU cores.**
   The biggest lever, and it doesn't change any results. Each of the 40 Step
   5 studies (family × outer fold) and each of the 40 Step 6 family/outer-fold
   groups is fully independent — different data, different checkpoint files,
   and (after the recent TensorBoard change) different log directories too.
   Fanning these out as separate OS processes (e.g., several
   `run_inner_model_selection.py --family X --outer-fold Y` invocations
   running concurrently) could cut wall-clock by close to the number of
   available cores, since each process stays internally single-threaded and
   deterministic.

   One thing to get right: PyTorch has no explicit `torch.set_num_threads()`
   call anywhere in the codebase, so each neural fit likely already tries to
   use all available cores for its own matrix ops. Running N such processes
   at once without capping that (`torch.set_num_threads(1)`,
   `OMP_NUM_THREADS=1`) would oversubscribe the CPU and could make things
   slower, not faster — so process fan-out needs a thread cap alongside it.

   One more constraint: the recent TensorBoard change made Step 6 share one
   writer across a family's three retraining seeds. Parallelizing at the
   family/outer-fold level (the big 40-way win) is completely safe, but going
   finer — parallelizing individual seeds within one family/fold — would have
   two processes fighting over the same log directory.

2. **Cut `candidate_budget_per_architecture` from 25 to ~10–12** for the six
   tunable families. A direct, linear reduction on the single largest
   multiplier in Step 5 — going from 25 to 10 would cut Step 5 fit count by
   well over half. One-line change in `experiment_contract.toml` (bump
   `contract_version`, rerun Step 1). Risk: a slightly worse hyperparameter
   configuration than a full 25-candidate search would have found — usually a
   minor effect for well-behaved search spaces, but not free.

3. **Shrink neural training's tail**: lower `early_stopping_patience`
   (25 → 10–12) and `maximum_epochs` (300 → 100–150). Patience mostly
   controls how long training waits *after* the best epoch before giving up,
   not which epoch turns out best, so trimming it usually costs little in
   selection quality while directly shrinking the worst case for all ~1,500
   neural fits.

4. **Narrow the sequence lookbacks compared**, from `[50, 100, 200]` down to
   `[50, 100]`, for a first pass. TCN and LSTM compute scales with sequence
   length, so dropping the 200-step option removes a meaningful chunk of
   exactly the two most expensive families' search space.

5. **Cap XGBoost's tree ceiling.** 2,000 trees with patience 50 is a wide
   window. Dropping the fixed `maximum_trees` value to ~800–1,000 (or
   tightening patience to ~25) bounds the worst case without necessarily
   changing where early stopping would have kicked in anyway on this dataset
   size.

6. **Stage the run instead of running everything at once.** Run the cheap
   families first (`--family mean_baseline cycle_only_baseline
   regularized_linear random_forest xgboost`), review results within hours,
   and let the three neural families run separately — overnight or in the
   background — since Step 5/6 already checkpoint per-study and
   `run_phase_2.py --from-step 5` resumes cleanly. Zero code risk, just a
   scheduling change, and it produces a partial comparison to look at
   immediately instead of waiting for the whole grid.

7. **Reduce Step 6 retraining seeds for an initial pass** — from 3 down to 1
   for the stochastic families (`random_forest`, `xgboost`, `mlp`, `tcn`,
   `lstm`), then only re-run the full 3-seed evaluation for whichever
   architecture(s) are actually under serious consideration. Cuts Step 6's
   stochastic-family cost by up to 3x for a first look, at the cost of not
   yet knowing seed-to-seed variance for the families not re-run.

8. **GPU acceleration**, if an NVIDIA card is available. The biggest
   theoretical per-fit speedup for the neural families and XGBoost, but also
   the most invasive: `torch.device("cpu")` is currently hardcoded in
   `neural_base.py`, so it needs real code changes, and PyTorch's CUDA
   determinism (`torch.use_deterministic_algorithms`) needs to be
   re-verified rather than assumed, since exact bit-reproducibility across
   GPU runs is a stricter guarantee than on CPU. Worth it if the hardware is
   sitting there unused; not worth the engineering time otherwise.

9. **Increase `batch_size`** (currently 64) for somewhat better CPU
   throughput per epoch. The smallest win on this list, and since it's a
   shared contract value across all neural families, changing it affects
   their comparability equally rather than favoring one — reasonable, just
   not where to start.

## Suggested starting point

Start with #1 (parallelize across studies with thread caps) and #3 (trim
patience/epochs) together — both are cheap to do, don't touch the science in
a way that would bias the comparison, and compound with each other. Add #2
(fewer candidates) if still not far enough under the deadline. Treat #8 (GPU)
as a last resort given the determinism re-verification cost.
