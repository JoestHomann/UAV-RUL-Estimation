"""
UAV RUL Estimation - v13 (XGBoost only, otherwise identical to v11)

v11 was an XGBoost + CatBoost ensemble. CatBoost never contributed much beyond
XGBoost in this project, so this version strips it out entirely - same sensor
tiering, same sample/feature weighting, same early-cycle specialist and
gating, same test-like evaluation - just one model instead of two. Simpler,
faster, and lets you isolate how much (if anything) CatBoost was actually
adding to v11's 0.89046 result.

Usage:
    python uav_rul_pipeline_v13.py                 # normal run: fit, submit
    python uav_rul_pipeline_v13.py --depth-sweep    # max_depth comparison
    python uav_rul_pipeline_v13.py --tune-xgb       # Optuna hyperparameter search
    python uav_rul_pipeline_v13.py --rul-cap-sweep  # RUL cap comparison
"""

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # one line per trial instead of a wall of logs
except ImportError as e:
    raise ImportError(
        "Optuna is not installed. Install it with: pip install optuna"
    ) from e

# --------------------------------------------------------------------------
# Config - tweak these first before touching the code below
# --------------------------------------------------------------------------
RUL_CAP = 125             # CONFIRMED via real Kaggle submission: capping is correct and
                          # necessary here - the hidden test ground truth is itself capped
                          # (probably near this value). An experiment training on uncapped
                          # targets scored 0.21 on the real leaderboard vs 0.879 with cap=125 -
                          # decisive, do not remove this again without strong new evidence.
N_EVAL_SEEDS = 10          # how many test-like cutoff assignments to average
ROLLING_WINDOWS = (5, 10, 20)

XGB_PARAMS = dict(
    n_estimators=3000,
    max_depth=5,
    learning_rate=0.015,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=1,  # Mac: >1 can collide with other OpenMP libs (segfault)
)
XGB_EARLY_STOPPING_ROUNDS = 100

DEPTH_SWEEP_VALUES = (4, 5, 6, 7, 8)
RUL_CAP_SWEEP_VALUES = (100, 110, 125, 140, 155, None)  # None = no cap at all
N_OPTUNA_TRIALS = 30
N_TUNING_EVAL_SEEDS = 3   # fewer seeds during tuning trials to keep them fast; full run re-checks with N_EVAL_SEEDS
WEIGHT_BANDWIDTH = 12.0    # Gaussian kernel width (in cycles) for test-similarity sample weighting
WEIGHT_FLOOR_FRAC = 0.05  # minimum weight relative to max, so no cycle range gets fully ignored

# Early specialist: a second model trained ONLY on short-history rows, since
# diagnose_error_by_cutoff showed this is by far the weakest region (R2~0.3-0.7
# there vs ~0.9+ for longer histories) - a single model has to compromise
# between "good at short" and "good at long", a dedicated specialist doesn't.
EARLY_GATE_MAX = 105       # test rows at or below this cutoff get the specialist blended in
EARLY_TRAIN_MAX = 150      # specialist trains on rows up to this cycle (a bit more than the
                          # gate range, so it has enough training rows to learn from)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(f"train: {train.shape}, test: {test.shape}")
    return train, test


def apply_rul_cap(df, cap=RUL_CAP):
    df = df.copy()
    if cap is not None:
        df["RUL"] = np.minimum(df["RUL"], cap)
    return df


def get_sensor_cols(train):
    tel_cols = [c for c in train.columns if c.startswith("telemetry")]
    stds = train[tel_cols].std()
    keep = stds[stds > 1e-6].index.tolist()
    print(f"Keeping {len(keep)}/{len(tel_cols)} non-constant sensors")
    return keep


def compute_sensor_tiers(train_raw, sensor_cols, strong_thresh=0.55, medium_thresh=0.30):
    """Ranks sensors by how strongly they correlate with RUL (row-level, Pearson +
    Spearman, take the stronger of the two). Sensors that barely relate to RUL get
    a much smaller feature set instead of the full 12-feature treatment - this is
    a data-driven reduction (grounded in actual RUL relevance), unlike the earlier
    sensor-to-sensor redundancy clustering attempt which barely removed anything
    and wasn't targeted at what actually matters for the target."""
    pearson = train_raw[sensor_cols].corrwith(train_raw["RUL"], method="pearson")
    spearman = train_raw[sensor_cols].corrwith(train_raw["RUL"], method="spearman")
    strength = pd.concat([pearson.abs(), spearman.abs()], axis=1).max(axis=1)

    strong = strength[strength >= strong_thresh].index.tolist()
    medium = strength[(strength >= medium_thresh) & (strength < strong_thresh)].index.tolist()
    weak = strength[strength < medium_thresh].index.tolist()

    print(f"Sensor tiers by |correlation to RUL|: strong={len(strong)}, "
          f"medium={len(medium)}, weak={len(weak)}")
    print(f"  strong (>= {strong_thresh}): {strong}")
    print(f"  medium ({medium_thresh}-{strong_thresh}): {medium}")
    print(f"  weak (< {medium_thresh}): {weak}")
    return {"strong": strong, "medium": medium, "weak": weak}


STRONG_FAMILIES = ("raw", "baseline_delta", "hist_mean", "hist_std",
                   "last_minus_hist_mean", "hist_slope", "roll5_mean", "roll5_std",
                   "roll10_mean", "roll10_std", "roll20_mean", "roll20_std")
MEDIUM_FAMILIES = ("raw", "baseline_delta", "hist_mean", "hist_std", "hist_slope",
                   "roll10_mean", "roll10_std")
WEAK_FAMILIES = ("raw", "hist_mean", "hist_std")


def compute_feature_weights(feature_cols, train_raw, sensor_cols, floor=0.15):
    """XGBoost's feature_weights parameter biases which features get considered
    at each split (via colsample_bytree/bynode sampling) - features with higher
    weight get included more often. This is the REAL mechanism for "more weight
    to better-correlated sensors": multiplying feature VALUES by a weight is a
    no-op for trees (they split on rank/threshold, not magnitude), so this is
    not just cosmetic - it changes what the model actually gets to consider.
    Each engineered column inherits its underlying sensor's |correlation to RUL|
    (floored so no sensor is ever fully excluded from consideration); non-sensor
    columns (flight_cycle etc.) get neutral weight 1.0."""
    pearson = train_raw[sensor_cols].corrwith(train_raw["RUL"], method="pearson")
    spearman = train_raw[sensor_cols].corrwith(train_raw["RUL"], method="spearman")
    strength = pd.concat([pearson.abs(), spearman.abs()], axis=1).max(axis=1)

    weights = []
    for col in feature_cols:
        sensor = col.split("__")[0]
        if sensor in strength.index:
            weights.append(max(strength[sensor], floor))
        else:
            weights.append(1.0)
    weights = np.array(weights, dtype=float)
    print(f"Feature weights: min={weights.min():.3f}, max={weights.max():.3f}, "
          f"mean={weights.mean():.3f} (floor={floor})")
    return weights


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def _slope(x):
    n = len(x)
    if n < 2:
        return pd.Series(np.zeros(n), index=x.index)
    idx = np.arange(1, n + 1)
    cs_x, cs_x2 = np.cumsum(idx), np.cumsum(idx**2)
    cs_y, cs_xy = np.cumsum(x.values), np.cumsum(idx * x.values)
    denom = (idx * cs_x2 - cs_x**2).astype(float)
    denom[denom == 0] = np.nan
    slope = (idx * cs_xy - cs_x * cs_y) / denom
    return pd.Series(slope, index=x.index)


def build_features(df, sensor_cols, windows=ROLLING_WINDOWS):
    """Original untiered version - kept for the reference/fallback model."""
    df = df.sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    g = df.groupby("uav_id", sort=False)
    feats = {"uav_id": df["uav_id"], "flight_cycle": df["flight_cycle"]}
    feats["flight_cycle_log"] = np.log1p(df["flight_cycle"])

    for c in sensor_cols:
        s = df[c]
        first_val = g[c].transform("first")
        feats[f"{c}__baseline_delta"] = s - first_val
        exp_mean = g[c].transform(lambda x: x.expanding().mean())
        exp_std = g[c].transform(lambda x: x.expanding().std())
        feats[f"{c}__hist_mean"] = exp_mean
        feats[f"{c}__hist_std"] = exp_std
        feats[f"{c}__last_minus_hist_mean"] = s - exp_mean
        feats[f"{c}__hist_slope"] = g[c].transform(_slope)
        for w in windows:
            feats[f"{c}__roll{w}_mean"] = g[c].transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            feats[f"{c}__roll{w}_std"] = g[c].transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std()
            )
        feats[c] = s

    return pd.DataFrame(feats).fillna(0.0)


def build_features_tiered(
    df,
    sensor_tiers,
    windows=ROLLING_WINDOWS,
    excluded_families=(),
):
    """Same feature mechanics as build_features, but each sensor only gets the
    feature families appropriate to its tier (strong/medium/weak RUL relevance).
    Cuts feature count substantially for the ~half of sensors that barely
    correlate with RUL at all, instead of spending 12 feature slots on each of
    them regardless of whether they carry any signal."""
    df = df.sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    g = df.groupby("uav_id", sort=False)
    feats = {"uav_id": df["uav_id"], "flight_cycle": df["flight_cycle"]}
    feats["flight_cycle_log"] = np.log1p(df["flight_cycle"])

    families_by_tier = {"strong": STRONG_FAMILIES, "medium": MEDIUM_FAMILIES, "weak": WEAK_FAMILIES}
    excluded_families = set(excluded_families)

    for tier, sensor_cols in sensor_tiers.items():
        families = tuple(
            family
            for family in families_by_tier[tier]
            if family not in excluded_families
        )
        for c in sensor_cols:
            s = df[c]
            exp_mean = g[c].transform(lambda x: x.expanding().mean()) if (
                "hist_mean" in families or "last_minus_hist_mean" in families
            ) else None

            if "raw" in families:
                feats[c] = s
            if "baseline_delta" in families:
                first_val = g[c].transform("first")
                feats[f"{c}__baseline_delta"] = s - first_val
            if "hist_mean" in families:
                feats[f"{c}__hist_mean"] = exp_mean
            if "hist_std" in families:
                feats[f"{c}__hist_std"] = g[c].transform(lambda x: x.expanding().std())
            if "last_minus_hist_mean" in families:
                feats[f"{c}__last_minus_hist_mean"] = s - exp_mean
            if "hist_slope" in families:
                feats[f"{c}__hist_slope"] = g[c].transform(_slope)
            for w in windows:
                if f"roll{w}_mean" in families:
                    feats[f"{c}__roll{w}_mean"] = g[c].transform(
                        lambda x, w=w: x.rolling(w, min_periods=1).mean()
                    )
                if f"roll{w}_std" in families:
                    feats[f"{c}__roll{w}_std"] = g[c].transform(
                        lambda x, w=w: x.rolling(w, min_periods=1).std()
                    )

    return pd.DataFrame(feats).fillna(0.0)


# --------------------------------------------------------------------------
# Test-like stable evaluation: reuse the real test set's cutoff lengths,
# randomly assigned to eligible training UAVs, averaged over several seeds.
# --------------------------------------------------------------------------
def get_test_cutoff_lengths(test_raw):
    """One cutoff length per test UAV = how many cycles it was truncated at."""
    return test_raw.groupby("uav_id")["flight_cycle"].max().to_numpy(dtype=int)


def _eligible_random_assignment(train_lifetimes, cutoffs, rng):
    """train_lifetimes: dict uav_id -> final_cycle. Assigns each cutoff (processed
    largest first, since large cutoffs are hardest to place) to a random eligible
    training UAV whose remaining life (final_cycle - 1) can support that cutoff."""
    available = list(train_lifetimes.items())
    assignments = []
    for cutoff in sorted(cutoffs, reverse=True):
        eligible = [(uid, fc) for uid, fc in available if fc - 1 >= cutoff]
        if not eligible:
            raise ValueError(
                f"No training UAV can support test-like cutoff {cutoff}. "
                "This means some test histories are longer than any available "
                "training UAV can support - check test/train lifetime ranges."
            )
        selected = eligible[int(rng.integers(0, len(eligible)))]
        available.remove(selected)
        assignments.append((selected[0], int(cutoff)))
    return assignments


def make_eval_set(train_feat, train_raw, test_cutoff_lengths, seed):
    lifetimes = train_raw.groupby("uav_id")["flight_cycle"].max().to_dict()
    rng = np.random.default_rng(seed)
    assignments = _eligible_random_assignment(lifetimes, test_cutoff_lengths, rng)

    rows = []
    for uid, cutoff in assignments:
        sub = train_feat[train_feat["uav_id_check"] == uid]
        row = sub[sub["flight_cycle"] == cutoff]
        if len(row) == 0:
            row = sub.iloc[[(sub["flight_cycle"] - cutoff).abs().values.argmin()]]
        rows.append(row.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_stable(train_feat, train_raw, test_cutoff_lengths, predict_fn, seeds=range(N_EVAL_SEEDS)):
    """predict_fn(eval_df) -> array of predictions. Returns mean R2 and per-seed R2s.
    Each seed = one full re-assignment of test-like cutoffs to training UAVs."""
    scores = []
    for seed in seeds:
        eval_df = make_eval_set(train_feat, train_raw, test_cutoff_lengths, seed)
        preds = predict_fn(eval_df)
        scores.append(r2_score(eval_df["RUL_raw"], preds))
    return float(np.mean(scores)), scores


def compute_test_similarity_weights(flight_cycles, test_cutoff_lengths,
                                     bandwidth=WEIGHT_BANDWIDTH, floor_frac=WEIGHT_FLOOR_FRAC):
    """Gaussian-kernel sample weight per training row: how close is this row's
    flight_cycle to one of the 100 REAL test cutoff lengths? Rows that look like
    a real test scenario get more training weight; rows at cycles the real test
    set never touches get down-weighted (but never to zero - floor_frac keeps a
    minimum weight so the model doesn't completely ignore any cycle range).
    Purely a function of flight_cycle (target-free), same idea as the teammate's
    Codex-generated 'weighted prefix' branch."""
    cycles = np.asarray(flight_cycles, dtype=float).reshape(-1, 1)
    cutoffs = np.asarray(test_cutoff_lengths, dtype=float).reshape(1, -1)
    kernel = np.exp(-0.5 * ((cycles - cutoffs) / bandwidth) ** 2)
    weight = kernel.sum(axis=1)
    weight = weight / weight.mean()
    floor = floor_frac * weight.max()
    return np.maximum(weight, floor)


def diagnose_error_by_cutoff(train_feat, train_raw, test_cutoff_lengths, predict_fn,
                              seeds=range(N_EVAL_SEEDS), n_buckets=4):
    """Pools (cutoff_length, abs_error) pairs across all eval seeds, then buckets
    by cutoff length (short vs. long sequences) to show WHERE the model struggles -
    early-life predictions (little degradation signal yet) usually hurt R2 most."""
    rows = []
    for seed in seeds:
        eval_df = make_eval_set(train_feat, train_raw, test_cutoff_lengths, seed)
        preds = predict_fn(eval_df)
        for cutoff, y_true, y_pred in zip(eval_df["flight_cycle"], eval_df["RUL_raw"], preds):
            rows.append((cutoff, y_true, y_pred))

    diag = pd.DataFrame(rows, columns=["cutoff", "y_true", "y_pred"])
    diag["abs_error"] = (diag["y_true"] - diag["y_pred"]).abs()
    diag["bucket"] = pd.qcut(diag["cutoff"], q=n_buckets, duplicates="drop")

    print(f"\nError breakdown by cutoff length (pooled over {len(list(seeds))} seeds, "
          f"n={len(diag)} eval rows):")
    for bucket, sub in diag.groupby("bucket", observed=True):
        bucket_r2 = r2_score(sub["y_true"], sub["y_pred"]) if len(sub) > 1 else float("nan")
        print(f"  cutoff {str(bucket):>18}: n={len(sub):4d}  "
              f"MAE={sub['abs_error'].mean():6.2f}  R2={bucket_r2:.4f}")
    return diag


# --------------------------------------------------------------------------
# XGBoost: direct on engineered features, no PCA, early stopping
# --------------------------------------------------------------------------
def fit_xgb_cv(train_feat, feature_cols, params=None, early_stopping_rounds=None, n_splits=5,
               verbose=True, sample_weight=None, feature_weights=None):
    """Returns fold models + the set of UAV groups each fold held out, so we
    can reuse them for evaluation on the stable test-like eval set. Accepts
    explicit params/early_stopping_rounds (used by Optuna tuning) - falls back
    to the module-level config if not given, so existing calls don't change.
    sample_weight (optional, same length as train_feat): applied to the
    TRAINING rows of each fold only - the held-out eval_set stays unweighted
    so early stopping still reflects genuine, unweighted generalization.
    feature_weights (optional, same length as feature_cols): biases which
    features get considered during column subsampling - NOT a per-value scale
    (trees are invariant to that); this changes what the model actually sees
    more or less often at each split."""
    if params is None:
        params = XGB_PARAMS
    if early_stopping_rounds is None:
        early_stopping_rounds = XGB_EARLY_STOPPING_ROUNDS

    X = train_feat[feature_cols].values
    y = train_feat["RUL_raw"].values
    groups = train_feat["uav_id_check"].values
    w = np.asarray(sample_weight) if sample_weight is not None else None
    fw = np.asarray(feature_weights) if feature_weights is not None else None

    gkf = GroupKFold(n_splits=n_splits)
    fold_models, fold_val_groups = [], []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds,
                                  feature_weights=fw)
        if verbose:
            print(f"  fold {len(fold_models)+1}/{n_splits}: fitting on {len(tr_idx)} rows...")
        model.fit(
            X[tr_idx], y[tr_idx],
            sample_weight=w[tr_idx] if w is not None else None,
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        if verbose:
            print(f"  fold {len(fold_models)+1}/{n_splits}: done, best_iteration={model.best_iteration}")
        fold_models.append(model)
        fold_val_groups.append(set(np.unique(groups[val_idx])))

    return fold_models, fold_val_groups


def xgb_predict_eval(eval_df, feature_cols, fold_models, fold_val_groups):
    preds = np.full(len(eval_df), np.nan)
    for model, val_groups in zip(fold_models, fold_val_groups):
        mask = eval_df["uav_id_check"].isin(val_groups).values
        if mask.any():
            preds[mask] = model.predict(eval_df.loc[mask, feature_cols].values)
    return preds



# --------------------------------------------------------------------------
# Standalone max_depth sweep for XGBoost. Not run by default (retrains 5
# folds per depth value) - run explicitly with --depth-sweep. Answers:
# "is XGBoost's capacity actually the bottleneck, or not?"
# --------------------------------------------------------------------------
def run_max_depth_sweep(train_feat, train_raw, feature_cols, test_cutoff_lengths,
                         depths=DEPTH_SWEEP_VALUES):
    print(f"\nRunning max_depth sweep over {depths} "
          f"(retrains {len(depths)} x 5-fold CV, this takes a while)...")
    results = []
    original_depth = XGB_PARAMS["max_depth"]
    try:
        for depth in depths:
            print(f"\n-- max_depth={depth} --")
            XGB_PARAMS["max_depth"] = depth
            fold_models, fold_val_groups = fit_xgb_cv(train_feat, feature_cols)
            mean_r2, scores = evaluate_stable(
                train_feat, train_raw, test_cutoff_lengths,
                lambda eval_df: xgb_predict_eval(eval_df, feature_cols, fold_models, fold_val_groups),
            )
            print(f"max_depth={depth}: stable R2 = {mean_r2:.4f} (std {np.std(scores):.4f})")
            results.append((depth, mean_r2, np.std(scores)))
    finally:
        XGB_PARAMS["max_depth"] = original_depth  # restore, in case sweep is followed by main()

    print("\nmax_depth sweep summary:")
    for depth, mean_r2, std in results:
        print(f"  depth={depth}: R2={mean_r2:.4f}  std={std:.4f}")
    best_depth = max(results, key=lambda r: r[1])[0]
    print(f"Best max_depth: {best_depth}")
    return results


# --------------------------------------------------------------------------
# Optuna hyperparameter tuning for XGBoost. Not run by default - run with
# --tune-xgb. Searches the space that matters most for tree models on
# tabular data with ~100 groups (depth, learning rate, subsampling,
# regularization). n_estimators stays fixed and high; early stopping decides
# the real tree count per trial/fold, same as everywhere else in this script.
# --------------------------------------------------------------------------
def tune_xgb_optuna(train_feat, train_raw, feature_cols, test_cutoff_lengths,
                     n_trials=N_OPTUNA_TRIALS):
    def objective(trial):
        params = dict(
            n_estimators=3000,
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            random_state=0,
            n_jobs=1,
        )
        fold_models, fold_val_groups = fit_xgb_cv(
            train_feat, feature_cols, params=params,
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS, verbose=False,
        )
        mean_r2, _ = evaluate_stable(
            train_feat, train_raw, test_cutoff_lengths,
            lambda eval_df: xgb_predict_eval(eval_df, feature_cols, fold_models, fold_val_groups),
            seeds=range(N_TUNING_EVAL_SEEDS),  # fewer seeds per trial to keep tuning fast
        )
        return mean_r2

    print(f"\nRunning Optuna tuning for XGBoost ({n_trials} trials, "
          f"{N_TUNING_EVAL_SEEDS} eval seeds per trial - this takes a while)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[lambda study, trial: print(
                       f"  trial {trial.number+1}/{n_trials}: R2={trial.value:.4f} "
                       f"(best so far: {study.best_value:.4f})"
                   )])

    print("\nBest params found:")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")
    print(f"Best trial R2 (on {N_TUNING_EVAL_SEEDS} eval seeds): {study.best_value:.4f}")

    print(f"\nRe-checking best params with the full {N_EVAL_SEEDS}-seed eval "
          f"(trial R2 above used only {N_TUNING_EVAL_SEEDS} seeds for speed)...")
    best_params = dict(study.best_params, n_estimators=3000, random_state=0, n_jobs=1)
    fold_models, fold_val_groups = fit_xgb_cv(train_feat, feature_cols, params=best_params)
    full_r2, full_scores = evaluate_stable(
        train_feat, train_raw, test_cutoff_lengths,
        lambda eval_df: xgb_predict_eval(eval_df, feature_cols, fold_models, fold_val_groups),
    )
    print(f"Full-seed stable R2 with best params: {full_r2:.4f} (std {np.std(full_scores):.4f})")
    print("\nCopy these into XGB_PARAMS in the config section to use them in the normal run:")
    print(f"  {best_params}")
    return study


# --------------------------------------------------------------------------
# RUL cap sweep. Not run by default - run with --rul-cap-sweep. Features
# don't depend on the cap (they're built from raw sensor values only), so we
# build the feature matrix ONCE and only swap the target column per cap
# value - much cheaper than rebuilding everything from scratch each time.
# --------------------------------------------------------------------------
def sweep_rul_cap(train_raw_uncapped, test, caps=RUL_CAP_SWEEP_VALUES):
    """IMPORTANT: always evaluates against the TRUE UNCAPPED ground truth, regardless
    of what cap the model was trained with. An earlier version of this function
    evaluated each cap against its own cap-matched "truth", which is circular and
    made every cap value look artificially fine locally (predicting a squashed
    target is easier when "correct" is defined by the same squashing). That
    version could not have detected that capping was actively hurting real
    (Kaggle) performance here - which it was, badly."""
    sensor_cols = get_sensor_cols(train_raw_uncapped)
    train_feat_base = build_features(train_raw_uncapped, sensor_cols)
    test_feat = build_features(test, sensor_cols)
    feature_cols = [c for c in train_feat_base.columns if c not in ("uav_id",)]
    test_cutoff_lengths = get_test_cutoff_lengths(test)

    true_uncapped_sorted = train_raw_uncapped.sort_values(["uav_id", "flight_cycle"]) \
        .reset_index(drop=True)["RUL"].values

    print(f"\nRunning RUL cap sweep over {caps} "
          f"(retrains 5-fold CV per cap value, this takes a while)...")
    results = []
    for cap in caps:
        cap_label = "no cap" if cap is None else str(cap)
        print(f"\n-- RUL_CAP={cap_label} (training target) --")
        train_raw_capped = train_raw_uncapped.copy()
        if cap is not None:
            train_raw_capped["RUL"] = np.minimum(train_raw_capped["RUL"], cap)
        train_raw_capped_sorted = train_raw_capped.sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)

        train_feat = train_feat_base.copy()
        train_feat["RUL_raw"] = train_raw_capped_sorted["RUL"].values  # what the model TRAINS on
        train_feat["uav_id_check"] = train_raw_capped_sorted["uav_id"].values

        fold_models, fold_val_groups = fit_xgb_cv(train_feat, feature_cols, verbose=False)

        # evaluate against the TRUE uncapped ground truth, not a cap-matched one
        train_feat_true = train_feat.copy()
        train_feat_true["RUL_raw"] = true_uncapped_sorted

        mean_r2, scores = evaluate_stable(
            train_feat_true, train_raw_uncapped, test_cutoff_lengths,
            lambda eval_df: xgb_predict_eval(eval_df, feature_cols, fold_models, fold_val_groups),
        )
        print(f"RUL_CAP={cap_label}: R2 vs TRUE uncapped ground truth = {mean_r2:.4f} (std {np.std(scores):.4f})")
        results.append((cap_label, mean_r2, np.std(scores)))

    print("\nRUL cap sweep summary (all evaluated against the same true uncapped ground truth):")
    for cap_label, mean_r2, std in results:
        print(f"  cap={cap_label}: R2={mean_r2:.4f}  std={std:.4f}")
    best_cap = max(results, key=lambda r: r[1])[0]
    print(f"Best RUL_CAP: {best_cap}")
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_full_pipeline(train_feat, test_feat, feature_cols, train_raw, test_cutoff_lengths,
                       sample_weight=None, feature_weights=None, label=""):
    """XGBoost-only routine (v13 = v11 minus the CatBoost half of the ensemble):
    CV fit, stable eval, final fit, test prediction. No blend step needed since
    there's only one model."""
    print(f"\n=== {label}: fitting XGBoost (5-fold GroupKFold, early stopping) ===")
    xgb_models, xgb_val_groups = fit_xgb_cv(train_feat, feature_cols, sample_weight=sample_weight,
                                             feature_weights=feature_weights)
    xgb_mean_r2, xgb_scores = evaluate_stable(
        train_feat, train_raw, test_cutoff_lengths,
        lambda eval_df: xgb_predict_eval(eval_df, feature_cols, xgb_models, xgb_val_groups),
    )
    print(f"{label}: XGBoost stable R2 = {xgb_mean_r2:.4f} (std {np.std(xgb_scores):.4f})")

    all_uav_ids = train_feat["uav_id_check"].unique()
    rng = np.random.RandomState(0)
    n_holdout_uavs = max(1, int(len(all_uav_ids) * 0.1))
    holdout_uav_ids = set(rng.choice(all_uav_ids, n_holdout_uavs, replace=False))
    holdout_mask = train_feat["uav_id_check"].isin(holdout_uav_ids).values
    holdout_idx = np.where(holdout_mask)[0]
    fit_idx = np.where(~holdout_mask)[0]

    X_full = train_feat[feature_cols].values
    y_full = train_feat["RUL_raw"].values
    w_full = np.asarray(sample_weight) if sample_weight is not None else None

    print(f"\n{label}: fitting final XGBoost on all training data...")
    xgb_final = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                                  feature_weights=np.asarray(feature_weights) if feature_weights is not None else None)
    xgb_final.fit(
        X_full[fit_idx], y_full[fit_idx],
        sample_weight=w_full[fit_idx] if w_full is not None else None,
        eval_set=[(X_full[holdout_idx], y_full[holdout_idx])],
        verbose=False,
    )
    print(f"{label}: final XGBoost best_iteration={xgb_final.best_iteration}")

    test_last = test_feat.sort_values("flight_cycle").groupby("uav_id").tail(1).reset_index(drop=True)
    test_pred = xgb_final.predict(test_last[feature_cols].values)

    submission = pd.DataFrame({"id": test_last["uav_id"], "RUL": test_pred})
    extras = {
        "xgb_final": xgb_final, "test_last": test_last, "test_pred": test_pred,
    }
    return submission, xgb_mean_r2, extras


def fit_early_specialist(train_feat, feature_cols, sample_weight=None, early_train_max=EARLY_TRAIN_MAX):
    """Trains an XGBoost model ONLY on rows with flight_cycle <= early_train_max -
    the region diagnose_error_by_cutoff flagged as by far the weakest (R2~0.3-0.7).
    A single model has to compromise between doing well there and doing well on
    long histories; this specialist only has to be good at the short-history
    case. Uses a UAV-grouped holdout (not random rows) for early stopping, same
    leakage precaution as the main final fit."""
    mask = train_feat["flight_cycle"] <= early_train_max
    early_feat = train_feat[mask].reset_index(drop=True)
    early_weight = np.asarray(sample_weight)[mask.values] if sample_weight is not None else None
    print(f"Early specialist: training on {len(early_feat)} rows "
          f"(flight_cycle <= {early_train_max}), {early_feat['uav_id_check'].nunique()} UAVs")

    all_uav_ids = early_feat["uav_id_check"].unique()
    rng = np.random.RandomState(1)
    n_holdout = max(1, int(len(all_uav_ids) * 0.15))
    holdout_ids = set(rng.choice(all_uav_ids, n_holdout, replace=False))
    holdout_mask = early_feat["uav_id_check"].isin(holdout_ids).values

    X = early_feat[feature_cols].values
    y = early_feat["RUL_raw"].values
    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
    model.fit(
        X[~holdout_mask], y[~holdout_mask],
        sample_weight=early_weight[~holdout_mask] if early_weight is not None else None,
        eval_set=[(X[holdout_mask], y[holdout_mask])],
        verbose=False,
    )
    print(f"Early specialist: best_iteration={model.best_iteration}")
    return model


def find_early_gate_weight(train_feat, train_raw, test_cutoff_lengths, feature_cols,
                            main_predict_fn, early_model, gate_max=EARLY_GATE_MAX, n_seeds=8):
    """Pools eval rows with flight_cycle <= gate_max across several test-like eval
    draws (need to pool since there are few such rows per draw), then grid-searches
    the blend weight between the main ensemble and the early specialist that
    maximizes R2 on just this short-history subset."""
    rows_true, rows_main, rows_early = [], [], []
    for seed in range(n_seeds):
        eval_df = make_eval_set(train_feat, train_raw, test_cutoff_lengths, seed)
        sub = eval_df[eval_df["flight_cycle"] <= gate_max]
        if len(sub) == 0:
            continue
        rows_true.append(sub["RUL_raw"].values)
        rows_main.append(main_predict_fn(sub))
        rows_early.append(early_model.predict(sub[feature_cols].values))

    y_true = np.concatenate(rows_true)
    main_pred = np.concatenate(rows_main)
    early_pred = np.concatenate(rows_early)

    best_gate_w, best_r2 = 0.0, r2_score(y_true, main_pred)  # gate_w=0 -> pure main, our fallback
    for gate_w in np.arange(0, 1.01, 0.05):
        blend = (1 - gate_w) * main_pred + gate_w * early_pred
        r2 = r2_score(y_true, blend)
        if r2 > best_r2:
            best_r2, best_gate_w = r2, gate_w
    print(f"Early specialist gate: best weight (specialist share) = {best_gate_w:.2f}, "
          f"R2 on short-history subset (n={len(y_true)}) = {best_r2:.4f} "
          f"(main ensemble alone scored {r2_score(y_true, main_pred):.4f} there)")
    return best_gate_w


def main():
    train_raw, test = load_data()
    train_raw = apply_rul_cap(train_raw, RUL_CAP)
    all_sensor_cols = get_sensor_cols(train_raw)
    test_cutoff_lengths = get_test_cutoff_lengths(test)
    print(f"Test-like cutoff lengths: n={len(test_cutoff_lengths)}, "
          f"min={test_cutoff_lengths.min()}, max={test_cutoff_lengths.max()}")

    # ================================================================
    # CANDIDATE ensemble: sensor-tiered features + test-similarity sample
    # weights + RUL-correlation feature weights (XGBoost only).
    # ================================================================
    sensor_tiers = compute_sensor_tiers(train_raw, all_sensor_cols)
    train_feat = build_features_tiered(train_raw, sensor_tiers)
    train_feat["RUL_raw"] = train_raw["RUL"].values
    train_feat["uav_id_check"] = train_raw["uav_id"].values
    test_feat = build_features_tiered(test, sensor_tiers)
    feature_cols = [c for c in train_feat.columns if c not in ("uav_id", "RUL_raw", "uav_id_check")]
    print(f"\nCandidate (tiered) feature matrix: {train_feat[feature_cols].shape}")

    sample_weight = compute_test_similarity_weights(train_feat["flight_cycle"], test_cutoff_lengths)
    print(f"Sample weights: min={sample_weight.min():.3f}, max={sample_weight.max():.3f}, "
          f"mean={sample_weight.mean():.3f}")
    feature_weights = compute_feature_weights(feature_cols, train_raw, all_sensor_cols)

    submission_no_early, xgb_r2, extras = run_full_pipeline(
        train_feat, test_feat, feature_cols, train_raw, test_cutoff_lengths,
        sample_weight=sample_weight, feature_weights=feature_weights,
        label="CANDIDATE (tiered+weighted+feature_weights, XGBoost only)",
    )
    submission_no_early.to_csv("submission_no_early.csv", index=False)
    print("\nSaved submission_no_early.csv (candidate, no early specialist - safety net)")

    # ================================================================
    # EARLY SPECIALIST: dedicated model for short-history rows, blended in
    # only for test UAVs at or below EARLY_GATE_MAX cycles.
    # ================================================================
    print(f"\n=== Fitting early specialist (rows with flight_cycle <= {EARLY_TRAIN_MAX}) ===")
    early_model = fit_early_specialist(train_feat, feature_cols, sample_weight=sample_weight)

    def main_ensemble_predict(eval_df):
        return extras["xgb_final"].predict(eval_df[feature_cols].values)

    gate_w = find_early_gate_weight(
        train_feat, train_raw, test_cutoff_lengths, feature_cols,
        main_ensemble_predict, early_model,
    )

    test_last = extras["test_last"]
    main_test_pred = extras["test_pred"]
    early_test_pred = early_model.predict(test_last[feature_cols].values)
    gated_mask = (test_last["flight_cycle"] <= EARLY_GATE_MAX).values
    final_pred = main_test_pred.copy()
    final_pred[gated_mask] = (
        (1 - gate_w) * main_test_pred[gated_mask] + gate_w * early_test_pred[gated_mask]
    )
    n_gated = gated_mask.sum()
    print(f"\nApplied early specialist to {n_gated}/{len(test_last)} test UAVs "
          f"(flight_cycle <= {EARLY_GATE_MAX}), gate weight = {gate_w:.2f}")

    submission = pd.DataFrame({"id": test_last["uav_id"], "RUL": final_pred})
    submission.to_csv("submission.csv", index=False)
    print("\nSaved submission.csv (candidate + early specialist - primary)")

    print("\n=== SUMMARY ===")
    print(f"Candidate (XGBoost only) R2: {xgb_r2:.4f}")
    print(f"Early specialist gate weight: {gate_w:.2f} (0.00 means specialist didn't help "
          f"and submission.csv == submission_no_early.csv)")
    print("\nIMPORTANT: local R2 has repeatedly NOT predicted real Kaggle direction in this "
          "project. Submit BOTH files if you have submissions to spare, and trust the real "
          "score - not just the local numbers above - before picking one as final.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-sweep", action="store_true",
                         help="Only run the max_depth comparison for XGBoost, skip the normal pipeline.")
    parser.add_argument("--tune-xgb", action="store_true",
                         help="Only run Optuna hyperparameter tuning for XGBoost, skip the normal pipeline.")
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS,
                         help=f"Number of Optuna trials for --tune-xgb (default {N_OPTUNA_TRIALS}).")
    parser.add_argument("--rul-cap-sweep", action="store_true",
                         help="Only run the RUL cap comparison, skip the normal pipeline.")
    args = parser.parse_args()

    if args.depth_sweep:
        train_raw, test = load_data()
        train_raw = apply_rul_cap(train_raw, RUL_CAP)
        sensor_cols = get_sensor_cols(train_raw)
        train_feat = build_features(train_raw, sensor_cols)
        train_feat["RUL_raw"] = train_raw["RUL"].values
        train_feat["uav_id_check"] = train_raw["uav_id"].values
        feature_cols = [c for c in train_feat.columns if c not in ("uav_id", "RUL_raw", "uav_id_check")]
        test_cutoff_lengths = get_test_cutoff_lengths(test)
        run_max_depth_sweep(train_feat, train_raw, feature_cols, test_cutoff_lengths)
    elif args.tune_xgb:
        train_raw, test = load_data()
        train_raw = apply_rul_cap(train_raw, RUL_CAP)
        sensor_cols = get_sensor_cols(train_raw)
        train_feat = build_features(train_raw, sensor_cols)
        train_feat["RUL_raw"] = train_raw["RUL"].values
        train_feat["uav_id_check"] = train_raw["uav_id"].values
        feature_cols = [c for c in train_feat.columns if c not in ("uav_id", "RUL_raw", "uav_id_check")]
        test_cutoff_lengths = get_test_cutoff_lengths(test)
        tune_xgb_optuna(train_feat, train_raw, feature_cols, test_cutoff_lengths, n_trials=args.n_trials)
    elif args.rul_cap_sweep:
        train_raw_uncapped, test = load_data()
        sweep_rul_cap(train_raw_uncapped, test)
    else:
        main()
