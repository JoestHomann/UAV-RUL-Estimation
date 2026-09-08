"""
UAV RUL Estimation - v4 (XGBoost + CatBoost ensemble)

Changes vs. v3:
  - Added CatBoost as a second, independently-tuned tree model. Same CV
    structure and same test-like stable eval as XGBoost, then a blend
    weight search decides how much each model contributes (could end up
    100% one model again - that's fine, the search will tell us).
  - Added a standalone max_depth sweep (run with --depth-sweep) to check
    whether XGBoost's capacity is actually the bottleneck before assuming
    it is. Not run by default since it retrains 5 folds per depth value.

Why this over just cranking up n_estimators/max_depth blindly: the
per-cutoff-length diagnostic (see diagnose_error_by_cutoff) showed R2 is
much worse for short sequences (early life, low degradation signal) than
long ones. That's an information problem, not obviously a capacity
problem - so before assuming "bigger model", we compare a second, very
different tree algorithm (CatBoost's ordered boosting + symmetric trees)
side by side, and separately test whether depth alone moves the needle.

Everything else (RUL capping, feature engineering, test-like stable
evaluation reusing real test cutoff lengths) is unchanged from v3.

Usage:
    python uav_rul_pipeline_v4.py                 # normal run: fit, blend, submit
    python uav_rul_pipeline_v4.py --depth-sweep    # only run the max_depth comparison
"""

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

try:
    import catboost as cb
except ImportError as e:
    raise ImportError(
        "CatBoost is not installed. Install it with: pip install catboost"
    ) from e

# --------------------------------------------------------------------------
# Config - tweak these first before touching the code below
# --------------------------------------------------------------------------
RUL_CAP = 125             # piecewise RUL cap; try 100-135 and see what helps
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

CATBOOST_PARAMS = dict(
    iterations=3000,
    depth=6,
    learning_rate=0.02,
    l2_leaf_reg=3.0,
    subsample=0.8,
    random_seed=0,
    thread_count=1,   # same Mac OpenMP caution as XGBoost
    verbose=False,
)
CATBOOST_EARLY_STOPPING_ROUNDS = 100

DEPTH_SWEEP_VALUES = (4, 5, 6, 7, 8)


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
    df["RUL"] = np.minimum(df["RUL"], cap)
    return df


def get_sensor_cols(train):
    tel_cols = [c for c in train.columns if c.startswith("telemetry")]
    stds = train[tel_cols].std()
    keep = stds[stds > 1e-6].index.tolist()
    print(f"Keeping {len(keep)}/{len(tel_cols)} non-constant sensors")
    return keep


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
def fit_xgb_cv(train_feat, feature_cols, n_splits=5):
    """Returns fold models + the set of UAV groups each fold held out, so we
    can reuse them for evaluation on the stable test-like eval set."""
    X = train_feat[feature_cols].values
    y = train_feat["RUL_raw"].values
    groups = train_feat["uav_id_check"].values

    gkf = GroupKFold(n_splits=n_splits)
    fold_models, fold_val_groups = [], []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
        print(f"  fold {len(fold_models)+1}/{n_splits}: fitting on {len(tr_idx)} rows...")
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
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
# CatBoost: same CV structure as XGBoost, independent model/hyperparameters
# --------------------------------------------------------------------------
def fit_catboost_cv(train_feat, feature_cols, n_splits=5):
    X = train_feat[feature_cols].values
    y = train_feat["RUL_raw"].values
    groups = train_feat["uav_id_check"].values

    gkf = GroupKFold(n_splits=n_splits)
    fold_models, fold_val_groups = [], []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = cb.CatBoostRegressor(
            **CATBOOST_PARAMS, early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS
        )
        print(f"  fold {len(fold_models)+1}/{n_splits}: fitting on {len(tr_idx)} rows...")
        model.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]))
        print(f"  fold {len(fold_models)+1}/{n_splits}: done, best_iteration={model.get_best_iteration()}")
        fold_models.append(model)
        fold_val_groups.append(set(np.unique(groups[val_idx])))

    return fold_models, fold_val_groups


def catboost_predict_eval(eval_df, feature_cols, fold_models, fold_val_groups):
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
# Main
# --------------------------------------------------------------------------
def main():
    train_raw, test = load_data()
    train_raw = apply_rul_cap(train_raw, RUL_CAP)
    sensor_cols = get_sensor_cols(train_raw)

    train_feat = build_features(train_raw, sensor_cols)
    train_feat["RUL_raw"] = train_raw["RUL"].values
    train_feat["uav_id_check"] = train_raw["uav_id"].values
    test_feat = build_features(test, sensor_cols)

    feature_cols = [c for c in train_feat.columns if c not in ("uav_id", "RUL_raw", "uav_id_check")]
    print("Feature matrix:", train_feat[feature_cols].shape)

    test_cutoff_lengths = get_test_cutoff_lengths(test)
    print(f"Test-like cutoff lengths: n={len(test_cutoff_lengths)}, "
          f"min={test_cutoff_lengths.min()}, max={test_cutoff_lengths.max()}")

    # ---- XGBoost: CV fit + stable multi-seed eval ----
    print("\nFitting XGBoost (5-fold GroupKFold, early stopping)...")
    xgb_models, xgb_val_groups = fit_xgb_cv(train_feat, feature_cols)
    xgb_mean_r2, xgb_scores = evaluate_stable(
        train_feat, train_raw, test_cutoff_lengths,
        lambda eval_df: xgb_predict_eval(eval_df, feature_cols, xgb_models, xgb_val_groups),
    )
    print(f"\nXGBoost stable R2 (mean over {N_EVAL_SEEDS} test-like seeds): "
          f"{xgb_mean_r2:.4f}  (std {np.std(xgb_scores):.4f})")

    diagnose_error_by_cutoff(
        train_feat, train_raw, test_cutoff_lengths,
        lambda eval_df: xgb_predict_eval(eval_df, feature_cols, xgb_models, xgb_val_groups),
    )

    # ---- CatBoost: CV fit + stable multi-seed eval ----
    print("\nFitting CatBoost (5-fold GroupKFold, early stopping)...")
    cat_models, cat_val_groups = fit_catboost_cv(train_feat, feature_cols)
    cat_mean_r2, cat_scores = evaluate_stable(
        train_feat, train_raw, test_cutoff_lengths,
        lambda eval_df: catboost_predict_eval(eval_df, feature_cols, cat_models, cat_val_groups),
    )
    print(f"\nCatBoost stable R2 (mean over {N_EVAL_SEEDS} test-like seeds): "
          f"{cat_mean_r2:.4f}  (std {np.std(cat_scores):.4f})")

    # ---- Blend weight search on ONE representative test-like draw ----
    eval_df = make_eval_set(train_feat, train_raw, test_cutoff_lengths, seed=1000)
    eval_df["xgb_pred"] = xgb_predict_eval(eval_df, feature_cols, xgb_models, xgb_val_groups)
    eval_df["cat_pred"] = catboost_predict_eval(eval_df, feature_cols, cat_models, cat_val_groups)

    best_w, best_r2 = None, -999
    for w in np.arange(0, 1.01, 0.05):
        blend = w * eval_df["xgb_pred"] + (1 - w) * eval_df["cat_pred"]
        r2 = r2_score(eval_df["RUL_raw"], blend)
        if r2 > best_r2:
            best_r2, best_w = r2, w
    print(f"\nBest blend weight (XGBoost share): {best_w:.2f}, R2 on this draw = {best_r2:.4f}")
    print(f"  XGBoost stable mean:  {xgb_mean_r2:.4f}")
    print(f"  CatBoost stable mean: {cat_mean_r2:.4f}")

    # ---- Final fit on ALL training data ----
    print("\nFitting final XGBoost on all training data...")
    X_full = train_feat[feature_cols].values
    y_full = train_feat["RUL_raw"].values
    # small internal holdout just for early stopping on the final fit.
    # IMPORTANT: must hold out whole UAVs, not random rows - otherwise rows
    # from the same UAV leak between "train" and "holdout" here (consecutive
    # cycles of one UAV look nearly identical after feature engineering),
    # early stopping becomes overly optimistic, and the final model trains
    # far longer than it should (silently overfitting).
    all_uav_ids = train_feat["uav_id_check"].unique()
    rng = np.random.RandomState(0)
    n_holdout_uavs = max(1, int(len(all_uav_ids) * 0.1))
    holdout_uav_ids = set(rng.choice(all_uav_ids, n_holdout_uavs, replace=False))
    holdout_mask = train_feat["uav_id_check"].isin(holdout_uav_ids).values
    holdout_idx = np.where(holdout_mask)[0]
    fit_idx = np.where(~holdout_mask)[0]

    xgb_final = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
    xgb_final.fit(
        X_full[fit_idx], y_full[fit_idx],
        eval_set=[(X_full[holdout_idx], y_full[holdout_idx])],
        verbose=False,
    )
    print(f"Final XGBoost best_iteration={xgb_final.best_iteration}")

    print("Fitting final CatBoost on all training data...")
    cat_final = cb.CatBoostRegressor(**CATBOOST_PARAMS, early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS)
    cat_final.fit(X_full[fit_idx], y_full[fit_idx], eval_set=(X_full[holdout_idx], y_full[holdout_idx]))
    print(f"Final CatBoost best_iteration={cat_final.get_best_iteration()}")

    test_last = test_feat.sort_values("flight_cycle").groupby("uav_id").tail(1).reset_index(drop=True)
    xgb_test_pred = xgb_final.predict(test_last[feature_cols].values)
    cat_test_pred = cat_final.predict(test_last[feature_cols].values)
    test_pred = best_w * xgb_test_pred + (1 - best_w) * cat_test_pred

    submission = pd.DataFrame({"id": test_last["uav_id"], "RUL": test_pred})
    submission.to_csv("submission.csv", index=False)
    print("\nSaved submission.csv with", len(submission), "rows")
    print(submission.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-sweep", action="store_true",
                         help="Only run the max_depth comparison for XGBoost, skip the normal pipeline.")
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
    else:
        main()
