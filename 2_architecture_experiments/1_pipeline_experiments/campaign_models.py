"""Fit-local weight normalization and complete training-side model selection for PE_31."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import io
from contextlib import redirect_stdout
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from campaign_data import weight_data, seeded_source
from confirmation_utils import select_uavs
from feature_comparison_models import fit_estimator, make_estimator, fit_run7
from followup_experiment_utils import input_path


def training_endpoints(dev, ids):
    selected = select_uavs(dev, ids)
    mask = selected.metadata.suite.eq("historical").to_numpy()
    from advanced_r2_utils import subset_dataset
    return subset_dataset(selected, mask)


def fit_candidate(training, calibration, held, source, policy, recipe, model_seed):
    ids = set(training.metadata.uav_id)
    calibration = training_endpoints(calibration, ids)
    if ids & set(held.metadata.uav_id):
        raise ValueError("Outer-held UAV leaked into model fitting")
    wf = seeded_source(source, model_seed, recipe)
    families = ["xgboost", "catboost"] if recipe["family"] == "blend" else [recipe["family"]]
    settings, cap = wf["simple"], float(wf["target_cap"])
    groups = calibration.metadata.uav_id.to_numpy()
    folds = GroupKFold(n_splits=int(wf["inner_fold_count"])).split(calibration.features, groups=groups)
    oof = np.full((len(calibration), len(families)), np.nan)
    iterations = {f: [] for f in families}; audits = []; fit_count = 0
    started = time.perf_counter()

    def train_model(family, data, rounds=None, stop=None):
        nonlocal fit_count
        data = weight_data(data, policy)
        if family == "extra_trees":
            regular = recipe["capacity"] == "regularized"
            model = ExtraTreesRegressor(n_estimators=source.get("campaign_extra_trees_estimators", 400),
                max_depth=12 if regular else None, min_samples_leaf=5 if regular else 1,
                max_features=.8, random_state=int(model_seed), n_jobs=1)
            model.fit(data.features, np.minimum(data.target, cap), sample_weight=data.sample_weights)
        else:
            model = make_estimator(family, settings, rounds, stopping=stop is not None)
            model = fit_estimator(model, family, data, cap, weight_data(stop, policy) if stop is not None else None)
        fit_count += 1
        return model

    for fold, (_, validation_idx) in enumerate(folds):
        validation_ids = set(groups[validation_idx]); available = ids - validation_ids
        rng = np.random.default_rng(np.random.SeedSequence([int(settings["selection_seed"]), fold]))
        stop_count = max(1, int(np.ceil(len(available) * settings["stopping_fraction"])))
        stopping_ids = set(rng.choice(sorted(available), stop_count, replace=False))
        fit_ids = available - stopping_ids
        if len(fit_ids) < 2:
            raise ValueError("Too few UAVs for nested early stopping")
        detail = {"fit_uavs": sorted(fit_ids), "stopping_uavs": sorted(stopping_ids), "oof_uavs": sorted(validation_ids)}
        for column, family in enumerate(families):
            count = None
            if family != "extra_trees":
                tuner = train_model(family, select_uavs(training, fit_ids), stop=select_uavs(training, stopping_ids))
                count = int(tuner.best_iteration if family == "xgboost" else tuner.get_best_iteration()) + 1
                iterations[family].append(count)
                del tuner
            model = train_model(family, select_uavs(training, available), count)
            oof[validation_idx, column] = model.predict(calibration.features.iloc[validation_idx])
            del model
        audits.append(detail)
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete training-side predictions")
    weights = 1. / calibration.metadata.groupby("uav_id").uav_id.transform("size").to_numpy(float)
    blend = 1.
    if len(families) == 2:
        blend = min((float(np.average((np.maximum(w*oof[:, 0]+(1-w)*oof[:, 1], 0)-calibration.target.to_numpy())**2,
                                     weights=weights)), float(w)) for w in settings["blend_weights"])[1]
    predictions = []
    final_rounds = {}
    for family in families:
        count = max(1, int(np.floor(np.median(iterations[family])+.5))) if iterations[family] else None
        final_rounds[family] = count
        model = train_model(family, training, count)
        predictions.append(model.predict(held.features)); del model
    prediction = predictions[0] if len(predictions) == 1 else blend*predictions[0]+(1-blend)*predictions[1]
    normalized = weight_data(training, policy)
    return np.maximum(prediction, 0), {"calibration_uavs": sorted(ids), "inner_selection": audits,
        "base_estimator_fits": fit_count, "fit_seconds": time.perf_counter()-started,
        "xgboost_weight": blend if len(families) == 2 else None, "final_iterations": final_rounds,
        "weight_sum": float(normalized.sample_weights.sum()), "weight_min": float(normalized.sample_weights.min()),
        "weight_max": float(normalized.sample_weights.max()), "recipe": recipe, "policy": policy, "model_seed": model_seed}


def reference_protocol(raw, training_ids, held, source, model_seed):
    """Original fitting/feature-order choices inside an outer split; exact replay is separate."""
    path = input_path(source["reference_implementation"])
    spec = importlib.util.spec_from_file_location("pe31_original_reference", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.XGB_PARAMS["random_state"] = int(model_seed)
    module.CATBOOST_PARAMS["random_seed"] = int(model_seed)
    # Suppress shared cwd logs in parallel benchmark workers; fitting is unchanged.
    module.CATBOOST_PARAMS["allow_writing_files"] = False
    training_raw = raw.loc[raw.uav_id.isin(training_ids)].copy()
    if set(training_ids) & set(held.metadata.uav_id):
        raise ValueError("Reference protocol includes outer-held UAVs")
    started = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        sensors = module.get_sensor_cols(training_raw)
        capped = module.apply_rul_cap(training_raw)
        features = module.build_features(capped, sensors)
        labels = capped.sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
        features["RUL_raw"] = labels.RUL; features["uav_id_check"] = labels.uav_id
        columns = [c for c in features if c not in ("uav_id", "RUL_raw", "uav_id_check")]
        xgb, xids = module.fit_xgb_cv(features, columns)
        cat, cids = module.fit_catboost_cv(features, columns)
        # A single capped-label draw restricted to the active training UAVs.
        rng = np.random.default_rng(1000)
        rows = []
        for uid, group in features.groupby("uav_id_check", sort=True):
            valid = group.loc[group.flight_cycle.lt(group.flight_cycle.max())]
            rows.append(valid.iloc[int(rng.integers(len(valid)))])
        calibration = pd.DataFrame(rows).reset_index(drop=True)
        xp = module.xgb_predict_eval(calibration, columns, xgb, xids)
        cp = module.catboost_predict_eval(calibration, columns, cat, cids)
        w = min((float(np.mean((a*xp+(1-a)*cp-calibration.RUL_raw.to_numpy())**2)), float(a))
                for a in np.arange(0, 1.01, .05))[1]
        del xgb, cat
        ids = features.uav_id_check.unique()
        stopping = set(np.random.RandomState(0).choice(ids, max(1, int(len(ids)*.1)), replace=False))
        mask = features.uav_id_check.isin(stopping).to_numpy()
        x = module.xgb.XGBRegressor(**module.XGB_PARAMS, early_stopping_rounds=module.XGB_EARLY_STOPPING_ROUNDS)
        c = module.cb.CatBoostRegressor(**module.CATBOOST_PARAMS, early_stopping_rounds=module.CATBOOST_EARLY_STOPPING_ROUNDS)
        matrix, y = features[columns].values, features.RUL_raw.values
        x.fit(matrix[~mask], y[~mask], eval_set=[(matrix[mask], y[mask])], verbose=False)
        c.fit(matrix[~mask], y[~mask], eval_set=(matrix[mask], y[mask]))
        evaluation = module.build_features(raw.loc[raw.uav_id.isin(set(held.metadata.uav_id))], sensors)
        evaluation = evaluation.set_index(["uav_id", "flight_cycle"], drop=False)
        query = evaluation.loc[list(zip(held.metadata.uav_id, held.metadata.cutoff)), columns].values
        prediction = w*x.predict(query)+(1-w)*c.predict(query)
    return prediction, {"calibration_uavs": sorted(training_ids), "final_stopping_uavs": sorted(stopping),
        "final_fit_uavs": sorted(set(ids)-stopping), "xgboost_weight": w, "base_estimator_fits": 12,
        "fit_seconds": time.perf_counter()-started, "original_feature_order": columns,
        "protocol_adaptation": "Outer-training-only single uniform feasible-cutoff draw; exact empirical-cutoff replay is separate.",
        "internal_calibration_labels_capped": True, "outer_metrics_use_raw_labels": True}
