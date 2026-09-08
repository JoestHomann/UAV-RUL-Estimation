"""Training-side calibration and early stopping for the two PE_28 recipes."""

import time
from types import MethodType
import numpy as np
from sklearn.model_selection import GroupKFold

from confirmation_utils import select_uavs
from followup_experiment_utils import input_path
from advanced_r2_utils import equal_uav_weights


def training_calibration(training, calibration):
    selected = select_uavs(calibration, set(training.metadata.uav_id))
    if list(training.features) != list(selected.features):
        raise ValueError("Calibration representation differs from model inputs")
    return selected


def fit_run7(training, calibration, held, workflow):
    from model_registry import ModelAdapterFactory
    from no_op_training_monitor import NoOpTrainingMonitor
    model = ModelAdapterFactory(input_path(workflow["specification"])).create(
        "residual_corrected_tree_ensemble", {"ensemble_contract_path": workflow["source_contract"]},
        seed=int(workflow["model_seed"]), allow_disabled=True, training_monitor=NoOpTrainingMonitor())
    if model.target_policy.mode != "piecewise_cap" or model.target_policy.maximum_rul != workflow["target_cap"]:
        raise ValueError("Run 7 fitting policy differs from PE_28")
    if model.internal_folds != workflow["inner_fold_count"]:
        raise ValueError("Run 7 internal calibration fold count differs")
    missing = set(model.residual_features) - set(training.features)
    if missing:
        raise ValueError(f"Residual features missing from representation: {missing}")
    selected = training_calibration(training, calibration)
    if set(selected.metadata.uav_id) & set(held.metadata.uav_id):
        raise ValueError("Outer held UAVs leaked into calibration")
    # Override only the calibration source. All model components and policies stay unchanged.
    model._calibration_data = MethodType(lambda self, data: training_calibration(data, selected), model)
    started = time.perf_counter()
    model.fit(training, None)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(held)
    return prediction, {"fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - started,
        "calibration_uavs": sorted(map(str, selected.metadata.uav_id.unique())),
        "xgboost_weight": float(model.xgboost_weight),
        "base_estimator_fits": 2 * len(model.member_seeds) * (model.internal_folds + 1)}


def make_estimator(family, settings, iterations=None, stopping=False):
    if family == "xgboost":
        from xgboost import XGBRegressor
        from models.tabular.xgboost import resolve_xgboost_device
        params = dict(settings["xgboost"])
        params["device"] = resolve_xgboost_device(params.get("device", "auto"))
        if iterations is not None:
            params["n_estimators"] = iterations
        if stopping:
            params["early_stopping_rounds"] = settings["early_stopping_rounds"]
        return XGBRegressor(**params)
    from catboost import CatBoostRegressor
    params = dict(settings["catboost"])
    if iterations is not None:
        params["iterations"] = iterations
    if stopping:
        params["early_stopping_rounds"] = settings["early_stopping_rounds"]
    return CatBoostRegressor(**params)


def fit_estimator(model, family, training, cap, stopping=None):
    y = np.minimum(training.target.to_numpy(float), cap)
    kwargs = {"sample_weight": training.sample_weights.to_numpy(float)}
    if stopping is not None:
        stop_y = np.minimum(stopping.target.to_numpy(float), cap)
        if family == "xgboost":
            kwargs.update(eval_set=[(stopping.features, stop_y)],
                sample_weight_eval_set=[stopping.sample_weights.to_numpy(float)], verbose=False)
        else:
            from catboost import Pool
            kwargs["eval_set"] = Pool(stopping.features, stop_y, weight=stopping.sample_weights.to_numpy(float))
    model.fit(training.features, y, **kwargs)
    return model


def fit_simple(training, calibration, held, workflow):
    """Inner OOF weight selection; stopping UAVs never include OOF/outer-held UAVs."""
    settings = workflow["simple"]
    cap = float(workflow["target_cap"])
    calibration = training_calibration(training, calibration)
    groups = calibration.metadata.uav_id.astype(str).to_numpy()
    all_uavs = set(groups)
    if all_uavs & set(held.metadata.uav_id):
        raise ValueError("Outer held UAVs leaked into calibration")
    predictions = np.full((len(calibration), 2), np.nan)
    rounds = {family: [] for family in ("xgboost", "catboost")}
    provenance = []
    started = time.perf_counter()
    splits = GroupKFold(n_splits=int(workflow["inner_fold_count"])).split(calibration.features, groups=groups)
    for fold, (_, held_index) in enumerate(splits):
        oof_uavs = set(groups[held_index])
        available = sorted(all_uavs - oof_uavs)
        count = max(1, int(np.ceil(len(available) * settings["stopping_fraction"])))
        if len(available) - count < 2:
            raise ValueError("Too few UAVs for training-side early stopping")
        rng = np.random.default_rng(np.random.SeedSequence([settings["selection_seed"], fold]))
        stop_uavs = set(rng.choice(available, count, replace=False))
        fit_uavs = set(available) - stop_uavs
        fit = select_uavs(training, fit_uavs)
        stop = select_uavs(training, stop_uavs)
        refit = select_uavs(training, set(available))
        fold_record = {"inner_fold": fold, "fit_uavs": sorted(fit_uavs), "stopping_uavs": sorted(stop_uavs),
                       "oof_uavs": sorted(oof_uavs), "iterations": {}}
        for column, family in enumerate(("xgboost", "catboost")):
            tuner = fit_estimator(make_estimator(family, settings, stopping=True), family, fit, cap, stop)
            best = tuner.best_iteration if family == "xgboost" else tuner.get_best_iteration()
            iterations = int(best) + 1
            if iterations < 1:
                raise ValueError("Early stopping returned no usable iteration")
            rounds[family].append(iterations)
            fold_record["iterations"][family] = iterations
            model = fit_estimator(make_estimator(family, settings, iterations), family, refit, cap)
            predictions[held_index, column] = model.predict(calibration.features.iloc[held_index])
            del tuner, model
        provenance.append(fold_record)
    if not np.isfinite(predictions).all():
        raise ValueError("Inner OOF blend predictions are incomplete")
    weights = equal_uav_weights(calibration.metadata)
    target = calibration.target.to_numpy(float)
    scored = [(float(np.average((np.maximum(w * predictions[:, 0] + (1-w) * predictions[:, 1], 0) - target)**2,
                               weights=weights)), float(w)) for w in settings["blend_weights"]]
    _, selected_weight = min(scored)
    final_rounds = {family: max(1, int(np.floor(np.median(values) + .5))) for family, values in rounds.items()}
    members = [fit_estimator(make_estimator(family, settings, final_rounds[family]), family, training, cap)
               for family in ("xgboost", "catboost")]
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.maximum(selected_weight * members[0].predict(held.features)
                            + (1-selected_weight) * members[1].predict(held.features), 0)
    return prediction, {"fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - started,
        "calibration_uavs": sorted(all_uavs), "xgboost_weight": selected_weight,
        "final_iterations": final_rounds, "inner_selection": provenance,
        "base_estimator_fits": 4 * int(workflow["inner_fold_count"]) + 2}
