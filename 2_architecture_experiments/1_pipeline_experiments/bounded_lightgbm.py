"""Four LightGBM recipes, with stopping and refitting confined to training UAVs."""
import time
import numpy as np

from confirmation_utils import select_uavs
from campaign_models import training_endpoints


RECIPES = {f'lgb_leaves{leaves}_leaf{leaf}': {'num_leaves': leaves, 'min_child_samples': leaf}
           for leaves in (7, 15) for leaf in (20, 80)}


def stopping_split(ids, seed, fraction=.2):
    values = np.asarray(sorted(ids))
    if len(values) < 6:
        raise ValueError('At least six training UAVs required for grouped stopping')
    rng = np.random.default_rng(seed)
    stop = set(rng.choice(values, max(1, int(np.ceil(len(values)*fraction))), replace=False))
    return set(values)-stop, stop


def fit_lightgbm(training, calibration, held, recipe, seed, workflow):
    import lightgbm as lgb
    ids = set(training.metadata.uav_id)
    if ids & set(held.metadata.uav_id):
        raise ValueError('Held UAVs leaked into LightGBM training')
    calibration = training_endpoints(calibration, ids)
    fit_ids, stop_ids = stopping_split(ids, seed)
    fit_data = select_uavs(training, fit_ids)
    stop_data = select_uavs(calibration, stop_ids)
    if set(stop_data.metadata.uav_id) != stop_ids:
        raise ValueError('Stopping endpoint coverage incomplete')
    params = dict(objective='regression', learning_rate=workflow['learning_rate'],
        max_depth=-1, colsample_bytree=.9, reg_lambda=1., min_child_weight=1e-3,
        n_jobs=workflow['cpu_threads'], deterministic=True, force_col_wise=True,
        random_state=int(seed), verbosity=-1, **RECIPES[recipe])
    start = time.perf_counter()
    tuner = lgb.LGBMRegressor(n_estimators=workflow['maximum_iterations'], **params)
    # Preserve actual 0.05 row weights. Stopping weights total one per UAV too.
    weights = 1./stop_data.metadata.groupby('uav_id').uav_id.transform('size').to_numpy(float)
    tuner.fit(fit_data.features, np.minimum(fit_data.target, 125), sample_weight=fit_data.sample_weights,
        eval_set=[(stop_data.features, stop_data.target)], eval_sample_weight=[weights],
        callbacks=[lgb.early_stopping(workflow['stopping_patience'], verbose=False)])
    rounds = max(1, int(tuner.best_iteration_))
    del tuner
    model = lgb.LGBMRegressor(n_estimators=rounds, **params)
    model.fit(training.features, np.minimum(training.target, 125), sample_weight=training.sample_weights)
    prediction = np.maximum(model.predict(held.features), 0.)
    return prediction, {'fit_uavs': sorted(fit_ids), 'stopping_uavs': sorted(stop_ids),
        'refit_uavs': sorted(ids), 'base_estimator_fits': 2, 'best_iteration': rounds,
        'weight_sum': float(training.sample_weights.sum()),
        'fit_seconds': time.perf_counter()-start, 'parameters': params}
