"""Training-side adaptation of the exact submitted simpler script for PE_32.

Preserves source features/order, capped targets, five grouped stopping folds,
seed zero, blend grid and final 90/10 UAV stopping split. Outer test labels are
never used. The original 100-cutoff assignment is scaled to the training fold
by drawing N cutoffs without replacement and rejecting infeasible assignments.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import time

import numpy as np
import pandas as pd

from followup_experiment_utils import input_path


def empirical_assignment(module, lifetimes, cutoffs, seed=1000):
    cutoffs = np.asarray(cutoffs, dtype=int)
    if len(lifetimes) > len(cutoffs) or not lifetimes or (cutoffs < 1).any():
        raise ValueError('Invalid training-side empirical cutoff assignment')
    rng = np.random.default_rng(seed)
    for attempt in range(1, 1001):
        sampled = rng.choice(cutoffs, size=len(lifetimes), replace=False)
        try:
            assignment = module._eligible_random_assignment(lifetimes, sampled, rng)
        except ValueError:
            continue
        return assignment, attempt
    raise ValueError('No feasible empirical assignment in 1,000 attempts; no silent fallback')


def fit_reference(raw, training_ids, held, source, cutoffs):
    ids = set(training_ids)
    if ids & set(held.metadata.uav_id) or len(ids) < 5:
        raise ValueError('Reference needs disjoint outer UAVs and at least five training UAVs')
    path = input_path(source['reference_implementation'])
    spec = importlib.util.spec_from_file_location('pe32_submitted_reference', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.CATBOOST_PARAMS['allow_writing_files'] = False
    training_raw = raw.loc[raw.uav_id.isin(ids)].sort_values(['uav_id','flight_cycle']).reset_index(drop=True)
    if set(training_raw.uav_id) != ids:
        raise ValueError('Reference training UAV coverage differs')
    started = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        sensors = module.get_sensor_cols(training_raw)
        capped = module.apply_rul_cap(training_raw)
        features = module.build_features(capped, sensors)
        features['RUL_raw'] = capped.RUL.to_numpy()
        features['uav_id_check'] = capped.uav_id.to_numpy()
        columns = [c for c in features if c not in ('uav_id','RUL_raw','uav_id_check')]
        xmodels, xids = module.fit_xgb_cv(features, columns)
        cmodels, cids = module.fit_catboost_cv(features, columns)
        assignments, attempts = empirical_assignment(module, training_raw.groupby('uav_id').flight_cycle.max().to_dict(),cutoffs)
        calibration = features.set_index(['uav_id_check','flight_cycle'],drop=False).loc[assignments].reset_index(drop=True)
        xp = module.xgb_predict_eval(calibration,columns,xmodels,xids)
        cp = module.catboost_predict_eval(calibration,columns,cmodels,cids)
        if not np.isfinite(xp).all() or not np.isfinite(cp).all():
            raise ValueError('Incomplete reference OOF predictions')
        weight = min((float(np.mean((a*xp+(1-a)*cp-calibration.RUL_raw.to_numpy())**2)),float(a))
                     for a in np.arange(0,1.01,.05))[1]
        del xmodels,cmodels
        stopping = set(np.random.RandomState(0).choice(features.uav_id_check.unique(),max(1,int(len(ids)*.1)),replace=False))
        mask = features.uav_id_check.isin(stopping).to_numpy()
        x = module.xgb.XGBRegressor(**module.XGB_PARAMS,early_stopping_rounds=module.XGB_EARLY_STOPPING_ROUNDS)
        c = module.cb.CatBoostRegressor(**module.CATBOOST_PARAMS,early_stopping_rounds=module.CATBOOST_EARLY_STOPPING_ROUNDS)
        matrix,y = features[columns].values,features.RUL_raw.values
        x.fit(matrix[~mask],y[~mask],eval_set=[(matrix[mask],y[mask])],verbose=False)
        c.fit(matrix[~mask],y[~mask],eval_set=(matrix[mask],y[mask]))
        # Remove unobserved future rows before feature construction, even though
        # the original expanding/rolling features are causal.
        query_raw = raw.loc[raw.uav_id.isin(held.metadata.uav_id)].copy()
        max_cutoff = held.metadata.groupby('uav_id').cutoff.max()
        query_raw = query_raw.loc[query_raw.flight_cycle.le(query_raw.uav_id.map(max_cutoff))].drop(columns='RUL',errors='ignore')
        evaluation = module.build_features(query_raw,sensors).set_index(['uav_id','flight_cycle'],drop=False)
        query = evaluation.loc[list(zip(held.metadata.uav_id,held.metadata.cutoff)),columns].values
        prediction = weight*x.predict(query)+(1-weight)*c.predict(query)
    return prediction, {'base_estimator_fits':12,'fit_seconds':time.perf_counter()-started,
        'model_seed':0,'xgboost_weight':weight,'final_fit_uavs':sorted(ids-stopping),
        'final_stopping_uavs':sorted(stopping),'calibration_uavs':sorted(ids),
        'calibration_assignments':[(str(uid),int(cutoff)) for uid,cutoff in assignments],
        'empirical_assignment_attempts':attempts,'original_feature_order':columns,
        'xgboost_best_iteration':int(x.best_iteration),'catboost_best_iteration':int(c.get_best_iteration()),
        'internal_labels_capped':True,'outer_labels_used_for_fitting':False,
        'protocol_adaptation':'N training-side empirical cutoffs without replacement, original greedy assignment; infeasible draws rejected',
        'final_model_refitted_on_stopping_uavs':False}
