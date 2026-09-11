"""Read-only-model audit of PE_31 matched predictions and observable test inputs.

Age reweighting and capped-label scores are retrospective diagnostics only.
This utility never fits a predictor or loads test RUL labels.
"""
import json
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score

from advanced_r2_utils import REPOSITORY_ROOT
from followup_experiment_utils import atomic_csv, atomic_json
from reproduce_simple_submission import sha

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'literature_and_planning/development_documentation/validation_transfer_audit_2026_09_11'
BINS = [0, 50, 100, 150, 200, 250, 10000]


def bands(values):
    return pd.cut(values, BINS, labels=False, include_lowest=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    campaign = ROOT / '2_architecture_experiments/1_pipeline_experiments/experiments/PE_31/runs/run_1'
    train_path, test_path = ROOT/'data/train.csv', ROOT/'data/test.csv'
    raw = pd.read_csv(train_path)
    # Explicit feature-only test read; any future RUL column is excluded.
    test = pd.read_csv(test_path, usecols=lambda c: c in ('uav_id', 'flight_cycle') or c.startswith('telemetry'))
    last = test.sort_values(['uav_id', 'flight_cycle']).groupby('uav_id').tail(1).copy()
    ep = pd.read_csv(campaign/'reporting/endpoints.csv')
    ep['age_band'] = bands(ep.cutoff)
    last['age_band'] = bands(last.flight_cycle)
    dist, fractions = [], []
    profiles = {name: g.cutoff for name, g in ep.groupby('suite')}
    profiles['test'] = last.flight_cycle
    for name, values in profiles.items():
        q = values.quantile([0, .1, .25, .5, .75, .9, 1]).tolist()
        dist.append(dict(profile=name, rows=len(values), mean=values.mean(), minimum=q[0],
            p10=q[1], p25=q[2], median=q[3], p75=q[4], p90=q[5], maximum=q[6],
            fraction_le100=float(values.le(100).mean()),
            wasserstein_to_test=float(wasserstein_distance(values, last.flight_cycle))))
        for band in range(len(BINS)-1):
            fractions.append({'profile': name, 'age_band': band, 'low': BINS[band], 'high': BINS[band+1],
                              'fraction': float(bands(values).eq(band).mean())})
    atomic_csv(OUT/'history_distributions.csv', pd.DataFrame(dist))
    atomic_csv(OUT/'history_band_fractions.csv', pd.DataFrame(fractions))

    prediction_path = campaign/'stages/benchmark/reporting/predictions.csv.gz'
    pred = pd.read_csv(prediction_path)
    pred = pred.loc[pred.method.isin(['run7', 'reference_protocol'])]
    keys = ['split_seed', 'outer_fold', 'uav_id', 'scenario', 'suite', 'endpoint_seed', 'cutoff', 'observed_rul']
    methods = {m: set(map(tuple, g[keys+['model_seed']].to_numpy())) for m,g in pred.groupby('method')}
    assert methods['run7'] == methods['reference_protocol'], 'Predictions are not matched'
    metrics = []
    for seed_mode, chosen in [('seed0', pred.loc[pred.model_seed.eq(0)]), ('two_seed_average', pred)]:
        averaged = chosen.groupby(keys+['method'], as_index=False).predicted_rul.mean()
        for (method, suite), group in averaged.groupby(['method','suite']):
            ages = bands(group.cutoff)
            base_weights = 1/group.groupby('uav_id').uav_id.transform('size').to_numpy(float)
            target_mass = bands(last.flight_cycle).value_counts(normalize=True)
            source_mass = pd.Series(base_weights).groupby(ages.to_numpy()).sum()/base_weights.sum()
            missing = [int(b) for b in target_mass.index if b not in source_mass]
            age_weight = base_weights * np.array([target_mass.get(b,0)/source_mass[b] for b in ages])
            for label_mode in ('raw', 'capped125_diagnostic'):
                y = group.observed_rul.to_numpy(float)
                if label_mode != 'raw':
                    y = np.minimum(y, 125)
                for weighting, weights in [('equal_uav', base_weights), ('test_age_bands_diagnostic', age_weight)]:
                    p = group.predicted_rul.to_numpy(float)
                    metrics.append({'seed_mode': seed_mode, 'method': method, 'suite': suite,
                        'labels': label_mode, 'weighting': weighting, 'rows': len(group),
                        'rmse': float(np.sqrt(np.average((p-y)**2,weights=weights))),
                        'r2': float(r2_score(y,p,sample_weight=weights)),
                        'bias': float(np.average(p-y,weights=weights)),
                        'missing_test_age_bands': json.dumps(missing),
                        'effective_rows_not_independent_uavs': float(weights.sum()**2/(weights**2).sum())})
    atomic_csv(OUT/'matched_metrics.csv', pd.DataFrame(metrics))

    # Compare raw endpoint sensor states both pooled and within fixed age bands.
    sensors = [c for c in raw if c.startswith('telemetry') and raw[c].std() > 1e-6]
    joined = ep.merge(raw[['uav_id','flight_cycle',*sensors]], left_on=['uav_id','cutoff'],
                      right_on=['uav_id','flight_cycle'], validate='many_to_one')
    sensor_rows = []
    for suite, group in joined.groupby('suite'):
        for age in [-1, *range(len(BINS)-1)]:
            a = group if age == -1 else group.loc[group.age_band.eq(age)]
            b = last if age == -1 else last.loc[last.age_band.eq(age)]
            if len(a) < 2 or len(b) < 2:
                continue
            for sensor in sensors:
                scale = float(a[sensor].quantile(.75)-a[sensor].quantile(.25))
                sensor_rows.append({'suite':suite,'age_band':age,'sensor':sensor,
                    'validation_rows':len(a),'test_rows':len(b),'validation_iqr':scale,
                    'median_shift':float(b[sensor].median()-a[sensor].median()),
                    'wasserstein':float(wasserstein_distance(a[sensor],b[sensor])),
                    'wasserstein_over_validation_iqr':float(wasserstein_distance(a[sensor],b[sensor])/scale) if scale > 1e-12 else None})
    atomic_csv(OUT/'endpoint_sensor_drift.csv', pd.DataFrame(sensor_rows))

    # Original-feature causality check on an actual UAV: mutate future values
    # and all labels, and compare the engineered prefix at the chosen cutoff.
    script = ROOT/'other_pipelines/uav_rul_pipeline_v4 (1).py'
    spec = importlib.util.spec_from_file_location('audit_original', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    unit = raw.loc[raw.uav_id.eq(raw.uav_id.iloc[0])].copy()
    cutoff = 100
    original = module.build_features(unit, sensors)
    altered = unit.copy()
    altered.loc[altered.flight_cycle.gt(cutoff),sensors] += 100000
    altered['RUL'] = -999999
    changed = module.build_features(altered, sensors)
    pd.testing.assert_frame_equal(original.loc[original.flight_cycle.le(cutoff)].reset_index(drop=True),
                                  changed.loc[changed.flight_cycle.le(cutoff)].reset_index(drop=True))
    from run_complete_pipeline_validation import build_feature_table
    meta = pd.DataFrame([{'uav_id':unit.uav_id.iloc[0],'cutoff':cutoff,'sample_id':'audit',
        'scenario':'audit','outer_fold':0,'lifetime_quantile':0,
        'terminal_lifetime':int(unit.flight_cycle.max()),'RUL':int(unit.flight_cycle.max())-cutoff}])
    run7_before = build_feature_table(unit,meta,feature_profile='extended')
    run7_after = build_feature_table(altered,meta,feature_profile='extended')
    feature_columns = [c for c in run7_before if c.startswith('feature__')]
    pd.testing.assert_frame_equal(run7_before[feature_columns],run7_after[feature_columns])
    original_assignment = module._eligible_random_assignment(raw.groupby('uav_id').flight_cycle.max().to_dict(),
        last.flight_cycle.to_numpy(int),np.random.default_rng(1000))
    original_calibration = pd.DataFrame(original_assignment,columns=['uav_id','cutoff'])
    original_calibration = original_calibration.merge(raw[['uav_id','flight_cycle','RUL']],
        left_on=['uav_id','cutoff'],right_on=['uav_id','flight_cycle'],validate='one_to_one')
    atomic_csv(OUT/'original_calibration_endpoints.csv',original_calibration)
    sorted_raw = raw.sort_values(['uav_id','flight_cycle']).reset_index(drop=True)
    alignment_ok = raw[['uav_id','flight_cycle','RUL']].reset_index(drop=True).equals(sorted_raw[['uav_id','flight_cycle','RUL']])
    submission_paths = {
        'run7': ROOT/'3_final_model_training_and_inference/runs/run_7/6_submission_verification/artifacts/submission.csv',
        'simpler_reproduction': campaign/'reproduction/submission.csv'}
    differences = last[['uav_id','flight_cycle','age_band']].copy()
    for method,path in submission_paths.items():
        differences = differences.merge(pd.read_csv(path).rename(columns={'id':'uav_id','RUL':method}),on='uav_id',validate='one_to_one')
    differences['simple_minus_run7'] = differences.simpler_reproduction-differences.run7
    atomic_csv(OUT/'test_prediction_disagreement.csv', differences)
    manifest = {'uses_test_labels':False,'new_model_fits':0,'matched_endpoint_identity_verified':True,
        'original_prefix_feature_causality_check_passed':True,'original_positional_label_alignment_valid_for_current_file':alignment_ok,
        'run7_prefix_feature_causality_check_passed':True,
        'original_calibration_fraction_raw_rul_over125':float(original_calibration.RUL.gt(125).mean()),
        'test_uavs':len(last),'training_uavs':raw.uav_id.nunique(),
        'matched_protocol_limitation':'PE31 reference used uniform feasible calibration cutoffs, not original test-like assignment.',
        'reweighted_and_capped_scores_are_posthoc_diagnostics':True,
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
            [train_path,test_path,script,prediction_path,campaign/'reporting/endpoints.csv',*submission_paths.values()]}}
    atomic_json(OUT/'audit_manifest.json',manifest)
    print(pd.DataFrame(dist).round(3).to_string(index=False))
    print(pd.DataFrame(metrics).query("seed_mode == 'two_seed_average' and labels == 'raw'").round(4).to_string(index=False))


if __name__ == '__main__':
    main()
