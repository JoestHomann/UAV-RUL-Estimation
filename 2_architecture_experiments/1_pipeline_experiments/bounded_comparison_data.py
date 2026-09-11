"""Frozen Run 7 inputs and PE_31 endpoint semantics for PE_32/33.

The LightGBM representation is A; the optional submitted-script baseline builds
its own original features inside each training fold.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import campaign_data  # initializes the existing adapters' import paths
from confirmation_utils import generated_partitions, evaluation_jobs, validate_partitions
from followup_experiment_utils import input_path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_contract(workflow):
    path = input_path(workflow['source_registration'])
    registration = json.loads(path.read_text(encoding='utf-8'))
    source = deepcopy(registration['workflow'])
    if registration['python_version'] != sys.version:
        raise ValueError('Python differs from the registered Run 7 comparison')
    for package, expected in registration['package_versions'].items():
        if version(package) != expected:
            raise ValueError(f'Registered package changed: {package}')
    unused = source['reference_implementation'].replace('\\', '/')
    paths = [path]
    for relative, expected in registration['input_sha256'].items():
        if relative.replace('\\', '/') == unused and not workflow.get('include_simple_baseline', False):
            continue
        original = input_path(relative)
        if sha(original) != expected:
            raise ValueError(f'Registered Run 7 source changed: {relative}')
        paths.append(original)
    for name in ('catboost', 'threadpoolctl'):
        if version(name) != source[f'{name}_version']:
            raise ValueError(f'Registered package changed: {name}')
    if source['target_cap'] != 125 or source['inner_fold_count'] != 4:
        raise ValueError('Expected the frozen cap125, four-calibration-fold Run 7 recipe')
    return source, paths


def prepare(workflow):
    source, paths = source_contract(workflow)
    adapter = campaign_data.prior.pe28.TabularDataAdapter(input_path(source['tabular_manifest']))
    training = adapter.load_training(source['feature_set'])
    historical = adapter.load_development(source['feature_set'])
    raw = pd.read_csv(input_path(source['raw_training']))
    campaign_data.prior.validate_raw(raw)
    counts = training.metadata.groupby('uav_id').size()
    if (len(training.features.columns) != 298 or not counts.eq(20).all()
            or training.sample_weights is None or not np.allclose(training.sample_weights, .05)):
        raise ValueError('Expected 298 features, 20 prefixes and total weight one per UAV')
    if set(raw.uav_id) != set(training.metadata.uav_id):
        raise ValueError('Raw/training UAV coverage differs')
    cutoff_path = input_path(workflow['test_cutoffs'])
    cutoffs = pd.read_csv(cutoff_path).final_cycle.to_numpy(int)
    endpoints = campaign_data.endpoint_suites(raw, historical, cutoffs, workflow['endpoint_seeds'])
    feature_meta = endpoints.copy()
    feature_meta['outer_fold'] = 0
    feature_meta['lifetime_quantile'] = 0
    feature_meta['terminal_lifetime'] = feature_meta.uav_id.map(raw.groupby('uav_id').flight_cycle.max())
    features = campaign_data.audit_tools.build_feature_table(raw, feature_meta, feature_profile='extended')
    development = replace(historical, features=features[list(training.features)].reset_index(drop=True),
        metadata=endpoints.drop(columns='RUL').reset_index(drop=True),
        target=endpoints.RUL.astype(float).reset_index(drop=True), sample_weights=None)
    outer, inner = generated_partitions(history_summary_path=source['history_summary'],
        split_seeds=workflow['split_seeds'], outer_fold_count=5, inner_fold_count=3)
    validate_partitions(outer, inner, expected_outer_folds=5, expected_inner_folds=3)
    paths.extend([cutoff_path, input_path(source['history_summary']),
        Path(campaign_data.__file__), Path(campaign_data.audit_tools.__file__),
        Path(campaign_data.audit_tools.build_feature_table.__code__.co_filename)])
    return source, {'training': training, 'development': development, 'raw': raw, 'cutoffs': cutoffs}, evaluation_jobs(
        outer, inner, include_inner=False), paths, outer, inner
