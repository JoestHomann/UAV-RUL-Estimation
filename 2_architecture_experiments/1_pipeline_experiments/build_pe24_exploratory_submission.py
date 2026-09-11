"""Build the requested PE_24 trial from PE_25 components and a final OOF gate.

The original depth-two gate recipe is fitted to all development OOF rows.
This is a final fit, not another validation result or a production promotion.
"""
from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from advanced_r2_utils import REPOSITORY_ROOT
from build_pe25_exploratory_submission import frame_hash, sha256
from followup_experiment_utils import atomic_csv, atomic_json, register_run, validate_saved_nested
from run_regime_tabpfn_confirmation import (
    CONTROL_DIAGNOSTICS, TabularDataAdapter, _aligned_predictions,
    cross_fit_regime_blend, pre_registration_contract,
)
from base import load_model_adapter

HERE = Path(__file__).resolve().parent


def main() -> None:
    config_path = HERE / 'experiments/PE_24/settings.toml'
    settings = tomllib.loads(config_path.read_text())['regime_tabpfn_workflows']['PE_24']
    source_root = HERE / 'experiments/PE_24/runs/run_1/reporting'
    root = source_root.parent / 'exploratory_submission'
    root.mkdir(parents=True, exist_ok=True)
    if json.loads((source_root / 'pre_registration.json').read_text()) != pre_registration_contract(settings):
        raise ValueError('PE_24 recipe differs from its original registration')
    source_manifest = json.loads((source_root / 'winner_manifest.json').read_text())
    if not (source_manifest.get('all_prediction_cells_completed') and
            source_manifest.get('nested_inner_predictions_complete')):
        raise ValueError('PE_24 is incomplete')
    if source_manifest.get('uses_test_labels') is not False or source_manifest.get('uses_locked_evaluation') is not False:
        raise ValueError('Gate inputs must be development predictions only')
    table = pd.read_csv(source_root / 'fold_predictions.csv')
    validate_saved_nested(table, pd.read_csv(source_root / 'outer_folds.csv'),
                          pd.read_csv(source_root / 'inner_folds.csv'),
                          methods={'control', 'tabpfn_full'}, outer_count=5, inner_count=4)
    training_oof = _aligned_predictions(table.loc[table.evaluation_level.eq('outer')])

    components_root = HERE / 'experiments/PE_25/runs/run_1/exploratory_submission_v2'
    component_path = components_root / 'component_predictions.csv'
    component_manifest = json.loads((components_root / 'submission_manifest.json').read_text())
    prior_registration = json.loads((components_root / 'pre_registration.json').read_text())
    if component_manifest.get('status') != 'complete' or component_manifest.get('uses_test_labels') is not False:
        raise ValueError('Need completed, unlabelled PE_25 components')
    if sha256(component_path) != component_manifest['component_predictions_sha256']:
        raise ValueError('PE_25 component checksum changed')
    if prior_registration['workflow']['source'] != settings:
        raise ValueError('PE_25 components used a different model recipe')
    components = pd.read_csv(component_path)
    adapter = TabularDataAdapter(REPOSITORY_ROOT / settings['tabular_manifest'])
    test = adapter.load_test(settings['feature_set'])
    if test.target is not None or len(test) != 100 or test.metadata.uav_id.duplicated().any():
        raise ValueError('Expected 100 unique, unlabelled test UAVs')
    if components.id.tolist() != test.metadata.uav_id.tolist():
        raise ValueError('Component/test identities or order differ')
    for field, frame in [('test_features_sha256', test.features), ('test_metadata_sha256', test.metadata)]:
        if frame_hash(frame) != prior_registration['workflow'][field]:
            raise ValueError(f'Test inputs changed: {field}')
    if set(test.metadata.uav_id) & set(training_oof.uav_id) or training_oof.uav_id.nunique() != 100:
        raise ValueError('Gate selection/test UAVs overlap or development is incomplete')

    model_path = REPOSITORY_ROOT / '3_final_model_training_and_inference/runs/run_7/4_final_model_training/artifacts/final_model.joblib'
    if sha256(model_path) != prior_registration['input_sha256'][str(model_path.relative_to(REPOSITORY_ROOT))]:
        raise ValueError('Saved Run 7 model differs from PE_25 component source')
    register_run(root, {
        'purpose': 'User-requested exploratory PE_24 leaderboard trial',
        'recipe': settings, 'selection': 'All 1,500 saved outer development OOF rows; equal UAV mass',
        'selection_score_is_unbiased_validation': False,
        'base_components': str(component_path.relative_to(REPOSITORY_ROOT)),
    }, [Path(__file__), HERE / 'build_pe25_exploratory_submission.py',
        HERE / 'run_regime_tabpfn_confirmation.py', HERE / 'followup_experiment_utils.py',
        config_path, model_path, component_path, components_root / 'submission_manifest.json',
        components_root / 'pre_registration.json',
        *[source_root / name for name in ['pre_registration.json', 'winner_manifest.json',
          'fold_predictions.csv', 'outer_folds.csv', 'inner_folds.csv']]])

    print('Reusing Run 7 and TabPFN; generating Run 7 uncertainty diagnostics', flush=True)
    model = load_model_adapter(model_path)
    if not hasattr(model, 'correction_strength'):
        model.correction_strength = float(model.contract.get('correction_strength', 1.0))
    diagnostics = model.predict_with_diagnostics(test).reset_index(drop=True)
    np.testing.assert_allclose(diagnostics.predicted_rul, components.control_prediction, rtol=1e-7, atol=1e-8)
    held = pd.DataFrame({'uav_id': components.id,
        'control_prediction': components.control_prediction,
        'challenger_prediction': components.challenger_prediction})
    for column in CONTROL_DIAGNOSTICS:
        held[column] = diagnostics[column].to_numpy(float)
    held['challenger_gap'] = held.challenger_prediction - held.control_prediction
    # A single group invokes the existing gate-fitting implementation once on
    # all development OOF rows. Held rows are unlabelled test UAVs, not a fold
    # whose score can be used as validation. No synthetic labels are introduced.
    held['outer_fold'] = 0
    training_oof['outer_fold'] = 0
    prediction, weights, provenance = cross_fit_regime_blend(held, training_oof,
        feature_columns=settings['gate_features'], maximum_depth=settings['gate_maximum_depth'],
        minimum_samples_leaf=settings['gate_minimum_samples_leaf'],
        maximum_tabpfn_weight=settings['gate_maximum_tabpfn_weight'],
        minimum_challenger_gap=settings['gate_minimum_challenger_gap'],
        random_state=settings['gate_random_state'])
    if not (np.isfinite(prediction).all() and np.isfinite(weights).all() and
            (weights >= 0).all() and (weights <= .5).all()):
        raise ValueError('Invalid gate predictions or weights')
    held['tabpfn_weight'] = weights
    held['RUL'] = np.maximum(prediction, 0)
    atomic_csv(root / 'test_components_and_weights.csv', held.drop(columns='outer_fold'))
    gate_record = dict(provenance[0])
    # The generic helper uses validation terminology for its application set.
    gate_record['test_uavs'] = gate_record.pop('validation_uavs')
    gate_record['test_rows'] = gate_record.pop('validation_rows')
    gate_record.pop('outer_fold')
    atomic_json(root / 'final_gate.json', {'fit_scope': 'development OOF only',
        'fit_metrics_are_not_validation': True, 'gate': gate_record})
    submission = held[['uav_id', 'RUL']].rename(columns={'uav_id': 'id'}).sort_values('id').reset_index(drop=True)
    if submission.id.tolist() != sorted(test.metadata.uav_id.tolist()):
        raise ValueError('Submission IDs do not match test inputs')
    output = root / 'submission_PE24_regime_tabpfn.csv'
    atomic_csv(output, submission)
    pd.testing.assert_frame_equal(pd.read_csv(output), submission)
    result = {'status': 'complete', 'method': 'regime_tabpfn_blend', 'exploratory': True,
        'production_model_replaced': False, 'validation_verdict_changed': False,
        'uses_test_labels': False, 'submitted_to_kaggle': False,
        'rows': len(submission), 'gate_training_rows': len(training_oof),
        'gate_training_uavs': int(training_oof.uav_id.nunique()),
        'additional_base_model_fits': 0, 'final_gate_fits': 1,
        'tabpfn_weight_min': float(weights.min()), 'tabpfn_weight_max': float(weights.max()),
        'tabpfn_weight_mean': float(weights.mean()),
        'submission': str(output.relative_to(REPOSITORY_ROOT)), 'submission_sha256': sha256(output)}
    atomic_json(root / 'submission_manifest.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
