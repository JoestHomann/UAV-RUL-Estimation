"""Prepare the user-requested PE_25 Kaggle trial without changing promotion results.

Select the final restricted rule on all saved development OOF predictions, then
apply it to the frozen Run 7 predictor and one full-training TabPFN fit. The
selection-set score is not a new validation result. No Kaggle upload is made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from advanced_r2_utils import REPOSITORY_ROOT, equal_uav_weights
from followup_experiment_utils import (
    aligned_methods, atomic_csv, atomic_json, register_run, route_predictions,
    validate_saved_nested,
)
from run_regime_tabpfn_confirmation import TabularDataAdapter
from run_tabular_prior import fit_tabpfn, fitting_target, tabpfn_dependency_status
from base import load_model_adapter


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def choose_rule(rows: pd.DataFrame, weights: list, thresholds: list) -> dict:
    candidates = []
    sample_weights = equal_uav_weights(rows)
    for weight in sorted(set(weights)):
        for threshold in sorted(set(thresholds)):
            prediction = route_predictions(rows, weight=weight, threshold=threshold,
                                           route_column='control_prediction')
            mse = float(np.average((prediction - rows.observed_rul.to_numpy(float)) ** 2,
                                   weights=sample_weights))
            candidates.append((mse, weight, threshold))
    mse, weight, threshold = min(candidates)
    return {'weight': weight, 'threshold': threshold,
            'selection_rmse_not_validation': float(np.sqrt(mse))}


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(json.dumps(list(frame.columns)).encode())
    return digest.hexdigest()


def main() -> None:
    p24 = HERE / 'experiments/PE_24/settings.toml'
    p25 = HERE / 'experiments/PE_25/settings.toml'
    source = tomllib.loads(p24.read_text())['regime_tabpfn_workflows']['PE_24']
    settings = tomllib.loads(p25.read_text())['restricted_tabpfn_workflows']['PE_25']
    root = HERE / 'experiments/PE_25/runs/run_1/exploratory_submission_v2'
    root.mkdir(parents=True, exist_ok=True)
    source_paths = [REPOSITORY_ROOT / settings[key] for key in
                    ('source_predictions', 'source_manifest', 'source_outer_folds', 'source_inner_folds')]
    saved, source_manifest, outer_path, inner_path = source_paths
    manifest = json.loads(source_manifest.read_text())
    if not (manifest.get('all_prediction_cells_completed') and manifest.get('nested_inner_predictions_complete')):
        raise ValueError('PE_24 must be complete')
    if manifest.get('uses_test_labels') is not False or manifest.get('uses_locked_evaluation') is not False:
        raise ValueError('Only development OOF predictions may select the final rule')
    table = pd.read_csv(saved)
    validate_saved_nested(table, pd.read_csv(outer_path), pd.read_csv(inner_path),
                          methods={'control', 'tabpfn_full'}, outer_count=5, inner_count=4)
    oof = aligned_methods(table.loc[table.evaluation_level.eq('outer')], 'tabpfn_full')
    rule = choose_rule(oof, settings['weights'], settings['prediction_thresholds'])
    # Every label below belongs to a training UAV. Each base OOF prediction
    # excluded that UAV; the resulting selection score is deliberately not held out.
    adapter = TabularDataAdapter(REPOSITORY_ROOT / source['tabular_manifest'])
    training = adapter.load_training(source['feature_set'])
    test = adapter.load_test(source['feature_set'])
    if test.target is not None or len(test) != 100 or test.metadata.uav_id.duplicated().any():
        raise ValueError('Expected 100 unique, unlabelled test UAVs')
    if set(training.metadata.uav_id) != set(oof.uav_id) or set(oof.uav_id) & set(test.metadata.uav_id):
        raise ValueError('Development/train/test UAV identities do not match')
    run7 = REPOSITORY_ROOT / '3_final_model_training_and_inference/runs/run_7'
    model_path = run7 / '4_final_model_training/artifacts/final_model.joblib'
    contract_path = run7 / '3_final_training_contract/artifacts/final_training_contract.json'
    contract = json.loads(contract_path.read_text())
    if contract['input_schema']['feature_names'] != list(training.features.columns):
        raise ValueError('Run 7 feature order differs from PE_24')
    if list(test.features.columns) != list(training.features.columns):
        raise ValueError('Train/test features differ')
    target = fitting_target(training, float(source['target_cap']))
    dependency = tabpfn_dependency_status(source['tabpfn'])
    if not dependency['ready']:
        raise RuntimeError(f'Pinned TabPFN dependency unavailable: {dependency}')
    registration = {
        'purpose': 'User-requested exploratory PE_25 submission; promotion verdict unchanged',
        'source': source, 'rule_grid': settings, 'final_rule': rule,
        'selection': 'All PE_24 outer development OOF rows; equal total weight per UAV',
        'training_features_sha256': frame_hash(training.features),
        'training_metadata_sha256': frame_hash(training.metadata),
        'training_target_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
        'test_features_sha256': frame_hash(test.features),
        'test_metadata_sha256': frame_hash(test.metadata),
        'tabpfn_dependency': dependency,
    }
    register_run(root, registration, [Path(__file__), p24, p25, *source_paths,
        model_path, contract_path, REPOSITORY_ROOT / source['source_contract'],
        REPOSITORY_ROOT / source['tabular_manifest'],
        HERE / 'run_tabular_prior.py', HERE / 'followup_experiment_utils.py'])
    atomic_json(root / 'frozen_rule.json', rule)
    print('Frozen final rule:', json.dumps(rule), flush=True)
    component_path = root / 'component_predictions.csv'
    if component_path.exists():
        components = pd.read_csv(component_path)
        if components.id.tolist() != test.metadata.uav_id.tolist():
            raise ValueError('Cached component IDs changed')
    else:
        print('Reusing the saved Run 7 model; verifying its submission predictions', flush=True)
        model = load_model_adapter(model_path)
        # The saved Run 7 predates this optional adapter attribute. Restore the
        # contract default in memory, then require agreement with its saved CSV.
        if not hasattr(model, 'correction_strength'):
            model.correction_strength = float(model.contract.get('correction_strength', 1.0))
        control = np.maximum(np.asarray(model.predict(test), float), 0)
        original = pd.read_csv(run7 / '6_submission_verification/artifacts/submission.csv').set_index('id')
        np.testing.assert_allclose(control, original.loc[test.metadata.uav_id, 'RUL'], rtol=1e-7, atol=1e-8)
        print('Fitting pinned TabPFN on 2,000 prefixes from all 100 training UAVs', flush=True)
        challenger = np.maximum(fit_tabpfn(training.features, target, test.features,
            {**source['tabpfn'], 'checkpoint': dependency['resolved_checkpoint']}), 0)
        components = pd.DataFrame({'id': test.metadata.uav_id.to_numpy(),
            'control_prediction': control, 'challenger_prediction': challenger})
        atomic_csv(component_path, components)
    if not np.isfinite(components[['control_prediction', 'challenger_prediction']].to_numpy(float)).all():
        raise ValueError('Nonfinite component predictions')
    prediction = route_predictions(components, weight=rule['weight'], threshold=rule['threshold'],
                                   route_column='control_prediction')
    submission = pd.DataFrame({'id': components.id, 'RUL': np.maximum(prediction, 0)}).sort_values('id')
    expected_ids = sorted(test.metadata.uav_id.tolist())
    if submission.id.tolist() != expected_ids or not np.isfinite(submission.RUL).all():
        raise ValueError('Invalid submission')
    submission_path = root / 'submission_PE25_restricted_tabpfn.csv'
    atomic_csv(submission_path, submission)
    pd.testing.assert_frame_equal(pd.read_csv(submission_path), submission.reset_index(drop=True))
    result = {'status': 'complete', 'method': 'restricted_tabpfn_blend', 'exploratory': True,
        'production_model_replaced': False, 'validation_verdict_changed': False,
        'submitted_to_kaggle': False, 'uses_test_labels': False, 'final_rule': rule,
        'selection_score_is_unbiased_validation': False, 'rows': len(submission),
        'training_uavs': 100, 'training_rows': len(training),
        'routed_test_uavs': int(components.control_prediction.le(rule['threshold']).sum()),
        'submission': str(submission_path.relative_to(REPOSITORY_ROOT)),
        'submission_sha256': sha256(submission_path),
        'component_predictions_sha256': sha256(component_path)}
    atomic_json(root / 'submission_manifest.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
