"""Leakage boundaries and real-model compatibility for engineered Run 10."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from contextlib import contextmanager
import shutil
import uuid
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '2_architecture_experiments/2_model_architecture_study'))
import run_engineered_feature_study as study
from engineered_feature_data import FeatureView, load_script, training_endpoints, validate_raw


@contextmanager
def scratch():
    path = ROOT / ('run10_scratch_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != ROOT.resolve() or not path.name.startswith('run10_scratch_'):
            raise ValueError('Unsafe temporary cleanup path')
        shutil.rmtree(path)


def synthetic():
    rng = np.random.default_rng(22)
    rows = []
    for uid in range(10):
        for cycle in range(1, 10):
            row = {'uav_id': f'UAV_{uid:04d}', 'flight_cycle': cycle, 'RUL': float(160-cycle)}
            row.update({f'telemetry_{j:02d}': rng.normal() + .1*cycle + .05*uid for j in range(22)})
            rows.append(row)
    return pd.DataFrame(rows)


def inputs():
    settings = json.loads(study.DEFAULT_SETTINGS.read_text())
    script = load_script(ROOT / settings['feature_script'])
    raw = synthetic()
    ids = sorted(raw.uav_id.unique())[:6]
    held = pd.DataFrame({'uav_id': sorted(raw.uav_id.unique())[6:], 'cutoff': [2, 5, 8, 9]})
    return settings, script, raw, ids, held


def _check_causality_and_training_only_scaling():
    _, script, raw, ids, held = inputs()
    validate_raw(raw)
    view = FeatureView(raw, ids, script)
    assert len(view.names) == 266
    direct = script.build_features(raw.drop(columns='RUL'), view.sensors).drop(columns='uav_id')
    np.testing.assert_allclose(view.values, direct, rtol=1e-6, atol=1e-6)
    altered = raw.copy()
    altered.loc[~altered.uav_id.isin(ids), 'RUL'] = 9999
    for uid, cutoff in held.itertuples(index=False, name=None):
        mask = altered.uav_id.eq(uid) & altered.flight_cycle.gt(cutoff)
        altered.loc[mask, view.sensors] += 10000
    changed = FeatureView(altered, ids, script)
    np.testing.assert_array_equal(view.center, changed.center)
    np.testing.assert_array_equal(view.scale, changed.scale)
    for family in study.CLASSES:
        before = view.dataset(held, family, 20)
        after = changed.dataset(held, family, 20)
        assert before.target is None
        if family in ('xgboost', 'mlp'):
            np.testing.assert_array_equal(before.features, after.features)
        elif family == 'trajectory_dtw_knn':
            for a, b in zip(before.trajectories, after.trajectories):
                np.testing.assert_array_equal(a, b)
            assert before.reference_library is None
        else:
            np.testing.assert_array_equal(before.sequences, after.sequences)
            assert not before.sequences[before.padding_mask].any()
            np.testing.assert_array_equal((~before.padding_mask).sum(axis=1), held.cutoff)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Evaluation UAV'):
        view.dataset(training_endpoints(raw, ids), 'lstm', 20)


def _check_real_model_fit_predict_and_save(family, tmp_path):
    settings, script, raw, ids, held = inputs()
    study.torch.set_num_threads(2)
    recipe, _ = study.read_recipe(family, 0, settings)
    view = FeatureView(raw, ids, script)
    training = view.dataset(training_endpoints(raw, ids, family == 'trajectory_dtw_knn'),
        family, recipe['lookback'], training=True, labels=True)
    validation = view.dataset(held, family, recipe['lookback'])
    model = study.make_model(family, recipe, 13,
                            iterations=None if family == 'trajectory_dtw_knn' else 2, smoke=True)
    model.fit(training, None)
    prediction = model.predict(validation)
    assert prediction.shape == (len(held),)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    if family == 'trajectory_dtw_knn':
        assert set(model.reference_library.metadata.uav_id) == set(ids)
        assert all((r <= 125).all() for r in model.reference_library.remaining_life)
    model.detach_training_monitor()
    model.save(tmp_path / 'model.joblib')
    from base import load_model_adapter
    restored = load_model_adapter(tmp_path / 'model.joblib')
    np.testing.assert_allclose(restored.predict(validation), prediction)


def _check_reporting_deterministic_dtw_is_not_three_independent_seeds(tmp_path):
    settings = json.loads(study.DEFAULT_SETTINGS.read_text())
    frames = []
    for family in study.CLASSES:
        for seed in ([13] if family == 'trajectory_dtw_knn' else [13, 37, 73]):
            frames.append(pd.DataFrame({'uav_id': ['a', 'b'], 'outer_fold': [0, 1],
                'model_family': family, 'model_seed': seed, 'suite': 'historical',
                'observed_rul': [10., 20.], 'predicted_rul': [11., 21.]}))
    study.reporting(tmp_path, frames, settings, complete=True)
    paired = pd.read_csv(tmp_path / 'reporting/paired_vs_xgboost.csv')
    assert len(paired) == 5 and paired.rmse_delta_vs_xgboost.eq(0).all()
    summary = pd.read_csv(tmp_path / 'reporting/summary.csv').set_index('model_family')
    assert summary.loc['trajectory_dtw_knn', 'seeds'] == 1


class EngineeredFeatureTests(unittest.TestCase):
    def test_causality(self):
        _check_causality_and_training_only_scaling()

    def test_six_real_models(self):
        for family in study.CLASSES:
            with self.subTest(family=family), scratch() as directory:
                _check_real_model_fit_predict_and_save(family, Path(directory))

    def test_paired_reporting(self):
        with scratch() as directory:
            _check_reporting_deterministic_dtw_is_not_three_independent_seeds(Path(directory))

    def test_cell_stopping_refit_resume_and_integrity(self):
        from unittest.mock import patch
        settings, _, raw, _, _ = inputs()
        recipe, _ = study.read_recipe('xgboost', 0, settings)
        recipe['hyperparameters']['maximum_trees'] = 3
        recipe['xgboost_patience'] = 2
        folds = pd.DataFrame({'uav_id': sorted(raw.uav_id.unique()), 'outer_fold': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]})
        endpoints = raw.loc[raw.flight_cycle.eq(5), ['uav_id', 'flight_cycle', 'RUL']].rename(columns={'flight_cycle': 'cutoff'})
        endpoints = endpoints.assign(scenario='historical_0', suite='historical', endpoint_seed=-1)
        original = study.make_model
        def cpu_model(*args, **kwargs):
            kwargs['smoke'] = True
            return original(*args, **kwargs)
        with scratch() as directory, patch.object(study, 'make_model', side_effect=cpu_model):
            rows = study.execute_cell(directory, settings, raw, folds, endpoints, recipe, 'xgboost', 0, 13, 'test')
            cell = directory / 'cells/xgboost__fold_0__seed_13'
            audit = json.loads((cell / 'fit_audit.json').read_text())
            self.assertFalse(set(audit['held_uavs']) & set(audit['train_uavs']))
            self.assertFalse(set(audit['stopping']['fit_uavs']) & set(audit['stopping']['stopping_uavs']))
            self.assertEqual(audit['training_rows'], 72)
            with patch.object(study, 'make_model', side_effect=AssertionError('A cached fit must not refit')):
                replay = study.execute_cell(directory, settings, raw, folds, endpoints, recipe, 'xgboost', 0, 13, 'test')
            np.testing.assert_allclose(rows.predicted_rul, replay.predicted_rul)
            with (cell / 'predictions.csv').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
                study.execute_cell(directory, settings, raw, folds, endpoints, recipe, 'xgboost', 0, 13, 'test')


if __name__ == '__main__':
    unittest.main()
