"""Grouped stopping, causal TCN inputs, checkpoints and hard experiment gates."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import multiprocessing
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT/'2_architecture_experiments/1_pipeline_experiments'
sys.path.insert(0, str(PIPELINE))
import run_bounded_comparison as runner
import bounded_contrastive as neural
import bounded_lightgbm as lightgbm
from confirmation_utils import EvaluationJob, select_uavs
from experiment_config import read_experiment_config
from run_experiment_definition import _execution_plan
from tabular_data_adapter import TabularDataset
from test_feature_comparison import raw_fixture, temporary_directory


def workflow(name='PE_32'):
    w = read_experiment_config(PIPELINE/f'experiments/{name}/settings.toml')['campaign_workflows'][name]
    # Legacy unit fixtures isolate the LGB/TCN branches; dedicated tests below
    # explicitly enable the real-config simpler baseline.
    return {**w, 'include_simple_baseline': False}


def fixture():
    raw = raw_fixture(count=15, cycles=80)
    rows = raw.loc[raw.flight_cycle.isin([10, 60])].reset_index(drop=True)
    metadata = rows[['uav_id', 'flight_cycle']].rename(columns={'flight_cycle': 'cutoff'})
    metadata['scenario'] = 'fixture'
    metadata['sample_id'] = [f's{i}' for i in range(len(rows))]
    data = TabularDataset(pd.DataFrame({'signal': rows.RUL, 'age': rows.flight_cycle}),
                          metadata, rows.RUL.astype(float), pd.Series(.5, index=rows.index))
    pieces = [replace(data, metadata=metadata.assign(suite=suite, endpoint_seed=seed, scenario=suite))
              for suite, seed in [('historical', -1), ('nominal', 1), ('unrestricted', 2)]]
    dev = replace(data, features=pd.concat([p.features for p in pieces], ignore_index=True),
                  metadata=pd.concat([p.metadata for p in pieces], ignore_index=True),
                  target=pd.concat([p.target for p in pieces], ignore_index=True), sample_weights=None)
    ids = sorted(metadata.uav_id.unique())
    jobs = [EvaluationJob(seed, fold, fold, 'outer', -1, frozenset(u for i, u in enumerate(ids) if i % 5 != fold),
                          frozenset(u for i, u in enumerate(ids) if i % 5 == fold))
            for seed in workflow()['split_seeds'] for fold in range(5)]
    return {'model_seed': 13}, {'training': data, 'development': dev, 'raw': raw}, jobs


def fake_control(training, calibration, held, source):
    assert not set(training.metadata.uav_id) & set(held.metadata.uav_id)
    assert set(calibration.metadata.uav_id) <= set(training.metadata.uav_id)
    return held.features.signal.to_numpy()+10., {'base_estimator_fits': 30}


def fake_bad_candidate(training, calibration, held, recipe, seed, workflow):
    assert not set(training.metadata.uav_id) & set(held.metadata.uav_id)
    return held.features.signal.to_numpy()+20., {'base_estimator_fits': 2}


class BoundedComparisonTests(unittest.TestCase):
    def test_simple_baseline_real_configuration_and_cache(self):
        actual = read_experiment_config(PIPELINE/'experiments/PE_32/settings.toml')['campaign_workflows']['PE_32']
        self.assertTrue(actual['include_simple_baseline'])
        self.assertEqual(runner.budget(actual)['screen_simple_base_estimator_fits_maximum'],60)
        source, data, jobs = fixture()
        data['cutoffs'] = np.array([10,60]*8)
        w = {**workflow(), 'include_simple_baseline':True,'max_workers':1}
        def simple(raw, ids, held, source, cutoffs):
            self.assertFalse(set(ids)&set(held.metadata.uav_id))
            return held.features.signal.to_numpy()+7, {'base_estimator_fits':12}
        with temporary_directory() as root, redirect_stdout(io.StringIO()), \
                patch.object(runner,'fit_run7',side_effect=fake_control), \
                patch.object(lightgbm,'fit_lightgbm',side_effect=fake_bad_candidate), \
                patch.object(runner.bounded_reference,'fit_reference',side_effect=simple) as reference:
            args=(root,jobs[0],source,data,w,['lgb_leaves7_leaf20'])
            rows=runner.task(args)
            self.assertEqual(reference.call_count,1)
            self.assertIn('simple_reproduction',set(rows.method))
            pd.testing.assert_frame_equal(rows,runner.task(args))
            self.assertEqual(reference.call_count,1)
            runner.reports.report(root,'screen',rows,w)
            runner.report_simple_comparisons(root,'screen',rows,w)
            comparisons=pd.read_csv(root/'stages/screen/reporting/paired_comparisons_vs_simple.csv')
            self.assertEqual(set(comparisons.reference),{'simple_reproduction'})
            self.assertTrue((comparisons.relative_rmse_improvement<0).all())

    def test_candidate_cannot_advance_when_simple_is_better(self):
        w={**workflow(),'include_simple_baseline':True}
        method='lgb_leaves7_leaf20_blend'
        summary=pd.DataFrame([{'method':m,'suite':s,'mean_fold_rmse':r,'r2':.92}
            for m,r in [(method,9.),('simple_reproduction',8.)]
            for s in ('historical','nominal','unrestricted')])
        comparisons=pd.DataFrame([{'method':method,'suite':s,'relative_rmse_improvement':.1,
            'fold_wins':8,'bootstrap_high':-.1} for s in ('historical','nominal','unrestricted')])
        decision=runner.lightgbm_decision(summary,comparisons,w,True)
        self.assertFalse(decision['eligible_for_final_review'])
        self.assertFalse(decision['candidates'][0]['checks']['beats_simple_baseline'])

    def test_submitted_reference_excludes_held_labels_and_future(self):
        source,data,jobs=fixture()
        source['reference_implementation']='other_pipelines/uav_rul_pipeline_v4 (1).py'
        job=jobs[0]; held=select_uavs(data['development'],job.validation_uavs)
        class SmallModel:
            def __init__(self,**kwargs): self.best_iteration=2
            def fit(self,x,y,**kwargs): self.mean=float(np.mean(y));return self
            def predict(self,x): return np.full(len(x),self.mean)
            def get_best_iteration(self): return self.best_iteration
        cutoffs=np.array([10,60]*8)
        with patch('xgboost.XGBRegressor',SmallModel),patch('catboost.CatBoostRegressor',SmallModel):
            p,a=runner.bounded_reference.fit_reference(data['raw'],job.training_uavs,held,source,cutoffs)
            changed=data['raw'].copy()
            mask=changed.uav_id.isin(job.validation_uavs)
            changed.loc[mask,'RUL']=999999
            sensors=[c for c in changed if c.startswith('telemetry')]
            changed.loc[mask&changed.flight_cycle.gt(60),sensors]=999999
            q,b=runner.bounded_reference.fit_reference(changed,job.training_uavs,replace(held,target=held.target+999999),source,cutoffs)
        np.testing.assert_array_equal(p,q)
        self.assertEqual(a['calibration_assignments'],b['calibration_assignments'])
        self.assertEqual(a['base_estimator_fits'],12)
        self.assertFalse(set(a['final_fit_uavs'])&set(a['final_stopping_uavs']))
        self.assertEqual(set(a['calibration_uavs']),set(job.training_uavs))

    def test_catalog_budget_and_hard_limits(self):
        for name in ('PE_32', 'PE_33'):
            w = workflow(name)
            runner.validate(w)
            _, _, steps = _execution_plan(PIPELINE/f'experiments/{name}/settings.toml', name)
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[0]['name'], 'validate_inputs')
        self.assertEqual(runner.budget(workflow())['screen_lightgbm_fits_maximum'], 320)
        self.assertEqual(runner.budget(workflow('PE_33'))['neural_fits_maximum'], 12)
        for change in ({'max_workers': 3}, {'split_seeds': [1, 2]}, {'model_seeds': [0]}, {'blend_weights': [0., 1.]}):
            with self.assertRaises(ValueError):
                runner.validate({**workflow(), **change})
        for change in ({'pilot_folds': [0, 1, 2]}, {'maximum_epochs': 31}, {'max_workers': 2}):
            with self.assertRaises(ValueError):
                runner.validate({**workflow('PE_33'), **change})

    def test_actual_lightgbm_held_labels_do_not_change_fit(self):
        _, data, jobs = fixture()
        job = jobs[0]
        train = select_uavs(data['training'], job.training_uavs)
        calibration = select_uavs(data['development'], job.training_uavs)
        held = select_uavs(data['development'], job.validation_uavs)
        w = {**workflow(), 'maximum_iterations': 5, 'stopping_patience': 2, 'cpu_threads': 1}
        p, audit = lightgbm.fit_lightgbm(train, calibration, held, 'lgb_leaves7_leaf20', 0, w)
        q, other = lightgbm.fit_lightgbm(train, calibration, replace(held, target=held.target+10000), 'lgb_leaves7_leaf20', 0, w)
        np.testing.assert_array_equal(p, q)
        self.assertFalse(set(audit['fit_uavs']) & set(audit['stopping_uavs']))
        self.assertEqual(set(audit['refit_uavs']), set(job.training_uavs))
        self.assertEqual(audit['best_iteration'], other['best_iteration'])
        self.assertEqual(audit['weight_sum'], len(job.training_uavs))
        with self.assertRaises(ValueError):
            lightgbm.fit_lightgbm(train, calibration, train, 'lgb_leaves7_leaf20', 0, w)

    def test_sequences_are_prefix_only_and_noise_preserves_padding_and_discrete_channels(self):
        _, data, jobs = fixture()
        train = select_uavs(data['training'], jobs[0].training_uavs)
        held = select_uavs(data['development'], jobs[0].validation_uavs)
        original = neural.Prefixes(data['raw'], train, 50)
        changed = data['raw'].copy()
        changed.loc[changed.flight_cycle.gt(60), list(neural.SENSORS_22)] = 1e9
        changed['RUL'] = -10000
        altered = neural.Prefixes(changed, train, 50)
        first, second = original.transform(held.metadata), altered.transform(held.metadata)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x, mask, _ = first
        augmented = neural.augment(x, mask, np.random.default_rng(0), .05)
        self.assertTrue(torch.equal(augmented[mask], x[mask]))
        untouched = sorted(set(range(len(neural.SENSORS_22)))-set(neural.NOISE_CHANNELS))
        self.assertTrue(torch.equal(augmented[:, :, untouched], x[:, :, untouched]))
        self.assertFalse(torch.equal(augmented, x))

    def test_pair_sampler_and_contrastive_gradients(self):
        _, data, _ = fixture()
        train = data['training']
        negative, distance = neural.pairs(train.metadata, train.target, np.random.default_rng(1), .08)
        np.testing.assert_array_equal(train.metadata.uav_id.to_numpy(), train.metadata.uav_id.to_numpy()[negative])
        self.assertTrue((distance > .08).all())
        a, p, n = [torch.randn(5, 4, requires_grad=True) for _ in range(3)]
        loss = neural.contrastive_loss(a, p, n, torch.ones(5), .2)
        loss.backward()
        self.assertTrue(all(torch.isfinite(t.grad).all() for t in (a, p, n)))
        zero = neural.contrastive_loss(a, p, n, torch.zeros(5), .2)
        self.assertEqual(float(zero.detach()), 0.)
        _, gap = neural.pairs(train.metadata, np.full(len(train), 200.), np.random.default_rng(1), .08)
        self.assertTrue((gap == 0).all())

    def test_actual_tcn_three_arms_and_held_label_isolation(self):
        _, data, jobs = fixture()
        job = jobs[0]
        train = select_uavs(data['training'], job.training_uavs)
        calibration = select_uavs(data['development'], job.training_uavs)
        held = select_uavs(data['development'], job.validation_uavs)
        w = {**workflow('PE_33'), 'maximum_epochs': 2, 'stopping_patience': 1, 'cpu_threads': 1}
        for arm in neural.ARMS:
            p, audit = neural.fit_contrastive(data['raw'], train, calibration, held, arm, 0, w)
            self.assertTrue(np.isfinite(p).all())
            self.assertLessEqual(audit['best_epoch'], 2)
            self.assertFalse(set(audit['fit_uavs']) & set(audit['stopping_uavs']))
            self.assertEqual(audit['neural_fits'], 2)
        q, other = neural.fit_contrastive(data['raw'], train, calibration, replace(held, target=held.target+1e6), arm, 0, w)
        np.testing.assert_array_equal(p, q)
        self.assertEqual(audit['best_epoch'], other['best_epoch'])

    def test_task_caches_control_across_seeds_and_rejects_tampering(self):
        source, data, jobs = fixture()
        w = {**workflow(), 'max_workers': 1}
        with temporary_directory() as root, redirect_stdout(io.StringIO()), patch.object(runner, 'fit_run7', side_effect=fake_control) as control, \
                patch.object(lightgbm, 'fit_lightgbm', side_effect=fake_bad_candidate) as candidate:
            args = (root, jobs[0], source, data, w, ['lgb_leaves7_leaf20'])
            first = runner.task(args)
            self.assertEqual(control.call_count, 4)
            self.assertEqual(candidate.call_count, 8)
            second = runner.task(args)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(control.call_count, 4)
            self.assertEqual(candidate.call_count, 8)
            selections = list((root/'cells').rglob('selection_*.json'))
            self.assertTrue(all(json.loads(p.read_text())['chosen']['weight'] == 0. for p in selections))
            path = next(p for p in (root/'cells').rglob('*.json') if not p.name.startswith('selection_'))
            content = json.loads(path.read_text())
            content['payload']['predictions'][0]['predicted_rul'] += 1
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                runner.task(args)

    def test_full_screen_failure_never_launches_confirmation(self):
        source, data, jobs = fixture()
        w = {**workflow(), 'max_workers': 1, 'bootstrap_repetitions': 20}
        config = PIPELINE/'experiments/PE_32/settings.toml'
        with temporary_directory() as root, redirect_stdout(io.StringIO()), \
                patch.object(runner.data_tools, 'prepare', return_value=(source, data, jobs, [], pd.DataFrame(), pd.DataFrame())), \
                patch.object(runner, 'fit_run7', side_effect=fake_control) as control, \
                patch.object(lightgbm, 'fit_lightgbm', side_effect=fake_bad_candidate) as candidate:
            decision = runner.run(w, root, 'all', config)
            self.assertEqual(decision['status'], 'stop_after_screen')
            self.assertFalse((root/'stages/confirmation').exists())
            self.assertEqual(control.call_count, 20)
            self.assertEqual(candidate.call_count, 160)
            resumed = runner.run(w, root, 'confirm', config)
            self.assertEqual(resumed, decision)
            self.assertEqual(control.call_count, 20)
            self.assertEqual(candidate.call_count, 160)
            with self.assertRaisesRegex(ValueError, 'Registered settings'):
                runner.run({**w, 'learning_rate': .02}, root, 'check', config)

    def test_success_confirms_only_one_recipe(self):
        source, data, jobs = fixture()
        w = {**workflow(), 'max_workers': 1, 'bootstrap_repetitions': 20}
        def good(training, calibration, held, recipe, seed, settings):
            return held.features.signal.to_numpy()+5., {'base_estimator_fits': 2}
        with temporary_directory() as root, redirect_stdout(io.StringIO()), \
                patch.object(runner.data_tools, 'prepare', return_value=(source, data, jobs, [], pd.DataFrame(), pd.DataFrame())), \
                patch.object(runner, 'fit_run7', side_effect=fake_control) as control, \
                patch.object(lightgbm, 'fit_lightgbm', side_effect=good) as candidate:
            decision = runner.run(w, root, 'all', PIPELINE/'experiments/PE_32/settings.toml')
            self.assertTrue(decision['eligible_for_final_review'])
            self.assertFalse(decision['promoted'])
            self.assertEqual(control.call_count, 60)
            self.assertEqual(candidate.call_count, 240)
            rows = pd.read_csv(root/'stages/confirmation/reporting/summary.csv')
            self.assertEqual(set(rows.method), {'run7', decision['selected_recipe'], decision['selected_recipe']+'_blend'})
            self.assertEqual(len(decision['candidates']), 1)

    def test_pilot_stops_at_two_folds(self):
        source, data, jobs = fixture()
        w = {**workflow('PE_33'), 'bootstrap_repetitions': 20}
        jobs = [replace(job, split_seed=w['split_seeds'][0]) for job in jobs[:5]]
        def fake_neural(raw, training, calibration, held, arm, seed, settings):
            return held.features.signal.to_numpy()+20., {'neural_fits': 2}
        with temporary_directory() as root, redirect_stdout(io.StringIO()), \
                patch.object(runner.data_tools, 'prepare', return_value=(source, data, jobs, [], pd.DataFrame(), pd.DataFrame())), \
                patch.object(runner, 'fit_run7', side_effect=fake_control) as control, \
                patch.object(neural, 'fit_contrastive', side_effect=fake_neural) as candidate:
            decision = runner.run(w, root, 'pilot', PIPELINE/'experiments/PE_33/settings.toml')
            self.assertEqual(decision['status'], 'stop_pilot')
            self.assertEqual(control.call_count, 2)
            self.assertEqual(candidate.call_count, 6)
            self.assertEqual(len(list((root/'cells').iterdir())), 2)
            self.assertFalse(decision['automatic_expansion'])

    def test_windows_spawn_reuses_serial_checkpoints(self):
        source, data, jobs = fixture()
        w = workflow()
        with temporary_directory() as root, redirect_stdout(io.StringIO()), \
                patch.object(runner, 'fit_run7', side_effect=fake_control), \
                patch.object(lightgbm, 'fit_lightgbm', side_effect=fake_bad_candidate):
            args = [(root, job, source, data, w, ['lgb_leaves7_leaf20']) for job in jobs[:2]]
            serial = [runner.task(argument) for argument in args]
            # Child processes have no mocks: they must validate and reuse every
            # cached prediction instead of trying to fit the incomplete source.
            with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
                parallel = list(pool.map(runner.task, args))
            for a, b in zip(serial, parallel):
                pd.testing.assert_frame_equal(a, b)

    def test_success_and_failure_decision_thresholds(self):
        w = workflow()
        method = 'lgb_leaves7_leaf20_blend'
        summary = pd.DataFrame([{'method': method, 'suite': suite, 'mean_fold_rmse': 9., 'r2': .91}
                                for suite in ('historical', 'nominal', 'unrestricted')])
        comparisons = pd.DataFrame([{'method': method, 'suite': suite, 'relative_rmse_improvement': .05,
                                     'fold_wins': 8, 'bootstrap_high': -.1}
                                    for suite in ('historical', 'nominal', 'unrestricted')])
        self.assertTrue(runner.lightgbm_decision(summary, comparisons, w, True)['eligible_for_final_review'])
        worse = summary.copy(); worse['r2'] = .89
        self.assertFalse(runner.lightgbm_decision(worse, comparisons, w, True)['eligible_for_final_review'])
        comparisons.loc[comparisons.suite.eq('unrestricted'), 'relative_rmse_improvement'] = -.06
        self.assertIsNone(runner.lightgbm_decision(summary, comparisons, w)['selected_recipe'])
        pilot_summary = pd.DataFrame([{'method': arm, 'suite': 'historical', 'mean_fold_rmse': value}
                                      for arm, value in [('tcn_mse', 20.), ('tcn_augmentation', 18.), ('tcn_contrastive', 18.)]])
        pilot_comparisons = pd.DataFrame([{'method': 'tcn_contrastive_blend', 'suite': suite,
                                          'relative_rmse_improvement': .02, 'fold_wins': 2}
                                         for suite in ('historical', 'nominal', 'unrestricted')])
        stopped = runner.pilot_decision(pilot_summary, pilot_comparisons, workflow('PE_33'))
        self.assertEqual(stopped['status'], 'stop_pilot')
        self.assertFalse(stopped['checks']['beats_tcn_augmentation'])
        pilot_summary.loc[pilot_summary.method.eq('tcn_contrastive'), 'mean_fold_rmse'] = 16.
        passed = runner.pilot_decision(pilot_summary, pilot_comparisons, workflow('PE_33'))
        self.assertEqual(passed['status'], 'worth_followup')
        self.assertFalse(passed['automatic_expansion'])
        self.assertFalse(passed['promoted'])


if __name__ == '__main__':
    unittest.main()
