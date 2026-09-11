"""Weight mass, nested isolation, cache integrity and pilot decisions."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT/'2_architecture_experiments/1_pipeline_experiments'
sys.path.insert(0,str(PIPELINE))
import adaptive_uav_weighting as adaptive
import run_adaptive_uav_weighting as runner
from experiment_config import read_experiment_config
from confirmation_utils import select_uavs
from test_bounded_comparison import fixture
from test_feature_comparison import temporary_directory


def workflow():
    return read_experiment_config(PIPELINE/'experiments/PE_34/settings.toml')['campaign_workflows']['PE_34']


def fake_control(training,calibration,held,source):
    assert not set(training.metadata.uav_id)&set(held.metadata.uav_id)
    assert set(calibration.metadata.uav_id) <= set(training.metadata.uav_id)
    # Depend on fitting labels, but not held labels. A leaked fitting UAV would
    # change this model's predictions and therefore downstream difficulty ranks.
    mean = float(training.target.mean())
    return held.features.signal.to_numpy()+mean/10, {'base_estimator_fits':30}


class FakeAdapter:
    internal_folds = 4
    target_policy = type('Policy',(),{'mode':'piecewise_cap','maximum_rul':125})()

    def _fit_members(self,training,prediction,*,retain):
        if retain:
            self.value = float(np.average(training.target,weights=training.sample_weights))
        return np.ones((len(prediction),6)), []

    def fit(self,training,validation):
        cal = self._calibration_data(training)
        groups = cal.metadata.uav_id.to_numpy()
        for _,idx in GroupKFold(4).split(cal.features,groups=groups):
            held = set(groups[idx])
            self._fit_members(select_uavs(training,set(groups)-held),select_uavs(cal,held),retain=False)
        self._fit_members(training,cal,retain=True)

    def predict(self,held):
        return np.full(len(held),self.value)


class AdaptiveWeightingTests(unittest.TestCase):
    def test_fixed_protocol_and_weight_mass(self):
        runner.validate(workflow())
        with self.assertRaises(ValueError):
            runner.validate({**workflow(),'multipliers':[1.,2.,10.]})
        _,data,_ = fixture()
        training = data['training']
        hard = sorted(training.metadata.uav_id.unique())[:4]
        for multiplier in (1.,1.5,2.):
            weighted = adaptive.reweight(training,hard,multiplier)
            self.assertAlmostEqual(weighted.sample_weights.sum(),training.sample_weights.sum())
            ratio = weighted.sample_weights.to_numpy()/training.sample_weights.to_numpy()
            flag = training.metadata.uav_id.isin(hard).to_numpy()
            self.assertAlmostEqual(ratio[flag][0]/ratio[~flag][0],multiplier)
        with self.assertRaises(ValueError):
            adaptive.reweight(training,['unknown'],2.)

    def test_uav_rmse_ranking_and_stable_ties(self):
        rows = pd.DataFrame({'uav_id':['a','a','b','c','d'],'scenario':['1','2','1','1','1'],
            'cutoff':[10,20,10,10,10],'suite':'historical','observed_rul':[0.]*5,
            'predicted_rul':[0.,10.,6.,1.,1.]})
        hard,ranking = adaptive.hardest_uavs(rows)
        self.assertEqual(hard,['a']) # sqrt(mean squares), not median absolute error
        self.assertEqual(len(ranking),4)
        rows.predicted_rul = 1.
        self.assertEqual(adaptive.hardest_uavs(rows)[0],['a'])
        with self.assertRaises(ValueError):
            adaptive.hardest_uavs(rows.assign(suite='unrestricted'))

    def test_nested_fit_local_weighting_cache_and_held_label_isolation(self):
        source,data,jobs = fixture();job = jobs[0]
        source.update(specification='3_final_model_training_and_inference/runs/run_7/1_winning_architecture_selection/artifacts/residual_ensemble_phase_2_specification.json',
            source_contract='3_final_model_training_and_inference/runs/run_7/1_winning_architecture_selection/artifacts/residual_ensemble_contract.json')
        with temporary_directory() as root, redirect_stdout(io.StringIO()), \
                patch('run_bounded_comparison.fit_run7',side_effect=fake_control) as control, \
                patch.object(adaptive.ModelAdapterFactory,'create',side_effect=lambda *a,**k:FakeAdapter()):
            engine = adaptive.AdaptiveEngine(root,job,source,data,workflow())
            first = engine.fit('adaptive_uav_1_5')
            calls = control.call_count
            self.assertEqual(calls,15) # 5 member-fit scopes, 3 difficulty fits each
            engine.fit('adaptive_uav_2_0')
            self.assertEqual(control.call_count,calls) # difficulty is shared across arms
            pd.testing.assert_frame_equal(first,engine.fit('adaptive_uav_1_5'))
            saved = json.loads((engine.directory/'adaptive_uav_1_5.json').read_text())
            audits = saved['payload']['result']['audit']['member_fits']
            for a in audits:
                self.assertTrue(set(a['hard_uavs']) <= set(a['fit_uavs']))
                self.assertFalse(set(a['fit_uavs'])&set(job.validation_uavs))
                self.assertAlmostEqual(a['original_weight_sum'],a['weighted_sum'])
                if not a['retain']:
                    self.assertFalse(set(a['fit_uavs'])&set(a['prediction_uavs']))
            # Perturb all outer-held labels in both training and development.
            changed = dict(data)
            for name in ('training','development'):
                d = data[name];target = d.target.copy()
                target.loc[d.metadata.uav_id.isin(job.validation_uavs)] += 100000
                changed[name] = replace(d,target=target)
            with temporary_directory() as other:
                second = adaptive.AdaptiveEngine(other,job,source,changed,workflow()).fit('adaptive_uav_1_5')
            np.testing.assert_array_equal(first.predicted_rul,second.predicted_rul)
            with self.assertRaises(ValueError):
                engine.difficulty(set(job.training_uavs)|set(job.validation_uavs))
            path = next(engine.directory.glob('difficulty_*.json'))
            content = json.loads(path.read_text());content['payload']['result']['hard_uavs']=['bad']
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError,'checksum'):
                engine.difficulty(content['payload']['contract']['training_uavs'])

    def test_pilot_gate_requires_global_improvement_and_never_promotes(self):
        summaries = [{'method':m,'suite':s,'mean_fold_rmse':9. if m!='simple_reproduction' else 11.}
            for m in [*adaptive.ARMS,'simple_reproduction'] for s in ('historical','nominal','unrestricted')]
        comparisons = [{'method':m,'suite':s,'relative_rmse_improvement':.03,'fold_wins':4,'bootstrap_high':-.1}
            for m in adaptive.ARMS for s in ('historical','nominal','unrestricted')]
        passed = runner.decision(pd.DataFrame(summaries),pd.DataFrame(comparisons),workflow())
        self.assertEqual(passed['status'],'worth_confirmation')
        self.assertFalse(passed['promoted']);self.assertFalse(passed['automatic_expansion'])
        for row in comparisons:
            if row['suite']=='historical':row['relative_rmse_improvement']=.01
        failed = runner.decision(pd.DataFrame(summaries),pd.DataFrame(comparisons),workflow())
        self.assertEqual(failed['status'],'stop_after_pilot')


if __name__ == '__main__':
    unittest.main()
