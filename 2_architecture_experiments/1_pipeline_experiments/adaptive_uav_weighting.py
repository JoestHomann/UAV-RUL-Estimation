"""One-pass, grouped-OOF UAV difficulty weighting with fit-local calibration.

Only base-estimator training weights change. Run 7's blend/residual calibration
objective remains unchanged. Every internal calibration fit estimates its own
difficulty weights without access to that calibration fold's held UAVs.
"""
from dataclasses import replace
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from campaign_models import training_endpoints
from campaign_reporting import KEYS
from confirmation_utils import select_uavs
from followup_experiment_utils import atomic_json, input_path
from run_bounded_comparison import Engine, checksum, data_digest
from model_registry import ModelAdapterFactory
from no_op_training_monitor import NoOpTrainingMonitor


ARMS = {'adaptive_uav_1_5': 1.5, 'adaptive_uav_2_0': 2.0}


def hardest_uavs(rows, fraction=.25):
    if rows.suite.nunique() != 1 or rows.suite.iloc[0] != 'historical':
        raise ValueError('Difficulty must use the predefined historical profile')
    if rows.duplicated(['uav_id','scenario','cutoff']).any():
        raise ValueError('Duplicate difficulty endpoints')
    if not np.isfinite(rows[['observed_rul','predicted_rul']].to_numpy(float)).all():
        raise ValueError('Nonfinite difficulty errors')
    errors = rows.assign(squared=(rows.predicted_rul-rows.observed_rul)**2).groupby('uav_id').squared.mean().pow(.5)
    ranked = errors.rename('rmse').reset_index().sort_values(['rmse','uav_id'],ascending=[False,True])
    count = max(1,int(np.ceil(len(ranked)*fraction)))
    return ranked.uav_id.iloc[:count].tolist(), ranked.to_dict('records')


def reweight(training, hard_ids, multiplier):
    if multiplier not in (1.,1.5,2.):
        raise ValueError('Unexpected difficulty multiplier')
    ids = set(training.metadata.uav_id)
    if not set(hard_ids) <= ids or training.sample_weights is None:
        raise ValueError('Weight selection does not belong to this fitting set')
    base = training.sample_weights.to_numpy(float)
    if not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Base weights must be finite and positive')
    relative = np.where(training.metadata.uav_id.isin(hard_ids),multiplier,1.)
    values = base*relative
    values *= base.sum()/values.sum()
    return replace(training,sample_weights=pd.Series(values,index=training.sample_weights.index))


def envelope(path, contract, build):
    """Check cache contents and their exact data/membership contract."""
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['sha256'] != checksum(saved['payload']) or saved['payload']['contract'] != contract:
            raise ValueError(f'Adaptive checkpoint checksum/contract changed: {path}')
        return saved['payload']['result']
    result = build()
    payload = {'contract':contract,'result':result}
    atomic_json(path,{'payload':payload,'sha256':checksum(payload)})
    return result


class AdaptiveEngine:
    def __init__(self,root,job,source,data,workflow):
        self.root,self.job,self.source,self.data,self.workflow = Path(root),job,source,data,workflow
        self.base = Engine(root,job,source,data,workflow)
        self.directory = self.root/'adaptive_cells'/f'{job.split_seed}_{job.outer_fold}'
        self.directory.mkdir(parents=True,exist_ok=True)

    def difficulty(self, ids):
        ids = set(ids)
        if not ids <= set(self.job.training_uavs):
            raise ValueError('Difficulty selection includes outer-held UAVs')
        training = select_uavs(self.data['training'],ids)
        calibration = training_endpoints(self.data['development'],ids)
        contract = {'training_uavs':sorted(ids),'fraction':self.workflow['hard_fraction'],
            'folds':self.workflow['difficulty_folds'],'training_digest':data_digest(training),
            'calibration_digest':data_digest(calibration)}
        path = self.directory/f'difficulty_{checksum(contract)}.json'
        def build():
            values = np.asarray(sorted(ids))
            partitions = KFold(self.workflow['difficulty_folds'],shuffle=True,
                               random_state=self.job.split_seed+self.job.outer_fold)
            frames,audits = [],[]
            for fit,held in partitions.split(values):
                fitting,excluded = set(values[fit]),set(values[held])
                if fitting & excluded:
                    raise ValueError('Difficulty fit/held UAV overlap')
                rows = self.base.fit(fitting,excluded,'run7',self.source['model_seed'])
                frames.append(rows.loc[rows.suite.eq('historical')])
                audits.append({'fit_uavs':sorted(fitting),'held_uavs':sorted(excluded)})
            rows = pd.concat(frames,ignore_index=True)
            if set(rows.uav_id) != ids:
                raise ValueError('Incomplete grouped OOF difficulty coverage')
            hard,ranked = hardest_uavs(rows,self.workflow['hard_fraction'])
            return {'hard_uavs':hard,'ranked_rmse':ranked,'partitions':audits,
                'training_uavs':sorted(ids),'prediction_checksum':checksum(rows.to_dict('records'))}
        return envelope(path,contract,build)

    def fit(self, method):
        multiplier = ARMS[method]
        ids = set(self.job.training_uavs)
        training = select_uavs(self.data['training'],ids)
        calibration = training_endpoints(self.data['development'],ids)
        held = select_uavs(self.data['development'],self.job.validation_uavs)
        contract = {'method':method,'multiplier':multiplier,'training_uavs':sorted(ids),
            'held_uavs':sorted(self.job.validation_uavs),'training_digest':data_digest(training),
            'calibration_digest':data_digest(calibration),'held_digest':data_digest(held)}
        def build():
            model = ModelAdapterFactory(input_path(self.source['specification'])).create(
                'residual_corrected_tree_ensemble',{'ensemble_contract_path':self.source['source_contract']},
                seed=int(self.source['model_seed']),allow_disabled=True,training_monitor=NoOpTrainingMonitor())
            if model.internal_folds != 4 or model.target_policy.mode != 'piecewise_cap' or model.target_policy.maximum_rul != 125:
                raise ValueError('Run 7 recipe changed')
            if list(training.features) != list(calibration.features):
                raise ValueError('Run 7 calibration feature order differs')
            model._calibration_data = MethodType(lambda model,data:training_endpoints(calibration,set(data.metadata.uav_id)),model)
            original_fit_members = model._fit_members
            audits = []
            def fit_members(model, fit_data, prediction_data, *, retain):
                local_ids = set(fit_data.metadata.uav_id)
                if not retain and local_ids & set(prediction_data.metadata.uav_id):
                    raise ValueError('Calibration-held UAVs leaked into member fitting')
                # No weighted model calls itself recursively: difficulty always
                # comes from unchanged Run 7 fits on only these local UAVs.
                difficulty = self.difficulty(local_ids)
                weighted = reweight(fit_data,difficulty['hard_uavs'],multiplier)
                audits.append({'fit_uavs':sorted(local_ids),'prediction_uavs':sorted(set(prediction_data.metadata.uav_id)),
                    'retain':retain,'hard_uavs':difficulty['hard_uavs'],
                    'original_weight_sum':float(fit_data.sample_weights.sum()),
                    'weighted_sum':float(weighted.sample_weights.sum()),
                    'minimum_row_weight':float(weighted.sample_weights.min()),
                    'maximum_row_weight':float(weighted.sample_weights.max())})
                print(f'PE_34 fold={self.job.outer_fold} {method}: fitting 6 members on {len(local_ids)} UAVs; hard={len(difficulty["hard_uavs"])}',flush=True)
                return original_fit_members(weighted,prediction_data,retain=retain)
            model._fit_members = MethodType(fit_members,model)
            started = time.perf_counter()
            model.fit(training,None)
            prediction = model.predict(held)
            if len(audits) != 5 or sum(a['retain'] for a in audits) != 1:
                raise ValueError('Expected four calibration member fits and one final member fit')
            if np.asarray(prediction).shape != (len(held),) or not np.isfinite(prediction).all():
                raise ValueError('Invalid adaptive predictions')
            rows = held.metadata[['uav_id','scenario','suite','endpoint_seed','cutoff']].copy()
            rows['observed_rul'] = held.target.to_numpy(float)
            rows['split_seed'],rows['outer_fold'],rows['model_seed'] = self.job.split_seed,self.job.outer_fold,self.source['model_seed']
            rows['predicted_rul'],rows['method'] = prediction,method
            return {'predictions':rows.to_dict('records'),'audit':{'base_estimator_fits':30,
                'elapsed_seconds_including_difficulty':time.perf_counter()-started,'member_fits':audits,
                'calibration_objective_unchanged':True,'one_reweighting_pass':True}}
        result = envelope(self.directory/f'{method}.json',contract,build)
        return pd.DataFrame(result['predictions'])[[*KEYS,'predicted_rul','method']]
