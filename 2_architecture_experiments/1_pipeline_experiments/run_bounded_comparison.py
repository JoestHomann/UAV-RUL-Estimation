"""PE_32 bounded LightGBM screen/confirmation; PE_33 optional TCN futility pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import bounded_comparison_data as data_tools
import bounded_lightgbm
import campaign_reporting as reports
from advanced_r2_utils import load_workflow
from campaign_models import fit_run7, training_endpoints
from confirmation_utils import select_uavs
from followup_experiment_utils import atomic_csv, atomic_json, register_run


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def data_digest(data):
    h = hashlib.sha256()
    for value in (data.features, data.metadata, data.target, data.sample_weights):
        h.update(b'None' if value is None else pd.util.hash_pandas_object(value, index=False).to_numpy().tobytes())
    h.update(json.dumps(list(data.features)).encode())
    return h.hexdigest()


def freeze(path, value):
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError(f'Frozen decision changed: {path}; use a new run directory')
    atomic_json(path, value)


def validate(w):
    if w['kind'] not in ('lightgbm', 'contrastive'):
        raise ValueError('Unknown comparison kind')
    for key in ('split_seeds', 'model_seeds', 'endpoint_seeds'):
        if not w[key] or len(set(w[key])) != len(w[key]) or any(type(v) is not int or v < 0 for v in w[key]):
            raise ValueError(f'Invalid seeds: {key}')
    for key in ('max_workers', 'cpu_threads', 'bootstrap_repetitions', 'stopping_patience'):
        if type(w[key]) is not int or w[key] < 1:
            raise ValueError(f'Invalid resource limit: {key}')
    if len(w['endpoint_seeds']) != 3 or w['max_workers'] > 2:
        raise ValueError('Bounded runs allow three endpoint draws and at most two workers')
    for key in ('nominal_maximum_regression', 'stress_maximum_regression', 'learning_rate'):
        if not 0 < w[key] < 1:
            raise ValueError(f'Invalid parameter: {key}')
    if w['kind'] == 'lightgbm':
        if len(w['split_seeds']) != 3 or len(w['model_seeds']) != 2 or w['selection_folds'] != 3:
            raise ValueError('LightGBM budget is one five-fold screen and two confirmation partitions, two model seeds')
        if w['blend_weights'] != [0., .1, .25, .5, 1.] or not 1 <= w['maximum_iterations'] <= 2000:
            raise ValueError('LightGBM grid or boosting budget changed')
        if not 0 < w['minimum_relative_improvement'] < 1 or not 0 < w['minimum_pooled_r2'] < 1:
            raise ValueError('Invalid LightGBM decision gate')
    else:
        if len(w['split_seeds']) != 1 or len(w['model_seeds']) != 1 or w['pilot_folds'] != [0, 1]:
            raise ValueError('Pilot is limited to the first two declared folds and one model seed')
        if (w['max_workers'] != 1 or not 1 <= w['maximum_epochs'] <= 30 or w['channels'] != 32
                or w['residual_blocks'] != 2 or w['lookback'] != 50 or w['fixed_blend_weight'] != .1):
            raise ValueError('Pilot architecture, blend or compute budget changed')
        for key in ('noise_std', 'contrastive_weight', 'temperature', 'minimum_negative_gap',
                    'standalone_minimum_improvement', 'blend_minimum_improvement', 'dropout'):
            if not 0 < w[key] < 1:
                raise ValueError(f'Invalid pilot parameter: {key}')
        if type(w['batch_size']) is not int or not 1 <= w['batch_size'] <= 256:
            raise ValueError('Invalid pilot batch size')


def budget(w):
    if w['kind'] == 'contrastive':
        return {'outer_folds': 2, 'arms': 3, 'neural_fits_maximum': 12,
                'run7_base_estimator_fits_maximum': 60, 'automatic_expansion': False}
    # Each outer task: 3 selection folds + refit. Run 7 is shared by both model
    # seeds and all four recipes. Each LGB call includes stopping fit and refit.
    return {'screen_outer_folds': 5, 'confirmation_outer_folds_maximum': 10,
            'lightgbm_recipes': 4, 'confirmation_recipes_maximum': 1,
            'screen_lightgbm_fits_maximum': 5*2*4*4*2,
            'confirmation_lightgbm_fits_maximum': 10*2*1*4*2,
            'screen_run7_base_estimator_fits_maximum': 5*4*30,
            'confirmation_run7_base_estimator_fits_maximum': 10*4*30}


class Engine:
    def __init__(self, root, job, source, data, workflow):
        self.root, self.job, self.source, self.data, self.workflow = Path(root), job, source, data, workflow
        self.directory = self.root/'cells'/f'{job.split_seed}_{job.outer_fold}'
        self.directory.mkdir(parents=True, exist_ok=True)

    def fit(self, train_ids, held_ids, method, seed):
        train_ids, held_ids = set(train_ids), set(held_ids)
        if not train_ids or not held_ids or train_ids & held_ids:
            raise ValueError('Empty or overlapping UAV membership')
        training = select_uavs(self.data['training'], train_ids)
        calibration = select_uavs(self.data['development'], train_ids)
        held = select_uavs(self.data['development'], held_ids)
        if set(training.metadata.uav_id) != train_ids or set(held.metadata.uav_id) != held_ids:
            raise ValueError('Requested UAV coverage incomplete')
        # Run 7 always uses its frozen seed; reuse exactly the same prediction
        # across candidate seeds rather than fitting the deterministic control twice.
        fit_seed = self.source['model_seed'] if method == 'run7' else seed
        identity = {'method': method, 'seed': fit_seed, 'train_uavs': sorted(train_ids), 'held_uavs': sorted(held_ids)}
        path = self.directory/f'{checksum(identity)}.json'
        contract = {**identity, 'training_digest': data_digest(training),
                    'calibration_digest': data_digest(calibration), 'held_digest': data_digest(held)}
        expected = held.metadata[['uav_id', 'scenario', 'suite', 'endpoint_seed', 'cutoff']].copy()
        expected['observed_rul'] = held.target.to_numpy(float)
        expected['split_seed'], expected['outer_fold'], expected['model_seed'] = self.job.split_seed, self.job.outer_fold, fit_seed
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            payload = saved['payload']
            if saved['sha256'] != checksum(payload) or payload['contract'] != contract:
                raise ValueError('Checkpoint checksum or data contract changed')
            rows = pd.DataFrame(payload['predictions'])
            if not rows[reports.KEYS].equals(expected[reports.KEYS]) or not np.isfinite(rows.predicted_rul).all():
                raise ValueError('Checkpoint endpoint coverage changed')
        else:
            print(f"{self.workflow['kind']} split={self.job.split_seed} fold={self.job.outer_fold} {method} seed={fit_seed}: {len(train_ids)} train / {len(held_ids)} held UAVs", flush=True)
            if method == 'run7':
                prediction, audit = fit_run7(training, training_endpoints(calibration, train_ids), held, self.source)
            elif method in bounded_lightgbm.RECIPES:
                prediction, audit = bounded_lightgbm.fit_lightgbm(training, calibration, held, method, fit_seed, self.workflow)
            else:
                from bounded_contrastive import fit_contrastive
                prediction, audit = fit_contrastive(self.data['raw'], training, calibration, held, method, fit_seed, self.workflow)
            if np.asarray(prediction).shape != (len(held),) or not np.isfinite(prediction).all():
                raise ValueError('Invalid fitted predictions')
            rows = expected.assign(predicted_rul=prediction)
            payload = {'contract': contract, 'audit': audit, 'predictions': rows.to_dict('records')}
            atomic_json(path, {'payload': payload, 'sha256': checksum(payload)})
        return rows[[*reports.KEYS, 'predicted_rul']].assign(model_seed=seed)


def choose_weight(candidate, control, w):
    scores = []
    for weight in w['blend_weights']:
        blended = reports.blended(candidate, control, weight)
        scores.append({'weight': weight, **{suite: reports.score(blended, suite)
                                           for suite in ('historical', 'nominal', 'unrestricted')}})
    eligible = [row for row in scores
                if row['nominal'] <= reports.score(control, 'nominal')*(1+w['nominal_maximum_regression'])
                and row['unrestricted'] <= reports.score(control, 'unrestricted')*(1+w['stress_maximum_regression'])]
    return min(eligible, key=lambda row: (row['historical'], row['weight'])), scores


def task(arguments):
    root, job, source, data, w, recipes = arguments
    engine = Engine(root, job, source, data, w)
    output = []
    with threadpool_limits(limits=w['cpu_threads']):
        for seed in w['model_seeds']:
            control = engine.fit(job.training_uavs, job.validation_uavs, 'run7', seed)
            output.append(control.assign(method='run7'))
            if w['kind'] == 'contrastive':
                from bounded_contrastive import ARMS
                for arm in ARMS:
                    rows = engine.fit(job.training_uavs, job.validation_uavs, arm, seed)
                    output.append(rows.assign(method=arm))
                    output.append(reports.blended(rows, control, w['fixed_blend_weight']).assign(method=arm+'_blend'))
                continue
            values = np.asarray(sorted(job.training_uavs))
            partitions = [(set(values[a]), set(values[b])) for a, b in KFold(
                w['selection_folds'], shuffle=True, random_state=job.split_seed+job.outer_fold).split(values)]
            inner_control = pd.concat([engine.fit(a, b, 'run7', seed) for a, b in partitions], ignore_index=True)
            for recipe in recipes:
                candidate = pd.concat([engine.fit(a, b, recipe, seed) for a, b in partitions], ignore_index=True)
                chosen, scores = choose_weight(candidate, inner_control, w)
                selection = {'recipe': recipe, 'seed': seed, 'chosen': chosen, 'scores': scores,
                    'selection_uses_outer_labels': False, 'training_uavs': sorted(job.training_uavs),
                    'inner_partitions': [{'train': sorted(a), 'held': sorted(b)} for a, b in partitions]}
                freeze(engine.directory/f'selection_{recipe}_{seed}.json', selection)
                rows = engine.fit(job.training_uavs, job.validation_uavs, recipe, seed)
                output.append(rows.assign(method=recipe))
                output.append(reports.blended(rows, control, chosen['weight']).assign(method=recipe+'_blend'))
    return pd.concat(output, ignore_index=True)


def execute(root, jobs, source, data, w, recipes):
    arguments = [(root, job, source, data, w, recipes) for job in jobs]
    if w['max_workers'] == 1:
        results = [task(args) for args in arguments]
    else:
        with ProcessPoolExecutor(w['max_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(task, arguments))
    return pd.concat(results, ignore_index=True)


def tables(root, stage):
    path = root/'stages'/stage/'reporting'
    return (pd.read_csv(path/'summary.csv'), pd.read_csv(path/'paired_comparisons.csv'))


def lightgbm_decision(summary, comparisons, w, confirmation=False):
    candidates = []
    for method in sorted(set(comparisons.method)):
        if not method.endswith('_blend'):
            continue
        c = comparisons.loc[comparisons.method.eq(method)].set_index('suite')
        s = summary.loc[summary.method.eq(method)].set_index('suite')
        checks = {
            'historical_gain': bool(c.loc['historical', 'relative_rmse_improvement'] >= w['minimum_relative_improvement']),
            'fold_wins': bool(c.loc['historical', 'fold_wins'] >= (8 if confirmation else 4)),
            'nominal_stable': bool(c.loc['nominal', 'relative_rmse_improvement'] >= -w['nominal_maximum_regression']),
            'stress_stable': bool(c.loc['unrestricted', 'relative_rmse_improvement'] >= -w['stress_maximum_regression'])}
        if confirmation:
            checks.update(r2_goal=bool(s.loc['historical', 'r2'] >= w['minimum_pooled_r2']),
                          conditional_bootstrap=bool(c.loc['historical', 'bootstrap_high'] < 0))
        candidates.append({'recipe': method.removesuffix('_blend'), 'checks': checks,
                           'mean_fold_rmse': float(s.loc['historical', 'mean_fold_rmse']), 'eligible': all(checks.values())})
    eligible = [row for row in candidates if row['eligible']]
    chosen = min(eligible, key=lambda row: (row['mean_fold_rmse'], row['recipe'])) if eligible else None
    return {'status': ('eligible_for_final_review' if confirmation else 'confirm_one_recipe') if chosen else
            ('retain_run7' if confirmation else 'stop_after_screen'),
            'selected_recipe': chosen['recipe'] if chosen else None, 'candidates': candidates,
            'promoted': False, 'eligible_for_final_review': bool(chosen) and confirmation,
            'automatic_production_replacement': False, 'new_splits_are_not_untouched_data': True}


def pilot_decision(summary, comparisons, w):
    historical = summary.loc[summary.suite.eq('historical')].set_index('method')
    contrastive = historical.loc['tcn_contrastive', 'mean_fold_rmse']
    checks = {f'beats_{arm}': bool(1-contrastive/historical.loc[arm, 'mean_fold_rmse'] >= w['standalone_minimum_improvement'])
              for arm in ('tcn_mse', 'tcn_augmentation')}
    blend = comparisons.loc[comparisons.method.eq('tcn_contrastive_blend')].set_index('suite')
    checks.update(blend_gain=bool(blend.loc['historical', 'relative_rmse_improvement'] >= w['blend_minimum_improvement']),
                  both_folds_win=bool(blend.loc['historical', 'fold_wins'] == 2),
                  nominal_stable=bool(blend.loc['nominal', 'relative_rmse_improvement'] >= -w['nominal_maximum_regression']),
                  stress_stable=bool(blend.loc['unrestricted', 'relative_rmse_improvement'] >= -w['stress_maximum_regression']))
    return {'status': 'worth_followup' if all(checks.values()) else 'stop_pilot', 'checks': checks,
            'promoted': False, 'automatic_expansion': False, 'automatic_production_replacement': False,
            'interpretation': 'Two-fold futility pilot only; even success requires a separately registered grouped study.',
            'new_splits_are_not_untouched_data': True}


def save_costs(root):
    costs = []
    for path in (root/'cells').rglob('*.json'):
        if path.name.startswith('selection_'):
            continue
        saved = json.loads(path.read_text(encoding='utf-8'))
        payload = saved['payload']
        if saved['sha256'] != checksum(payload):
            raise ValueError('Fit audit checksum changed')
        costs.append({'cell': str(path.relative_to(root)), 'method': payload['contract']['method'],
                      **{key: payload['audit'].get(key, 0) for key in ('base_estimator_fits', 'neural_fits', 'fit_seconds')}})
    atomic_csv(root/'reporting'/'fit_costs.csv', pd.DataFrame(costs))


def run(w, root, stage, config_path):
    validate(w)
    permitted = ('check', 'all', 'screen', 'confirm') if w['kind'] == 'lightgbm' else ('check', 'all', 'pilot')
    if stage not in permitted:
        raise ValueError('Stage not available for this experiment')
    root = Path(root)
    reporting = root/'reporting'
    reporting.mkdir(parents=True, exist_ok=True)
    source, data, jobs, paths, outer, inner = data_tools.prepare(w)
    modules = ['run_bounded_comparison.py', 'bounded_comparison_data.py', 'bounded_lightgbm.py',
               'campaign_reporting.py', 'campaign_models.py', 'campaign_data.py',
               'followup_experiment_utils.py', 'confirmation_utils.py', 'advanced_r2_utils.py']
    if w['kind'] == 'contrastive':
        modules.append('bounded_contrastive.py')
        import bounded_contrastive
        paths.append(Path(bounded_contrastive.TCNRegressor.forward.__code__.co_filename))
    paths.extend(Path(__file__).parent/name for name in modules)
    paths.extend([Path(__file__), config_path])
    if not (reporting/'pre_registration.json').exists() and any((root/'cells').rglob('*.json')):
        raise ValueError('Cannot resume orphan checkpoints without registration')
    registered_workflow = {**w, 'source_workflow': source,
                          'lightgbm_version': version('lightgbm'),
                          **({'torch_version': version('torch')} if w['kind'] == 'contrastive' else {})}
    if stage != 'check' or (reporting/'pre_registration.json').exists():
        register_run(reporting, registered_workflow, paths)
    readiness = {'ready': True, 'kind': w['kind'], **budget(w), 'parallel_workers': w['max_workers'],
                 'cpu_threads_per_worker': w['cpu_threads'], 'features': len(data['training'].features.columns),
                 'suite_counts': data['development'].metadata.groupby('suite').size().to_dict(),
                 'uses_test_labels': False, 'unused_alternative_script_not_required': True}
    atomic_json(reporting/'input_verification.json', readiness)
    atomic_csv(reporting/'endpoints.csv', data['development'].metadata.assign(observed_rul=data['development'].target))
    atomic_csv(reporting/'outer_folds.csv', outer)
    atomic_csv(reporting/'partition_builder_inner_folds.csv', inner)
    if stage == 'check':
        return readiness
    (reporting/'winner_manifest.json').unlink(missing_ok=True)
    if w['kind'] == 'contrastive':
        selected_jobs = [job for job in jobs if job.outer_fold in w['pilot_folds']]
        if len(selected_jobs) != 2:
            raise ValueError('Expected exactly two pilot folds')
        rows = execute(root, selected_jobs, source, data, w, [])
        reports.report(root, 'pilot', rows, w)
        decision = pilot_decision(*tables(root, 'pilot'), w)
        atomic_json(root/'stages/pilot/reporting/winner_manifest.json', decision)
    else:
        if stage in ('all', 'screen'):
            screen_jobs = [job for job in jobs if job.split_seed == w['split_seeds'][0]]
            if len(screen_jobs) != 5:
                raise ValueError('Expected five screen folds')
            rows = execute(root, screen_jobs, source, data, w, list(bounded_lightgbm.RECIPES))
            reports.report(root, 'screen', rows, w)
            decision = lightgbm_decision(*tables(root, 'screen'), w)
            # Freeze the one screening choice before any confirmation is fitted.
            selection = {'decision': decision, 'predictions_sha256': checksum(rows.to_dict('records'))}
            freeze(reporting/'screen_selection.json', selection)
            atomic_json(root/'stages/screen/reporting/winner_manifest.json', decision)
        if stage in ('all', 'confirm'):
            path = reporting/'screen_selection.json'
            if not path.exists():
                raise ValueError('Complete the registered screen before confirmation')
            # Reassemble screening predictions from validated cell checkpoints.
            # This also rechecks training-only weights and avoids trusting editable reports.
            if stage == 'confirm':
                screen_jobs = [job for job in jobs if job.split_seed == w['split_seeds'][0]]
                # Explicit confirmation must not silently train a missing screen.
                for job in screen_jobs:
                    directory = root/'cells'/f'{job.split_seed}_{job.outer_fold}'
                    expected = 4 + len(w['model_seeds'])*len(bounded_lightgbm.RECIPES)*4
                    if len([p for p in directory.glob('*.json') if not p.name.startswith('selection_')]) != expected:
                        raise ValueError('Screen checkpoint set incomplete; resume --stage screen first')
                rows = execute(root, screen_jobs, source, data, w, list(bounded_lightgbm.RECIPES))
                reports.report(root, 'screen', rows, w)
                decision = lightgbm_decision(*tables(root, 'screen'), w)
                freeze(path, {'decision': decision, 'predictions_sha256': checksum(rows.to_dict('records'))})
                atomic_json(root/'stages/screen/reporting/winner_manifest.json', decision)
            recipe = decision['selected_recipe']
            if recipe is not None:
                confirmation_jobs = [job for job in jobs if job.split_seed in w['split_seeds'][1:]]
                if len(confirmation_jobs) != 10:
                    raise ValueError('Expected ten confirmation folds')
                rows = execute(root, confirmation_jobs, source, data, w, [recipe])
                reports.report(root, 'confirmation', rows, w)
                decision = lightgbm_decision(*tables(root, 'confirmation'), w, confirmation=True)
                atomic_json(root/'stages/confirmation/reporting/winner_manifest.json', decision)
    save_costs(root)
    atomic_json(reporting/'winner_manifest.json', {**decision, 'requested_stage': stage, 'completed': True,
                'retained_production_model': 'phase3_run_7', 'uses_test_labels': False})
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--workflow', required=True)
    parser.add_argument('--stage', choices=['check', 'screen', 'confirm', 'pilot', 'all'], default='all')
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, 'campaign_workflows', args.workflow)
    print(json.dumps(run(workflow, root, args.stage, args.config.resolve()), indent=2, sort_keys=True))
