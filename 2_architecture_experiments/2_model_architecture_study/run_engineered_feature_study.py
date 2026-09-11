"""Run 10: six architectures using the original v4 engineered feature recipe.

Run --check first. Completed family/fold/seed cells are verified and reused.
The entry point never modifies the shared Phase 2 settings or a PE campaign.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import torch

from engineered_feature_data import HERE, ROOT, FeatureView, load_script, training_endpoints, validate_raw
from models.neural.lstm import LSTMAdapter
from models.neural.mlp import MLPAdapter
from models.neural.multiscale_cnn import MultiScaleCNNAdapter
from models.neural.transformer import TransformerAdapter
from models.neural.neural_base import NeuralTrainingConfig
from models.tabular.xgboost import XGBoostAdapter
from models.trajectory.trajectory_dtw_knn import TrajectoryDTWKNNAdapter
from no_op_training_monitor import NoOpTrainingMonitor
from policies import TargetPolicy, PredictionPolicy

CLASSES = {'xgboost': XGBoostAdapter, 'mlp': MLPAdapter, 'lstm': LSTMAdapter,
           'multiscale_cnn': MultiScaleCNNAdapter, 'transformer': TransformerAdapter,
           'trajectory_dtw_knn': TrajectoryDTWKNNAdapter}
DEFAULT_SETTINGS = HERE / 'engineered_run_10_settings.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


@contextmanager
def failure_status(root):
    try:
        yield
    except Exception as error:
        write_json(root / 'reporting/progress.json', {'status': 'failed', 'pid': os.getpid(),
            'error': f'{type(error).__name__}: {error}',
            'completed_cells': len(list((root / 'cells').glob('*/complete.json')))})
        raise


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a+b') as stream:
        stream.write(b'0'); stream.flush(); stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class CurveMonitor(NoOpTrainingMonitor):
    def __init__(self, path):
        self.path = path

    def log_training_step(self, *, step, scalars, **kwargs):
        record = {'step': int(step), 'time': time.time(), **{k: float(v) for k, v in scalars.items()}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        if step == 1 or step % 20 == 0:
            print(f'{self.path.stem}: iteration {step} {scalars}', flush=True)
        return True


def read_recipe(family, fold, settings):
    source = HERE / f'runs/run_{settings["selection_runs"][family]}'
    selection = source / f'5_inner_model_selection/studies/{family}__outer_{fold:02d}__selected.json'
    value = json.loads(selection.read_text(encoding='utf-8'))
    if value['model_family'] != family or value['outer_fold'] != fold:
        raise ValueError('Saved selection identity mismatch')
    recipe = {'hyperparameters': json.loads(value['hyperparameters_json']), 'lookback': value['lookback'],
        'prior_refit_iterations': value['outer_retraining_iterations'],
        'source': str(selection.relative_to(ROOT)), 'source_sha256': sha(selection),
        'prior_inner_rmse': value['mean_inner_rmse']}
    paths = [selection]
    spec = source / '1_architecture_study_settings/artifacts/experiment_specification.json'
    if spec.exists():
        prior = json.loads(spec.read_text())['settings']
        recipe['neural_training'] = prior['neural_training']
        recipe['xgboost_patience'] = prior['architectures'].get('xgboost', {}).get('early_stopping_patience') or 35
        paths.append(spec)
    else:
        table = source / '7_architecture_comparison/architecture_study_settings.csv'
        frame = pd.read_csv(table, keep_default_na=False)
        values = dict(frame.iloc[:, :2].itertuples(index=False, name=None))
        recipe['neural_training'] = {key: int(float(values[f'neural_training.{key}']))
            for key in ('batch_size', 'maximum_epochs', 'early_stopping_patience', 'gradient_clip_global_norm')}
        recipe['xgboost_patience'] = int(values.get('architectures.xgboost.early_stopping_patience') or 35)
        paths.append(table)
    if family == 'xgboost' and any(recipe['hyperparameters'].get(k, 'none') != 'none'
            for k in ('signal_compression_strategy', 'fault_mode_strategy')):
        raise ValueError('Source selection would change the requested feature set')
    return recipe, paths


def endpoint_suites(raw, historical, cutoffs, seeds):
    # Match the PE_31/32 endpoint semantics without importing their full runner.
    rows = historical[['uav_id', 'cutoff', 'scenario']].assign(suite='historical', endpoint_seed=-1).to_dict('records')
    for uid, group in raw.groupby('uav_id', sort=True):
        for suite in ('nominal', 'unrestricted'):
            valid = group.loc[group.RUL.ge(1) & (group.RUL.le(125) if suite == 'nominal' else True), 'flight_cycle']
            possible = cutoffs[np.isin(cutoffs, valid)]
            if not len(possible):
                raise ValueError(f'No feasible cutoff: {uid}/{suite}')
            salt = int(hashlib.sha256(str(uid).encode()).hexdigest()[:8], 16)
            for seed in seeds:
                rng = np.random.default_rng(np.random.SeedSequence([seed, salt, int(suite == 'nominal')]))
                rows.append({'uav_id': uid, 'cutoff': int(rng.choice(possible)), 'scenario': f'{suite}_{seed}',
                             'suite': suite, 'endpoint_seed': seed})
    endpoints = pd.DataFrame(rows)
    return endpoints.merge(raw.rename(columns={'flight_cycle': 'cutoff'})[['uav_id', 'cutoff', 'RUL']],
                           on=['uav_id', 'cutoff'], validate='many_to_one')


def prepare(settings_path):
    settings = json.loads(settings_path.read_text())
    if settings['target_cap'] != 125 or settings['max_workers'] != 1 or set(settings['families']) != set(CLASSES):
        raise ValueError('Unexpected study scope')
    paths = [settings_path, *(ROOT / settings[k] for k in
             ('feature_script', 'raw_training', 'outer_folds', 'historical_endpoints', 'test_cutoffs'))]
    raw = pd.read_csv(ROOT / settings['raw_training']).sort_values(['uav_id', 'flight_cycle']).reset_index(drop=True)
    validate_raw(raw)
    folds = pd.read_csv(ROOT / settings['outer_folds'])[['uav_id', 'outer_fold']]
    if folds.uav_id.duplicated().any() or set(folds.uav_id) != set(raw.uav_id) or set(folds.outer_fold) != set(range(5)):
        raise ValueError('Outer fold membership incomplete')
    historical = pd.read_csv(ROOT / settings['historical_endpoints'], usecols=['uav_id', 'cutoff', 'scenario'])
    cutoffs = pd.read_csv(ROOT / settings['test_cutoffs']).final_cycle.to_numpy(int)
    endpoints = endpoint_suites(raw, historical, cutoffs, settings['endpoint_seeds'])
    recipes = {}
    for family in settings['families']:
        for fold in range(5):
            recipes[f'{family}/{fold}'], source_paths = read_recipe(family, fold, settings)
            paths.extend(source_paths)
    # Freeze all reused model implementations and data adapters as well as this runner.
    paths.extend((HERE / '4_model_adapters').rglob('*.py'))
    paths.extend([Path(__file__), HERE / 'engineered_feature_data.py',
        HERE / '2_tabular_data_adapter/tabular_data_adapter.py',
        HERE / '3_sequence_data_adapter/sequence_data_adapter.py',
        HERE / '3_trajectory_data_adapter/trajectory_data_adapter.py'])
    contract = {'settings': settings, 'recipes': recipes, 'python': sys.version,
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'torch', 'scikit-learn', 'xgboost', 'catboost', 'joblib')},
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        'protocol': 'dense_unit_rows; cap125; train-only stopping then full outer-train refit; no q55',
        'exploratory_previously_used_UAVs': True}
    return settings, raw, folds, endpoints, recipes, contract


def make_model(family, recipe, seed, *, iterations=None, monitor=None, smoke=False):
    kwargs = dict(hyperparameters=recipe['hyperparameters'], seed=seed,
                  training_monitor=monitor or NoOpTrainingMonitor())
    if family == 'xgboost':
        kwargs.update(early_stopping_patience=recipe['xgboost_patience'], training_iterations=iterations,
                      device='cpu' if smoke else 'auto')
    elif family != 'trajectory_dtw_knn':
        kwargs.update(training_config=NeuralTrainingConfig(**recipe['neural_training']), training_epochs=iterations)
    model = CLASSES[family](**kwargs)
    if smoke and family not in ('xgboost', 'trajectory_dtw_knn'):
        model.device = torch.device('cpu')
    model.configure_policies(TargetPolicy(mode='piecewise_cap', maximum_rul=125.), PredictionPolicy())
    return model


def stopping_ids(ids, fold, settings):
    fit, stop = train_test_split(sorted(ids), test_size=settings['stopping_fraction'],
                                random_state=settings['stopping_split_seed'] + fold)
    return sorted(fit), sorted(stop)


def process_created(pid):
    """Read Windows process identity/liveness without an extra dependency."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: process no longer exists.
            return None
        raise ctypes.WinError(error)
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 259:
            return None
        created, ended, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in (created, ended, system, user))):
            raise ctypes.WinError(ctypes.get_last_error())
        return ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 10_000_000 - 11644473600
    finally:
        kernel.CloseHandle(handle)


def execute_cell(root, settings, raw, folds, endpoints, recipe, family, fold, seed, registration):
    cell = root / 'cells' / f'{family}__fold_{fold}__seed_{seed}'
    completion = cell / 'complete.json'
    held_ids = set(folds.loc[folds.outer_fold.eq(fold), 'uav_id'])
    train_ids = set(folds.uav_id) - held_ids
    held = endpoints.loc[endpoints.uav_id.isin(held_ids)].reset_index(drop=True)
    identity = {'registration': registration, 'family': family, 'fold': fold, 'seed': seed}
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved['identity'] != identity or any(sha(cell / name) != checksum for name, checksum in saved['sha256'].items()):
            raise ValueError(f'Checkpoint changed: {cell}')
        print(f'Reusing {cell.name}', flush=True)
        return pd.read_csv(cell / 'predictions.csv')
    cell.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f'Starting {cell.name}: {len(train_ids)} train / {len(held_ids)} held UAVs', flush=True)
    script = load_script(ROOT / settings['feature_script'])
    trajectory = family == 'trajectory_dtw_knn'
    iterations, stopping_audit = None, None
    if not trajectory:
        fit_ids, stop_ids = stopping_ids(train_ids, fold, settings)
        view = FeatureView(raw, fit_ids, script)
        training = view.dataset(training_endpoints(raw, fit_ids), family, recipe['lookback'], training=True, labels=True)
        stopping = view.dataset(endpoints.loc[endpoints.uav_id.isin(stop_ids) & endpoints.suite.eq('historical')],
                                family, recipe['lookback'], labels=True)
        # Prevent a singleton final BatchNorm batch while keeping the recorded batch size.
        if family == 'mlp' and len(training) % recipe['neural_training']['batch_size'] == 1:
            raise ValueError('MLP singleton final batch: choose a documented batching policy before running')
        model = make_model(family, recipe, seed, monitor=CurveMonitor(cell / 'stopping_curve.jsonl'))
        summary = model.fit(training, stopping)
        iterations = summary.best_epoch_or_iteration or summary.epochs_or_iterations
        if not iterations or iterations < 1:
            raise ValueError('No usable stopping iteration')
        stopping_audit = {'fit_uavs': fit_ids, 'stopping_uavs': stop_ids, 'summary': summary.to_dict()}
        del model, training, stopping, view
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    view = FeatureView(raw, train_ids, script)
    training = view.dataset(training_endpoints(raw, train_ids, trajectory), family, recipe['lookback'], training=True, labels=True)
    # Held labels are deliberately absent from the model-facing dataset.
    validation = view.dataset(held, family, recipe['lookback'], labels=False)
    model = make_model(family, recipe, seed, iterations=iterations, monitor=CurveMonitor(cell / 'refit_curve.jsonl'))
    summary = model.fit(training, None)
    prediction = model.predict(validation)
    if prediction.shape != (len(held),) or not np.isfinite(prediction).all():
        raise ValueError('Invalid held prediction coverage')
    rows = held.rename(columns={'RUL': 'observed_rul'}).assign(
        model_family=family, outer_fold=fold, model_seed=seed, predicted_rul=prediction)
    write_csv(cell / 'predictions.csv', rows)
    write_csv(cell / 'feature_scaler.csv', pd.DataFrame({'feature': view.names, 'center': view.center, 'scale': view.scale}))
    model.engineered_feature_schema = view.names
    model.engineered_feature_center = view.center
    model.engineered_feature_scale = view.scale
    model.engineered_sensor_columns = view.sensors
    model.detach_training_monitor()
    model.save(cell / 'model.joblib')
    write_json(cell / 'fit_audit.json', {'recipe': recipe, 'train_uavs': sorted(train_ids), 'held_uavs': sorted(held_ids),
        'stopping': stopping_audit, 'refit_summary': summary.to_dict(), 'feature_count': len(view.names),
        'training_rows': len(training), 'held_rows': len(held), 'refit_iterations': iterations,
        'elapsed_seconds': time.time()-started, 'test_labels_loaded': False, 'q55_enabled': False})
    write_json(completion, {'identity': identity, 'sha256': {name: sha(cell / name)
        for name in ('predictions.csv', 'fit_audit.json', 'feature_scaler.csv', 'model.joblib')}})
    print(f'Completed {cell.name} in {(time.time()-started)/60:.1f} minutes', flush=True)
    del model, training, validation, view
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def metrics(frame):
    error = frame.predicted_rul.to_numpy() - frame.observed_rul.to_numpy()
    denominator = np.square(frame.observed_rul - frame.observed_rul.mean()).sum()
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(np.abs(error))),
        'r2': float(1 - np.sum(error**2)/denominator) if denominator > 0 else None,
        'bias': float(np.mean(error)), 'rows': len(frame), 'uavs': frame.uav_id.nunique()}


def reporting(root, frames, settings, complete=False):
    all_rows = pd.concat(frames, ignore_index=True)
    directory = root / 'reporting'
    write_csv(directory / 'predictions.csv', all_rows)
    fold_rows = []
    for keys, group in all_rows.groupby(['model_family', 'outer_fold', 'model_seed', 'suite']):
        fold_rows.append(dict(zip(['model_family', 'outer_fold', 'model_seed', 'suite'], keys)) | metrics(group))
    write_csv(directory / 'fold_metrics.csv', pd.DataFrame(fold_rows))
    summaries = []
    for (family, suite), group in all_rows.groupby(['model_family', 'suite']):
        summaries.append({'model_family': family, 'suite': suite, 'completed_folds': group.outer_fold.nunique(),
                          'seeds': group.model_seed.nunique(), **metrics(group)})
    write_csv(directory / 'summary.csv', pd.DataFrame(summaries))
    if complete:
        comparisons = []
        # Average squared errors over model seeds first; resample entire UAVs,
        # preserving their endpoint draws. DTW is deterministic and is run once.
        all_rows['squared_error'] = (all_rows.predicted_rul - all_rows.observed_rul)**2
        for suite, subset in all_rows.groupby('suite'):
            mse = subset.groupby(['uav_id', 'model_family']).squared_error.mean().unstack()
            if mse.isna().any().any():
                raise ValueError('Incomplete paired UAV coverage')
            rng = np.random.default_rng(settings['bootstrap_seed'])
            draws = rng.integers(0, len(mse), size=(settings['bootstrap_repetitions'], len(mse)))
            baseline = mse.xgboost.to_numpy()
            for family in settings['families']:
                if family == 'xgboost':
                    continue
                candidate = mse[family].to_numpy()
                deltas = np.sqrt(candidate[draws].mean(axis=1)) - np.sqrt(baseline[draws].mean(axis=1))
                comparisons.append({'model_family': family, 'suite': suite,
                    'rmse_delta_vs_xgboost': float(np.sqrt(candidate.mean())-np.sqrt(baseline.mean())),
                    'ci_lower': float(np.quantile(deltas, .025)), 'ci_upper': float(np.quantile(deltas, .975))})
        write_csv(directory / 'paired_vs_xgboost.csv', pd.DataFrame(comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--wait-for-created', type=float)
    args = parser.parse_args()
    settings, raw, folds, endpoints, recipes, contract = prepare(args.settings.resolve())
    root = HERE / settings['run_directory']
    with run_lock(root), failure_status(root):
        registration_path = root / 'reporting/pre_registration.json'
        if registration_path.exists() and json.loads(registration_path.read_text()) != contract:
            raise ValueError('Registered inputs/settings changed; use a new run directory')
        write_json(registration_path, contract)
        write_json(root / 'reporting/hyperparameter_sources.json', recipes)
        write_csv(root / 'reporting/outer_folds.csv', folds)
        write_csv(root / 'reporting/endpoints.csv', endpoints)
        if args.check:
            checks = []
            script = load_script(ROOT / settings['feature_script'])
            schema = None
            for fold in range(5):
                train_ids = set(folds.loc[folds.outer_fold.ne(fold), 'uav_id'])
                inner_ids, stop = stopping_ids(train_ids, fold, settings)
                for stage, ids in [('stopping', inner_ids), ('refit', train_ids)]:
                    view = FeatureView(raw, ids, script)
                    if schema is not None and schema != view.names:
                        raise ValueError('Feature schema differs across fits')
                    schema = view.names
                    count = len(training_endpoints(raw, ids))
                    batch = recipes[f'mlp/{fold}']['neural_training']['batch_size']
                    if count % batch == 1:
                        raise ValueError('MLP singleton batch in preflight')
                    checks.append({'fold': fold, 'stage': stage, 'train_uavs': len(ids), 'training_rows': count,
                                   'features': len(view.names), 'sensor_count': len(view.sensors)})
            write_csv(root / 'reporting/feature_columns.csv', pd.DataFrame({'feature': schema}))
            write_json(root / 'reporting/input_verification.json', {'status': 'passed', 'checks': checks,
                'outer_cells': 80, 'fits_maximum': 155, 'test_labels_loaded': False})
            print('Preflight passed: 6 families, 5 folds, 3 stochastic seeds, 80 cells / at most 155 fits.', flush=True)
            return
        if not (root / 'reporting/input_verification.json').exists():
            raise ValueError('Run --check before starting training')
        if args.wait_for_pid:
            if args.wait_for_created is None:
                raise ValueError('Waiting requires process creation time to guard PID reuse')
            while True:
                created = process_created(args.wait_for_pid)
                if created is None or abs(created-args.wait_for_created) > .01:
                    break
                write_json(root / 'reporting/progress.json', {'status': 'queued', 'waiting_for_pid': args.wait_for_pid,
                    'pid': os.getpid(), 'updated_at': time.time()})
                time.sleep(30)
        # A queued job must not silently consume files edited during the wait.
        if any(sha(ROOT / name) != checksum for name, checksum in contract['input_sha256'].items()):
            raise ValueError('Registered input changed while queued; review before starting')
        torch.set_num_threads(settings['cpu_threads'])
        frames = []
        registration = digest(contract)
        with threadpool_limits(limits=settings['cpu_threads']):
            # Each fold first covers every family before advancing to the next.
            for fold in range(5):
                for family in settings['families']:
                    seeds = [settings['model_seeds'][0]] if family == 'trajectory_dtw_knn' else settings['model_seeds']
                    for seed in seeds:
                        write_json(root / 'reporting/progress.json', {'status': 'running', 'pid': os.getpid(),
                            'completed_cells': len(frames), 'total_cells': 80, 'family': family, 'fold': fold, 'seed': seed})
                        frame = execute_cell(root, settings, raw, folds, endpoints, recipes[f'{family}/{fold}'],
                                             family, fold, seed, registration)
                        frames.append(frame)
                        reporting(root, frames, settings)
        reporting(root, frames, settings, complete=True)
        write_json(root / 'reporting/progress.json', {'status': 'complete', 'completed_cells': len(frames), 'total_cells': 80})
        print('Architecture Run 10 complete. See reporting/summary.csv and paired_vs_xgboost.csv.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Keep the traceback and checkpoint files; a resume reruns only unfinished cells.
        print(f'Architecture study failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
