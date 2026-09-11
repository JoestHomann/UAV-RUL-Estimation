"""PE_34: bounded three-arm adaptive UAV weighting pilot on matched PE_32 folds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import pandas as pd
from threadpoolctl import threadpool_limits

import adaptive_uav_weighting as adaptive
import bounded_comparison_data as data_tools
import campaign_reporting as reports
from advanced_r2_utils import REPOSITORY_ROOT, load_workflow
from followup_experiment_utils import atomic_csv, atomic_json, input_path, register_run
from run_bounded_comparison import Engine, checksum, report_simple_comparisons


def validate(w):
    if (w['kind'] != 'adaptive_uav' or w['multipliers'] != [1.,1.5,2.] or w['hard_fraction'] != .25
            or w['difficulty_folds'] != 3 or len(w['split_seeds']) != 1
            or w['max_workers'] not in (1,2) or w['cpu_threads'] not in (1,2,4)
            or w['include_simple_baseline'] is not True or w['bootstrap_repetitions'] < 100):
        raise ValueError('PE_34 is a fixed five-fold, three-arm, one-pass pilot')


def reuse_baselines(root, jobs, source, data, w):
    """Copy only validated compatible prediction cells; never mutate PE_32."""
    registration_path = input_path(w['reuse_registration'])
    saved = json.loads(registration_path.read_text())
    if saved['workflow']['source_workflow'] != source:
        raise ValueError('PE_32 and PE_34 source recipes differ')
    paths = [registration_path]
    for relative, expected in saved['input_sha256'].items():
        path = input_path(relative)
        if data_tools.sha(path) != expected:
            raise ValueError(f'PE_32 registered source changed: {relative}')
        paths.append(path)
    source_root = registration_path.parent.parent
    records = []
    for job in jobs:
        directory = source_root/'cells'/f'{job.split_seed}_{job.outer_fold}'
        engine = Engine(root,job,source,data,w)
        copied = []
        for path in sorted(directory.glob('*.json')):
            if path.name.startswith('selection_'):
                continue
            saved_cell = json.loads(path.read_text())
            payload = saved_cell['payload']
            if saved_cell['sha256'] != checksum(payload):
                raise ValueError(f'PE_32 cell checksum changed: {path}')
            c = payload['contract']
            if c['method'] not in ('run7','simple_reproduction'):
                continue
            dest = engine.directory/path.name
            if dest.exists() and data_tools.sha(dest) != data_tools.sha(path):
                raise ValueError('Reused PE_34 cell differs from its PE_32 source')
            if not dest.exists():
                shutil.copy2(path,dest)
            # Since the exact path exists, this validates all data/endpoint
            # contracts without running any fitting or changing the source.
            expected_name = checksum({k:c[k] for k in ('method','seed','train_uavs','held_uavs')})+'.json'
            if expected_name != path.name:
                raise ValueError('Unexpected PE_32 cell identity')
            engine.fit(c['train_uavs'],c['held_uavs'],c['method'],source['model_seed'])
            copied.append(c['method'])
            records.append({'source':str(path.relative_to(REPOSITORY_ROOT)),
                'destination':str(dest.relative_to(Path(root))), 'method':c['method'],
                'sha256':data_tools.sha(path),'saved_base_fits':payload['audit']['base_estimator_fits']})
            paths.append(path)
        if copied.count('run7') != 4 or copied.count('simple_reproduction') != 1:
            raise ValueError('Need all PE_32 screen control/difficulty and simpler baseline cells')
    return paths, records


def task(args):
    root,job,source,data,w = args
    with threadpool_limits(limits=w['cpu_threads']):
        engine = adaptive.AdaptiveEngine(root,job,source,data,w)
        rows = [engine.base.fit(job.training_uavs,job.validation_uavs,method,source['model_seed']).assign(method=method)
                for method in ('run7','simple_reproduction')]
        for method in adaptive.ARMS:
            rows.append(engine.fit(method))
        return pd.concat(rows,ignore_index=True)


def decision(summary, comparisons, w):
    candidates = []
    simple = summary.loc[summary.method.eq('simple_reproduction') & summary.suite.eq('historical')].iloc[0]
    for method in adaptive.ARMS:
        c = comparisons.loc[comparisons.method.eq(method)].set_index('suite')
        s = summary.loc[summary.method.eq(method)].set_index('suite')
        checks = {'historical_gain':bool(c.loc['historical','relative_rmse_improvement'] >= w['minimum_relative_improvement']),
            'fold_wins':bool(c.loc['historical','fold_wins'] >= w['minimum_fold_wins']),
            'conditional_bootstrap':bool(c.loc['historical','bootstrap_high'] < 0),
            'beats_simple_baseline':bool(s.loc['historical','mean_fold_rmse'] < simple.mean_fold_rmse),
            'nominal_stable':bool(c.loc['nominal','relative_rmse_improvement'] >= -w['nominal_maximum_regression']),
            'stress_stable':bool(c.loc['unrestricted','relative_rmse_improvement'] >= -w['stress_maximum_regression'])}
        candidates.append({'method':method,'mean_fold_rmse':float(s.loc['historical','mean_fold_rmse']),
                           'checks':checks,'eligible':all(checks.values())})
    eligible = [c for c in candidates if c['eligible']]
    chosen = min(eligible,key=lambda c:(c['mean_fold_rmse'],c['method'])) if eligible else None
    return {'status':'worth_confirmation' if chosen else 'stop_after_pilot','completed':True,
        'candidates':candidates,'selected_method':chosen['method'] if chosen else None,
        'promoted':False,'automatic_expansion':False,'automatic_production_replacement':False,
        'retained_production_model':'phase3_run_7','requires_separate_confirmation':True,
        'same_previously_inspected_uavs_and_pe32_screen_partition':True,'uses_test_labels':False}


def save_costs(root, reused):
    rows = []
    reused_paths = {r['destination'] for r in reused}
    for path in (root/'cells').rglob('*.json'):
        payload = json.loads(path.read_text())['payload']
        rows.append({'cell':str(path.relative_to(root)),'method':payload['contract']['method'],
            'base_estimator_fits':payload['audit']['base_estimator_fits'],
            'reused_from_pe32':str(path.relative_to(root)) in reused_paths})
    for path in (root/'adaptive_cells').rglob('adaptive_uav_*.json'):
        payload = json.loads(path.read_text())['payload']
        rows.append({'cell':str(path.relative_to(root)),'method':payload['contract']['method'],
            'base_estimator_fits':payload['result']['audit']['base_estimator_fits'],'reused_from_pe32':False})
    atomic_csv(root/'reporting/fit_costs.csv',pd.DataFrame(rows))


@contextmanager
def run_lock(root):
    import msvcrt
    root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error:
            raise RuntimeError('Another PE_34 process holds the run lock') from error
        try:
            yield
        finally:
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def run(w,root,config_path,check_only=False):
    validate(w)
    root = Path(root);reporting = root/'reporting';reporting.mkdir(parents=True,exist_ok=True)
    source,data,jobs,paths,outer,inner = data_tools.prepare(w)
    if len(jobs) != 5:
        raise ValueError('Expected exactly five outer pilot jobs')
    paths2,reused = reuse_baselines(root,jobs,source,data,w)
    paths.extend(paths2)
    paths.extend([Path(__file__),Path(adaptive.__file__),config_path])
    # No new model fits happen during preflight. Freeze even the preflight so
    # copied checkpoints cannot outlive changes to scientific settings/code.
    register_run(reporting,{**w,'source_workflow':source},paths)
    atomic_csv(reporting/'reused_cells.csv',pd.DataFrame(reused))
    atomic_csv(reporting/'outer_folds.csv',outer)
    atomic_csv(reporting/'endpoints.csv',data['development'].metadata.assign(observed_rul=data['development'].target))
    readiness = {'ready':True,'outer_folds':5,'adaptive_arms':2,'difficulty_folds':3,
        'weighted_base_fits':300,'additional_difficulty_base_fits_maximum':1800,
        'new_base_fits_maximum':2100,'reused_base_fits':sum(r['saved_base_fits'] for r in reused),
        'new_catboost_fits':0,'parallel_workers':w['max_workers'],'cpu_threads_per_worker':w['cpu_threads'],
        'internal_calibration_difficulty_is_fit_local':True,'uses_test_labels':False,'automatic_expansion':False}
    atomic_json(reporting/'input_verification.json',readiness)
    if check_only:
        return readiness
    progress = {'status':'running','pid':os.getpid(),'completed_outer_folds':[], 'total_outer_folds':5}
    atomic_json(reporting/'progress.json',progress)
    results = []
    arguments = [(root,j,source,data,w) for j in jobs]
    try:
        if w['max_workers'] == 1:
            for job,args in zip(jobs,arguments):
                results.append(task(args));progress['completed_outer_folds'].append(job.outer_fold)
                atomic_json(reporting/'progress.json',progress);save_costs(root,reused)
        else:
            with ProcessPoolExecutor(w['max_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(task,args):job for job,args in zip(jobs,arguments)}
                for future in as_completed(futures):
                    results.append(future.result());progress['completed_outer_folds'].append(futures[future].outer_fold)
                    atomic_json(reporting/'progress.json',progress);save_costs(root,reused)
        rows = pd.concat(results,ignore_index=True)
        reports.report(root,'pilot',rows,w)
        report_simple_comparisons(root,'pilot',rows,w)
        directory = root/'stages/pilot/reporting'
        result = decision(pd.read_csv(directory/'summary.csv'),pd.read_csv(directory/'paired_comparisons.csv'),w)
        atomic_json(directory/'winner_manifest.json',result)
        atomic_json(reporting/'winner_manifest.json',result)
        progress['status'] = 'complete';atomic_json(reporting/'progress.json',progress)
        save_costs(root,reused)
        return result
    except BaseException as error:
        progress.update(status='failed',error=f'{type(error).__name__}: {error}')
        atomic_json(reporting/'progress.json',progress)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--workflow',default='PE_34')
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    w,_,root = load_workflow(args.config,'campaign_workflows',args.workflow)
    with run_lock(root):
        print(json.dumps(run(w,root,args.config.resolve(),args.check),indent=2),flush=True)
