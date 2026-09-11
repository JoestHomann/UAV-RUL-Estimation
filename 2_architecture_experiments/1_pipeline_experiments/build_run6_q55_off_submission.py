"""Create the requested Run 6 q=0.55-off ablation from its frozen model.

Only the external conditional-quantile adjustment is bypassed. The model's
intrinsic residual calibration, target cap and nonnegative bound are retained.
Run 7 already has no conditional-quantile handler and is not modified.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from advanced_r2_utils import REPOSITORY_ROOT
from followup_experiment_utils import atomic_csv, atomic_json, register_run
from reproduce_simple_submission import sha

sys.path.insert(0,str(REPOSITORY_ROOT/'3_final_model_training_and_inference/5_test_inference'))
import run_test_inference as inference


def main():
    source_root = REPOSITORY_ROOT/'3_final_model_training_and_inference/runs/run_6'
    root = source_root/'exploratory_submissions/q55_off'
    root.mkdir(parents=True,exist_ok=True)
    contract_path = source_root/'3_final_training_contract/artifacts/final_training_contract.json'
    model_path = source_root/'4_final_model_training/artifacts/final_model.joblib'
    original_path = source_root/'6_submission_verification/artifacts/submission.csv'
    contract = json.loads(contract_path.read_text())
    calibrator_payload = contract['conditional_calibrator']
    if (contract['status'] != 'frozen' or contract['model_family'] != 'calibrated_tree_blend'
            or calibrator_payload['quantile'] != .55
            or contract['prediction_policy']['calibration'] != 'conditional_quantile'
            or contract['preprocessing']['separate_artifact']):
        raise ValueError('Expected the frozen Run 6 q=0.55 conditional calibration contract')
    test = inference.load_final_test_data(contract,None)
    if test.target is not None or test.sample_weights is not None or len(test) != 100:
        raise ValueError('Expected 100 unlabelled test endpoints')
    ids = test.metadata.uav_id.astype(str)
    if ids.duplicated().any() or set(ids)&set(contract['training']['uav_ids']):
        raise ValueError('Test identity validation failed')
    from build_pe25_exploratory_submission import frame_hash
    registration = {'purpose':'User-requested Run 6 q55-off test submission',
        'removed_operation':'External ConditionalQuantileCalibrator.apply at q=0.55',
        'intrinsic_model_calibration_retained':True,
        'feature_digest':frame_hash(test.features),'metadata_digest':frame_hash(test.metadata),
        'test_labels_loaded':False,'new_model_fits':0}
    register_run(root,registration,[Path(__file__),contract_path,model_path,original_path,
        Path(inference.__file__),REPOSITORY_ROOT/'3_final_model_training_and_inference/phase_3_data.py',
        REPOSITORY_ROOT/'2_architecture_experiments/2_model_architecture_study/4_model_adapters/policies.py'])
    model = inference.load_model_adapter(model_path)
    unadjusted = np.asarray(model.predict(test),float)
    minimum = float(contract['prediction_minimum'])
    if unadjusted.shape != (100,) or not np.isfinite(unadjusted).all() or (unadjusted < minimum).any():
        raise ValueError('Invalid unadjusted model predictions')
    calibrator = inference.ConditionalQuantileCalibrator.from_dict(calibrator_payload)
    adjusted = calibrator.apply(unadjusted,prediction_minimum=minimum)
    original = pd.read_csv(original_path).set_index('id')
    if set(original.index) != set(ids):
        raise ValueError('Source submission IDs differ')
    # Prove that bypassing this one adjustment is the only prediction change.
    np.testing.assert_allclose(adjusted,original.loc[ids,'RUL'].to_numpy(),rtol=1e-7,atol=1e-8)
    details = pd.DataFrame({'id':ids.to_numpy(),'q55_on':adjusted,'q55_off':unadjusted})
    details['removed_subtraction'] = details.q55_off-details.q55_on
    if (details.removed_subtraction < -1e-8).any():
        raise ValueError('Disabling subtraction unexpectedly reduced a prediction')
    details = details.sort_values('id').reset_index(drop=True)
    submission = details[['id','q55_off']].rename(columns={'q55_off':'RUL'})
    path = root/'submission_run6_q55_off.csv'
    atomic_csv(path,submission)
    pd.testing.assert_frame_equal(pd.read_csv(path),submission)
    atomic_csv(root/'prediction_changes.csv',details)
    # Detect whether this is numerically a repeat of an earlier submission.
    equivalents = []
    for run in (5,7):
        previous = REPOSITORY_ROOT/f'3_final_model_training_and_inference/runs/run_{run}/6_submission_verification/artifacts/submission.csv'
        if previous.is_file():
            values = pd.read_csv(previous).set_index('id').loc[submission.id,'RUL'].to_numpy()
            if np.allclose(values,submission.RUL.to_numpy(),rtol=1e-7,atol=1e-8):
                equivalents.append(f'phase3_run_{run}')
    result = {'status':'complete','source_model':'phase3_run_6','q55_enabled':False,
        'source_q55_submission_reproduced':True,'new_model_fits':0,'test_labels_loaded':False,
        'rows':100,'changed_predictions':int((details.removed_subtraction > 1e-8).sum()),
        'mean_prediction_increase':float(details.removed_subtraction.mean()),
        'maximum_prediction_increase':float(details.removed_subtraction.max()),
        'numerically_equivalent_prior_submissions':equivalents,
        'submission_sha256':sha(path),'submission':str(path.relative_to(REPOSITORY_ROOT)),
        'original_model_sha256':sha(model_path),'submitted_to_kaggle':False,
        'current_run7_already_has_q55_disabled':True}
    atomic_json(root/'submission_manifest.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
