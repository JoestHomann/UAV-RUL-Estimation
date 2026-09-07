"""Read-only diagnostics of saved development predictions; no model selection/fits."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
KEYS = ["outer_fold", "uav_id", "scenario", "cutoff", "observed_rul"]


def metrics(rows, prediction="predicted_rul"):
    y = rows.observed_rul.to_numpy(float)
    e = rows[prediction].to_numpy(float) - y
    return {
        "rows": len(rows), "uavs": rows.uav_id.nunique(),
        "r2": float(1 - np.sum(e**2) / np.sum((y-y.mean())**2)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))), "bias": float(e.mean()),
        "sse": float(np.sum(e**2)),
    }


def load_run(run, candidate):
    path = ROOT / f"3_final_model_training_and_inference/runs/run_{run}/2_final_configuration_search/artifacts/final_search_oof_predictions.csv"
    rows = pd.read_csv(path)
    rows = rows.loc[rows.candidate_number.eq(candidate)].copy()
    assert not rows.duplicated(KEYS).any()
    return rows


def main():
    public = pd.read_csv(OUT / "kaggle_scores.csv")
    public6 = float(public.loc[public.submission_description.eq("run_6"), "public_r2"].item())
    public7 = float(public.loc[public.submission_description.eq("run_7"), "public_r2"].item())
    old = load_run(6, 12)
    new = load_run(7, 1)
    paired = old[KEYS + ["predicted_rul"]].merge(
        new[KEYS + ["predicted_rul"]], on=KEYS,
        suffixes=("_run6", "_run7"), validate="one_to_one")
    assert len(paired) == len(old) == len(new) == 500
    summary = {"scope": "Reused development predictions; descriptive, conditional on prior selection",
               "run6": metrics(old), "run7": metrics(new),
               "unique_uav_cutoff_pairs": len(new.drop_duplicates(["uav_id", "cutoff"])),
               "duplicate_uav_cutoff_rows": int(new.duplicated(["uav_id", "cutoff"]).sum()),
               "run7_unique_endpoint_metrics": metrics(new.drop_duplicates(["uav_id", "cutoff"])),
               "public_score_source": "User-supplied Kaggle screenshots; kaggle_scores.csv",
               "public_run6_recorded_r2": public6,
               "public_run7_recorded_r2": public7,
               "public_run7_r2_gain_vs_run6": public7-public6,
               "public_run7_mse_reduction_vs_run6": float(1-(1-public7)/(1-public6)),
               "public_run7_rmse_reduction_vs_run6": float(1-np.sqrt((1-public7)/(1-public6))),
               "public_rmse_reduction_needed_from_run6": float(1-np.sqrt(.1/(1-public6))),
               "public_mse_reduction_needed_from_run7": float(1-.1/(1-public7)),
               "public_rmse_reduction_needed_from_run7": float(1-np.sqrt(.1/(1-public7)))}
    band_results = []
    for column, bins, labels in [
        ("observed_rul", [0,25,50,75,100,125,np.inf], ["1-25","26-50","51-75","76-100","101-125",">125"]),
        ("cutoff", [0,100,200,300,np.inf], ["<=100","101-200","201-300",">300"]),
    ]:
        for run, frame in [(6, old), (7, new)]:
            frame = frame.copy()
            frame["band"] = pd.cut(frame[column], bins, labels=labels)
            total_sse = metrics(frame)["sse"]
            for band, group in frame.groupby("band", observed=True):
                m = metrics(group)
                band_results.append({"run":run,"grouping":column,"band":str(band),
                                     **m,"sse_share":m["sse"]/total_sse})
    pd.DataFrame(band_results).to_csv(OUT / "error_bands.csv", index=False)
    per_uav = new.assign(squared_error=new.residual**2).groupby("uav_id").agg(
        sse=("squared_error","sum"), rows=("squared_error","size"))
    per_uav = per_uav.sort_values("sse", ascending=False)
    summary["run7_top10_uavs_sse_share"] = float(per_uav.head(10).sse.sum()/per_uav.sse.sum())
    per_uav.to_csv(OUT / "per_uav_error.csv")
    rng = np.random.default_rng(20260907)
    ids = paired.uav_id.unique()
    groups = [paired.loc[paired.uav_id.eq(u)] for u in ids]
    boot = []
    for _ in range(3000):
        selected = rng.integers(0, len(ids), len(ids))
        sample = pd.concat([groups[i] for i in selected], ignore_index=True)
        a = metrics(sample, "predicted_rul_run6")
        b = metrics(sample, "predicted_rul_run7")
        boot.append([b["r2"], b["rmse"], b["rmse"]-a["rmse"], b["r2"]-a["r2"]])
    ci = np.quantile(boot, [.025,.975], axis=0)
    summary["conditional_uav_bootstrap_95_intervals"] = dict(zip(
        ["run7_pooled_r2", "run7_pooled_rmse", "rmse_delta_run7_minus_run6", "r2_delta_run7_minus_run6"], ci.T.tolist()))
    summary["bootstrap_replicates"] = len(boot)
    summary["bootstrap_caveat"] = "Resamples fixed OOF predictions, not model training/selection; excludes selection uncertainty."
    transitions = []
    for uav, rows in new.groupby("uav_id"):
        rows = rows.drop_duplicates("cutoff").sort_values("cutoff")
        for a, b in zip(rows.itertuples(), list(rows.itertuples())[1:]):
            transitions.append({"uav_id":uav, "elapsed_cycles":b.cutoff-a.cutoff,
                "prediction_change":b.predicted_rul-a.predicted_rul,
                "failure_cycle_estimate_change":(b.cutoff+b.predicted_rul)-(a.cutoff+a.predicted_rul)})
    transition_frame = pd.DataFrame(transitions)
    summary["available_endpoint_transitions"] = len(transition_frame)
    summary["increasing_rul_transition_fraction"] = float(transition_frame.prediction_change.gt(0).mean())
    transition_frame.to_csv(OUT / "endpoint_consistency.csv", index=False)
    # Deliberately optimistic fit-and-score diagnostic: reject low-potential
    # blend ideas, never promote them from weights learned on these targets.
    prefix = ROOT / "2_architecture_experiments"
    temporal = pd.read_csv(prefix / "2_model_architecture_study/runs/run_7/5_inner_model_selection/selected_inner_predictions.csv.gz")
    control = pd.read_csv(prefix / "1_pipeline_experiments/experiments/PE_3/runs/run_1/PE3_ensemble_calibration/reporting/method_predictions.csv.gz")
    control = control.loc[control.method.eq("blend_xgb_0.50__calibrated")]
    pe11 = pd.read_csv(prefix / "1_pipeline_experiments/experiments/PE_11/runs/run_1/reporting/method_predictions.csv.gz")
    pe11 = pe11.loc[pe11.method.eq("residual_corrected")]
    alignment = ["outer_fold", "inner_fold", "uav_id", "scenario", "cutoff", "observed_rul"]
    blend_results = []
    for control_name, control_rows in [("PE3", control), ("PE11", pe11)]:
        for family, candidate in temporal.groupby("model_family"):
            joined = control_rows[alignment+["predicted_rul"]].merge(
                candidate[alignment+["predicted_rul"]], on=alignment,
                suffixes=("_tree", "_temporal"), validate="one_to_one")
            assert len(joined) == len(control_rows) == len(candidate)
            y = joined.observed_rul.to_numpy(float)
            e = joined.predicted_rul_tree.to_numpy(float)-y
            d = joined.predicted_rul_temporal.to_numpy(float)-joined.predicted_rul_tree.to_numpy(float)
            alpha = float(np.clip(-np.mean(e*d)/np.mean(d*d),0,1))
            baseline_rmse = float(np.sqrt(np.mean(e*e)))
            oracle_rmse = float(np.sqrt(np.mean((e+alpha*d)**2)))
            blend_results.append({"control":control_name,"temporal":family,
                "rows":len(joined),"oracle_alpha":alpha,
                "baseline_pooled_rmse":baseline_rmse,"oracle_pooled_rmse":oracle_rmse,
                "optimistic_rmse_improvement":1-oracle_rmse/baseline_rmse,
                "scope":"Weight fit and scored on same reused development targets; NOT validation"})
    pd.DataFrame(blend_results).to_csv(OUT / "optimistic_blend_screen.csv", index=False)
    (OUT / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(band_results).loc[lambda x: x.run.eq(7)].to_string(index=False))
    print(pd.DataFrame(blend_results).to_string(index=False))


if __name__ == "__main__":
    main()
