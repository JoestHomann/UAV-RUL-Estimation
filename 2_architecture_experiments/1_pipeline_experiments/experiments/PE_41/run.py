"""Matched, resumable test of which difference reference v13 should keep: baseline, history mean, or both."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import shutil
import time
import tomllib

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

BASELINE = "baseline"
DROP_HISTORY = "drop_history_difference"
DROP_BASELINE = "drop_baseline_difference"
DROP_HISTORY_MATCHED = "drop_history_difference_matched"
DROP_BOTH = "drop_both"
# Reporting order, and the order the arms are fitted in.
ARM_ORDER = [BASELINE, DROP_HISTORY, DROP_BASELINE, DROP_HISTORY_MATCHED, DROP_BOTH]
# Only these three are removals v13 could actually adopt, so only these three
# form the multiplicity family. The matched arm exists to make the head-to-head
# fair and is reported with the declared secondary contrasts.
PRIMARY_REMOVALS = [DROP_HISTORY, DROP_BASELINE, DROP_BOTH]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def load_source(path):
    spec = importlib.util.spec_from_file_location("pe41_v13", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def channel_of(column):
    return column.split("__")[0]


def family_columns(columns, suffix):
    return [column for column in columns if column.endswith("__" + suffix)]


def build_removals(columns, config):
    """Return the exact columns each arm deletes from v13's 153.

    v13 does not give the two families the same reach. The strong tier receives
    all twelve feature families, the medium tier only seven, and the medium tier
    keeps the baseline difference while dropping the history-mean difference. So
    telemetry 07 carries a baseline difference with no history-mean counterpart,
    and removing one family is not the same size of change as removing the other.
    `drop_history_difference_matched` also deletes that unpaired column, which
    makes it the same feature count as `drop_baseline_difference` and turns the
    head-to-head into a comparison of references rather than of feature counts.
    """
    baseline_difference = family_columns(columns, config["baseline_difference_suffix"])
    history_difference = family_columns(columns, config["history_difference_suffix"])
    paired = {channel_of(column) for column in history_difference}
    unpaired = [column for column in baseline_difference if channel_of(column) not in paired]
    return {
        BASELINE: [],
        DROP_HISTORY: list(history_difference),
        DROP_BASELINE: list(baseline_difference),
        DROP_HISTORY_MATCHED: history_difference + unpaired,
        DROP_BOTH: baseline_difference + history_difference,
    }


def partitions(data, config):
    groups = data.uav_id.to_numpy()
    result = []
    for fold, (outer_train, held) in enumerate(GroupKFold(config["outer_folds"]).split(data, groups=groups)):
        ids = data.iloc[outer_train].uav_id.unique()
        rng = np.random.RandomState(config["stopping_seed"])
        stop_ids = rng.choice(ids, max(1, int(len(ids) * config["stopping_uav_fraction"])), replace=False)
        stopping = outer_train[np.isin(groups[outer_train], stop_ids)]
        fitting = outer_train[~np.isin(groups[outer_train], stop_ids)]
        sets = [set(groups[idx]) for idx in (fitting, stopping, held)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(fitting) + len(stopping) + len(held) == len(data)
        result.append((fold, fitting, stopping, held))
    return result


def prepare(config):
    source = ROOT / config["source"]
    module = load_source(source)
    contract = json.loads((ROOT / config["feature_contract"]).read_text())
    tiers = {tier: contract["sensor_tiers"][tier] for tier in ("strong", "medium", "weak")}
    data = pd.read_csv(ROOT / config["train"]).sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    assert not data.duplicated(["uav_id", "flight_cycle"]).any()
    assert np.isfinite(data.select_dtypes(include="number").to_numpy()).all()
    raw_y = data.RUL.to_numpy().copy()
    data = module.apply_rul_cap(data)
    # The test file contributes only UAV IDs/cutoffs, never telemetry or labels.
    cutoffs = module.get_test_cutoff_lengths(pd.read_csv(ROOT / config["test_cutoffs"], usecols=["uav_id", "flight_cycle"]))
    features = module.build_features_tiered(data, tiers)
    columns = [c for c in features if c != "uav_id"]
    assert columns == contract["feature_columns"] and len(columns) == 153
    assert np.isfinite(features[columns].to_numpy()).all()
    assert np.array_equal(features.uav_id, data.uav_id)
    assert np.array_equal(features.flight_cycle, data.flight_cycle)

    removals = build_removals(columns, config)
    assert list(removals) == ARM_ORDER
    known = set(columns)
    for arm, dropped in removals.items():
        assert len(set(dropped)) == len(dropped), arm
        assert set(dropped) <= known, arm
    assert removals[DROP_HISTORY] and removals[DROP_BASELINE], "a family under test is absent from v13"
    # The two families never share a column, so dropping both is exactly their union.
    assert not set(removals[DROP_HISTORY]) & set(removals[DROP_BASELINE])
    assert set(removals[DROP_BOTH]) == set(removals[DROP_HISTORY]) | set(removals[DROP_BASELINE])
    # The matched arm is the history removal plus the baseline-difference columns
    # that have no history-mean counterpart, and nothing else.
    assert set(removals[DROP_HISTORY]) <= set(removals[DROP_HISTORY_MATCHED])
    assert set(removals[DROP_HISTORY_MATCHED]) <= set(removals[DROP_BOTH])
    assert len(removals[DROP_HISTORY_MATCHED]) == len(removals[DROP_BASELINE]), "head-to-head is not feature-matched"

    catalogs = {arm: [c for c in columns if c not in set(dropped)] for arm, dropped in removals.items()}
    assert catalogs[BASELINE] == columns
    for arm, dropped in removals.items():
        assert len(catalogs[arm]) == 153 - len(dropped), arm
    # Ensure the full-history computation is identical to a genuinely truncated
    # trajectory, including rolling/expanding summaries (no future observations).
    probe_id = data.uav_id.iloc[0]
    probe = data[(data.uav_id == probe_id) & (data.flight_cycle <= 37)].copy()
    prefix_features = module.build_features_tiered(probe, tiers)
    np.testing.assert_allclose(prefix_features[columns].to_numpy(), features.loc[probe.index, columns].to_numpy(),
                               rtol=1e-12, atol=1e-12)
    folds = partitions(data, config)
    fold_by_row = np.full(len(data), -1, dtype=int)
    membership = []
    for fold, fitting, stopping, held in folds:
        fold_by_row[held] = fold
        for role, indices in (("fit", fitting), ("early_stop", stopping), ("score", held)):
            membership.extend({"fold": fold, "role": role, "uav_id": uid} for uid in data.iloc[indices].uav_id.unique().tolist())
    endpoints = []
    lifetimes = data.groupby("uav_id").flight_cycle.max().to_dict()
    lookup = {(row.uav_id, row.flight_cycle): idx for idx, row in data[["uav_id", "flight_cycle"]].iterrows()}
    for seed in config["evaluation_seeds"]:
        assigned = module._eligible_random_assignment(lifetimes, cutoffs, np.random.default_rng(seed))
        for uid, cutoff in assigned:
            idx = lookup[(uid, cutoff)]
            endpoints.append(dict(evaluation_seed=seed, uav_id=uid, flight_cycle=cutoff, row_index=idx,
                                  fold=int(fold_by_row[idx]), target_capped=float(data.RUL.iloc[idx]), target_raw=float(raw_y[idx])))
    endpoints = pd.DataFrame(endpoints)
    assert (endpoints.groupby(["evaluation_seed", "uav_id"]).size() == 1).all()
    assert endpoints.uav_id.nunique() == data.uav_id.nunique()
    assert (endpoints.groupby("uav_id").size() == len(config["evaluation_seeds"])).all()
    return module, data, features, tiers, removals, catalogs, folds, pd.DataFrame(membership), endpoints, cutoffs


def metrics(frame, target="target_capped"):
    actual, predicted = frame[target], frame.prediction
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
            "bias": float(np.mean(predicted - actual))}


def indexed_arm(predictions, arm):
    frame = predictions[predictions.arm == arm].set_index(["evaluation_seed", "uav_id", "flight_cycle"]).sort_index()
    assert not frame.empty, arm
    return frame


def bootstrap_rmse(predictions, arms, uavs, draws):
    """Paired UAV bootstrap RMSE per arm, drawn on one shared set of resamples."""
    reference = None
    result = {}
    for arm in arms:
        frame = indexed_arm(predictions, arm)
        if reference is None:
            reference = frame
        else:
            assert frame.index.equals(reference.index)
            np.testing.assert_array_equal(frame.target_capped, reference.target_capped)
        squared = (frame.prediction - frame.target_capped) ** 2
        uav_mse = squared.groupby(level="uav_id").mean().reindex(uavs).to_numpy()
        result[arm] = np.sqrt(uav_mse[draws].mean(axis=1))
    return result


def paired_change(boot, summary, fold_metrics, treatment, reference, family):
    changes = boot[treatment] - boot[reference]
    lo, hi = np.quantile(changes, [0.025, 0.975])
    adjusted_lo, adjusted_hi = np.quantile(changes, [0.025 / family, 1 - 0.025 / family])
    rmse = summary.set_index("arm").rmse
    delta = float(rmse[treatment] - rmse[reference])
    folds = fold_metrics.set_index(["arm", "fold"]).rmse
    reference_folds = folds[reference]
    return dict(arm=treatment, reference=reference, rmse_change=delta,
                rmse_change_percent=100 * delta / float(rmse[reference]),
                improved_folds=int((folds[treatment] < reference_folds).sum()), total_folds=len(reference_folds),
                uav_bootstrap_ci95_low=float(lo), uav_bootstrap_ci95_high=float(hi),
                familywise_ci95_low=float(adjusted_lo), familywise_ci95_high=float(adjusted_hi),
                comparisons_adjusted=family,
                interpretation="removal_candidate" if adjusted_hi < 0 else "removal_harmful" if adjusted_lo > 0 else "inconclusive")


def additivity(boot, summary, pair, singles):
    """Interaction term: what dropping both families costs beyond the two single drops."""
    rmse = summary.set_index("arm").rmse
    combined = float(rmse[pair] - rmse[BASELINE])
    parts = [float(rmse[single] - rmse[BASELINE]) for single in singles]
    samples = boot[pair] - boot[BASELINE] - sum(boot[single] - boot[BASELINE] for single in singles)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"pair": pair, "singles": list(singles), "pair_change": combined,
            "sum_of_single_changes": float(sum(parts)),
            "interaction": combined - float(sum(parts)),
            "interaction_ci95_low": float(lo), "interaction_ci95_high": float(hi),
            "interpretation": "inconclusive" if lo <= 0 <= hi else
                              "complementary" if lo > 0 else "redundant"}


def reproduction_note(output, config, catalogs, registration):
    """Bitwise comparison against PE_39's baseline fits, when that run is present."""
    reference = config.get("reproduction_check") or ""
    if not reference:
        return {"status": "skipped", "reason": "no reference run configured"}
    source = ROOT / reference
    if not (source / "registration.json").exists():
        return {"status": "skipped", "reason": f"{reference} not found"}
    old = json.loads((source / "registration.json").read_text())
    shared = {arm for arm in catalogs if (source / "cells" / f"{arm}__fold_0.csv").exists()}
    inputs_match = all(registration["hashes"][key] == old["hashes"][key]
                       for key in ("source", "feature_contract", "train", "test_cutoffs"))
    versions_match = registration["versions"] == old["versions"]
    settings_match = all(registration[key] == old[key] for key in ("xgboost_params", "early_stopping_rounds", "target_cap"))
    compared, mismatched, worst = [], [], 0.0
    for arm in sorted(shared):
        for fold in range(config["outer_folds"]):
            new_path = output / "cells" / f"{arm}__fold_{fold}.csv"
            old_path = source / "cells" / f"{arm}__fold_{fold}.csv"
            if not new_path.exists():
                continue
            new, previous = pd.read_csv(new_path), pd.read_csv(old_path)
            keys = ["evaluation_seed", "uav_id", "flight_cycle"]
            merged = new.merge(previous[keys + ["prediction"]], on=keys, suffixes=("", "_reference"))
            assert len(merged) == len(new)
            difference = float(np.abs(merged.prediction - merged.prediction_reference).max())
            worst = max(worst, difference)
            compared.append(f"{arm}__fold_{fold}")
            if difference > 0:
                mismatched.append({"cell": f"{arm}__fold_{fold}", "max_abs_difference": difference})
    note = {"status": "checked" if compared else "skipped", "reference_run": reference,
            "arms": sorted(shared), "cells_compared": len(compared),
            "inputs_match": inputs_match, "settings_match": settings_match, "versions_match": versions_match,
            "max_abs_prediction_difference": worst, "mismatched_cells": mismatched,
            "identical": not mismatched}
    if mismatched and inputs_match and settings_match and versions_match:
        raise RuntimeError(
            "PE_39's baseline was not reproduced bitwise although inputs, settings and library "
            f"versions are identical (worst difference {worst:.6g}). Investigate before reporting: "
            "the protocol is supposed to be deterministic.")
    return note


def report(output, config, catalogs, removals, registration=None):
    predictions = []
    for arm in catalogs:
        for fold in range(config["outer_folds"]):
            frame = pd.read_csv(output / "cells" / f"{arm}__fold_{fold}.csv")
            frame["arm"] = arm
            predictions.append(frame)
    predictions = pd.concat(predictions, ignore_index=True)
    write_csv(output / "reporting/predictions.csv", predictions)
    rows = []
    for arm, frame in predictions.groupby("arm", sort=False):
        row = {"arm": arm, "features": len(catalogs[arm]), "removed": len(removals[arm]), **metrics(frame)}
        row.update({f"raw_{k}": v for k, v in metrics(frame, "target_raw").items()})
        row["mean_test_like_seed_r2"] = float(np.mean([metrics(part)["r2"] for _, part in frame.groupby("evaluation_seed")]))
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("arm").loc[list(catalogs)].reset_index()
    write_csv(output / "reporting/summary.csv", summary)
    fold_metrics = pd.DataFrame([{"arm": arm, "fold": fold, **metrics(part)}
                                 for (arm, fold), part in predictions.groupby(["arm", "fold"])])
    write_csv(output / "reporting/fold_metrics.csv", fold_metrics)
    seed_metrics = pd.DataFrame([{"arm": arm, "evaluation_seed": seed, **metrics(part)}
                                 for (arm, seed), part in predictions.groupby(["arm", "evaluation_seed"])])
    write_csv(output / "reporting/seed_metrics.csv", seed_metrics)

    uavs = sorted(predictions.uav_id.unique())
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.integers(0, len(uavs), size=(config["bootstrap_repetitions"], len(uavs)))
    boot = bootstrap_rmse(predictions, list(catalogs), uavs, draws)
    paired = pd.DataFrame([paired_change(boot, summary, fold_metrics, arm, BASELINE, len(PRIMARY_REMOVALS))
                           for arm in PRIMARY_REMOVALS])
    write_csv(output / "reporting/paired_ablation.csv", paired)
    interaction = additivity(boot, summary, DROP_BOTH, [DROP_HISTORY, DROP_BASELINE])
    write_json(output / "reporting/additivity.json", interaction)
    contrasts = [tuple(item) for item in config["secondary_contrasts"]]
    secondary = pd.DataFrame([paired_change(boot, summary, fold_metrics, treatment, reference, len(contrasts))
                              for treatment, reference in contrasts])
    write_csv(output / "reporting/secondary_contrasts.csv", secondary)

    regions = []
    for arm, frame in predictions.groupby("arm"):
        for region, mask in (("short_history_le105", frame.flight_cycle <= 105),
                             ("longer_history_gt105", frame.flight_cycle > 105),
                             ("late_life_raw_rul_le30", frame.target_raw <= 30)):
            subset = frame[mask]
            if len(subset) > 1:
                regions.append({"arm": arm, "region": region, "rows": len(subset), "uavs": subset.uav_id.nunique(), **metrics(subset)})
    write_csv(output / "reporting/region_metrics.csv", pd.DataFrame(regions))
    note = reproduction_note(output, config, catalogs, registration) if registration else {"status": "skipped", "reason": "report-only"}
    write_json(output / "reporting/reproduction_check.json", note)

    head_to_head = secondary[(secondary.arm == DROP_BASELINE) & (secondary.reference == DROP_HISTORY_MATCHED)]
    text = [
        "# PE_41: which difference reference should v13 keep", "",
        f"Completed {len(catalogs) * config['outer_folds']} matched fits. v13 measures how far a channel has "
        "moved in two ways at once: against the UAV's first observed value (`baseline_delta`) and against its "
        "own expanding mean (`last_minus_hist_mean`). Nothing in the register has ever asked whether both are "
        "needed. Every arm below changes only which of those two families is present.", "",
        "Primary target: RUL capped at 125, matching v13. Negative RMSE change favors removal. "
        "Each UAV has ten test-like cutoffs and equal total evaluation weight.", "",
        "| Arm | Features | Removed | RMSE | R2 | Mean cutoff-seed R2 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        text.append(f"| {row.arm} | {row.features} | {row.removed} | {row.rmse:.4f} | {row.r2:.5f} | {row.mean_test_like_seed_r2:.5f} |")
    text += ["", f"| Removal | RMSE change | Fold wins | Paired UAV 95% interval | "
             f"{len(PRIMARY_REMOVALS)}-comparison adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in paired.sort_values("rmse_change").itertuples():
        text.append(f"| {row.arm} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    text += ["", "## Why the head-to-head needs a fifth arm", "",
             "v13 does not give the two families the same reach. The strong tier receives all twelve feature "
             "families; the medium tier receives seven, and those seven include the baseline difference but not "
             f"the history-mean difference. So `{DROP_BASELINE}` removes "
             f"{len(removals[DROP_BASELINE])} columns while `{DROP_HISTORY}` removes "
             f"{len(removals[DROP_HISTORY])}, and comparing them directly would confound the reference with the "
             f"feature count. `{DROP_HISTORY_MATCHED}` also deletes the unpaired baseline-difference column, so it "
             f"holds {len(catalogs[DROP_HISTORY_MATCHED])} features against "
             f"`{DROP_BASELINE}`'s {len(catalogs[DROP_BASELINE])}. That contrast, and only that contrast, answers "
             "which reference is the better one.", ""]
    if not head_to_head.empty:
        row = head_to_head.iloc[0]
        favored = ("keeping the history-mean difference" if row.rmse_change < 0 else "keeping the baseline difference")
        text += [f"Matched head-to-head: {row.rmse_change:+.4f} cycles, 95% interval "
                 f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}], "
                 f"{row.improved_folds}/{row.total_folds} folds. The point estimate favors {favored}; "
                 "read the interval before saying more.", ""]
    text += ["## Are the two references redundant?", "",
             "The interaction term is the change from dropping both families minus the two single changes. A "
             "positive value means the two references carry complementary information, so the second removal "
             "costs more than the first predicts. A negative value means they are substitutes and the second "
             "removal is nearly free once the first has happened.", "",
             f"- Both dropped: {interaction['pair_change']:+.4f} cycles",
             f"- Sum of single drops: {interaction['sum_of_single_changes']:+.4f} cycles",
             f"- Interaction: {interaction['interaction']:+.4f} cycles, 95% interval "
             f"[{interaction['interaction_ci95_low']:+.4f}, {interaction['interaction_ci95_high']:+.4f}] "
             f"({interaction['interpretation']})", "",
             "Pre-declared secondary contrasts, adjusted over their own family. They describe where any effect "
             "sits; only the comparisons against the baseline are decisive.", "",
             "| Contrast | RMSE change | Fold wins | Paired UAV 95% interval | Adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in secondary.itertuples():
        text.append(f"| {row.arm} vs. {row.reference} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    if note.get("status") == "checked":
        outcome = ("reproduced every shared PE_39 fit bitwise" if note["identical"] else
                   f"differed from PE_39 by at most {note['max_abs_prediction_difference']:.6g} cycles "
                   f"(inputs_match={note['inputs_match']}, versions_match={note['versions_match']})")
        text += ["", f"Reproduction check: this run refits rather than imports the shared arms "
                 f"({', '.join(note['arms'])}) and {outcome}."]
    text += ["", "These are development ablations on previously examined UAVs with historically selected, frozen "
             "feature tiers. Bootstrap intervals resample UAVs and condition on the fitted models; they do not "
             "measure variation from retraining on different splits. Ten cutoff seeds are repeated endpoints, "
             "not ten independent training runs.", "",
             "One main XGBoost per arm/fold, no early specialist or gating. Outer scoring UAVs are excluded from "
             "feature-weight calculation, fitting, and early stopping. A fixed 10% of outer-training UAVs supplies "
             "unweighted early stopping; the model is not refit, matching v13's final-fit holdout approach. "
             "Feature weights use only actual fitting UAVs. Tree settings and all-cycle sample-weight mechanics "
             "come directly from v13.", "",
             "This result covers exactly these two families under v13's frozen tiers. It does not say which "
             "reference a differently tiered feature set would prefer, and an inconclusive head-to-head is a real "
             "outcome: it would mean the two references are interchangeable and the cheaper one may be kept on "
             "simplicity grounds alone. No production model or submission was changed.", ""]
    (output / "reporting/report.md").write_text("\n".join(text), encoding="utf-8")
    write_json(output / "reporting/completion.json",
               {"status": "complete", "fits": len(catalogs) * config["outer_folds"], "arms": len(catalogs),
                "primary_target": "cap125", "bootstrap_unit": "uav_id", "production_changed": False})
    print(summary[["arm", "features", "removed", "rmse", "r2"]].to_string(index=False), flush=True)
    print(paired[["arm", "rmse_change", "improved_folds", "interpretation"]].to_string(index=False), flush=True)
    print(f"interaction={interaction['interaction']:+.4f} "
          f"[{interaction['interaction_ci95_low']:+.4f}, {interaction['interaction_ci95_high']:+.4f}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate features, partitions and endpoints without fitting")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"]
    module, data, features, tiers, removals, catalogs, folds, membership, endpoints, cutoffs = prepare(config)
    declared = {arm for pair in config["secondary_contrasts"] for arm in pair}
    assert declared <= set(catalogs), sorted(declared - set(catalogs))
    hashes = {name: sha(ROOT / config[name]) for name in ("source", "feature_contract", "train", "test_cutoffs")}
    hashes.update(runner=sha(__file__), settings=sha(HERE / "settings.toml"))
    registration = {"config": config, "hashes": hashes, "xgboost_params": module.XGB_PARAMS,
                    "early_stopping_rounds": module.XGB_EARLY_STOPPING_ROUNDS, "target_cap": module.RUL_CAP,
                    "rolling_windows": list(module.ROLLING_WINDOWS), "sensor_tiers": tiers,
                    "removed_columns": removals, "feature_columns": catalogs,
                    "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__},
                    "protocol": "main_xgboost_only; frozen_historical_tiers; inner_10pct_uav_early_stop; no_refit; "
                                "fit_only_feature_weights; v13_GroupKFold; ten_test_cutoff_seeds"}
    manifest = output / "registration.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != registration:
            raise RuntimeError("Inputs, code or settings changed. Choose a new study.run before running.")
    else:
        write_json(manifest, registration)
        shutil.copyfile(ROOT / config["source"], output / "v13_source_snapshot.py")
    write_csv(output / "fold_membership.csv", membership)
    write_csv(output / "evaluation_endpoints.csv", endpoints)
    write_json(output / "feature_catalog.json", catalogs)
    write_json(output / "input_verification.json",
               {"status": "passed", "feature_counts": {k: len(v) for k, v in catalogs.items()},
                "removed_counts": {k: len(v) for k, v in removals.items()},
                "uavs": int(data.uav_id.nunique()), "training_rows": len(data), "evaluation_rows": len(endpoints),
                "causal_prefix_check": True, "disjoint_fit_stopping_score_uavs": True,
                "both_removes_union_of_singles": True, "head_to_head_feature_matched": True})
    print(f"Validated {len(catalogs)} arms, {len(folds)} folds, {len(endpoints)} matched endpoints", flush=True)
    if args.check:
        return
    if args.report_only:
        report(output, config, catalogs, removals, registration)
        return
    all_columns = [c for c in features if c != "uav_id"]
    matrix = features[all_columns].to_numpy()
    y = data.RUL.to_numpy()
    sensors = [sensor for tier in tiers.values() for sensor in tier]
    total = len(catalogs) * config["outer_folds"]
    completed = 0
    for fold, fitting, stopping, held in folds:
        with redirect_stdout(io.StringIO()):
            feature_weights = module.compute_feature_weights(all_columns, data.iloc[fitting], sensors)
        sample_weights = module.compute_test_similarity_weights(data.iloc[fitting].flight_cycle, cutoffs)
        selected_endpoints = endpoints[endpoints.fold == fold].copy()
        assert set(selected_endpoints.uav_id) <= set(data.iloc[held].uav_id)
        for arm, columns in catalogs.items():
            cell_csv = output / "cells" / f"{arm}__fold_{fold}.csv"
            cell_json = cell_csv.with_suffix(".json")
            if cell_json.exists():
                if json.loads(cell_json.read_text())["prediction_sha256"] != sha(cell_csv):
                    raise RuntimeError(f"Checkpoint corrupted: {cell_csv}")
                completed += 1
                print(f"Reused {completed}/{total} {arm} fold {fold}", flush=True)
                continue
            indices = [all_columns.index(column) for column in columns]
            print(f"Fitting {completed + 1}/{total} {arm} fold {fold}: {len(columns)} features", flush=True)
            started = time.perf_counter()
            model = xgb.XGBRegressor(**module.XGB_PARAMS, early_stopping_rounds=module.XGB_EARLY_STOPPING_ROUNDS,
                                     feature_weights=feature_weights[indices])
            model.fit(matrix[np.ix_(fitting, indices)], y[fitting], sample_weight=sample_weights,
                      eval_set=[(matrix[np.ix_(stopping, indices)], y[stopping])], verbose=False)
            selected_endpoints["prediction"] = model.predict(matrix[np.ix_(selected_endpoints.row_index.to_numpy(), indices)])
            assert np.isfinite(selected_endpoints.prediction).all()
            write_csv(cell_csv, selected_endpoints)
            elapsed = time.perf_counter() - started
            write_json(cell_json, {"arm": arm, "fold": fold, "features": len(columns),
                                   "best_iteration": int(model.best_iteration), "fit_seconds": elapsed,
                                   "prediction_sha256": sha(cell_csv), "metrics": metrics(selected_endpoints),
                                   "removed_columns": removals[arm],
                                   "feature_weights": dict(zip(columns, feature_weights[indices].tolist())),
                                   "sample_weight_min": float(sample_weights.min()), "sample_weight_max": float(sample_weights.max())})
            completed += 1
            print(f"Completed {completed}/{total} in {elapsed:.1f}s; iteration={model.best_iteration}; "
                  f"RMSE={metrics(selected_endpoints)['rmse']:.4f}", flush=True)
    report(output, config, catalogs, removals, registration)


if __name__ == "__main__":
    main()
