"""Matched, resumable study of how telemetry 07 should be represented in the v13 main XGBoost."""
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

# Telemetry 07 is degradation-associated and state-like. Each arm gives it one
# numeric depth and either keeps or drops the discrete-state block. Every other
# channel, and every other v13 setting, is identical in all six arms.
ARMS = {
    "baseline": {"numeric": "medium", "state": False},
    "numeric_strong": {"numeric": "strong", "state": False},
    "state_only": {"numeric": "none", "state": True},
    "medium_plus_state": {"numeric": "medium", "state": True},
    "strong_plus_state": {"numeric": "strong", "state": True},
    "drop_07": {"numeric": "none", "state": False},
}
EXPECTED_FEATURES = {"baseline": 153, "numeric_strong": 158, "state_only": 157,
                     "medium_plus_state": 164, "strong_plus_state": 169, "drop_07": 146}
# v13 STRONG_FAMILIES minus MEDIUM_FAMILIES, in v13's own family order.
STRONG_ONLY_FAMILIES = ("last_minus_hist_mean", "roll5_mean", "roll5_std", "roll20_mean", "roll20_std")
STATE_SUFFIXES = ("state", "state_n_unique", "state_n_transitions", "state_transition_rate",
                  "state_run_length", "state_run_fraction", "state_max_so_far")


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
    spec = importlib.util.spec_from_file_location("pe38_v13", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def channel_columns(columns, channel):
    """Every column produced from one channel, without prefix collisions."""
    return [c for c in columns if c == channel or c.startswith(channel + "__")]


def state_feature_names(channel, n_levels):
    return [f"{channel}__{suffix}" for suffix in STATE_SUFFIXES] + \
           [f"{channel}__state_frac_l{index}" for index in range(n_levels)]


def master_order(contract_columns, channel, extras, state_names):
    """Contract order, with the extra numeric and state columns kept next to the channel."""
    block = channel_columns(contract_columns, channel)
    end = contract_columns.index(block[-1]) + 1
    return contract_columns[:end] + list(extras) + list(state_names) + contract_columns[end:]


def derive_levels(values, tolerance):
    """Discrete levels of an already-quantized channel, ascending."""
    levels = []
    for value in np.sort(np.unique(np.asarray(values, dtype=float))):
        if not levels or value - levels[-1] > tolerance:
            levels.append(float(value))
    return levels


def assign_states(values, levels):
    """Nearest-level state index, so a value unseen while fitting still maps to a known state."""
    distance = np.abs(np.asarray(values, dtype=float)[:, None] - np.asarray(levels, dtype=float)[None, :])
    return distance.argmin(axis=1).astype(int)


def build_state_features(frame, states, n_levels, channel):
    """Causal per-UAV summaries of the discrete state sequence.

    Every column at cycle n uses only cycles 1..n of that UAV: the current state,
    how many distinct states and transitions have occurred, the transition rate,
    the length of the current run, the highest state reached, and the share of
    observed cycles spent in each state. Rates use the within-UAV observation
    count, which equals flight_cycle in this dataset and is verified in prepare().
    """
    assert isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0
    names = state_feature_names(channel, n_levels)
    states = np.asarray(states, dtype=int)
    assert len(states) == len(frame) and states.min() >= 0 and states.max() < n_levels
    values = np.zeros((len(frame), len(names)), dtype=float)
    for positions in frame.groupby("uav_id", sort=False).indices.values():
        order = np.sort(positions)
        sequence = states[order]
        observed = np.arange(1, len(sequence) + 1, dtype=float)
        changed = np.concatenate(([False], sequence[1:] != sequence[:-1]))
        transitions = np.cumsum(changed).astype(float)
        seen = np.zeros(n_levels, dtype=bool)
        counts = np.zeros(n_levels, dtype=float)
        unique = np.empty(len(sequence), dtype=float)
        run = np.empty(len(sequence), dtype=float)
        dwell = np.empty((len(sequence), n_levels), dtype=float)
        run_length = 0
        for position, state in enumerate(sequence):
            seen[state] = True
            counts[state] += 1.0
            run_length = 1 if changed[position] else run_length + 1
            unique[position] = seen.sum()
            run[position] = run_length
            dwell[position] = counts
        block = [sequence.astype(float), unique, transitions, transitions / observed,
                 run, run / observed, np.maximum.accumulate(sequence).astype(float)]
        block.extend(dwell[:, level] / observed for level in range(n_levels))
        values[order] = np.column_stack(block)
    return pd.DataFrame(values, columns=names, index=frame.index)


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
    channel = config["channel"]
    n_levels = config["state_levels"]
    source = ROOT / config["source"]
    module = load_source(source)
    contract = json.loads((ROOT / config["feature_contract"]).read_text())
    tiers = {tier: contract["sensor_tiers"][tier] for tier in ("strong", "medium", "weak")}
    data = pd.read_csv(ROOT / config["train"]).sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    assert not data.duplicated(["uav_id", "flight_cycle"]).any()
    assert np.isfinite(data.select_dtypes(include="number").to_numpy()).all()
    # State rates divide by the within-UAV observation count; verify it is the cycle number.
    assert np.array_equal(data.groupby("uav_id").cumcount().to_numpy() + 1, data.flight_cycle.to_numpy())
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
    # The deeper numeric arm is v13's own strong-tier recipe, obtained by moving the
    # channel between tiers rather than by writing new feature code.
    promoted = {tier: [s for s in sensors if s != channel] for tier, sensors in tiers.items()}
    promoted["strong"] = promoted["strong"] + [channel]
    strong_features = module.build_features_tiered(data, promoted)
    extras = [f"{channel}__{family}" for family in STRONG_ONLY_FAMILIES]
    assert sorted(set(strong_features) - set(features)) == sorted(extras)
    assert not set(features) - set(strong_features)
    np.testing.assert_allclose(strong_features[columns].to_numpy(), features[columns].to_numpy(), rtol=1e-12, atol=1e-12)
    assert np.isfinite(strong_features[extras].to_numpy()).all()

    state_names = state_feature_names(channel, n_levels)
    master = master_order(columns, channel, extras, state_names)
    numeric_master = [c for c in master if c not in set(state_names)]
    assert sorted(master) == sorted(columns + extras + state_names)
    medium_block = channel_columns(columns, channel)
    strong_block = medium_block + extras
    assert len(medium_block) == 7 and len(strong_block) == 12
    keep = {"none": set(), "medium": set(medium_block), "strong": set(strong_block)}
    catalogs = {}
    for arm, recipe in ARMS.items():
        selected = keep[recipe["numeric"]] | (set(state_names) if recipe["state"] else set())
        catalogs[arm] = [c for c in master if c not in set(strong_block) | set(state_names) or c in selected]
    for arm, cols in catalogs.items():
        assert len(cols) == EXPECTED_FEATURES[arm], (arm, len(cols), EXPECTED_FEATURES[arm])
        assert len(set(cols)) == len(cols)
    assert catalogs["baseline"] == columns

    numeric = pd.concat([features[columns], strong_features[extras]], axis=1)[numeric_master]
    assert list(numeric.columns) == numeric_master

    folds = partitions(data, config)
    channel_values = data[channel].to_numpy()
    global_levels = derive_levels(channel_values, config["state_level_tolerance"])
    fold_levels, fold_states = {}, {}
    for fold, fitting, _, _ in folds:
        levels = derive_levels(channel_values[fitting], config["state_level_tolerance"])
        if len(levels) != n_levels:
            raise RuntimeError(f"Fold {fold} fitting UAVs show {len(levels)} states, not {n_levels}. "
                               "Re-declare state_levels or the level derivation before running.")
        fold_levels[fold] = levels
        frame = build_state_features(data, assign_states(channel_values, levels), n_levels, channel)
        assert list(frame.columns) == state_names and np.isfinite(frame.to_numpy()).all()
        fold_states[fold] = frame

    # A truncated trajectory must reproduce the retained rows exactly: no feature,
    # numeric or state, may read an observation after the prediction cutoff.
    probe_id = data.uav_id.iloc[0]
    probe = data[(data.uav_id == probe_id) & (data.flight_cycle <= 37)].copy()
    np.testing.assert_allclose(module.build_features_tiered(probe, tiers)[columns].to_numpy(),
                               features.loc[probe.index, columns].to_numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(module.build_features_tiered(probe, promoted)[extras].to_numpy(),
                               strong_features.loc[probe.index, extras].to_numpy(), rtol=1e-12, atol=1e-12)
    truncated = probe.reset_index(drop=True)
    for fold, levels in fold_levels.items():
        rebuilt = build_state_features(truncated, assign_states(truncated[channel].to_numpy(), levels), n_levels, channel)
        np.testing.assert_allclose(rebuilt.to_numpy(), fold_states[fold].loc[probe.index].to_numpy(), rtol=1e-12, atol=1e-12)

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
    return dict(module=module, data=data, numeric=numeric, numeric_columns=numeric_master, tiers=tiers,
                promoted_tiers=promoted, catalogs=catalogs, master=master, state_names=state_names,
                folds=folds, fold_levels=fold_levels, fold_states=fold_states, global_levels=global_levels,
                membership=pd.DataFrame(membership), endpoints=endpoints, cutoffs=cutoffs)


def metrics(frame, target="target_capped"):
    actual, predicted = frame[target], frame.prediction
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
            "bias": float(np.mean(predicted - actual))}


def paired_change(predictions, summary, fold_metrics, treatment, reference, uavs, draws, family, labels):
    """Paired UAV bootstrap of the RMSE difference between two matched arms."""
    indexed = {}
    for arm in (treatment, reference):
        frame = predictions[predictions.arm == arm].set_index(["evaluation_seed", "uav_id", "flight_cycle"]).sort_index()
        indexed[arm] = frame
    assert indexed[treatment].index.equals(indexed[reference].index)
    np.testing.assert_array_equal(indexed[treatment].target_capped, indexed[reference].target_capped)
    boot = {}
    for arm, frame in indexed.items():
        squared = (frame.prediction - frame.target_capped) ** 2
        uav_mse = squared.groupby(level="uav_id").mean().reindex(uavs).to_numpy()
        boot[arm] = np.sqrt(uav_mse[draws].mean(axis=1))
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
                interpretation=labels[0] if adjusted_hi < 0 else labels[1] if adjusted_lo > 0 else "inconclusive")


def report(output, config, catalogs):
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
        row = {"arm": arm, "features": len(catalogs[arm]), **metrics(frame)}
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
    labels = ("representation_candidate", "representation_harmful")
    primary = [arm for arm in catalogs if arm != "baseline"]
    paired = pd.DataFrame([paired_change(predictions, summary, fold_metrics, arm, "baseline", uavs, draws, len(primary), labels)
                           for arm in primary]).sort_values("rmse_change")
    write_csv(output / "reporting/paired_representations.csv", paired)
    contrasts = [tuple(pair) for pair in config["secondary_contrasts"]]
    secondary = pd.DataFrame([paired_change(predictions, summary, fold_metrics, treatment, reference, uavs, draws, len(contrasts), labels)
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

    text = [f"# PE_38: v13 XGBoost {config['channel']} representation study", "",
            "Completed 30 matched fits: the unchanged v13 representation, a deeper numeric recipe, "
            "a discrete-state recipe, both combinations, and removal of the channel.", "",
            "Primary target: RUL capped at 125, matching v13. Negative RMSE change favors the arm over the "
            "unchanged baseline. Each UAV has ten test-like cutoffs and equal total evaluation weight.", "",
            "| Arm | Telemetry 07 representation | Features | RMSE | R2 | Mean cutoff-seed R2 |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    described = {"baseline": "medium numeric (v13 as shipped)", "numeric_strong": "strong-tier numeric",
                 "state_only": "discrete state block only", "medium_plus_state": "medium numeric + state block",
                 "strong_plus_state": "strong numeric + state block", "drop_07": "channel removed"}
    for row in summary.itertuples():
        text.append(f"| {row.arm} | {described[row.arm]} | {row.features} | {row.rmse:.4f} | {row.r2:.5f} | {row.mean_test_like_seed_r2:.5f} |")
    text += ["", "| Arm | RMSE change vs. baseline | Fold wins | Paired UAV 95% interval | Five-comparison adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in paired.itertuples():
        text.append(f"| {row.arm} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    text += ["", "Pre-declared secondary contrasts, adjusted over their own family. They describe where any "
             "effect sits; only the five baseline comparisons above are decisive.", "",
             "| Contrast | RMSE change | Fold wins | Paired UAV 95% interval | Adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in secondary.itertuples():
        text.append(f"| {row.arm} vs. {row.reference} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    text += ["", "The state block is derived per fold from the fitting UAVs only, and every state column is causal: "
             "at each cycle it reads the current state, the distinct states and transitions so far, the transition "
             "rate, the current run length, the highest state reached, and the share of observed cycles spent in "
             "each state. For a tree the ordinal state code is an order-preserving relabel of the raw value, so "
             "`state_only` retains the channel's current level and drops only its numeric history.", "",
             "These are development ablations on previously examined UAVs with historically selected, frozen "
             "feature tiers. Bootstrap intervals resample UAVs and condition on the fitted models; they do not "
             "measure variation from retraining on different splits. Ten cutoff seeds are repeated endpoints, "
             "not ten independent training runs.", "",
             "One main XGBoost per arm/fold, no early specialist or gating. Outer scoring UAVs are excluded from "
             "state-level derivation, feature-weight calculation, fitting, and early stopping. A fixed 10% of "
             "outer-training UAVs supplies unweighted early stopping; the model is not refit, matching v13's "
             "final-fit holdout approach. Feature weights use only actual fitting UAVs, and every state column "
             "inherits the channel's own correlation weight. Tree settings and all-cycle sample-weight mechanics "
             "come directly from v13.", "",
             "A winning representation still needs confirmation. These arms change one channel only; the result "
             "does not transfer to other channels and does not license combining it with other pending changes. "
             "No production model or submission was changed.", ""]
    (output / "reporting/report.md").write_text("\n".join(text), encoding="utf-8")
    write_json(output / "reporting/completion.json",
               {"status": "complete", "fits": len(catalogs) * config["outer_folds"], "arms": len(catalogs),
                "primary_target": "cap125", "bootstrap_unit": "uav_id", "production_changed": False})
    print(summary[["arm", "features", "rmse", "r2"]].to_string(index=False), flush=True)
    print(paired[["arm", "rmse_change", "improved_folds", "interpretation"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate features, states, partitions and endpoints without fitting")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"]
    prepared = prepare(config)
    module, data, catalogs = prepared["module"], prepared["data"], prepared["catalogs"]
    declared = {arm for pair in config["secondary_contrasts"] for arm in pair}
    assert declared <= set(catalogs), sorted(declared - set(catalogs))
    hashes = {name: sha(ROOT / config[name]) for name in ("source", "feature_contract", "train", "test_cutoffs")}
    hashes.update(runner=sha(__file__), settings=sha(HERE / "settings.toml"))
    registration = {"config": config, "hashes": hashes, "xgboost_params": module.XGB_PARAMS,
                    "early_stopping_rounds": module.XGB_EARLY_STOPPING_ROUNDS, "target_cap": module.RUL_CAP,
                    "rolling_windows": list(module.ROLLING_WINDOWS), "sensor_tiers": prepared["tiers"],
                    "arms": ARMS, "feature_columns": catalogs, "state_features": prepared["state_names"],
                    "state_levels_global": prepared["global_levels"],
                    "state_levels_by_fold": {str(fold): levels for fold, levels in prepared["fold_levels"].items()},
                    "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__},
                    "protocol": "main_xgboost_only; frozen_historical_tiers; inner_10pct_uav_early_stop; no_refit; "
                                "fit_only_feature_weights; fit_only_state_levels; causal_state_block; v13_GroupKFold; "
                                "ten_test_cutoff_seeds"}
    manifest = output / "registration.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != registration:
            raise RuntimeError("Inputs, code or settings changed. Choose a new study.run before running.")
    else:
        write_json(manifest, registration)
        shutil.copyfile(ROOT / config["source"], output / "v13_source_snapshot.py")
    write_csv(output / "fold_membership.csv", prepared["membership"])
    write_csv(output / "evaluation_endpoints.csv", prepared["endpoints"])
    write_json(output / "feature_catalog.json", catalogs)
    write_json(output / "input_verification.json",
               {"status": "passed", "feature_counts": {k: len(v) for k, v in catalogs.items()},
                "uavs": int(data.uav_id.nunique()), "training_rows": len(data), "evaluation_rows": len(prepared["endpoints"]),
                "causal_prefix_check": True, "causal_state_prefix_check": True,
                "disjoint_fit_stopping_score_uavs": True, "fold_state_levels_match_global": all(
                    np.allclose(levels, prepared["global_levels"]) for levels in prepared["fold_levels"].values())})
    print(f"Validated {len(catalogs)} arms, {len(prepared['folds'])} folds, {len(prepared['endpoints'])} matched endpoints", flush=True)
    if args.check:
        return
    if args.report_only:
        report(output, config, catalogs)
        return

    endpoints, cutoffs = prepared["endpoints"], prepared["cutoffs"]
    numeric_columns = prepared["numeric_columns"]
    numeric_matrix = prepared["numeric"].to_numpy()
    y = data.RUL.to_numpy()
    sensors = [sensor for tier in prepared["tiers"].values() for sensor in tier]
    total = len(catalogs) * config["outer_folds"]
    completed = 0
    for fold, fitting, stopping, held in prepared["folds"]:
        combined_columns = numeric_columns + prepared["state_names"]
        matrix = np.column_stack([numeric_matrix, prepared["fold_states"][fold].to_numpy()])
        with redirect_stdout(io.StringIO()):
            feature_weights = module.compute_feature_weights(combined_columns, data.iloc[fitting], sensors)
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
            indices = [combined_columns.index(column) for column in columns]
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
                                   "state_levels": prepared["fold_levels"][fold],
                                   "feature_weights": dict(zip(columns, feature_weights[indices].tolist())),
                                   "sample_weight_min": float(sample_weights.min()), "sample_weight_max": float(sample_weights.max())})
            completed += 1
            print(f"Completed {completed}/{total} in {elapsed:.1f}s; iteration={model.best_iteration}; "
                  f"RMSE={metrics(selected_endpoints)['rmse']:.4f}", flush=True)
    report(output, config, catalogs)


if __name__ == "__main__":
    main()
