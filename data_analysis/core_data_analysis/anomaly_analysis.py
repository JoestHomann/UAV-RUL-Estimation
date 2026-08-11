"""Detect robust extremes, jumps, persistent shifts, and similar UAV histories."""

from __future__ import annotations

from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_common import (
    CYCLE_COLUMN,
    DARK_BLUE,
    GRAY,
    ID_COLUMN,
    LIGHT_BLUE,
    ORANGE,
    TARGET_COLUMN,
    is_effectively_constant,
    load_dataset,
    make_parser,
    robust_scale,
    save_figure,
    save_table,
    selected_channels,
    style_axis,
)


def training_reference(
    train: pd.DataFrame, channels: list[str]
) -> pd.DataFrame:
    records = []
    for channel in channels:
        constant = is_effectively_constant(train[channel])
        level_median, level_scale = robust_scale(train[channel])
        differences = train.groupby(ID_COLUMN, sort=False)[channel].diff().dropna()
        jump_median, jump_scale = robust_scale(differences)
        q25, q75 = train[channel].quantile([0.25, 0.75])
        if constant:
            level_scale = 0.0
            jump_scale = 0.0
        records.append(
            {
                "channel": channel,
                "effectively_constant": constant,
                "level_median": level_median,
                "level_robust_scale": level_scale,
                "train_iqr": 0.0 if constant else float(q75 - q25),
                "jump_median": jump_median,
                "jump_robust_scale": jump_scale,
            }
        )
    return pd.DataFrame.from_records(records).set_index("channel")


def flag_rows(
    data: pd.DataFrame,
    split: str,
    channels: list[str],
    reference: pd.DataFrame,
    *,
    robust_z_threshold: float,
    jump_z_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extreme_parts = []
    jump_parts = []
    summary_records = []
    for channel in channels:
        level_scale = float(reference.loc[channel, "level_robust_scale"])
        if level_scale > 0:
            robust_z = (
                data[channel] - reference.loc[channel, "level_median"]
            ) / level_scale
        else:
            robust_z = pd.Series(np.zeros(len(data)), index=data.index)
        extreme_flag = robust_z.abs() > robust_z_threshold
        if extreme_flag.any():
            selected = data.loc[
                extreme_flag,
                [ID_COLUMN, CYCLE_COLUMN, channel]
                + ([TARGET_COLUMN] if TARGET_COLUMN in data.columns else []),
            ].copy()
            selected = selected.rename(columns={channel: "value"})
            selected.insert(0, "split", split)
            selected.insert(3, "channel", channel)
            selected["robust_z"] = robust_z.loc[extreme_flag].to_numpy()
            extreme_parts.append(selected)

        previous = data.groupby(ID_COLUMN, sort=False)[channel].shift(1)
        difference = data[channel] - previous
        jump_scale = float(reference.loc[channel, "jump_robust_scale"])
        if jump_scale > 0:
            jump_z = (
                difference - reference.loc[channel, "jump_median"]
            ) / jump_scale
        else:
            jump_z = pd.Series(np.zeros(len(data)), index=data.index)
        jump_flag = jump_z.abs() > jump_z_threshold
        if jump_flag.any():
            selected = data.loc[
                jump_flag,
                [ID_COLUMN, CYCLE_COLUMN, channel]
                + ([TARGET_COLUMN] if TARGET_COLUMN in data.columns else []),
            ].copy()
            selected = selected.rename(columns={channel: "value"})
            selected.insert(0, "split", split)
            selected.insert(3, "channel", channel)
            selected["previous_value"] = previous.loc[jump_flag].to_numpy()
            selected["cycle_to_cycle_change"] = difference.loc[jump_flag].to_numpy()
            selected["robust_jump_z"] = jump_z.loc[jump_flag].to_numpy()
            jump_parts.append(selected)

        summary_records.append(
            {
                "split": split,
                "channel": channel,
                "rows": len(data),
                "robust_extreme_rows": int(extreme_flag.sum()),
                "robust_extreme_rows_percent": 100.0 * float(extreme_flag.mean()),
                "robust_extreme_uavs": int(data.loc[extreme_flag, ID_COLUMN].nunique()),
                "jump_rows": int(jump_flag.sum()),
                "jump_rows_percent": 100.0 * float(jump_flag.mean()),
                "jump_uavs": int(data.loc[jump_flag, ID_COLUMN].nunique()),
            }
        )

    base_columns = [
        "split",
        ID_COLUMN,
        CYCLE_COLUMN,
        "channel",
        "value",
    ]
    extremes = (
        pd.concat(extreme_parts, ignore_index=True)
        if extreme_parts
        else pd.DataFrame(columns=[*base_columns, "robust_z"])
    )
    jumps = (
        pd.concat(jump_parts, ignore_index=True)
        if jump_parts
        else pd.DataFrame(
            columns=[
                *base_columns,
                "previous_value",
                "cycle_to_cycle_change",
                "robust_jump_z",
            ]
        )
    )
    return extremes, jumps, pd.DataFrame.from_records(summary_records)


def candidate_shift(
    cycles: np.ndarray,
    values: np.ndarray,
    *,
    window: int,
    level_scale: float,
    threshold: float,
) -> dict[str, float | int | bool]:
    if len(values) < 2 * window or level_scale <= 0:
        return {
            "shift_cycle": np.nan,
            "pre_shift_median": np.nan,
            "post_shift_median": np.nan,
            "candidate_shift_size": 0.0,
            "candidate_shift_score": 0.0,
            "persistence_ratio": 0.0,
            "permanent_shift_flag": False,
        }
    series = pd.Series(values)
    past = series.rolling(window, min_periods=window).median().to_numpy()
    future = (
        series.iloc[::-1]
        .rolling(window, min_periods=window)
        .median()
        .iloc[::-1]
        .to_numpy()
    )
    boundaries = np.arange(window, len(values) - window + 1)
    changes = future[boundaries] - past[boundaries - 1]
    best_position = int(np.argmax(np.abs(changes)))
    boundary = int(boundaries[best_position])
    change = float(changes[best_position])
    pre_median = float(past[boundary - 1])
    post_median = float(future[boundary])
    tail_median = float(np.median(values[boundary:]))
    persistence_ratio = (
        abs(tail_median - pre_median) / abs(change) if abs(change) > 0 else 0.0
    )
    same_direction = np.sign(tail_median - pre_median) == np.sign(change)
    score = abs(change) / level_scale
    return {
        "shift_cycle": int(cycles[boundary]),
        "pre_shift_median": pre_median,
        "post_shift_median": post_median,
        "candidate_shift_size": change,
        "candidate_shift_score": score,
        "persistence_ratio": persistence_ratio,
        "permanent_shift_flag": bool(
            score >= threshold and same_direction and persistence_ratio >= 0.5
        ),
    }


def find_shifts(
    data: pd.DataFrame,
    split: str,
    channels: list[str],
    reference: pd.DataFrame,
    *,
    window: int,
    threshold: float,
) -> pd.DataFrame:
    records = []
    for uav_id, history in data.groupby(ID_COLUMN, sort=True):
        cycles = history[CYCLE_COLUMN].to_numpy(dtype=int)
        for channel in channels:
            level_scale = float(reference.loc[channel, "train_iqr"])
            if level_scale <= 0:
                level_scale = float(reference.loc[channel, "level_robust_scale"])
            records.append(
                {
                    "split": split,
                    ID_COLUMN: uav_id,
                    "channel": channel,
                    "history_length": len(history),
                    **candidate_shift(
                        cycles,
                        history[channel].to_numpy(dtype=float),
                        window=window,
                        level_scale=level_scale,
                        threshold=threshold,
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def normalized_histories(
    data: pd.DataFrame,
    channels: list[str],
    reference: pd.DataFrame,
) -> dict[str, np.ndarray]:
    usable = [
        channel
        for channel in channels
        if float(reference.loc[channel, "level_robust_scale"]) > 0
        and not is_effectively_constant(data[channel])
    ]
    result = {}
    for uav_id, history in data.groupby(ID_COLUMN, sort=True):
        matrix = history[usable].to_numpy(dtype=float)
        centers = reference.loc[usable, "level_median"].to_numpy(dtype=float)
        scales = reference.loc[usable, "level_robust_scale"].to_numpy(dtype=float)
        result[str(uav_id)] = (matrix - centers) / scales
    return result


def compare_history_pair(
    first: np.ndarray,
    second: np.ndarray,
    *,
    minimum_overlap: int,
    maximum_points: int,
) -> tuple[int, float, float] | None:
    overlap = min(len(first), len(second))
    if overlap < minimum_overlap:
        return None
    count = min(overlap, maximum_points)
    positions = np.unique(np.linspace(0, overlap - 1, count).round().astype(int))
    difference = first[positions] - second[positions]
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    mean_absolute = float(np.mean(np.abs(difference)))
    return overlap, rmse, mean_absolute


def copied_history_search(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channels: list[str],
    reference: pd.DataFrame,
    *,
    minimum_overlap: int,
    maximum_points: int,
    near_duplicate_threshold: float,
    keep_pairs: int,
) -> pd.DataFrame:
    histories = {
        "train": normalized_histories(train, channels, reference),
        "test": normalized_histories(test, channels, reference),
    }
    records = []
    comparisons = [
        (
            "train-train",
            (("train", left, "train", right) for left, right in combinations(histories["train"], 2)),
        ),
        (
            "test-test",
            (("test", left, "test", right) for left, right in combinations(histories["test"], 2)),
        ),
        (
            "train-test",
            (("train", left, "test", right) for left, right in product(histories["train"], histories["test"])),
        ),
    ]
    for comparison_type, pairs in comparisons:
        type_records = []
        for split_a, uav_a, split_b, uav_b in pairs:
            result = compare_history_pair(
                histories[split_a][uav_a],
                histories[split_b][uav_b],
                minimum_overlap=minimum_overlap,
                maximum_points=maximum_points,
            )
            if result is None:
                continue
            overlap, rmse, mean_absolute = result
            type_records.append(
                {
                    "comparison": comparison_type,
                    "split_a": split_a,
                    "uav_a": uav_a,
                    "split_b": split_b,
                    "uav_b": uav_b,
                    "overlap_cycles": overlap,
                    "sampled_points": min(overlap, maximum_points),
                    "robust_scaled_rmse": rmse,
                    "robust_scaled_mean_absolute_difference": mean_absolute,
                    "near_duplicate": rmse <= near_duplicate_threshold,
                }
            )
        ordered = sorted(type_records, key=lambda row: row["robust_scaled_rmse"])
        retained = ordered[:keep_pairs]
        seen_a: set[tuple[str, str]] = set()
        seen_b: set[tuple[str, str]] = set()
        for row in ordered:
            key_a = (row["split_a"], row["uav_a"])
            key_b = (row["split_b"], row["uav_b"])
            if key_a not in seen_a or key_b not in seen_b:
                retained.append(row)
                seen_a.add(key_a)
                seen_b.add(key_b)
        unique_retained = {
            (row["split_a"], row["uav_a"], row["split_b"], row["uav_b"]): row
            for row in retained
        }
        records.extend(unique_retained.values())
    return pd.DataFrame.from_records(records)


def history_lengths(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for split, data in [("train", train), ("test", test)]:
        lengths = data.groupby(ID_COLUMN)[CYCLE_COLUMN].max().rename("history_length")
        q25, q75 = lengths.quantile([0.25, 0.75])
        iqr = q75 - q25
        table = lengths.reset_index()
        table.insert(0, "split", split)
        table["lower_iqr_bound"] = q25 - 1.5 * iqr
        table["upper_iqr_bound"] = q75 + 1.5 * iqr
        table["unusual_history_length"] = (
            (table["history_length"] < table["lower_iqr_bound"])
            | (table["history_length"] > table["upper_iqr_bound"])
        )
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def make_uav_priority(
    train: pd.DataFrame,
    test: pd.DataFrame,
    extremes: pd.DataFrame,
    jumps: pd.DataFrame,
    shifts: pd.DataFrame,
    lengths: pd.DataFrame,
    copied_pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split, data in [("train", train), ("test", test)]:
        base = data.groupby(ID_COLUMN).size().rename("row_count").reset_index()
        base.insert(0, "split", split)
        split_extremes = extremes.loc[extremes["split"] == split]
        split_jumps = jumps.loc[jumps["split"] == split]
        extreme_counts = split_extremes.groupby(ID_COLUMN).size().rename("robust_extreme_rows")
        jump_counts = split_jumps.groupby(ID_COLUMN).size().rename("jump_rows")
        split_shifts = shifts.loc[shifts["split"] == split]
        shift_summary = split_shifts.groupby(ID_COLUMN).agg(
            permanent_shift_channels=("permanent_shift_flag", "sum"),
            maximum_shift_score=("candidate_shift_score", "max"),
        )
        base = base.merge(extreme_counts, on=ID_COLUMN, how="left")
        base = base.merge(jump_counts, on=ID_COLUMN, how="left")
        base = base.merge(shift_summary, on=ID_COLUMN, how="left")
        base = base.merge(
            lengths.loc[lengths["split"] == split, [
                ID_COLUMN,
                "history_length",
                "unusual_history_length",
            ]],
            on=ID_COLUMN,
            how="left",
        )
        rows.append(base)
    priority = pd.concat(rows, ignore_index=True).fillna(
        {"robust_extreme_rows": 0, "jump_rows": 0, "permanent_shift_channels": 0}
    )
    priority["robust_flags_per_100_cycles"] = (
        100.0 * priority["robust_extreme_rows"] / priority["row_count"]
    )
    priority["jump_flags_per_100_cycles"] = (
        100.0 * priority["jump_rows"] / priority["row_count"]
    )

    nearest: dict[tuple[str, str], float] = {}
    for row in copied_pairs.itertuples(index=False):
        value = float(row.robust_scaled_rmse)
        for split, uav_id in [(row.split_a, row.uav_a), (row.split_b, row.uav_b)]:
            key = (split, str(uav_id))
            nearest[key] = min(nearest.get(key, np.inf), value)
    priority["nearest_history_rmse"] = [
        nearest.get((row.split, str(row.uav_id)), np.nan)
        for row in priority.itertuples(index=False)
    ]

    scored_parts = []
    for _, group in priority.groupby("split", sort=False):
        group = group.copy()
        components = pd.DataFrame(index=group.index)
        components["extreme"] = group["robust_flags_per_100_cycles"].rank(pct=True)
        components["jump"] = group["jump_flags_per_100_cycles"].rank(pct=True)
        components["shift"] = group["permanent_shift_channels"].rank(pct=True)
        components["copy"] = (-group["nearest_history_rmse"]).rank(pct=True)
        components["length"] = group["unusual_history_length"].astype(float)
        group["anomaly_priority_score"] = components.mean(axis=1, skipna=True)
        scored_parts.append(group)
    return pd.concat(scored_parts, ignore_index=True).sort_values(
        ["split", "anomaly_priority_score"], ascending=[True, False]
    )


def channel_summary(
    row_summary: pd.DataFrame,
    shifts: pd.DataFrame,
) -> pd.DataFrame:
    shift_summary = (
        shifts.groupby(["split", "channel"])
        .agg(
            uavs=(ID_COLUMN, "nunique"),
            permanent_shift_uavs=("permanent_shift_flag", "sum"),
            maximum_shift_score=("candidate_shift_score", "max"),
        )
        .reset_index()
    )
    shift_summary["permanent_shift_uavs_percent"] = (
        100.0 * shift_summary["permanent_shift_uavs"] / shift_summary["uavs"]
    )
    return row_summary.merge(
        shift_summary.drop(columns="uavs"),
        on=["split", "channel"],
        how="left",
    )


def plot_anomalies(
    summary: pd.DataFrame,
    priority: pd.DataFrame,
    top_uavs: int,
    output_dir,
    dpi: int,
) -> object:
    train_summary = summary.loc[summary["split"] == "train"].sort_values(
        "robust_extreme_rows_percent"
    )
    channels = train_summary["channel"].tolist()
    y = np.arange(len(channels))
    figure, axes = plt.subplots(1, 4, figsize=(20, 10), constrained_layout=True)
    axes[0].barh(y, train_summary["robust_extreme_rows_percent"], color=DARK_BLUE)
    axes[0].set_yticks(y, channels)
    axes[0].set_title("Train robust extremes")
    axes[0].set_xlabel("Rows (%)")
    axes[1].barh(y, train_summary["jump_rows_percent"], color=LIGHT_BLUE)
    axes[1].set_yticks(y, channels)
    axes[1].set_title("Train large jumps")
    axes[1].set_xlabel("Rows (%)")
    axes[2].barh(y, train_summary["permanent_shift_uavs_percent"], color=ORANGE)
    axes[2].set_yticks(y, channels)
    axes[2].set_title("Candidate persistent shifts")
    axes[2].set_xlabel("Training UAVs (%)")

    top = priority.groupby("split", group_keys=False).head(top_uavs).copy()
    prefixes = top["split"].map({"train": "Train", "test": "Test"})
    top["label"] = prefixes + ":" + top[ID_COLUMN].astype(str)
    top = top.sort_values("anomaly_priority_score")
    axes[3].barh(
        np.arange(len(top)),
        top["anomaly_priority_score"],
        color=np.where(top["split"].eq("train"), DARK_BLUE, ORANGE),
    )
    axes[3].set_yticks(np.arange(len(top)), top["label"], fontsize=7)
    axes[3].set_title(f"Top {top_uavs} review candidates per split")
    axes[3].set_xlabel("Composite diagnostic rank")
    for axis in axes:
        style_axis(axis)
    figure.suptitle(
        "Telemetry anomaly diagnostics\nFlags are review candidates, not automatic deletion rules",
        fontsize=14,
    )
    return save_figure(figure, output_dir, "anomaly_summary.png", dpi)


def main() -> None:
    parser = make_parser(
        "Flag robust extremes, jumps, persistent shifts, and similar histories.",
        "anomalies",
        include_test=True,
    )
    parser.add_argument("--robust-z-threshold", type=float, default=6.0)
    parser.add_argument("--jump-z-threshold", type=float, default=6.0)
    parser.add_argument("--shift-window", type=int, default=10)
    parser.add_argument(
        "--shift-threshold",
        type=float,
        default=3.0,
        help="Minimum adjacent-window median shift in train-IQR units.",
    )
    parser.add_argument("--minimum-copy-overlap", type=int, default=30)
    parser.add_argument("--copy-maximum-points", type=int, default=100)
    parser.add_argument("--near-duplicate-threshold", type=float, default=1e-3)
    parser.add_argument("--keep-copy-pairs", type=int, default=50)
    parser.add_argument("--top-uavs", type=int, default=10)
    args = parser.parse_args()
    if min(args.robust_z_threshold, args.jump_z_threshold, args.shift_threshold) <= 0:
        parser.error("Anomaly thresholds must be greater than zero")
    if args.shift_window < 2:
        parser.error("--shift-window must be at least 2")

    channels = selected_channels(args)
    train = load_dataset(args.train_csv, channels, require_rul=True)
    test = load_dataset(args.test_csv, channels, require_rul=False)
    reference = training_reference(train, channels)

    extreme_parts = []
    jump_parts = []
    row_summaries = []
    shift_parts = []
    for split, data in [("train", train), ("test", test)]:
        extremes, jumps, summary = flag_rows(
            data,
            split,
            channels,
            reference,
            robust_z_threshold=args.robust_z_threshold,
            jump_z_threshold=args.jump_z_threshold,
        )
        extreme_parts.append(extremes)
        jump_parts.append(jumps)
        row_summaries.append(summary)
        shift_parts.append(
            find_shifts(
                data,
                split,
                channels,
                reference,
                window=args.shift_window,
                threshold=args.shift_threshold,
            )
        )
    extremes = pd.concat(extreme_parts, ignore_index=True)
    jumps = pd.concat(jump_parts, ignore_index=True)
    shifts = pd.concat(shift_parts, ignore_index=True)
    row_summary = pd.concat(row_summaries, ignore_index=True)
    summary = channel_summary(row_summary, shifts)
    lengths = history_lengths(train, test)
    copied_pairs = copied_history_search(
        train,
        test,
        channels,
        reference,
        minimum_overlap=args.minimum_copy_overlap,
        maximum_points=args.copy_maximum_points,
        near_duplicate_threshold=args.near_duplicate_threshold,
        keep_pairs=args.keep_copy_pairs,
    )
    priority = make_uav_priority(
        train, test, extremes, jumps, shifts, lengths, copied_pairs
    )
    manual_candidates = priority.groupby("split", group_keys=False).head(
        args.top_uavs
    )

    paths = [
        save_table(reference.reset_index(), args.output_dir, "train_robust_reference.csv"),
        save_table(extremes, args.output_dir, "robust_extreme_rows.csv"),
        save_table(jumps, args.output_dir, "large_jump_rows.csv"),
        save_table(shifts, args.output_dir, "persistent_shift_candidates.csv"),
        save_table(lengths, args.output_dir, "history_length_flags.csv"),
        save_table(copied_pairs, args.output_dir, "most_similar_uav_histories.csv"),
        save_table(summary, args.output_dir, "anomaly_channel_summary.csv"),
        save_table(priority, args.output_dir, "uav_anomaly_priority.csv"),
        save_table(
            manual_candidates,
            args.output_dir,
            "manual_review_candidates.csv",
        ),
        plot_anomalies(summary, priority, args.top_uavs, args.output_dir, args.dpi),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
