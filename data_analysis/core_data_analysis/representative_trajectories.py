"""Plot short-, median-, and long-history train/test UAV trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_common import (
    CYCLE_COLUMN,
    ID_COLUMN,
    is_effectively_constant,
    load_dataset,
    make_parser,
    robust_scale,
    save_figure,
    save_table,
    selected_channels,
)


CATEGORIES = ["short", "median", "long"]
QUANTILES = [0.10, 0.50, 0.90]


def choose_representatives(lengths: pd.Series) -> pd.DataFrame:
    available = set(lengths.index)
    records = []
    for category, quantile in zip(CATEGORIES, QUANTILES):
        target = float(lengths.quantile(quantile))
        selected = min(
            available,
            key=lambda uav_id: (abs(float(lengths.loc[uav_id]) - target), str(uav_id)),
        )
        available.remove(selected)
        records.append(
            {
                "category": category,
                ID_COLUMN: selected,
                "history_length": int(lengths.loc[selected]),
                "target_quantile": quantile,
                "target_length": target,
            }
        )
    return pd.DataFrame.from_records(records)


def choose_age_matched_test(
    test_lengths: pd.Series,
    train_selection: pd.DataFrame,
) -> pd.DataFrame:
    available = set(test_lengths.index)
    records = []
    for row in train_selection.itertuples(index=False):
        selected = min(
            available,
            key=lambda uav_id: (
                abs(float(test_lengths.loc[uav_id]) - row.history_length),
                str(uav_id),
            ),
        )
        available.remove(selected)
        records.append(
            {
                "category": row.category,
                ID_COLUMN: selected,
                "history_length": int(test_lengths.loc[selected]),
                "matched_train_length": int(row.history_length),
                "age_difference": int(test_lengths.loc[selected] - row.history_length),
            }
        )
    return pd.DataFrame.from_records(records)


def training_robust_parameters(
    train: pd.DataFrame, channels: list[str]
) -> pd.DataFrame:
    records = []
    for channel in channels:
        center, scale = robust_scale(train[channel])
        if is_effectively_constant(train[channel]):
            scale = 0.0
        records.append(
            {
                "channel": channel,
                "train_median": center,
                "train_robust_scale": scale,
            }
        )
    return pd.DataFrame.from_records(records).set_index("channel")


def scaled_history(
    history: pd.DataFrame,
    channels: list[str],
    parameters: pd.DataFrame,
    rolling_median: int,
) -> pd.DataFrame:
    values = history[[CYCLE_COLUMN, *channels]].copy()
    for channel in channels:
        scale = float(parameters.loc[channel, "train_robust_scale"])
        if scale <= 0:
            values[channel] = 0.0
        else:
            values[channel] = (
                values[channel] - parameters.loc[channel, "train_median"]
            ) / scale
    if rolling_median > 1:
        values[channels] = values[channels].rolling(
            rolling_median, center=True, min_periods=1
        ).median()
    return values


def main() -> None:
    parser = make_parser(
        "Plot representative short-, median-, and long-history train/test UAVs.",
        "representative_trajectories",
        include_test=True,
    )
    parser.add_argument(
        "--rolling-median",
        type=int,
        default=5,
        help="Smoothing window used only in the visualisation (default: 5).",
    )
    parser.add_argument(
        "--color-limit",
        type=float,
        default=3.0,
        help="Symmetric robust-Z colour limit (default: 3).",
    )
    args = parser.parse_args()
    if args.rolling_median < 1:
        parser.error("--rolling-median must be at least 1")
    if args.color_limit <= 0:
        parser.error("--color-limit must be greater than zero")

    channels = selected_channels(args)
    train = load_dataset(args.train_csv, channels, require_rul=True)
    test = load_dataset(args.test_csv, channels, require_rul=False)
    train_lengths = train.groupby(ID_COLUMN)[CYCLE_COLUMN].max()
    test_lengths = test.groupby(ID_COLUMN)[CYCLE_COLUMN].max()
    train_selection = choose_representatives(train_lengths)
    train_selection.insert(0, "split", "train")
    test_selection = choose_age_matched_test(
        test_lengths, train_selection.drop(columns="split")
    )
    test_selection.insert(0, "split", "test")
    selection = pd.concat([train_selection, test_selection], ignore_index=True)

    parameters = training_robust_parameters(train, channels)
    scaled_parts = []
    matrices: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
    for row in selection.itertuples(index=False):
        source = train if row.split == "train" else test
        history = source.loc[source[ID_COLUMN] == row.uav_id]
        scaled = scaled_history(
            history, channels, parameters, args.rolling_median
        )
        exported = scaled.copy()
        exported.insert(0, ID_COLUMN, row.uav_id)
        exported.insert(0, "category", row.category)
        exported.insert(0, "split", row.split)
        scaled_parts.append(exported)
        matrices[(row.split, row.category)] = (scaled, row.uav_id)

    scaled_trajectories = pd.concat(scaled_parts, ignore_index=True)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row_index, split in enumerate(["train", "test"]):
        for column_index, category in enumerate(CATEGORIES):
            axis = axes[row_index, column_index]
            scaled, uav_id = matrices[(split, category)]
            matrix = scaled[channels].to_numpy(dtype=float).T
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-args.color_limit,
                vmax=args.color_limit,
                extent=(
                    float(scaled[CYCLE_COLUMN].min()),
                    float(scaled[CYCLE_COLUMN].max()),
                    len(channels) - 0.5,
                    -0.5,
                ),
            )
            axis.set_title(
                f"{split.title()} {category}: {uav_id}\n"
                f"{int(scaled[CYCLE_COLUMN].max())} observed cycles",
                fontsize=10,
            )
            axis.set_xlabel("Flight cycle")
            axis.set_yticks(np.arange(len(channels)), channels, fontsize=7)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.75, pad=0.02)
    colorbar.set_label("Train-referenced robust Z value")
    figure.suptitle(
        "Representative UAV telemetry trajectories\n"
        "Test UAVs are selected to match the train representatives' observed ages",
        fontsize=14,
    )

    paths = [
        save_table(selection, args.output_dir, "selected_uavs.csv"),
        save_table(parameters.reset_index(), args.output_dir, "train_scaling.csv"),
        save_table(
            scaled_trajectories,
            args.output_dir,
            "selected_scaled_trajectories.csv",
        ),
        save_figure(
            figure,
            args.output_dir,
            "representative_trajectories.png",
            args.dpi,
        ),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
