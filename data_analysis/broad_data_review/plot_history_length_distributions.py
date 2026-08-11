"""Plot train and test UAV history-length distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plotting_common import (
    CYCLE_COLUMN,
    DARK_BLUE,
    ID_COLUMN,
    LIGHT_BLUE,
    load_dataset,
    make_parser,
    save_csv,
    save_figure,
    style_axis,
)
import matplotlib.pyplot as plt


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    probability = np.arange(1, len(ordered) + 1) / len(ordered)
    return ordered, probability


def main() -> None:
    parser = make_parser(
        "Plot train and test UAV history-length distributions.",
        "history_length_distributions",
        include_test=True,
        include_channels=False,
    )
    args = parser.parse_args()
    train = load_dataset(args.train_csv, [])
    test = load_dataset(args.test_csv, [])

    lengths = pd.concat(
        [
            train.groupby(ID_COLUMN)[CYCLE_COLUMN]
            .size()
            .rename("history_length")
            .to_frame()
            .assign(split="train"),
            test.groupby(ID_COLUMN)[CYCLE_COLUMN]
            .size()
            .rename("history_length")
            .to_frame()
            .assign(split="test"),
        ]
    ).reset_index()
    train_lengths = lengths.loc[lengths["split"] == "train", "history_length"].to_numpy()
    test_lengths = lengths.loc[lengths["split"] == "test", "history_length"].to_numpy()
    upper = int(max(train_lengths.max(), test_lengths.max()))
    bins = np.arange(0, upper + 26, 25)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].hist(
        train_lengths, bins=bins, alpha=0.62, color=DARK_BLUE, label="Train"
    )
    axes[0].hist(
        test_lengths, bins=bins, alpha=0.52, color=LIGHT_BLUE, label="Test"
    )
    axes[0].set(xlabel="History length (cycles)", ylabel="Number of UAVs", title="Histogram")
    axes[0].legend(frameon=False)

    for values, color, label in [
        (train_lengths, DARK_BLUE, "Train"),
        (test_lengths, LIGHT_BLUE, "Test"),
    ]:
        x_values, probabilities = ecdf(values)
        axes[1].step(x_values, probabilities, where="post", color=color, label=label)
    axes[1].set(
        xlabel="History length (cycles)",
        ylabel="Cumulative fraction of UAVs",
        title="Empirical CDF",
    )
    axes[1].legend(frameon=False)

    axes[2].boxplot(
        [train_lengths, test_lengths],
        tick_labels=["Train", "Test"],
        patch_artist=True,
        boxprops={"facecolor": LIGHT_BLUE, "alpha": 0.5},
        medianprops={"color": DARK_BLUE, "linewidth": 2},
    )
    axes[2].set(ylabel="History length (cycles)", title="Box plots")
    for position, values in enumerate([train_lengths, test_lengths], start=1):
        axes[2].text(
            position,
            0.98,
            f"median={np.median(values):.1f}\nrange={values.min()}–{values.max()}",
            transform=axes[2].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
    for axis in axes:
        style_axis(axis)
    figure.suptitle("Per-UAV history-length distributions", fontsize=14)

    csv_path = save_csv(
        lengths.set_index(["split", ID_COLUMN]),
        args.output_dir,
        "history_lengths.csv",
    )
    figure_path = save_figure(
        figure, args.output_dir, "history_length_distributions.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
