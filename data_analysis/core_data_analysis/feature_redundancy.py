"""Compare row-level and UAV-level telemetry redundancy."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from core_common import (
    ID_COLUMN,
    is_effectively_constant,
    load_dataset,
    make_parser,
    save_figure,
    save_table,
    selected_channels,
)


def matrix_to_long(matrix: pd.DataFrame, name: str) -> pd.DataFrame:
    result = matrix.copy()
    result.index.name = "channel"
    return result.reset_index().melt(
        id_vars="channel", var_name="other_channel", value_name=name
    )


def main() -> None:
    parser = make_parser(
        "Calculate row-level and UAV-summary telemetry correlation structure.",
        "feature_redundancy",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="Absolute correlation used to define strong redundancy (default: 0.90).",
    )
    args = parser.parse_args()
    if not 0 < args.correlation_threshold < 1:
        parser.error("--correlation-threshold must be in (0, 1)")

    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels, require_rul=True)
    uav_means = data.groupby(ID_COLUMN, sort=True)[channels].mean()
    matrices = {
        "row_pearson": data[channels].corr(method="pearson"),
        "row_spearman": data[channels].corr(method="spearman"),
        "uav_pearson": uav_means.corr(method="pearson"),
        "uav_spearman": uav_means.corr(method="spearman"),
    }

    nonconstant = [
        channel for channel in channels if not is_effectively_constant(data[channel])
    ]
    constants = [channel for channel in channels if channel not in nonconstant]
    for matrix in matrices.values():
        matrix.loc[constants, :] = np.nan
        matrix.loc[:, constants] = np.nan
    combined_strength = np.maximum(
        matrices["row_spearman"].loc[nonconstant, nonconstant].abs().to_numpy(),
        matrices["uav_spearman"].loc[nonconstant, nonconstant].abs().to_numpy(),
    )
    combined_strength = np.nan_to_num(combined_strength, nan=0.0)
    np.fill_diagonal(combined_strength, 1.0)
    distance = np.clip(1.0 - combined_strength, 0.0, 1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    if len(nonconstant) > 1:
        linkage = hierarchy.linkage(squareform(distance), method="average")
        ordered_nonconstant = [nonconstant[index] for index in hierarchy.leaves_list(linkage)]
        cluster_ids = hierarchy.fcluster(
            linkage,
            t=1.0 - args.correlation_threshold,
            criterion="distance",
        )
    else:
        ordered_nonconstant = nonconstant
        cluster_ids = np.ones(len(nonconstant), dtype=int)
    order = ordered_nonconstant + constants

    pair_records = []
    for left_index, channel_a in enumerate(nonconstant):
        for channel_b in nonconstant[left_index + 1 :]:
            values = {
                name: matrix.loc[channel_a, channel_b]
                for name, matrix in matrices.items()
            }
            finite_values = [abs(float(value)) for value in values.values() if pd.notna(value)]
            maximum = max(finite_values, default=np.nan)
            if pd.notna(maximum) and maximum >= args.correlation_threshold:
                pair_records.append(
                    {
                        "channel_a": channel_a,
                        "channel_b": channel_b,
                        **values,
                        "maximum_absolute_correlation": maximum,
                    }
                )
    strong_pairs = pd.DataFrame.from_records(pair_records)
    if not strong_pairs.empty:
        strong_pairs = strong_pairs.sort_values(
            "maximum_absolute_correlation", ascending=False
        )

    cluster_map = dict(zip(nonconstant, cluster_ids))
    cluster_records = []
    for channel in channels:
        peers = strong_pairs.loc[
            (strong_pairs.get("channel_a", pd.Series(dtype=str)) == channel)
            | (strong_pairs.get("channel_b", pd.Series(dtype=str)) == channel)
        ] if not strong_pairs.empty else pd.DataFrame()
        cluster_records.append(
            {
                "channel": channel,
                "effectively_constant": channel in constants,
                "cluster_id": int(cluster_map[channel]) if channel in cluster_map else np.nan,
                "strongly_correlated_peer_count": len(peers),
                "maximum_absolute_correlation": (
                    float(peers["maximum_absolute_correlation"].max())
                    if not peers.empty
                    else np.nan
                ),
            }
        )
    clusters = pd.DataFrame.from_records(cluster_records)

    figure, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#E6E6E6")
    image = None
    titles = {
        "row_pearson": "Row-level Pearson",
        "row_spearman": "Row-level Spearman",
        "uav_pearson": "UAV-mean Pearson",
        "uav_spearman": "UAV-mean Spearman",
    }
    for axis, (name, matrix) in zip(axes.flat, matrices.items()):
        plotted = matrix.loc[order, order]
        image = axis.imshow(plotted, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        axis.set_title(titles[name])
        axis.set_xticks(np.arange(len(order)), order, rotation=90, fontsize=6)
        axis.set_yticks(np.arange(len(order)), order, fontsize=6)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.75, pad=0.02)
    colorbar.set_label("Correlation")
    figure.suptitle(
        "Telemetry redundancy at row and UAV-summary levels\n"
        f"Ordering uses absolute Spearman structure; strong-pair threshold = {args.correlation_threshold:.2f}",
        fontsize=14,
    )

    paths = []
    for name, matrix in matrices.items():
        paths.append(save_table(matrix.reset_index(names="channel"), args.output_dir, f"{name}.csv"))
    paths.extend(
        [
            save_table(strong_pairs, args.output_dir, "strongly_correlated_pairs.csv"),
            save_table(clusters, args.output_dir, "correlation_clusters.csv"),
            save_figure(
                figure,
                args.output_dir,
                "correlation_heatmaps.png",
                args.dpi,
            ),
        ]
    )
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
