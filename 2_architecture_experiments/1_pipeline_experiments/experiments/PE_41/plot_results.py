"""Plot completed PE_41 effects without fitting or selecting a model."""
from pathlib import Path
import importlib.util
import json
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pe41_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

LABELS = {
    runner.DROP_HISTORY: "History-mean difference removed",
    runner.DROP_BASELINE: "Baseline difference removed",
    runner.DROP_BOTH: "Both removed",
}
HEAD_TO_HEAD = "History-mean reference kept\nrather than the baseline reference"


def draw(axis, rows, labels):
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#D9D9D9", alpha=0.45, linewidth=0.7)
    axis.axvline(0, color="#D55E00", linewidth=1.25, zorder=2)
    for index, row in enumerate(rows.itertuples()):
        axis.plot([row.familywise_ci95_low, row.familywise_ci95_high], [index, index],
                  color="#56B4E9", linewidth=8, alpha=0.30, solid_capstyle="butt")
        axis.plot([row.uav_bootstrap_ci95_low, row.uav_bootstrap_ci95_high], [index, index],
                  color="#0072B2", linewidth=3, solid_capstyle="butt")
        axis.scatter(row.rmse_change, index, color="#0072B2", s=38,
                     edgecolor="white", linewidth=0.7, zorder=3)
    axis.set_ylim(len(rows) - 0.5, -0.5)
    axis.set_yticks(range(len(rows)), labels)
    axis.tick_params(axis="y", labelleft=True, labelright=False, length=0, pad=10)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.7)


def main():
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"] / "reporting"
    if not (output / "completion.json").exists():
        raise RuntimeError("Finish the matched ablation before plotting.")
    diagnostics = pd.DataFrame([
        {key: record[key] for key in ("arm", "fold", "features", "best_iteration", "fit_seconds")}
        for path in sorted((output.parent / "cells").glob("*.json"))
        for record in [json.loads(path.read_text())]
    ])
    diagnostics["selected_trees"] = diagnostics.best_iteration + 1
    diagnostics.to_csv(output / "fit_diagnostics.csv", index=False)

    effects = pd.read_csv(output / "paired_ablation.csv").set_index("arm").loc[runner.PRIMARY_REMOVALS]
    secondary = pd.read_csv(output / "secondary_contrasts.csv")
    head = secondary[(secondary.arm == runner.DROP_BASELINE)
                     & (secondary.reference == runner.DROP_HISTORY_MATCHED)]
    if head.empty:
        raise RuntimeError("The matched head-to-head contrast is missing from the report.")

    # Match the presentation palette, with axis text but no title or captions.
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.edgecolor": "#30343B", "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans",
                         "font.size": 11, "axes.labelsize": 12,
                         "axes.labelcolor": "#30343B", "xtick.color": "#30343B",
                         "ytick.color": "#30343B"})
    fig, (upper, lower) = plt.subplots(2, 1, figsize=(11.6, 4.6), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1.35]})
    fig.subplots_adjust(left=0.40, right=0.985, top=0.985, bottom=0.19, hspace=0.35)
    draw(upper, effects.reset_index(), [LABELS[arm] for arm in effects.index])
    upper.set_ylabel("Versus v13", labelpad=10)
    upper.tick_params(axis="x", length=0)
    draw(lower, head, [HEAD_TO_HEAD])
    lower.set_ylabel("Matched pair", labelpad=10)
    lower.set_xlabel("RMSE change (cycles)", labelpad=12)
    lower.tick_params(axis="x", labelbottom=True, length=3, width=0.7, pad=6)

    fig.savefig(output / "difference_reference_effects.png", dpi=240, facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output / "difference_reference_effects.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    report = output / "report.md"
    marker = "![Difference-reference effects](difference_reference_effects.png)"
    content = report.read_text(encoding="utf-8")
    if marker not in content:
        content += "\n" + marker + "\n"
    report.write_text(content, encoding="utf-8")
    print(output / "difference_reference_effects.png")


if __name__ == "__main__":
    main()
