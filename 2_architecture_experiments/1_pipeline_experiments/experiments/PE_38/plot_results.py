"""Plot completed PE_38 paired effects without fitting or selecting a model."""
from pathlib import Path
import importlib.util
import json
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pe38_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

LABELS = {"numeric_strong": "strong-tier numeric", "state_only": "state block only",
          "medium_plus_state": "medium numeric + state", "strong_plus_state": "strong numeric + state",
          "drop_07": "channel removed"}


def main():
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"] / "reporting"
    if not (output / "completion.json").exists():
        raise RuntimeError("Finish the matched representation study before plotting.")
    diagnostics = pd.DataFrame([
        {key: record[key] for key in ("arm", "fold", "features", "best_iteration", "fit_seconds")}
        for path in sorted((output.parent / "cells").glob("*.json"))
        for record in [json.loads(path.read_text())]
    ])
    diagnostics["selected_trees"] = diagnostics.best_iteration + 1
    diagnostics.to_csv(output / "fit_diagnostics.csv", index=False)
    effects = pd.read_csv(output / "paired_representations.csv")
    order = [arm for arm in runner.ARMS if arm != "baseline"]
    effects = effects.set_index("arm").loc[order]
    # Match the presentation palette, with axis text but no title or captions.
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.edgecolor": "#30343B", "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans",
                         "font.size": 11, "axes.labelsize": 12,
                         "axes.labelcolor": "#30343B", "xtick.color": "#30343B",
                         "ytick.color": "#30343B"})
    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.subplots_adjust(left=0.26, right=0.985, top=0.985, bottom=0.17)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D9D9D9", alpha=0.45, linewidth=0.7)
    ax.axvline(0, color="#D55E00", linewidth=1.25, zorder=2)
    for index, row in enumerate(effects.itertuples()):
        ax.plot([row.familywise_ci95_low, row.familywise_ci95_high], [index, index],
                color="#56B4E9", linewidth=8, alpha=0.30, solid_capstyle="butt")
        ax.plot([row.uav_bootstrap_ci95_low, row.uav_bootstrap_ci95_high], [index, index],
                color="#0072B2", linewidth=3, solid_capstyle="butt")
        ax.scatter(row.rmse_change, index, color="#0072B2", s=38,
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_ylim(len(effects) - 0.5, -0.5)
    ax.set_yticks(range(len(effects)), [LABELS[arm] for arm in order])
    ax.set_ylabel(f"{config['channel']} representation", labelpad=12)
    ax.set_xlabel("RMSE change versus the 153-feature v13 baseline (cycles)", labelpad=12)
    ax.tick_params(axis="x", labelbottom=True, labeltop=False, length=3, width=0.7, pad=6)
    ax.tick_params(axis="y", labelleft=True, labelright=False, length=0, pad=10)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.savefig(output / "representation_effects.png", dpi=240, facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output / "representation_effects.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    report = output / "report.md"
    marker = "![Paired representation effects](representation_effects.png)"
    content = report.read_text(encoding="utf-8")
    if marker not in content:
        content += "\n" + marker + "\n"
    report.write_text(content, encoding="utf-8")
    print(output / "representation_effects.png")


if __name__ == "__main__":
    main()
