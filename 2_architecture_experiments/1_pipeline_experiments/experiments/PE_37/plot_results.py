"""Plot completed PE_37 paired effects without fitting or selecting a model."""
from pathlib import Path
import json
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent


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
    effects = pd.read_csv(output / "paired_additions.csv").set_index("arm")
    order = [f"add_{channel.removeprefix('telemetry_')}" for channel in config["channels"]]
    effects = effects.loc[order]
    # Match the presentation palette; export graphics only, without any text.
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.edgecolor": "#30343B", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.025)
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
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False,
                   labeltop=False, labelright=False, length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.savefig(output / "addition_effects.png", dpi=240, facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output / "addition_effects.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    report = output / "report.md"
    marker = "![Paired channel-addition effects](addition_effects.png)"
    content = report.read_text(encoding="utf-8")
    if marker not in content:
        content += "\n" + marker + "\n"
    report.write_text(content, encoding="utf-8")
    print(output / "addition_effects.png")


if __name__ == "__main__":
    main()
