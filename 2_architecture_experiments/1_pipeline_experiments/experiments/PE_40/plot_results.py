"""Plot completed PE_40 paired effects without fitting or selecting a model."""
from pathlib import Path
import importlib.util
import json
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pe40_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

LABELS = {"similarity_uav_equal": "similarity + UAV-equal",
          "uav_equal_only": "UAV-equal only",
          "unweighted": "no weighting"}


def main():
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"] / "reporting"
    if not (output / "completion.json").exists():
        raise RuntimeError("Finish the matched weighting study before plotting.")
    diagnostics = pd.DataFrame([
        {key: record[key] for key in ("arm", "fold", "best_iteration", "fit_seconds", "uav_total_ratio")}
        for path in sorted((output.parent / "cells").glob("*.json"))
        for record in [json.loads(path.read_text())]
    ])
    diagnostics["selected_trees"] = diagnostics.best_iteration + 1
    diagnostics.to_csv(output / "fit_diagnostics.csv", index=False)
    effects = pd.read_csv(output / "paired_weighting.csv").set_index("arm")
    order = [arm for arm in runner.ARMS if arm != "baseline"]
    effects = effects.loc[order]
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.edgecolor": "#30343B", "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans",
                         "font.size": 11, "axes.labelsize": 12,
                         "axes.labelcolor": "#30343B", "xtick.color": "#30343B",
                         "ytick.color": "#30343B"})
    fig, ax = plt.subplots(figsize=(10, 3.9))
    fig.subplots_adjust(left=0.27, right=0.985, top=0.985, bottom=0.21)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D9D9D9", alpha=0.45, lw=0.7)
    ax.axvline(0, color="#D55E00", lw=1.25, zorder=2)
    for index, row in enumerate(effects.itertuples()):
        ax.plot([row.familywise_ci95_low, row.familywise_ci95_high], [index, index],
                color="#56B4E9", lw=8, alpha=0.30, solid_capstyle="butt")
        ax.plot([row.uav_bootstrap_ci95_low, row.uav_bootstrap_ci95_high], [index, index],
                color="#0072B2", lw=3, solid_capstyle="butt")
        ax.scatter(row.rmse_change, index, color="#0072B2", s=38, edgecolor="white", lw=0.7, zorder=3)
    ax.set_ylim(len(effects) - 0.5, -0.5)
    ax.set_yticks(range(len(effects)), [LABELS[arm] for arm in order])
    ax.set_ylabel("Training-row weighting", labelpad=12)
    ax.set_xlabel("RMSE change versus v13 as shipped (cycles)", labelpad=12)
    ax.tick_params(axis="x", labelbottom=True, length=3, width=0.7, pad=6)
    ax.tick_params(axis="y", labelleft=True, length=0, pad=10)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    for ext in ("png", "svg"):
        fig.savefig(output / f"weighting_effects.{ext}", dpi=240, facecolor="white",
                    bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    report = output / "report.md"
    marker = "![Paired weighting effects](weighting_effects.png)"
    content = report.read_text(encoding="utf-8")
    if marker not in content:
        content += "\n" + marker + "\n"
    report.write_text(content, encoding="utf-8")
    print(output / "weighting_effects.png")


if __name__ == "__main__":
    main()
