"""Plot completed PE_35 paired effects without fitting or selecting a model."""
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
    effects = pd.read_csv(output / "paired_ablation.csv").set_index("arm")
    summary = pd.read_csv(output / "summary.csv").set_index("arm")
    order = [f"drop_{channel.removeprefix('telemetry_')}" for channel in config["channels"]] + ["drop_all_eight"]
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
    fig.savefig(output / "ablation_effects.png", dpi=240, facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output / "ablation_effects.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    report = output / "report.md"
    marker = "![Paired channel-removal effects](ablation_effects.png)"
    content = report.read_text(encoding="utf-8")
    if marker not in content:
        content += "\n" + marker + "\n"
    heading = "\n## Observed outcome\n"
    content = content.split(heading)[0]
    individual = effects.drop(index="drop_all_eight")
    all_eight = effects.loc["drop_all_eight"]
    findings = [heading, f"Baseline pooled RMSE: {summary.loc['baseline', 'rmse']:.4f} cycles; pooled R2: {summary.loc['baseline', 'r2']:.5f}.", ""]
    if (individual.rmse_change > 0).all():
        findings += ["Every individual channel removal slightly worsened pooled RMSE.", ""]
    findings += [f"Removing all eight changed pooled RMSE by {all_eight.rmse_change:+.4f} cycles ({all_eight.rmse_change_percent:+.2f}%), winning {int(all_eight.improved_folds)}/5 folds. Its ordinary paired 95% interval is [{all_eight.uav_bootstrap_ci95_low:+.4f}, {all_eight.uav_bootstrap_ci95_high:+.4f}] cycles.", ""]
    if (effects.interpretation == "inconclusive").all():
        findings += ["None of the nine comparisons excludes zero after adjustment for multiple comparisons. Keep the existing feature set provisionally; this screen does not justify an automatic removal or prove that every retained channel is necessary.", ""]
    report.write_text(content + "\n".join(findings), encoding="utf-8")
    print(output / "ablation_effects.png")


if __name__ == "__main__":
    main()
