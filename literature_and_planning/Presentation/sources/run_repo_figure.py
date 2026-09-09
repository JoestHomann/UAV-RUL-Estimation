"""Run a repository Phase 0 plotting script unchanged, at projection size.

The Phase 0 figures in the repository are drawn on canvases sized for reading
on a monitor (11 x 9 in to 19 x 10 in). Scaled down onto the 10 x 5.625 in
template used for this talk, their axis labels land at three to four points and
stop being readable from a lecture room.

This runner imports the repository script *as it is* -- no repository file is
copied with edits, and every number, colour, ordering and label still comes
from the repository code -- and overrides exactly three presentation
quantities before calling its ``main()``:

  1. the figure canvas size, so the figure is placed on the slide at 1:1
     and a point in the figure is a point on the slide;
  2. the tick-label size used by the shared ``style_axis`` helper, which
     hard-codes ``labelsize=8``;
  3. the matplotlib base font size, which the scripts inherit for titles,
     axis labels, legends and annotations;
  4. the figure-title size, which a few scripts hard-code at 14 pt. On a
     narrow canvas a 14 pt title is wider than the plot and ``bbox_inches
     ="tight"`` then pads the whole figure out to the title's width.

Nothing else is patched. Run, for example:

    python run_repo_figure.py \
        --script <repo>/0_data_analysis/broad_data_review/plot_constant_features.py \
        --figure-size 9.0 4.10 --label-size 6.4 --font-size 8 \
        -- --train-csv <repo>/data/train.csv --output-dir figures_repo/constants
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_module(path: Path):
    directory = str(path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def patch_style_axis(module, label_size: float) -> None:
    """Replace the shared style_axis with the same helper at a new labelsize."""
    for helper_name in ("plotting_common", "core_common"):
        helper = sys.modules.get(helper_name)
        if helper is None:
            continue

        def styled(axis, _size=label_size):
            axis.grid(True, alpha=0.2)
            axis.tick_params(labelsize=_size)

        helper.style_axis = styled
        if hasattr(module, "style_axis"):
            module.style_axis = styled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--figure-size", type=float, nargs=2, required=True,
                        metavar=("WIDTH_IN", "HEIGHT_IN"))
    parser.add_argument("--label-size", type=float, default=6.5)
    parser.add_argument("--font-size", type=float, default=8.0)
    parser.add_argument("--suptitle-size", type=float, default=9.5)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    width, height = args.figure_size
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.font_size + 1,
        "axes.labelsize": args.font_size,
        "legend.fontsize": args.font_size - 1,
        "figure.titlesize": args.font_size + 2,
        "savefig.bbox": "tight",
    })

    from matplotlib.figure import Figure

    original_suptitle = Figure.suptitle

    def suptitle(self, text, *call_args, **kwargs):
        kwargs["fontsize"] = args.suptitle_size
        return original_suptitle(self, text, *call_args, **kwargs)

    Figure.suptitle = suptitle

    original_subplots = plt.subplots

    def subplots(*call_args, **kwargs):
        kwargs["figsize"] = (width, height)
        return original_subplots(*call_args, **kwargs)

    plt.subplots = subplots

    module = load_module(args.script)
    patch_style_axis(module, args.label_size)

    passthrough = args.script_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    sys.argv = [str(args.script), *passthrough]
    module.main()


if __name__ == "__main__":
    main()
