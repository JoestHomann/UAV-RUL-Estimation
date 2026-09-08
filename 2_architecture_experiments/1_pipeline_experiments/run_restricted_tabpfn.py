"""Screen one low-predicted-RUL TabPFN rule using PE_24's saved nested fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from advanced_r2_utils import load_workflow, method_report
from followup_experiment_utils import (
    aligned_methods, atomic_csv, atomic_json, input_path, register_run,
    select_routed_blend, validate_saved_nested,
)


def run(workflow: dict, root: Path, *, check_only: bool = False) -> dict:
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    if any(not 0.0 <= float(weight) <= 0.5 for weight in workflow["weights"]):
        raise ValueError("PE_25 TabPFN weights must stay in [0, 0.5]")
    names = ["source_predictions", "source_manifest", "source_outer_folds", "source_inner_folds"]
    paths = {name: input_path(workflow[name]) for name in names}
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    if not manifest.get("all_prediction_cells_completed") or not manifest.get("nested_inner_predictions_complete"):
        raise ValueError("PE_25 requires a completed nested PE_24 run")
    if manifest.get("uses_test_labels") is not False or manifest.get("uses_locked_evaluation") is not False:
        raise ValueError("PE_25 accepts development predictions only")
    table = pd.read_csv(paths["source_predictions"])
    outer = pd.read_csv(paths["source_outer_folds"])
    inner = pd.read_csv(paths["source_inner_folds"])
    if sorted(outer.split_seed.unique()) != sorted(workflow["source_split_seeds"]):
        raise ValueError("PE_25 source split seeds changed")
    validate_saved_nested(table, outer, inner, methods={"control", "tabpfn_full"},
                          outer_count=int(workflow["outer_fold_count"]), inner_count=int(workflow["inner_fold_count"]))
    held = aligned_methods(table.loc[table.evaluation_level.eq("outer")], "tabpfn_full")
    training = aligned_methods(table.loc[table.evaluation_level.eq("inner")], "tabpfn_full")
    estimate, provenance = select_routed_blend(
        held, training, weights=workflow["weights"], thresholds=workflow["prediction_thresholds"],
        route_column="control_prediction",
    )
    readiness = {"ready": True, "outer_folds": int(held.outer_fold.nunique()),
                 "outer_rows": len(held), "source_model_fits": len(table.groupby(
                     ["split_seed", "source_outer_fold", "evaluation_level", "inner_fold", "method"])),
                 "requires_model_training": False, "uses_test_labels": False, "uses_locked_evaluation": False}
    if check_only:
        atomic_json(reporting / "input_verification.json", readiness)
        return readiness
    register_run(reporting, workflow, [*paths.values(), Path(__file__),
        Path(__file__).with_name("followup_experiment_utils.py"),
        Path(__file__).with_name("confirmation_utils.py"), Path(__file__).with_name("advanced_r2_utils.py")])
    atomic_csv(reporting / "selection_provenance.csv", provenance)
    primary = "restricted_tabpfn_blend"
    result = method_report(
        held, {"control": held.control_prediction.to_numpy(float), primary: estimate},
        root=root, control="control", promotion_allowed=False,
        promotion_eligible_methods={primary},
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(workflow["minimum_relative_rmse_improvement"]),
        minimum_pooled_r2=float(workflow["minimum_pooled_r2"]),
        require_bootstrap_improvement=True,
    )
    result.update({**readiness, "retained_production_model": "phase3_run_7",
        "hypothesis_source": "PE_24 post-hoc predicted-RUL diagnostics",
        "requires_independent_confirmation": True,
        "evaluation_caveat": "Reuses PE_24 endpoints; new inner-only selection does not erase prior hypothesis selection.",
        "rule": "control + weight * (control <= threshold) * (TabPFN - control)",
        "selection_scope": "exploratory_reused_nested_predictions"})
    atomic_json(reporting / "winner_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_25")
    parser.add_argument("--check", action="store_true", help="Validate source coverage without producing a comparison")
    parser.add_argument("--force", action="store_true", help="Rebuild reports; cannot change a registered recipe")
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, "restricted_tabpfn_workflows", args.workflow)
    print(json.dumps(run(workflow, root, check_only=args.check), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
