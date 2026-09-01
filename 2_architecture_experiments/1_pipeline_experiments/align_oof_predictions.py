"""Align independently generated OOF predictions on immutable endpoint keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_config import read_experiment_config
from experiment_paths import repository_path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_KEYS = [
    "outer_fold",
    "inner_fold",
    "validation_row",
    "uav_id",
    "scenario",
    "cutoff",
    "observed_rul",
]


class OOFAlignmentError(ValueError):
    """Explain why OOF sources cannot be paired without ambiguity."""


def _source_table(name: str, source: dict[str, Any]) -> pd.DataFrame:
    value = source.get("path")
    if not isinstance(value, str) or not value:
        raise OOFAlignmentError(f"OOF source {name!r} has no path")
    path = repository_path(REPOSITORY_ROOT, value)
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise OOFAlignmentError(f"Cannot read OOF source {name!r}: {error}") from error
    selectors = source.get("selectors", {})
    if not isinstance(selectors, dict):
        raise OOFAlignmentError(f"OOF source {name!r} selectors must be a table")
    for column, expected in selectors.items():
        if column not in table.columns:
            raise OOFAlignmentError(f"OOF source {name!r} has no {column!r} column")
        table = table.loc[table[column].astype(str).eq(str(expected))]
    prediction_column = str(source.get("prediction_column", "predicted_rul"))
    if prediction_column not in table.columns:
        raise OOFAlignmentError(
            f"OOF source {name!r} has no prediction column {prediction_column!r}"
        )
    return table.rename(columns={prediction_column: f"prediction__{name}"})


def align_sources(
    sources: dict[str, Any],
    *,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Return one exact, duplicate-free row per endpoint and OOF source."""

    keys = list(DEFAULT_KEYS if keys is None else keys)
    if len(sources) < 2:
        raise OOFAlignmentError("At least two OOF sources are required")
    aligned: pd.DataFrame | None = None
    expected_rows: int | None = None
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise OOFAlignmentError(f"OOF source {name!r} must be a table")
        table = _source_table(str(name), source)
        missing = sorted(set(keys) - set(table.columns))
        if missing:
            raise OOFAlignmentError(f"OOF source {name!r} is missing keys {missing}")
        if table.duplicated(keys).any():
            duplicate = table.loc[table.duplicated(keys, keep=False), keys].iloc[0]
            raise OOFAlignmentError(
                f"OOF source {name!r} has duplicate endpoint {duplicate.to_dict()}"
            )
        prediction = f"prediction__{name}"
        subset = table[[*keys, prediction]].copy()
        if expected_rows is None:
            expected_rows = len(subset)
            aligned = subset
        else:
            assert aligned is not None
            before = len(aligned)
            aligned = aligned.merge(
                subset,
                on=keys,
                how="inner",
                validate="one_to_one",
            )
            if len(aligned) != before or len(subset) != expected_rows:
                raise OOFAlignmentError(
                    f"OOF source {name!r} does not contain exactly the same endpoints"
                )
    assert aligned is not None
    prediction_columns = [column for column in aligned if column.startswith("prediction__")]
    if aligned[prediction_columns].isna().any().any():
        raise OOFAlignmentError("Aligned OOF predictions contain missing values")
    return aligned.sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_7")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config.get("stacking_workflows", {}).get(args.workflow)
    if not isinstance(workflow, dict):
        raise OOFAlignmentError(f"Unknown stacking workflow {args.workflow!r}")
    sources = workflow.get("oof_sources")
    if not isinstance(sources, dict):
        raise OOFAlignmentError("stacking workflow has no oof_sources table")
    keys = workflow.get("alignment_keys", DEFAULT_KEYS)
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise OOFAlignmentError("alignment_keys must be a list of column names")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned = align_sources(sources, keys=keys)
    output = output_dir / "aligned_oof_predictions.csv.gz"
    aligned.to_csv(output, index=False, compression="gzip")
    manifest = {
        "status": "complete",
        "rows": len(aligned),
        "keys": keys,
        "sources": list(sources),
        "artifact": output.relative_to(REPOSITORY_ROOT).as_posix(),
    }
    (output_dir / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
