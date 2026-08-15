"""Load traceable tabular feature views for Phase 2 model experiments.

The adapter reads only the copies created inside this step. It never falls back
to Phase 1 files, changes row order, scales values, imputes values, or transforms
the target. Its role is limited to selecting an ordered feature set, separating
features from metadata and labels, and applying the fixed UAV fold assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype


STEP_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = STEP_DIR / "artifacts"
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_DIR / "tabular_dataset_manifest.json"


class TabularAdapterError(ValueError):
    """Represent a readable adapter configuration or usage failure."""


@dataclass(frozen=True)
class TabularDataset:
    """Keep model inputs, labels, weights, and identifying metadata separate.

    DataFrames remain mutable pandas objects even though this dataclass is
    frozen. The frozen container prevents accidentally replacing one component
    with another while leaving normal pandas operations available to callers.
    """

    features: pd.DataFrame
    metadata: pd.DataFrame
    target: pd.Series | None
    sample_weights: pd.Series | None

    def __len__(self) -> int:
        """Return the number of observations in this dataset view."""

        return len(self.features)


@dataclass(frozen=True)
class TabularSplit:
    """Pair one training view with its corresponding validation view."""

    training: TabularDataset
    validation: TabularDataset


class TabularDataAdapter:
    """Provide column-selective access to copied tabular Phase 1 artifacts.

    The manifest defines file locations and column roles. The copied feature
    catalog remains the single source for exact feature names and their order.
    No hidden DataFrame cache is used; callers decide how long returned views
    should remain in memory.
    """

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> None:
        self.manifest_path = manifest_path.resolve()
        self.artifact_dir = self.manifest_path.parent
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """Read the Step 2 manifest and require its core mappings."""

        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise TabularAdapterError(
                "Tabular adapter artifacts are missing. Run "
                "build_tabular_data_adapter.py first."
            ) from error
        except json.JSONDecodeError as error:
            raise TabularAdapterError(
                f"Tabular dataset manifest is invalid JSON: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise TabularAdapterError("Tabular dataset manifest must be an object")
        if not isinstance(payload.get("files"), dict):
            raise TabularAdapterError("Manifest is missing its files object")
        if not isinstance(payload.get("feature_sets"), dict):
            raise TabularAdapterError("Manifest is missing its feature_sets object")
        return payload

    def _file_entry(self, name: str) -> dict[str, Any]:
        """Return one named manifest entry and explain unknown names clearly."""

        entry = self.manifest["files"].get(name)
        if not isinstance(entry, dict):
            raise TabularAdapterError(f"Manifest does not define file {name!r}")
        return entry

    def _copied_path(self, name: str) -> Path:
        """Resolve one copied file while keeping access inside artifacts."""

        entry = self._file_entry(name)
        relative_path = entry.get("copied_path")
        if not isinstance(relative_path, str):
            raise TabularAdapterError(
                f"Manifest file {name!r} has no valid copied_path"
            )
        supplied = Path(relative_path)
        if supplied.is_absolute():
            raise TabularAdapterError(
                f"Manifest copied path must be relative: {relative_path}"
            )

        resolved = (self.artifact_dir / supplied).resolve()
        try:
            resolved.relative_to(self.artifact_dir.resolve())
        except ValueError as error:
            raise TabularAdapterError(
                f"Manifest copied path escapes artifacts: {relative_path}"
            ) from error
        if not resolved.is_file():
            raise TabularAdapterError(
                f"Copied artifact is missing: {resolved}. Run the Step 2 builder."
            )
        return resolved

    def feature_names(self, feature_set: str) -> list[str]:
        """Return exact feature names in the copied catalog's original order."""

        if feature_set not in self.manifest["feature_sets"]:
            available = ", ".join(sorted(self.manifest["feature_sets"]))
            raise TabularAdapterError(
                f"Unknown feature set {feature_set!r}. Available: {available}"
            )

        catalog_entry = self._file_entry("feature_catalog")
        feature_column = catalog_entry.get("feature_name_column")
        if not isinstance(feature_column, str):
            raise TabularAdapterError(
                "Feature catalog entry has no valid feature_name_column"
            )

        try:
            catalog = pd.read_csv(
                self._copied_path("feature_catalog"),
                usecols=[feature_column, feature_set],
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise TabularAdapterError(
                f"Cannot read feature set {feature_set!r} from the catalog: {error}"
            ) from error

        # A Boolean membership column avoids treating arbitrary non-empty text
        # as true. The feature catalog is copied unchanged from Phase 1.
        if not is_bool_dtype(catalog[feature_set].dtype):
            raise TabularAdapterError(
                f"Feature-set membership column {feature_set!r} must be Boolean"
            )
        names = catalog.loc[catalog[feature_set], feature_column].tolist()
        if not names:
            raise TabularAdapterError(f"Feature set {feature_set!r} is empty")
        if len(names) != len(set(names)):
            raise TabularAdapterError(
                f"Feature set {feature_set!r} contains duplicate feature names"
            )
        return names

    def _load_feature_dataset(
        self,
        dataset_name: str,
        feature_set: str,
    ) -> TabularDataset:
        """Read one dataset using only its interface and selected features."""

        entry = self._file_entry(dataset_name)
        if entry.get("category") != "feature_dataset":
            raise TabularAdapterError(f"File {dataset_name!r} is not a feature dataset")

        metadata_columns = entry.get("metadata_columns")
        target_column = entry.get("target_column")
        weight_column = entry.get("sample_weight_column")
        if not isinstance(metadata_columns, list) or not all(
            isinstance(column, str) for column in metadata_columns
        ):
            raise TabularAdapterError(
                f"Dataset {dataset_name!r} has invalid metadata columns"
            )

        feature_columns = self.feature_names(feature_set)
        selected_columns = list(metadata_columns)
        if isinstance(target_column, str):
            selected_columns.append(target_column)
        if isinstance(weight_column, str):
            selected_columns.append(weight_column)
        selected_columns.extend(feature_columns)

        try:
            table = pd.read_csv(
                self._copied_path(dataset_name),
                usecols=selected_columns,
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise TabularAdapterError(
                f"Cannot load dataset {dataset_name!r}: {error}"
            ) from error
        if table.empty:
            raise TabularAdapterError(f"Dataset {dataset_name!r} is empty")

        # Explicit column selection restores catalog feature order regardless of
        # how pandas internally handles the usecols collection.
        features = table.loc[:, feature_columns].copy()
        metadata = table.loc[:, metadata_columns].copy()
        target = table[target_column].copy() if isinstance(target_column, str) else None
        sample_weights = (
            table[weight_column].copy() if isinstance(weight_column, str) else None
        )
        return TabularDataset(
            features=features,
            metadata=metadata,
            target=target,
            sample_weights=sample_weights,
        )

    def load_training(self, feature_set: str) -> TabularDataset:
        """Load all training-prefix rows for one feature set."""

        return self._load_feature_dataset("training", feature_set)

    def load_development(self, feature_set: str) -> TabularDataset:
        """Load development-scenario rows used only for inner selection."""

        return self._load_feature_dataset("development", feature_set)

    def load_locked(self, feature_set: str) -> TabularDataset:
        """Explicitly load locked-scenario rows for outer evaluation."""

        return self._load_feature_dataset("locked", feature_set)

    def load_test(self, feature_set: str) -> TabularDataset:
        """Load unlabelled test endpoints for later final prediction."""

        return self._load_feature_dataset("test", feature_set)

    def _fold_uav_ids(
        self,
        *,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> tuple[set[str], set[str]]:
        """Return training and validation UAV IDs for one requested split."""

        if inner_fold is None:
            try:
                folds = pd.read_csv(
                    self._copied_path("outer_folds"),
                    usecols=["uav_id", "outer_fold"],
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                raise TabularAdapterError(
                    f"Cannot read outer folds: {error}"
                ) from error

            available = set(folds["outer_fold"].astype(int))
            if outer_fold not in available:
                raise TabularAdapterError(
                    f"Unknown outer fold {outer_fold}. Available: {sorted(available)}"
                )
            validation_mask = folds["outer_fold"] == outer_fold
        else:
            try:
                folds = pd.read_csv(
                    self._copied_path("inner_folds"),
                    usecols=["outer_fold", "uav_id", "inner_fold"],
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                raise TabularAdapterError(
                    f"Cannot read inner folds: {error}"
                ) from error

            available_outer = set(folds["outer_fold"].astype(int))
            if outer_fold not in available_outer:
                raise TabularAdapterError(
                    f"Unknown outer fold {outer_fold}. "
                    f"Available: {sorted(available_outer)}"
                )
            folds = folds.loc[folds["outer_fold"] == outer_fold]
            available_inner = set(folds["inner_fold"].astype(int))
            if inner_fold not in available_inner:
                raise TabularAdapterError(
                    f"Unknown inner fold {inner_fold} for outer fold {outer_fold}. "
                    f"Available: {sorted(available_inner)}"
                )
            validation_mask = folds["inner_fold"] == inner_fold

        training_ids = set(folds.loc[~validation_mask, "uav_id"].astype(str))
        validation_ids = set(folds.loc[validation_mask, "uav_id"].astype(str))
        if not training_ids or not validation_ids:
            raise TabularAdapterError("Requested fold produced an empty UAV partition")
        return training_ids, validation_ids

    @staticmethod
    def _select_uavs(
        dataset: TabularDataset,
        uav_ids: set[str],
        *,
        purpose: str,
    ) -> TabularDataset:
        """Filter a dataset by UAV ID while preserving source row order."""

        if "uav_id" not in dataset.metadata.columns:
            raise TabularAdapterError("Dataset metadata does not contain uav_id")
        mask = dataset.metadata["uav_id"].astype(str).isin(uav_ids)
        if not mask.any():
            raise TabularAdapterError(f"No rows found for {purpose}")

        # Resetting labels after filtering does not reorder observations. It
        # keeps features, metadata, targets, and weights aligned from row zero.
        target = (
            dataset.target.loc[mask].reset_index(drop=True)
            if dataset.target is not None
            else None
        )
        sample_weights = (
            dataset.sample_weights.loc[mask].reset_index(drop=True)
            if dataset.sample_weights is not None
            else None
        )
        return TabularDataset(
            features=dataset.features.loc[mask].reset_index(drop=True),
            metadata=dataset.metadata.loc[mask].reset_index(drop=True),
            target=target,
            sample_weights=sample_weights,
        )

    def get_inner_selection_split(
        self,
        outer_fold: int,
        inner_fold: int,
        feature_set: str,
    ) -> TabularSplit:
        """Build an inner training/development split from fixed UAV groups.

        Training rows come from the training-prefix table. Validation rows come
        from the five development scenarios for the held-out inner-fold UAVs.
        Locked scenarios are never loaded by this method.
        """

        training_ids, validation_ids = self._fold_uav_ids(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
        )
        training = self._select_uavs(
            self.load_training(feature_set),
            training_ids,
            purpose="inner training",
        )
        validation = self._select_uavs(
            self.load_development(feature_set),
            validation_ids,
            purpose="inner development validation",
        )
        return TabularSplit(training=training, validation=validation)

    def get_locked_outer_evaluation_split(
        self,
        outer_fold: int,
        feature_set: str,
    ) -> TabularSplit:
        """Build outer training and explicitly locked evaluation views.

        Training contains prefixes from the 80 outer-training UAVs. Validation
        contains the twenty locked scenarios for the 20 held-out outer-fold
        UAVs. This method is intentionally named to make locked-data access
        visible at every call site.
        """

        training_ids, validation_ids = self._fold_uav_ids(outer_fold=outer_fold)
        training = self._select_uavs(
            self.load_training(feature_set),
            training_ids,
            purpose="outer training",
        )
        validation = self._select_uavs(
            self.load_locked(feature_set),
            validation_ids,
            purpose="locked outer validation",
        )
        return TabularSplit(training=training, validation=validation)
