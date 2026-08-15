"""Construct causal, fold-scaled telemetry windows for sequence models.

This adapter reads only the copied Step 3 inputs. Each prediction endpoint is
converted into a trailing telemetry window containing cycles at or before its
cutoff. Short histories are padded on the left, and a Boolean mask marks padded
positions with True.

The adapter also fits robust channel scaling from the active training UAVs and
applies the resulting parameters to training and validation windows. It never
uses validation UAVs to estimate preprocessing values. Age side features remain
unscaled and are returned separately from telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Collection

import numpy as np
from numpy.typing import NDArray
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = STEP_DIR / "artifacts"
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_DIR / "sequence_dataset_manifest.json"


class SequenceAdapterError(ValueError):
    """Represent a readable sequence configuration or usage failure."""


@dataclass(frozen=True)
class SequenceDataset:
    """Hold one model-ready sequence view and its non-sequence information.

    Shapes are:

    - sequences: observations by lookback positions by telemetry channels;
    - padding_mask: observations by lookback positions;
    - side_features: observations by age-side-feature columns.

    A True padding-mask value identifies a synthetic left-padding position.
    Real observations are False. Padded sequence values always remain zero,
    including after robust scaling.
    """

    sequences: NDArray[np.float32]
    padding_mask: NDArray[np.bool_]
    side_features: NDArray[np.float32]
    metadata: pd.DataFrame
    target: pd.Series | None
    sample_weights: pd.Series | None
    channel_names: tuple[str, ...]
    side_feature_names: tuple[str, ...]
    lookback: int
    scaled: bool

    def __post_init__(self) -> None:
        """Reject misaligned arrays before they reach a model adapter."""

        row_count = self.sequences.shape[0]
        expected_sequence_shape = (
            row_count,
            self.lookback,
            len(self.channel_names),
        )
        if self.sequences.shape != expected_sequence_shape:
            raise SequenceAdapterError(
                "Sequence array shape does not match rows, lookback, and channels"
            )
        if self.padding_mask.shape != (row_count, self.lookback):
            raise SequenceAdapterError("Padding-mask shape does not match sequences")
        if self.side_features.shape != (
            row_count,
            len(self.side_feature_names),
        ):
            raise SequenceAdapterError("Side-feature shape does not match rows")
        if len(self.metadata) != row_count:
            raise SequenceAdapterError("Metadata row count does not match sequences")
        if self.target is not None and len(self.target) != row_count:
            raise SequenceAdapterError("Target row count does not match sequences")
        if self.sample_weights is not None and len(self.sample_weights) != row_count:
            raise SequenceAdapterError("Weight row count does not match sequences")

    def __len__(self) -> int:
        """Return the number of prediction endpoints represented."""

        return self.sequences.shape[0]

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Return True for real observations and False for left padding."""

        return ~self.padding_mask


@dataclass(frozen=True)
class RobustChannelScaler:
    """Store fold-fitted robust scaling parameters for telemetry channels."""

    channel_names: tuple[str, ...]
    centers: NDArray[np.float64]
    scales: NDArray[np.float64]
    scale_methods: tuple[str, ...]
    iqrs: NDArray[np.float64]
    standard_deviations: NDArray[np.float64]
    data_ranges: NDArray[np.float64]
    variation_tolerances: NDArray[np.float64]
    fit_uavs: int
    fit_rows: int

    def transform(self, dataset: SequenceDataset) -> SequenceDataset:
        """Scale real readings and keep every padded position equal to zero."""

        if dataset.channel_names != self.channel_names:
            raise SequenceAdapterError(
                "Scaler channels do not match the sequence dataset channels"
            )

        # Boolean indexing flattens the first two dimensions and keeps channels
        # as columns. Only genuine observations participate in the transform.
        transformed = dataset.sequences.astype(np.float64, copy=True)
        valid = dataset.valid_mask
        transformed[valid] = (
            transformed[valid] - self.centers
        ) / self.scales
        transformed[dataset.padding_mask] = 0.0
        if not np.isfinite(transformed).all():
            raise SequenceAdapterError("Scaled sequence values are not finite")

        return replace(
            dataset,
            sequences=transformed.astype(np.float32),
            scaled=True,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return readable per-channel parameters for later model artifacts."""

        return pd.DataFrame(
            {
                "channel": self.channel_names,
                "center": self.centers,
                "scale": self.scales,
                "scale_method": self.scale_methods,
                "iqr": self.iqrs,
                "standard_deviation": self.standard_deviations,
                "data_range": self.data_ranges,
                "variation_tolerance": self.variation_tolerances,
                "fit_uavs": self.fit_uavs,
                "fit_rows": self.fit_rows,
            }
        )


@dataclass(frozen=True)
class SequenceSplit:
    """Pair scaled training and validation views with their fitted scaler."""

    training: SequenceDataset
    validation: SequenceDataset
    channel_scaler: RobustChannelScaler


def fit_robust_channel_scaler(
    values: NDArray[np.float64],
    channel_names: tuple[str, ...],
    *,
    fit_uavs: int,
    relative_variation_tolerance: float = 1e-12,
) -> RobustChannelScaler:
    """Fit median and IQR scales to telemetry from training UAVs only.

    The primary scale is IQR divided by 1.349, matching Phase 1 preprocessing.
    Channels with a meaningful range but negligible IQR use standard deviation.
    Truly constant channels would use scale 1.0, although the contract already
    excludes the six constant telemetry channels from this representation.
    """

    if values.ndim != 2 or values.shape[1] != len(channel_names):
        raise SequenceAdapterError(
            "Scaler input must be rows by the configured telemetry channels"
        )
    if values.shape[0] < 2:
        raise SequenceAdapterError("At least two telemetry rows are needed to scale")
    if not np.isfinite(values).all():
        raise SequenceAdapterError("Scaler training values are not finite")

    centers = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqrs = q75 - q25
    iqr_scales = iqrs / 1.349
    standard_deviations = np.std(values, axis=0, ddof=1)
    data_ranges = np.ptp(values, axis=0)
    magnitudes = np.max(np.abs(values), axis=0)
    tolerances = relative_variation_tolerance * np.maximum(magnitudes, 1.0)

    has_meaningful_range = data_ranges > tolerances
    use_iqr = has_meaningful_range & (iqr_scales > tolerances)
    use_standard_deviation = has_meaningful_range & ~use_iqr
    scales = np.where(
        use_iqr,
        iqr_scales,
        np.where(use_standard_deviation, standard_deviations, 1.0),
    )
    methods = tuple(
        np.where(
            use_iqr,
            "iqr",
            np.where(
                use_standard_deviation,
                "standard_deviation_fallback",
                "unit_fallback",
            ),
        ).tolist()
    )
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise SequenceAdapterError("Fitted channel scales must be finite and positive")

    return RobustChannelScaler(
        channel_names=channel_names,
        centers=centers,
        scales=scales,
        scale_methods=methods,
        iqrs=iqrs,
        standard_deviations=standard_deviations,
        data_ranges=data_ranges,
        variation_tolerances=tolerances,
        fit_uavs=fit_uavs,
        fit_rows=values.shape[0],
    )


class SequenceDataAdapter:
    """Load copied histories and construct causal windows for fixed UAV folds.

    No hidden cache is used. Each public loading call reads only the configured
    telemetry columns and the requested endpoint table. This keeps data access
    explicit and lets the later experiment runner decide when arrays can be
    released or reused.
    """

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> None:
        self.manifest_path = manifest_path.resolve()
        self.artifact_dir = self.manifest_path.parent
        self.manifest = self._load_manifest()
        self.channel_names = self._configured_text_tuple("channels")
        self.side_feature_names = self._configured_text_tuple("side_features")
        self.lookbacks = self._configured_lookbacks()

    def _load_manifest(self) -> dict[str, Any]:
        """Read the generated sequence manifest and require its core mappings."""

        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise SequenceAdapterError(
                "Sequence adapter artifacts are missing. Run "
                "build_sequence_data_adapter.py first."
            ) from error
        except json.JSONDecodeError as error:
            raise SequenceAdapterError(
                f"Sequence dataset manifest is invalid JSON: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise SequenceAdapterError("Sequence dataset manifest must be an object")
        if not isinstance(payload.get("files"), dict):
            raise SequenceAdapterError("Manifest is missing its files object")
        return payload

    def _configured_text_tuple(self, key: str) -> tuple[str, ...]:
        """Read one non-empty list of unique names from the manifest."""

        values = self.manifest.get(key)
        if not isinstance(values, list) or not values:
            raise SequenceAdapterError(f"Manifest field {key!r} must be a list")
        if not all(isinstance(value, str) for value in values):
            raise SequenceAdapterError(f"Manifest field {key!r} must contain text")
        if len(values) != len(set(values)):
            raise SequenceAdapterError(f"Manifest field {key!r} has duplicates")
        return tuple(values)

    def _configured_lookbacks(self) -> tuple[int, ...]:
        """Read the positive, unique sequence-window alternatives."""

        values = self.manifest.get("lookbacks")
        if not isinstance(values, list) or not values:
            raise SequenceAdapterError("Manifest lookbacks must be a non-empty list")
        if not all(isinstance(value, int) and value > 0 for value in values):
            raise SequenceAdapterError("Every sequence lookback must be positive")
        if len(values) != len(set(values)):
            raise SequenceAdapterError("Sequence lookbacks contain duplicates")
        return tuple(values)

    def _file_entry(self, name: str) -> dict[str, Any]:
        """Return one named manifest file entry."""

        entry = self.manifest["files"].get(name)
        if not isinstance(entry, dict):
            raise SequenceAdapterError(f"Manifest does not define file {name!r}")
        return entry

    def _copied_path(self, name: str) -> Path:
        """Resolve a copied input without permitting artifact-directory escape."""

        entry = self._file_entry(name)
        relative_path = entry.get("copied_path")
        if not isinstance(relative_path, str):
            raise SequenceAdapterError(
                f"Manifest file {name!r} has no valid copied_path"
            )
        supplied = Path(relative_path)
        if supplied.is_absolute():
            raise SequenceAdapterError(
                f"Manifest copied path must be relative: {relative_path}"
            )

        resolved = (self.artifact_dir / supplied).resolve()
        try:
            resolved.relative_to(self.artifact_dir.resolve())
        except ValueError as error:
            raise SequenceAdapterError(
                f"Manifest copied path escapes artifacts: {relative_path}"
            ) from error
        if not resolved.is_file():
            raise SequenceAdapterError(
                f"Copied artifact is missing: {resolved}. Run the Step 3 builder."
            )
        return resolved

    def _validate_lookback(self, lookback: int) -> None:
        """Reject sequence lengths outside the fixed experiment alternatives."""

        if lookback not in self.lookbacks:
            raise SequenceAdapterError(
                f"Unknown lookback {lookback}. Available: {list(self.lookbacks)}"
            )

    def _load_endpoints(
        self,
        endpoint_name: str,
        uav_ids: Collection[str] | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Read one endpoint table and optionally retain selected UAVs only."""

        entry = self._file_entry(endpoint_name)
        if entry.get("category") != "prediction_endpoints":
            raise SequenceAdapterError(
                f"Manifest file {endpoint_name!r} is not an endpoint table"
            )
        try:
            endpoints = pd.read_csv(self._copied_path(endpoint_name))
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise SequenceAdapterError(
                f"Cannot read endpoint table {endpoint_name!r}: {error}"
            ) from error

        if uav_ids is not None:
            selected_ids = {str(uav_id) for uav_id in uav_ids}
            endpoints = endpoints.loc[
                endpoints["uav_id"].astype(str).isin(selected_ids)
            ]
        if endpoints.empty:
            raise SequenceAdapterError(
                f"Endpoint selection {endpoint_name!r} produced no rows"
            )

        # Filtering preserves the original file order. Resetting only supplies
        # aligned zero-based labels for the generated NumPy arrays.
        return endpoints.reset_index(drop=True), entry

    def _load_histories(
        self,
        history_name: str,
        uav_ids: Collection[str],
    ) -> dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]]:
        """Load ordered cycle and channel arrays for the requested UAVs."""

        selected_ids = {str(uav_id) for uav_id in uav_ids}
        if not selected_ids:
            raise SequenceAdapterError("At least one history UAV is required")

        columns = ["uav_id", "flight_cycle", *self.channel_names]
        try:
            table = pd.read_csv(
                self._copied_path(history_name),
                usecols=columns,
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise SequenceAdapterError(
                f"Cannot read telemetry history {history_name!r}: {error}"
            ) from error
        table = table.loc[table["uav_id"].astype(str).isin(selected_ids)]
        if table.empty:
            raise SequenceAdapterError("No telemetry histories match the endpoints")

        histories: dict[
            str,
            tuple[NDArray[np.int64], NDArray[np.float64]],
        ] = {}
        for uav_id, history in table.groupby("uav_id", sort=False):
            cycles = history["flight_cycle"].to_numpy(dtype=np.int64)
            values = history[list(self.channel_names)].to_numpy(dtype=np.float64)
            if len(cycles) == 0 or np.any(np.diff(cycles) <= 0):
                raise SequenceAdapterError(
                    f"History {uav_id!r} is empty, unordered, or duplicated"
                )
            if not np.isfinite(values).all():
                raise SequenceAdapterError(
                    f"History {uav_id!r} contains non-finite telemetry"
                )
            histories[str(uav_id)] = (cycles, values)

        missing = sorted(selected_ids - set(histories))
        if missing:
            raise SequenceAdapterError(
                f"Telemetry histories are missing requested UAVs: {missing}"
            )
        return histories

    def _side_feature_values(self, cutoffs: NDArray[np.int64]) -> NDArray[np.float32]:
        """Create the configured age features from each causal cutoff."""

        columns: list[NDArray[np.float64]] = []
        for name in self.side_feature_names:
            if name == "flight_cycle":
                columns.append(cutoffs.astype(np.float64))
            elif name == "log1p_flight_cycle":
                columns.append(np.log1p(cutoffs.astype(np.float64)))
            else:
                raise SequenceAdapterError(f"Unsupported side feature {name!r}")
        return np.column_stack(columns).astype(np.float32)

    def _build_dataset(
        self,
        endpoint_name: str,
        lookback: int,
        uav_ids: Collection[str] | None = None,
    ) -> SequenceDataset:
        """Construct raw causal windows for one endpoint table.

        The search position must land exactly on the declared cutoff. Only rows
        ending at that position are copied, so no cycle after the endpoint can
        enter the sequence. Histories shorter than the lookback are right-aligned
        and all earlier positions remain zero padding.
        """

        self._validate_lookback(lookback)
        endpoints, entry = self._load_endpoints(endpoint_name, uav_ids)
        history_name = entry.get("history_file")
        if not isinstance(history_name, str):
            raise SequenceAdapterError(
                f"Endpoint table {endpoint_name!r} has no history file"
            )
        endpoint_uavs = endpoints["uav_id"].astype(str).unique().tolist()
        histories = self._load_histories(history_name, endpoint_uavs)

        row_count = len(endpoints)
        channel_count = len(self.channel_names)
        sequences = np.zeros(
            (row_count, lookback, channel_count),
            dtype=np.float32,
        )
        padding_mask = np.ones((row_count, lookback), dtype=np.bool_)

        for row_index, endpoint in endpoints.iterrows():
            uav_id = str(endpoint["uav_id"])
            cutoff = int(endpoint["cutoff"])
            cycles, values = histories[uav_id]
            stop = int(np.searchsorted(cycles, cutoff, side="right"))
            if stop == 0 or cycles[stop - 1] != cutoff:
                raise SequenceAdapterError(
                    f"History {uav_id!r} has no reading at cutoff {cutoff}"
                )

            start = max(0, stop - lookback)
            window = values[start:stop]
            valid_length = len(window)
            destination_start = lookback - valid_length
            sequences[row_index, destination_start:, :] = window.astype(np.float32)
            padding_mask[row_index, destination_start:] = False

        metadata_columns = entry.get("metadata_columns")
        if not isinstance(metadata_columns, list):
            raise SequenceAdapterError(
                f"Endpoint table {endpoint_name!r} has invalid metadata columns"
            )
        missing_metadata = [
            column for column in metadata_columns if column not in endpoints.columns
        ]
        if missing_metadata:
            raise SequenceAdapterError(
                f"Endpoint table {endpoint_name!r} is missing {missing_metadata}"
            )

        target_name = entry.get("target_column")
        weight_name = entry.get("sample_weight_column")
        target = (
            endpoints[target_name].copy()
            if isinstance(target_name, str) and target_name in endpoints.columns
            else None
        )
        sample_weights = (
            endpoints[weight_name].copy()
            if isinstance(weight_name, str) and weight_name in endpoints.columns
            else None
        )
        if isinstance(target_name, str) and target is None:
            raise SequenceAdapterError(
                f"Endpoint table {endpoint_name!r} is missing target {target_name!r}"
            )
        if isinstance(weight_name, str) and sample_weights is None:
            raise SequenceAdapterError(
                f"Endpoint table {endpoint_name!r} is missing weights {weight_name!r}"
            )

        cutoffs = endpoints["cutoff"].to_numpy(dtype=np.int64)
        return SequenceDataset(
            sequences=sequences,
            padding_mask=padding_mask,
            side_features=self._side_feature_values(cutoffs),
            metadata=endpoints.loc[:, metadata_columns].copy(),
            target=target,
            sample_weights=sample_weights,
            channel_names=self.channel_names,
            side_feature_names=self.side_feature_names,
            lookback=lookback,
            scaled=False,
        )

    def load_training(self, lookback: int) -> SequenceDataset:
        """Build raw causal windows for all training prefixes."""

        return self._build_dataset("training_endpoints", lookback)

    def load_development(self, lookback: int) -> SequenceDataset:
        """Build raw windows for all development-scenario endpoints."""

        return self._build_dataset("development_endpoints", lookback)

    def load_locked(self, lookback: int) -> SequenceDataset:
        """Explicitly build raw windows for all locked endpoints."""

        return self._build_dataset("locked_endpoints", lookback)

    def load_test(self, lookback: int) -> SequenceDataset:
        """Build raw windows for unlabelled test endpoints."""

        return self._build_dataset("test_endpoints", lookback)

    def outer_fold_labels(self) -> tuple[int, ...]:
        """Return the actual outer-fold labels stored in the copied table."""

        try:
            folds = pd.read_csv(
                self._copied_path("outer_folds"),
                usecols=["outer_fold"],
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise SequenceAdapterError(f"Cannot read outer folds: {error}") from error
        return tuple(
            sorted(int(value) for value in folds["outer_fold"].unique())
        )

    def inner_fold_labels(self, outer_fold: int) -> tuple[int, ...]:
        """Return actual inner-fold labels for one outer-training partition."""

        try:
            folds = pd.read_csv(
                self._copied_path("inner_folds"),
                usecols=["outer_fold", "inner_fold"],
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise SequenceAdapterError(f"Cannot read inner folds: {error}") from error
        folds = folds.loc[folds["outer_fold"].astype(int) == outer_fold]
        if folds.empty:
            available = self.outer_fold_labels()
            raise SequenceAdapterError(
                f"Unknown outer fold {outer_fold}. Available: {list(available)}"
            )
        return tuple(
            sorted(int(value) for value in folds["inner_fold"].unique())
        )

    def _fold_uav_ids(
        self,
        *,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> tuple[set[str], set[str]]:
        """Return fixed training and validation UAV IDs for one fold."""

        if inner_fold is None:
            try:
                folds = pd.read_csv(
                    self._copied_path("outer_folds"),
                    usecols=["uav_id", "outer_fold"],
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                raise SequenceAdapterError(
                    f"Cannot read outer folds: {error}"
                ) from error
            available = set(folds["outer_fold"].astype(int))
            if outer_fold not in available:
                raise SequenceAdapterError(
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
                raise SequenceAdapterError(
                    f"Cannot read inner folds: {error}"
                ) from error
            available_outer = set(folds["outer_fold"].astype(int))
            if outer_fold not in available_outer:
                raise SequenceAdapterError(
                    f"Unknown outer fold {outer_fold}. "
                    f"Available: {sorted(available_outer)}"
                )
            folds = folds.loc[folds["outer_fold"] == outer_fold]
            available_inner = set(folds["inner_fold"].astype(int))
            if inner_fold not in available_inner:
                raise SequenceAdapterError(
                    f"Unknown inner fold {inner_fold} for outer fold {outer_fold}. "
                    f"Available: {sorted(available_inner)}"
                )
            validation_mask = folds["inner_fold"] == inner_fold

        training_ids = set(folds.loc[~validation_mask, "uav_id"].astype(str))
        validation_ids = set(folds.loc[validation_mask, "uav_id"].astype(str))
        if not training_ids or not validation_ids:
            raise SequenceAdapterError("Requested fold produced an empty UAV partition")
        return training_ids, validation_ids

    def fit_channel_scaler(
        self,
        training_uav_ids: Collection[str],
    ) -> RobustChannelScaler:
        """Fit channel parameters using complete histories of training UAVs.

        Only UAV membership controls this fit; validation histories are never
        read. Using complete training-UAV histories makes channel normalization
        independent of the candidate lookback while preserving the fold boundary.
        """

        selected_ids = {str(uav_id) for uav_id in training_uav_ids}
        histories = self._load_histories("raw_train", selected_ids)
        values = np.concatenate(
            [history_values for _, history_values in histories.values()],
            axis=0,
        )
        scaling = self.manifest.get("scaling", {})
        tolerance = scaling.get("relative_variation_tolerance", 1e-12)
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise SequenceAdapterError(
                "Relative variation tolerance must be a positive number"
            )
        return fit_robust_channel_scaler(
            values,
            self.channel_names,
            fit_uavs=len(selected_ids),
            relative_variation_tolerance=float(tolerance),
        )

    def get_inner_selection_split(
        self,
        outer_fold: int,
        inner_fold: int,
        lookback: int,
    ) -> SequenceSplit:
        """Return safely scaled inner-training and development windows."""

        training_ids, validation_ids = self._fold_uav_ids(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
        )
        raw_training = self._build_dataset(
            "training_endpoints",
            lookback,
            training_ids,
        )
        raw_validation = self._build_dataset(
            "development_endpoints",
            lookback,
            validation_ids,
        )
        scaler = self.fit_channel_scaler(training_ids)
        return SequenceSplit(
            training=scaler.transform(raw_training),
            validation=scaler.transform(raw_validation),
            channel_scaler=scaler,
        )

    def get_locked_outer_evaluation_split(
        self,
        outer_fold: int,
        lookback: int,
    ) -> SequenceSplit:
        """Return outer-training and explicitly locked scaled windows."""

        training_ids, validation_ids = self._fold_uav_ids(outer_fold=outer_fold)
        raw_training = self._build_dataset(
            "training_endpoints",
            lookback,
            training_ids,
        )
        raw_validation = self._build_dataset(
            "locked_endpoints",
            lookback,
            validation_ids,
        )
        scaler = self.fit_channel_scaler(training_ids)
        return SequenceSplit(
            training=scaler.transform(raw_training),
            validation=scaler.transform(raw_validation),
            channel_scaler=scaler,
        )
