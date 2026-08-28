"""Validate the Phase 2 architecture study settings and its Phase 1 inputs.

This module owns all Step 1 verification logic.  It is both a command-line
check and an importable guard for the builder and later Phase 2 steps.  The
verifier is intentionally read-only: it reports problems but never changes an
input or artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Annotated, Any, Literal, TypeAlias

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)


STEP_DIR = Path(__file__).resolve().parent
PHASE_ROOT = STEP_DIR.parent
REPOSITORY_ROOT = PHASE_ROOT.parent
DEFAULT_SETTINGS_PATH = STEP_DIR / "architecture_study_settings.toml"

# The three directory constants anchor every default path to this source file,
# not to the caller's current working directory.  The scripts can therefore be
# run from the repository root, the step folder, or another local directory.

# Phase 1 names every generated model input with this prefix.  Counting these
# columns separately lets the verifier distinguish metadata such as "uav_id"
# and "RUL" from features that may later be passed to a model.
FEATURE_PREFIX = "feature__"

# The named search parameters are part of the executable interface between the
# settings and the future model adapters.  Keeping this registry here means a
# misspelled parameter fails now instead of being silently ignored during tuning.
EXPECTED_SEARCH_PARAMETERS: dict[str, set[str]] = {
    "mean_baseline": set(),
    "cycle_only_baseline": set(),
    "regularized_linear": {
        "variant",
        "ridge_alpha",
        "elastic_net_alpha",
        "elastic_net_l1_ratio",
    },
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    },
    "extra_trees": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    },
    "xgboost": {
        "maximum_trees",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    },
    "catboost": {
        "maximum_trees",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "bagging_temperature",
        "rsm",
        "boosting_type",
    },
    "mlp": {"hidden_layers", "dropout", "learning_rate", "weight_decay"},
    "tcn": {
        "residual_blocks",
        "channels",
        "kernel_size",
        "dilation_base",
        "dropout",
        "learning_rate",
        "weight_decay",
    },
    "multiscale_cnn": {
        "branch_channels",
        "kernel_sizes",
        "dropout",
        "learning_rate",
        "weight_decay",
    },
    "sensor_graph_tcn": {
        "graph_hidden",
        "graph_layers",
        "graph_neighbors",
        "temporal_blocks",
        "temporal_channels",
        "kernel_size",
        "dropout",
        "learning_rate",
        "weight_decay",
    },
    "lstm": {
        "layers",
        "hidden_units",
        "direction",
        "dropout",
        "learning_rate",
        "weight_decay",
    },
    "transformer": {
        "encoder_layers",
        "model_width",
        "attention_heads",
        "feed_forward_ratio",
        "position_encoding",
        "dropout",
        "learning_rate",
        "weight_decay",
    },
    "rbf_svr": {"c", "gamma", "epsilon"},
    "trajectory_dtw_knn": {
        "neighbors",
        "reference_pool_size",
        "max_points",
        "warping_window",
        "distance_power",
    },
}

EXPECTED_VARIANTS: dict[str, set[str]] = {
    "mean_baseline": {"mean"},
    "cycle_only_baseline": {"weighted_linear"},
    "regularized_linear": {"ridge", "elastic_net"},
    "random_forest": {"random_forest"},
    "extra_trees": {"extra_trees"},
    "xgboost": {"xgboost"},
    "catboost": {"catboost"},
    "mlp": {"mlp"},
    "tcn": {"tcn"},
    "multiscale_cnn": {"multiscale_cnn"},
    "sensor_graph_tcn": {"sensor_graph_tcn"},
    "lstm": {"lstm"},
    "transformer": {"transformer_encoder"},
    "rbf_svr": {"rbf_svr"},
    "trajectory_dtw_knn": {"dtw_knn"},
}

# Step 1 is a boundary between dataset construction and model experiments.
# Requiring the complete named set prevents a newly added Phase 1 dependency
# from being forgotten or a required dependency from disappearing silently.
EXPECTED_PHASE_1_ARTIFACTS = {
    "verification_report",
    "fold_config",
    "outer_folds",
    "inner_folds",
    "scenario_config",
    "development_scenarios",
    "locked_scenarios",
    "training_prefix_config",
    "training_prefixes",
    "training_features",
    "development_features",
    "locked_features",
    "test_features",
    "feature_catalog",
    "preprocessing_config",
    "metric_specification",
}


# ---------------------------------------------------------------------------
# Strict schema for the TOML architecture study settings
# ---------------------------------------------------------------------------
# These models validate structure and relationships only. They do not train a
# model, read telemetry, or modify an upstream artifact.


class SettingsError(ValueError):
    """Represent one readable settings-validation or upstream-artifact failure.

    A dedicated exception keeps expected validation failures separate from
    programming errors.  Both command-line scripts catch this exception and
    show its message without exposing an unnecessary Python traceback.
    """


class StrictModel(BaseModel):
    """Make every settings section reject unknown fields and loose coercion.

    "extra='forbid'" catches misspelled settings instead of ignoring them.
    Strict mode prevents surprising conversions, such as interpreting the
    string value "25" as the integer 25.  All schema classes inherit these
    rules so validation behaviour is consistent throughout the settings file.
    """

    model_config = ConfigDict(extra="forbid", strict=True)


ScalarValue: TypeAlias = str | int | float | bool


class FixedParameter(StrictModel):
    """Describe a hyperparameter that has one non-tunable value.

    Fixed parameters still live in "search" so every model adapter receives
    one uniform configuration mapping, regardless of whether a value is tuned.
    """

    kind: Literal["fixed"]
    value: ScalarValue


class CategoricalParameter(StrictModel):
    """Describe a finite set of scalar values sampled during tuning.

    This covers unordered alternatives such as tree depths or a choice between
    Ridge and Elastic Net.  The tuning implementation must select only values
    listed here rather than inventing intermediate values.
    """

    kind: Literal["categorical"]
    values: list[ScalarValue]

    @field_validator("values")
    @classmethod
    def values_must_be_unique(cls, values: list[ScalarValue]) -> list[ScalarValue]:
        """Reject empty choices and duplicates before tuning begins."""

        if not values:
            raise ValueError("must contain at least one value")

        # Include the type in the identity because Python otherwise considers
        # values such as "True" and "1" equal even though they represent
        # different configuration choices.
        identities = [(type(value).__name__, value) for value in values]
        if len(identities) != len(set(identities)):
            raise ValueError("must not contain duplicate values")
        return values


class CategoricalIntegerSequencesParameter(StrictModel):
    """Describe choices whose individual values are integer layer layouts.

    A normal categorical parameter stores scalars.  Neural hidden-layer shapes
    instead look like "[128, 64]", so they receive an explicit type rather
    than being accepted through an unstructured "Any" value.
    """

    kind: Literal["categorical_integer_sequences"]
    values: list[list[PositiveInt]]

    @field_validator("values")
    @classmethod
    def sequences_must_be_unique(
        cls, values: list[list[PositiveInt]]
    ) -> list[list[PositiveInt]]:
        """Require at least one unique, non-empty positive layer layout."""

        if not values:
            raise ValueError("must contain at least one sequence")
        identities = [tuple(value) for value in values]
        if len(identities) != len(set(identities)):
            raise ValueError("must not contain duplicate sequences")
        return values


class IntegerRangeParameter(StrictModel):
    """Describe a stepped integer interval for a future tuning sampler.

    The schema records the endpoints and a positive step explicitly, avoiding
    ambiguous floating-point conversion when a parameter must be integral.
    """

    kind: Literal["integer_range"]
    low: int
    high: int
    step: PositiveInt = 1

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> IntegerRangeParameter:
        """Ensure a future sampler can traverse the interval."""

        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        return self


class UniformParameter(StrictModel):
    """Describe a continuous interval sampled uniformly on a linear scale.

    This is appropriate for bounded fractions such as dropout or subsampling,
    where equal absolute differences should receive equal sampling density.
    """

    kind: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> UniformParameter:
        """Reject zero-width or reversed intervals."""

        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        return self


class LogUniformParameter(StrictModel):
    """Describe a positive interval sampled uniformly in logarithmic space.

    Learning rates and regularization strengths often span several orders of
    magnitude, so equal multiplicative changes receive equal search emphasis.
    """

    kind: Literal["log_uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def bounds_are_positive_and_ordered(self) -> LogUniformParameter:
        """Logarithmic sampling requires positive, increasing bounds."""

        if self.low <= 0:
            raise ValueError("low must be greater than zero")
        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        return self


SearchParameter = Annotated[
    FixedParameter
    | CategoricalParameter
    | CategoricalIntegerSequencesParameter
    | IntegerRangeParameter
    | UniformParameter
    | LogUniformParameter,
    Field(discriminator="kind"),
]

# Pydantic reads "kind" first and then selects exactly one model from the
# discriminated union above.  Consequently, a "fixed" parameter cannot
# accidentally receive "low" and "high" fields, and a range cannot omit
# either boundary.


class ExecutionSpecification(StrictModel):
    """Define the default Step 5/6 concurrency, an execution/performance knob.

    Unlike every other section in this file, this is not a scientific
    setting: it only changes wall-clock time, never results.  "run_phase_2.py"
    is therefore still allowed to override it for one-off runs with its
    "--max-workers" flag without touching this file.
    """

    max_workers: PositiveInt | Literal["auto"]


class StudySpecification(StrictModel):
    """Define which architectures the study runs, and in what priority.

    "enabled" is the single on/off switch for every declared architecture,
    checked against the other three lists below by "sections_are_consistent".
    """

    architectures_to_run: list[str]
    conditional_architectures: list[str]
    optional_architectures: list[str]
    enabled: dict[str, bool]


class TuningSpecification(StrictModel):
    """Define settings shared by all within-architecture tuning runs.

    This section separates automated hyperparameter optimization—which remains
    enabled—from the manual comparison between architecture families.
    """

    candidate_budget_per_architecture: PositiveInt
    search_seed: int
    retraining_seeds: list[int]

    @field_validator("retraining_seeds")
    @classmethod
    def retraining_seeds_must_be_unique(cls, values: list[int]) -> list[int]:
        """Ensure repeated stochastic fits represent distinct runs."""

        if not values:
            raise ValueError("must contain at least one seed")
        if len(values) != len(set(values)):
            raise ValueError("must not contain duplicate seeds")
        return values


class EvaluationSpecification(StrictModel):
    """Define common metrics, reporting groups, and uncertainty settings.

    Centralizing these values prevents individual architecture adapters from
    reporting a more favorable metric or using a different weighting policy.
    """

    metrics: list[Literal["r2", "rmse", "mae", "bias"]]
    reported_groups: list[
        Literal[
            "overall",
            "scenario",
            "outer_fold",
            "age_band",
            "lifetime_quantile",
        ]
    ]
    prediction_minimum: float
    bootstrap_repetitions: PositiveInt
    bootstrap_seed: int


class RepresentationSpecification(StrictModel):
    """Describe the tabular and raw-sequence inputs available to adapters.

    Tabular models consume Phase 1 engineered feature rows.  Sequence models
    consume ordered telemetry windows plus age side features.  Both interfaces
    are declared here so adapters can be exchanged without changing folds,
    targets, or the evaluation protocol.
    """

    tabular_feature_sets: list[str]
    sequence_channel_count: PositiveInt
    sequence_channels: list[str]
    sequence_lookbacks: list[PositiveInt]
    sequence_side_features: list[
        Literal["flight_cycle", "log1p_flight_cycle"]
    ]

    @model_validator(mode="after")
    def sequence_channels_are_explicit_and_unique(
        self,
    ) -> RepresentationSpecification:
        """Keep the declared sequence count synchronized with channel names."""

        if len(self.sequence_channels) != self.sequence_channel_count:
            raise ValueError(
                "sequence_channel_count must equal the number of sequence_channels"
            )
        if len(self.sequence_channels) != len(set(self.sequence_channels)):
            raise ValueError("sequence_channels must not contain duplicates")
        if not self.tabular_feature_sets:
            raise ValueError("tabular_feature_sets must not be empty")
        if len(self.tabular_feature_sets) != len(set(self.tabular_feature_sets)):
            raise ValueError("tabular_feature_sets must not contain duplicates")
        if any(not name.strip() for name in self.tabular_feature_sets):
            raise ValueError("tabular_feature_sets must contain non-empty names")
        return self


class PreprocessingSpecification(StrictModel):
    """Assign fold-fitted preprocessing modes to architecture groups.

    The literal fit scope makes the leakage boundary part of the validated
    settings: transformations may learn statistics from training UAVs only.
    """

    scaled_tabular_architectures: list[str]
    unscaled_tree_architectures: list[str]


class NeuralTrainingSpecification(StrictModel):
    """Hold optimization choices shared by every neural architecture.

    Shared defaults keep architecture comparisons interpretable while each
    neural family tunes its own capacity, dropout, and learning rate.
    """

    batch_size: PositiveInt
    maximum_epochs: PositiveInt
    early_stopping_patience: PositiveInt
    gradient_clip_global_norm: float = Field(gt=0)


class ArchitectureSpecification(StrictModel):
    """Describe one executable architecture and its tuning alternatives.

    "feature_sets" and "lookbacks" are mutually exclusive because tabular
    and sequence adapters expose different input structures.  The optional
    early-stopping field is currently used by XGBoost; neural early stopping is
    shared through "NeuralTrainingSpecification".
    """

    status: Literal["included", "conditional", "optional"]
    representation: Literal["none", "tabular", "sequence", "trajectory"]
    feature_sets: list[str] = Field(default_factory=list)
    lookbacks: list[PositiveInt] = Field(default_factory=list)
    variants: list[str]
    early_stopping_patience: PositiveInt | None = None
    search: dict[str, SearchParameter] = Field(default_factory=dict)

    @model_validator(mode="after")
    def representation_fields_are_consistent(self) -> ArchitectureSpecification:
        """Reject input settings that do not match the representation type."""

        if not self.variants:
            raise ValueError("variants must not be empty")
        if len(self.variants) != len(set(self.variants)):
            raise ValueError("variants must not contain duplicates")

        if self.representation == "none" and (self.feature_sets or self.lookbacks):
            raise ValueError("a representation-free model cannot define inputs")
        if self.representation == "tabular":
            if not self.feature_sets or self.lookbacks:
                raise ValueError("tabular models need feature_sets and no lookbacks")
        if self.representation == "sequence":
            if not self.lookbacks or self.feature_sets:
                raise ValueError("sequence models need lookbacks and no feature_sets")
        if self.representation == "trajectory" and (
            self.feature_sets or self.lookbacks
        ):
            raise ValueError(
                "trajectory models use the trajectory adapter and no feature_sets "
                "or lookbacks"
            )
        return self


class ArtifactSpecification(StrictModel):
    """Describe the lightweight structural checks for one Phase 1 artifact.

    CSV files are checked through dimensions and required columns.  JSON files
    are checked through required dotted keys and selected exact scalar values.
    The settings file intentionally stores no file hashes.
    """

    path: str
    format: Literal["csv", "json"]
    rows: int | None = Field(default=None, ge=0)
    total_columns: int | None = Field(default=None, ge=0)
    feature_columns: int | None = Field(default=None, ge=0)
    required_columns: list[str] = Field(default_factory=list)
    required_json_keys: list[str] = Field(default_factory=list)
    required_json_values: dict[str, ScalarValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fields_match_file_format(self) -> ArtifactSpecification:
        """Prevent CSV-only and JSON-only expectations from being mixed."""

        csv_only_values = (self.rows, self.total_columns, self.feature_columns)
        if self.format == "csv":
            if self.rows is None or self.total_columns is None:
                raise ValueError("CSV artifacts require rows and total_columns")
            if not self.required_columns:
                raise ValueError("CSV artifacts require required_columns")
            if self.required_json_keys or self.required_json_values:
                raise ValueError("CSV artifacts cannot define JSON requirements")
        elif any(value is not None for value in csv_only_values):
            raise ValueError("JSON artifacts cannot define CSV dimensions")
        elif self.required_columns:
            raise ValueError("JSON artifacts cannot define required_columns")
        return self


class PhaseOneSpecification(StrictModel):
    """Collect the Phase 1 interface consumed by the architecture study.

    Expected global counts and per-file specifications make upstream changes
    visible before they can alter an experiment unnoticed.
    """

    expected_training_uavs: PositiveInt
    expected_outer_folds: PositiveInt
    expected_inner_folds_per_outer_fold: PositiveInt
    expected_development_scenarios: PositiveInt
    expected_locked_scenarios: PositiveInt
    expected_prefixes_per_training_uav: PositiveInt | None = None
    minimum_prefixes_per_training_uav: PositiveInt | None = None
    maximum_prefixes_per_training_uav: PositiveInt | None = None
    expected_generated_features: PositiveInt
    expected_feature_sets: dict[str, PositiveInt]
    artifacts: dict[str, ArtifactSpecification]

    @model_validator(mode="after")
    def prefix_count_is_exact_or_bounded(self) -> PhaseOneSpecification:
        exact = self.expected_prefixes_per_training_uav
        bounds = (
            self.minimum_prefixes_per_training_uav,
            self.maximum_prefixes_per_training_uav,
        )
        if exact is not None and any(value is not None for value in bounds):
            raise ValueError(
                "declare either expected_prefixes_per_training_uav or the "
                "minimum/maximum prefix bounds, not both"
            )
        if exact is None:
            if any(value is None for value in bounds):
                raise ValueError(
                    "variable prefix counts require both "
                    "minimum_prefixes_per_training_uav and "
                    "maximum_prefixes_per_training_uav"
                )
            assert bounds[0] is not None and bounds[1] is not None
            if bounds[0] > bounds[1]:
                raise ValueError("minimum prefixes cannot exceed maximum prefixes")
        return self


class ArchitectureStudySettings(StrictModel):
    """Represent the complete, typed interface for Phase 2 experiments.

    A successfully created instance guarantees both local field validity and
    semantic agreement among architecture, representation, and Phase 1 sections.
    It does not yet guarantee that the declared files exist; that read-only check
    is performed by "verify_phase_1_inputs" below.
    """

    settings_version: PositiveInt
    # Selects the runs/run_<n>/ folder that Steps 5, 6 and 7 use. It is edited
    # by hand and never advanced by the pipeline, so an interrupted run resumes
    # into the folder it started in.
    run_number: PositiveInt
    execution: ExecutionSpecification
    study: StudySpecification
    tuning: TuningSpecification
    evaluation: EvaluationSpecification
    representations: RepresentationSpecification
    preprocessing: PreprocessingSpecification
    neural_training: NeuralTrainingSpecification
    architectures: dict[str, ArchitectureSpecification]
    phase_1: PhaseOneSpecification

    @model_validator(mode="after")
    def sections_are_consistent(self) -> ArchitectureStudySettings:
        """Validate relationships that span otherwise independent sections.

        Pydantic validates each field locally.  This method handles semantic
        relationships, for example that an architecture appears in exactly one
        status list and that every named search parameter is recognized by its
        future adapter.
        """

        to_run = set(self.study.architectures_to_run)
        conditional = set(self.study.conditional_architectures)
        optional = set(self.study.optional_architectures)
        listed = to_run | conditional | optional

        if len(listed) != sum(map(len, (to_run, conditional, optional))):
            raise ValueError("architecture status lists must not overlap")
        if listed != set(self.architectures):
            raise ValueError(
                "study architecture lists must exactly match the architecture tables"
            )
        if set(self.study.enabled) != listed:
            raise ValueError(
                "study.enabled must contain exactly one entry per declared "
                "architecture"
            )

        # Derive the expected per-model status from the three top-level lists.
        # This prevents a table from claiming "optional" while also appearing
        # in the list of architectures the study runs.
        expected_status = {
            **{name: "included" for name in to_run},
            **{name: "conditional" for name in conditional},
            **{name: "optional" for name in optional},
        }
        for name, architecture in self.architectures.items():
            if architecture.status != expected_status[name]:
                raise ValueError(f"architectures.{name}.status is inconsistent")

            # Search dictionaries need a separate name check because their keys
            # are dynamic mappings rather than normal Pydantic model fields.
            expected_parameters = EXPECTED_SEARCH_PARAMETERS.get(name)
            if expected_parameters is None:
                raise ValueError(f"unsupported architecture {name!r}")
            if set(architecture.search) != expected_parameters:
                raise ValueError(
                    f"architectures.{name}.search must contain exactly "
                    f"{sorted(expected_parameters)}"
                )
            if set(architecture.variants) != EXPECTED_VARIANTS[name]:
                raise ValueError(
                    f"architectures.{name}.variants must contain exactly "
                    f"{sorted(EXPECTED_VARIANTS[name])}"
                )

            unknown_features = set(architecture.feature_sets) - set(
                self.representations.tabular_feature_sets
            )
            if unknown_features:
                raise ValueError(
                    f"architectures.{name} uses unknown feature sets "
                    f"{sorted(unknown_features)}"
                )
            unknown_lookbacks = set(architecture.lookbacks) - set(
                self.representations.sequence_lookbacks
            )
            if unknown_lookbacks:
                raise ValueError(
                    f"architectures.{name} uses unknown lookbacks "
                    f"{sorted(unknown_lookbacks)}"
                )

        if set(self.phase_1.artifacts) != EXPECTED_PHASE_1_ARTIFACTS:
            raise ValueError(
                "phase_1.artifacts must contain exactly "
                f"{sorted(EXPECTED_PHASE_1_ARTIFACTS)}"
            )
        if set(self.phase_1.expected_feature_sets) != set(
            self.representations.tabular_feature_sets
        ):
            raise ValueError(
                "phase_1.expected_feature_sets must exactly match "
                "representations.tabular_feature_sets"
            )
        return self


class ArtifactCheck(StrictModel):
    """Summarize the structure actually observed for one input file.

    These observations enter the generated JSON artifact.  They provide a
    compact audit trail without copying full datasets or storing file hashes.
    """

    path: str
    format: Literal["csv", "json"]
    rows: int | None = None
    columns: int | None = None
    feature_columns: int | None = None


class VerificationSummary(StrictModel):
    """Provide deterministic verification metadata for the JSON artifact.

    The status can only be "passed" because failures raise "SettingsError"
    and prevent artifact generation entirely.
    """

    status: Literal["passed"]
    checked_artifacts: int
    phase_1_assertions: int
    artifacts: dict[str, ArtifactCheck]


# ---------------------------------------------------------------------------
# Settings loading and portable path handling
# ---------------------------------------------------------------------------


def _format_pydantic_error(error: ValidationError) -> str:
    """Convert Pydantic errors into messages that point into the TOML file.

    Pydantic exposes errors as structured dictionaries.  That structure is
    useful to code, but a reader editing TOML needs a location such as
    "architectures.lstm.lookbacks". Joining each location component with a
    dot produces that familiar notation while preserving Pydantic's exact
    explanation of the invalid value.
    """

    messages: list[str] = []
    for detail in error.errors(include_url=False):
        # "loc" can contain both field names and list indices.  Converting
        # every component to text handles both without special cases.
        location = ".".join(str(part) for part in detail["loc"])
        messages.append(f"{location}: {detail['msg']}")
    return "\n".join(messages)


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> ArchitectureStudySettings:
    """Read the human-authored TOML settings and validate it completely.

    Parsing and schema validation are deliberately separate operations.  A
    TOML syntax error means the file itself cannot be read, whereas a Pydantic
    error means the TOML is syntactically valid but violates an experiment
    rule.  Both cases are converted to "SettingsError" so callers get a
    consistent, readable failure interface.
    """

    try:
        # "tomllib" expects a binary stream and performs no type coercion of
        # our own.  The resulting Python mapping is validated in the next step.
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SettingsError(f"Cannot read settings {path}: {error}") from error

    try:
        # This invokes field validation and the cross-section model validator.
        # No downstream file is inspected until this in-memory settings object passes.
        return ArchitectureStudySettings.model_validate(payload)
    except ValidationError as error:
        raise SettingsError(
            "Architecture study settings schema validation failed:\n"
            f"{_format_pydantic_error(error)}"
        ) from error


def repository_relative_path(path: Path) -> str:
    """Represent a path portably and reject files outside this repository.

    Generated metadata must not contain developer-specific absolute paths such
    as "C:\\Users\\...". The resolved containment check also prevents a path
    containing ".." from merely looking repository-relative while actually
    referring to an external file.
    """

    try:
        # POSIX separators make the generated JSON stable across operating
        # systems even though the current development machine uses Windows.
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise SettingsError(f"Path is outside the repository: {path}") from error


def _resolve_input_path(relative_path: str) -> Path:
    """Resolve one declared Phase 1 input safely from the repository root.

    The settings file stores repository-relative paths for portability.  This
    helper centralizes path handling and refuses both absolute paths and
    relative paths that escape through parent-directory components.
    """

    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise SettingsError(f"Artifact path must be relative: {relative_path}")

    # Resolve first so that embedded ".." components and symbolic links are
    # accounted for before the containment check is applied.
    resolved = (REPOSITORY_ROOT / supplied).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise SettingsError(
            f"Artifact path escapes the repository: {relative_path}"
        ) from error
    return resolved


def _nested_json_value(payload: dict[str, Any], dotted_path: str) -> Any:
    """Read a nested JSON value named with settings-friendly dotted notation.

    For example, "uncertainty.unit" means
    "payload['uncertainty']['unit']". The settings can therefore describe
    nested requirements without reproducing an entire upstream JSON document.
    A missing object or key raises "KeyError" with the original full path so
    the caller can report exactly which requirement failed.
    """

    current: Any = payload
    for part in dotted_path.split("."):
        # Each intermediate value must be a JSON object.  Lists are not
        # traversed because the current settings file only needs named object keys.
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


# ---------------------------------------------------------------------------
# Phase 1 artifact verification
# ---------------------------------------------------------------------------


def _verify_json_artifact(
    name: str,
    path: Path,
    specification: ArtifactSpecification,
) -> tuple[ArtifactCheck, dict[str, Any]]:
    """Validate the small, stable interface of one Phase 1 JSON artifact.

    The verifier intentionally does not compare the complete document.  It
    checks the keys that Phase 2 consumes and exact values that define critical
    interfaces, such as the number of folds.  This keeps Phase 2 sensitive to
    meaningful upstream changes without coupling it to unrelated metadata.

    The parsed payload is returned as well as the summary because selected JSON
    files participate in later cross-artifact consistency checks.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SettingsError(
            f"{name}: cannot read valid JSON from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SettingsError(f"{name}: top-level JSON value must be an object")

    # Accumulate every local problem before failing.  A user can then correct
    # all missing or changed values in one edit-and-verify cycle.
    problems: list[str] = []

    # Presence checks establish the required interface even when the concrete
    # value is allowed to vary or is checked against another file later.
    for key in specification.required_json_keys:
        try:
            _nested_json_value(payload, key)
        except KeyError:
            problems.append(f"missing required JSON key {key!r}")
    # Exact-value checks protect settings that must agree with this experiment,
    # for example five outer folds or leakage verification status "passed".
    for key, expected in specification.required_json_values.items():
        try:
            observed = _nested_json_value(payload, key)
        except KeyError:
            problems.append(f"missing required JSON value {key!r}")
            continue
        if observed != expected:
            problems.append(f"{key!r} is {observed!r}, expected {expected!r}")
    if problems:
        raise SettingsError(f"{name}: " + "; ".join(problems))

    # Only lightweight, deterministic observations enter the generated
    # specification; the complete upstream payload remains an input artifact.
    return (
        ArtifactCheck(
            path=repository_relative_path(path),
            format="json",
        ),
        payload,
    )


def _verify_csv_artifact(
    name: str,
    path: Path,
    specification: ArtifactSpecification,
) -> ArtifactCheck:
    """Validate the structural interface of one Phase 1 tabular artifact.

    The checks answer four questions needed before an architecture can consume
    the file: can pandas read it, does it contain the expected observations,
    are required metadata columns present, and does it expose the expected
    number of engineered features?  Values are not re-audited here because
    their semantic and leakage checks belong to Phase 1.
    """

    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise SettingsError(f"{name}: cannot read CSV {path}: {error}") from error

    # As with JSON validation, collect all local discrepancies so one run gives
    # a useful diagnostic instead of stopping at the first missing column.
    problems: list[str] = []
    missing = [
        column
        for column in specification.required_columns
        if column not in table.columns
    ]
    if missing:
        problems.append(f"missing columns {missing}")
    if len(table) != specification.rows:
        problems.append(f"has {len(table)} rows, expected {specification.rows}")
    if len(table.columns) != specification.total_columns:
        problems.append(
            f"has {len(table.columns)} columns, expected {specification.total_columns}"
        )

    # Phase 1 uses a naming convention for model features.  Counting by prefix
    # avoids hard-coding 606 names here and excludes identifiers/targets.
    feature_columns = sum(column.startswith(FEATURE_PREFIX) for column in table.columns)
    if (
        specification.feature_columns is not None
        and feature_columns != specification.feature_columns
    ):
        problems.append(
            f"has {feature_columns} feature columns, "
            f"expected {specification.feature_columns}"
        )
    if problems:
        raise SettingsError(f"{name}: " + "; ".join(problems))

    return ArtifactCheck(
        path=repository_relative_path(path),
        format="csv",
        rows=int(len(table)),
        columns=int(len(table.columns)),
        feature_columns=int(feature_columns),
    )


def _verify_cross_artifact_consistency(
    settings: ArchitectureStudySettings,
    json_payloads: dict[str, dict[str, Any]],
) -> int:
    """Verify agreements that cannot be established one file at a time.

    Individual artifact checks prove that files have the required structures.
    This function proves that selected contents agree with each other and with
    the Phase 2 settings.  Without these comparisons, two individually valid
    files could still describe different feature sets, metrics, or validation
    protocols.

    Returns:
        The number of passed Phase 1 assertions, recorded in the generated
        verification summary for a concise audit trail.
    """

    report = json_payloads["verification_report"]
    assertions = report.get("assertions")
    if not isinstance(assertions, dict) or not assertions:
        raise SettingsError(
            "verification_report: assertions must be a non-empty object"
        )
    # "is not True" deliberately rejects false, null, zero, and strings such
    # as "passed". Every assertion must be the JSON boolean true.
    failed_assertions = sorted(
        name for name, passed in assertions.items() if passed is not True
    )
    if failed_assertions:
        raise SettingsError(
            "verification_report: failed Phase 1 assertions "
            f"{failed_assertions}"
        )

    # The same feature-set sizes appear in the Phase 1 verification report and
    # preprocessing configuration.  Requiring three-way agreement prevents a
    # stale feature catalog from silently entering the architecture study.
    expected_sets = dict(settings.phase_1.expected_feature_sets)
    if report.get("feature_sets") != expected_sets:
        raise SettingsError(
            "verification_report: feature-set counts do not match the settings"
        )

    preprocessing = json_payloads["preprocessing_config"]
    if preprocessing.get("feature_counts") != expected_sets:
        raise SettingsError(
            "preprocessing_config: feature-set counts do not match the settings"
        )

    # Phase 2 must report exactly the metrics and groups defined by Phase 1.
    # Sets are suitable for metric names; report-group order is retained because
    # it also defines a stable output and plotting order.
    metrics = json_payloads["metric_specification"]
    if set(metrics.get("metrics", {})) != set(settings.evaluation.metrics):
        raise SettingsError(
            "metric_specification: metric names do not match the settings"
        )
    if metrics.get("reported_groups") != settings.evaluation.reported_groups:
        raise SettingsError(
            "metric_specification: reported groups do not match the settings"
        )
    return len(assertions)


def _verify_training_prefix_counts(settings: ArchitectureStudySettings) -> None:
    """Check the declared exact count or bounds against each training UAV."""

    specification = settings.phase_1.artifacts["training_prefixes"]
    path = _resolve_input_path(specification.path)
    try:
        prefixes = pd.read_csv(path, usecols=["uav_id"])
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise SettingsError(
            f"training_prefixes: cannot inspect per-UAV counts: {error}"
        ) from error
    counts = prefixes.groupby("uav_id", sort=False).size()
    if len(counts) != settings.phase_1.expected_training_uavs:
        raise SettingsError(
            "training_prefixes: observed prefix groups do not match "
            "expected_training_uavs"
        )

    exact = settings.phase_1.expected_prefixes_per_training_uav
    if exact is not None:
        invalid = counts[counts != exact]
        if not invalid.empty:
            raise SettingsError(
                "training_prefixes: per-UAV counts are outside the exact "
                f"contract of {exact}"
            )
        return

    minimum = settings.phase_1.minimum_prefixes_per_training_uav
    maximum = settings.phase_1.maximum_prefixes_per_training_uav
    assert minimum is not None and maximum is not None
    invalid = counts[(counts < minimum) | (counts > maximum)]
    if not invalid.empty:
        raise SettingsError(
            "training_prefixes: per-UAV counts are outside the declared "
            f"bounds [{minimum}, {maximum}]"
        )


# ---------------------------------------------------------------------------
# Public verification API and command-line entry point
# ---------------------------------------------------------------------------


def verify_phase_1_inputs(settings: ArchitectureStudySettings) -> VerificationSummary:
    """Run format-specific and cross-file checks for every declared input.

    This is the public verification entry point used by both the standalone
    verifier and the builder.  It performs no writes.  Returning a typed summary
    makes the observed input structure available to the deterministic JSON
    specification that later pipeline steps will consume.
    """

    artifact_checks: dict[str, ArtifactCheck] = {}
    json_payloads: dict[str, dict[str, Any]] = {}

    # Each artifact is resolved and checked according to its declared format.
    # JSON payloads are retained temporarily because a few contain values that
    # must be compared after every individual file has passed.
    for name, specification in settings.phase_1.artifacts.items():
        path = _resolve_input_path(specification.path)
        if not path.is_file():
            raise SettingsError(f"{name}: required artifact does not exist: {path}")

        if specification.format == "json":
            check, payload = _verify_json_artifact(name, path, specification)
            artifact_checks[name] = check
            json_payloads[name] = payload
        else:
            artifact_checks[name] = _verify_csv_artifact(name, path, specification)

    # Cross-file checks happen last so they can assume all required JSON files
    # exist, parse successfully, and contain their minimum required interface.
    _verify_training_prefix_counts(settings)
    assertion_count = _verify_cross_artifact_consistency(settings, json_payloads)
    return VerificationSummary(
        status="passed",
        checked_artifacts=len(artifact_checks),
        phase_1_assertions=assertion_count,
        artifacts=artifact_checks,
    )


def load_and_verify_settings(
    path: Path = DEFAULT_SETTINGS_PATH,
) -> tuple[ArchitectureStudySettings, VerificationSummary]:
    """Provide one mandatory gate for every consumer of the settings.

    Keeping schema and input verification behind this single function prevents
    the builder—and later model-running code—from accidentally validating only
    half of the experiment interface.
    """

    settings = load_settings(path)
    summary = verify_phase_1_inputs(settings)
    return settings, summary


def main() -> None:
    """Run the read-only settings check from the command line."""

    # The CLI accepts file locations only.  Scientific settings intentionally
    # cannot be overridden with flags because that would create experiments not
    # represented by the versioned TOML source of truth.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="TOML settings location; experiment values cannot be overridden.",
    )
    args = parser.parse_args()

    try:
        settings, summary = load_and_verify_settings(args.settings)
        source = repository_relative_path(args.settings)
    except SettingsError as error:
        print(f"Architecture study settings verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    # Keep success output concise but include the quantities a human should
    # check before moving on to model execution.
    print("Architecture study settings verification passed")
    print(f"Settings: {source}")
    print(f"Settings version: {settings.settings_version}")
    print(f"Phase 1 artifacts checked: {summary.checked_artifacts}")
    print(f"Passed Phase 1 assertions: {summary.phase_1_assertions}")


if __name__ == "__main__":
    main()
