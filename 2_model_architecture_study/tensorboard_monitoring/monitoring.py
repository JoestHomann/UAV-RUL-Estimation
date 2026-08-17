"""Keep all TensorBoard-specific behavior outside the numbered Phase 2 steps.

The training and evaluation runners send small dictionaries of scalar values
to this module. They never construct TensorBoard writers directly. This keeps
the experiment code readable while giving every fitted model a stable,
human-readable run path in the dashboard.

Generated event files are a live monitoring aid. The CSV and JSON artifacts
written by Steps 5 through 7 remain the authoritative scientific results.

One writer is shared by every fit inside a study (one model family evaluated
on one outer fold) instead of opening a fresh writer per candidate or inner
fold. This keeps the generated directory count fixed at one study instead of
growing with the candidate budget and inner-fold count, while distinct tag
prefixes still let every individual fit's curves be selected separately in
TensorBoard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Literal, Mapping

import numpy as np


MONITORING_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = MONITORING_DIR / "logs"

# Neural epochs are inexpensive to record and are the natural optimization
# unit. XGBoost can execute thousands of rounds per fit, so recording every
# tenth round prevents excessive event files while retaining a useful curve.
NEURAL_LOG_INTERVAL = 1
XGBOOST_LOG_INTERVAL = 10

# SummaryWriter performs asynchronous disk writes. A short flush interval keeps
# the browser current, while explicit flush calls make write failures visible to
# the training process instead of silently losing an entire fit's monitoring.
WRITER_FLUSH_SECONDS = 5


class TensorBoardMonitoringError(RuntimeError):
    """Explain a missing dependency, unsafe path, or event-writing failure."""


@dataclass(frozen=True)
class TrainingRunContext:
    """Identify one model fit without timestamps or generated run identifiers."""

    stage: Literal["step_5", "step_6"]
    model_family: str
    representation: str
    outer_fold: int
    seed: int
    configuration_id: str
    candidate_number: int | None = None
    inner_fold: int | None = None
    feature_set: str | None = None
    lookback: int | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous contexts before any existing log is replaced."""

        if not self.model_family or not self.configuration_id:
            raise TensorBoardMonitoringError(
                "A monitored fit requires a model family and configuration ID"
            )
        if self.outer_fold < 0 or self.seed < 0:
            raise TensorBoardMonitoringError(
                "Fold labels and model seeds must be nonnegative"
            )
        if self.stage == "step_5":
            if self.candidate_number is None or self.candidate_number <= 0:
                raise TensorBoardMonitoringError(
                    "A Step 5 fit requires a positive candidate number"
                )
            if self.inner_fold is None or self.inner_fold < 0:
                raise TensorBoardMonitoringError(
                    "A Step 5 fit requires a nonnegative inner-fold label"
                )
        elif self.candidate_number is not None or self.inner_fold is not None:
            raise TensorBoardMonitoringError(
                "Step 6 run paths must not contain candidate or inner-fold labels"
            )

    def study_parts(self) -> tuple[str, str, str]:
        """Return the stable per-study key shared by every fit inside it."""

        return (self.stage, self.model_family, f"outer_fold_{self.outer_fold:02d}")

    def tag_prefix(self) -> str:
        """Return the tag namespace that keeps this fit's curves distinct.

        Every scalar and text value this fit writes is namespaced under this
        prefix inside the study's single shared writer, so TensorBoard's
        regex filter box (for example "^candidate_007/") recovers exactly
        what a dedicated per-fit directory used to provide.
        """

        if self.stage == "step_5":
            return (
                f"candidate_{self.candidate_number:03d}/"
                f"inner_fold_{self.inner_fold:02d}"
            )
        return f"seed_{self.seed:03d}"


def _summary_writer_class() -> type[Any]:
    """Import SummaryWriter only inside the isolated monitoring package."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError) as error:
        raise TensorBoardMonitoringError(
            "TensorBoard is mandatory for Phase 2 but is not available. "
            "Install the declared dependencies with: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r "
            "2_model_architecture_study\\requirements.txt"
        ) from error
    return SummaryWriter


def ensure_tensorboard_available() -> str:
    """Fail before expensive work when the mandatory dependency is unavailable."""

    try:
        installed_version = version("tensorboard")
    except PackageNotFoundError as error:
        raise TensorBoardMonitoringError(
            "TensorBoard is mandatory for Phase 2 but is not installed. "
            "Install the Phase 2 requirements before starting the pipeline."
        ) from error
    _summary_writer_class()
    return installed_version


def _validated_log_root(log_root: Path) -> Path:
    """Resolve and create the dedicated root used only for event files."""

    resolved = log_root.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _safe_run_directory(log_root: Path, parts: tuple[str, ...]) -> Path:
    """Resolve one run and prove that replacement cannot escape the log root."""

    if not parts:
        raise TensorBoardMonitoringError("A TensorBoard run path cannot be empty")
    for part in parts:
        if not part or part in {".", ".."} or Path(part).name != part:
            raise TensorBoardMonitoringError(
                f"Unsafe TensorBoard run-path component: {part!r}"
            )
    root = _validated_log_root(log_root)
    run_directory = root.joinpath(*parts).resolve()
    if not run_directory.is_relative_to(root):
        raise TensorBoardMonitoringError(
            f"Refusing to write TensorBoard events outside {root}"
        )
    return run_directory


def _replace_run_directory(run_directory: Path, log_root: Path) -> None:
    """Remove only one validated generated run before a clean replacement."""

    root = _validated_log_root(log_root)
    resolved = run_directory.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise TensorBoardMonitoringError(
            f"Refusing to replace unsafe TensorBoard directory {resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _json_text(value: Mapping[str, Any]) -> str:
    """Create readable TensorBoard text for nested experiment settings."""

    return json.dumps(dict(value), indent=2, sort_keys=True, default=str)


def _finite_scalar(value: Any) -> float | None:
    """Convert optional numeric values and omit metrics that are undefined."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TensorBoardMonitoringError(
            f"TensorBoard scalar value is not numeric: {value!r}"
        ) from error
    return numeric if math.isfinite(numeric) else None


def _input_feature_count(dataset: Any, representation: str) -> int:
    """Describe model inputs without retaining or writing any training rows."""

    if representation == "none":
        return 0
    features = getattr(dataset, "features", None)
    if features is not None and hasattr(features, "shape"):
        return int(features.shape[1])
    sequences = getattr(dataset, "sequences", None)
    side_features = getattr(dataset, "side_features", None)
    if sequences is not None and getattr(sequences, "ndim", 0) == 3:
        side_count = (
            int(side_features.shape[1])
            if side_features is not None and getattr(side_features, "ndim", 0) == 2
            else 0
        )
        return int(sequences.shape[2]) + side_count
    return 0


def calculate_regression_metrics(
    targets: Any,
    predictions: Any,
) -> dict[str, float]:
    """Calculate the four development metrics displayed for Step 5 fits."""

    observed = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if observed.shape != predicted.shape or not len(observed):
        raise TensorBoardMonitoringError(
            "Targets and predictions must be nonempty arrays with equal shapes"
        )
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise TensorBoardMonitoringError(
            "Cannot monitor non-finite targets or predictions"
        )
    residual = predicted - observed
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": (
            float("nan")
            if denominator <= 0.0
            else 1.0 - float(np.sum(np.square(residual))) / denominator
        ),
        "bias": float(np.mean(residual)),
    }


def calculate_age_band_regression_metrics(
    targets: Any,
    predictions: Any,
    cutoffs: Any,
) -> dict[str, float]:
    """Calculate permitted development metrics inside the fixed age bands."""

    observed = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    ages = np.asarray(cutoffs, dtype=np.float64).reshape(-1)
    if observed.shape != predicted.shape or observed.shape != ages.shape:
        raise TensorBoardMonitoringError(
            "Age-band targets, predictions, and cutoffs must have equal shapes"
        )
    if not np.isfinite(ages).all():
        raise TensorBoardMonitoringError("Age-band cutoffs must be finite")

    bands = {
        "1_50": (ages >= 1.0) & (ages <= 50.0),
        "51_100": (ages >= 51.0) & (ages <= 100.0),
        "101_200": (ages >= 101.0) & (ages <= 200.0),
        "over_200": ages > 200.0,
    }
    result: dict[str, float] = {}
    for label, mask in bands.items():
        if not np.any(mask):
            continue
        result[f"age_band/{label}/rows"] = float(np.sum(mask))
        metrics = calculate_regression_metrics(observed[mask], predicted[mask])
        for metric, value in metrics.items():
            result[f"age_band/{label}/{metric}"] = value
    return result


class TensorBoardStudyMonitor:
    """Own one shared SummaryWriter for every fit inside one study.

    A study is one model family evaluated on one outer fold: every candidate
    and inner fold in Step 5, or every retraining seed in Step 6. Sharing one
    writer across the whole study keeps the generated directory count fixed
    at one per study regardless of the candidate budget or inner-fold count,
    while ``TrainingRunContext.tag_prefix()`` keeps each fit's curves
    separately selectable inside that one run.
    """

    def __init__(
        self,
        *,
        stage: Literal["step_5", "step_6"],
        model_family: str,
        outer_fold: int,
        log_root: Path = DEFAULT_LOG_ROOT,
    ) -> None:
        if stage not in {"step_5", "step_6"}:
            raise TensorBoardMonitoringError(f"Unsupported monitoring stage {stage!r}")
        if not model_family:
            raise TensorBoardMonitoringError(
                "A monitored study requires a model family"
            )
        if outer_fold < 0:
            raise TensorBoardMonitoringError("Fold labels must be nonnegative")
        self.stage = stage
        self.model_family = model_family
        self.outer_fold = outer_fold
        self._study_key = (stage, model_family, f"outer_fold_{outer_fold:02d}")
        self.log_root = _validated_log_root(log_root)
        self.log_directory = _safe_run_directory(
            self.log_root,
            self._study_key + ("fit_progress",),
        )
        _replace_run_directory(self.log_directory, self.log_root)
        writer_class = _summary_writer_class()
        try:
            self._writer = writer_class(
                log_dir=str(self.log_directory),
                max_queue=10,
                flush_secs=WRITER_FLUSH_SECONDS,
                purge_step=0,
            )
        except Exception as error:
            raise TensorBoardMonitoringError(
                f"Cannot create TensorBoard writer at {self.log_directory}: {error}"
            ) from error
        self._closed = False
        self._writer_failed = False

    def __enter__(self) -> TensorBoardStudyMonitor:
        """Return this monitor for a readable with statement in each runner."""

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        """Close the shared writer without hiding an error from the study."""

        try:
            self.close()
        except TensorBoardMonitoringError:
            if exception is None:
                raise
        return False

    def _perform_write(self, action: Callable[[], None]) -> None:
        """Execute and flush one event group so disk failures stop the fit."""

        if self._closed:
            raise TensorBoardMonitoringError(
                "Cannot write to a closed TensorBoard study monitor"
            )
        try:
            action()
            self._writer.flush()
        except Exception as error:
            self._writer_failed = True
            raise TensorBoardMonitoringError(
                f"TensorBoard failed to write {self.log_directory}: {error}"
            ) from error

    def close(self) -> None:
        """Flush and close this writer exactly once."""

        if self._closed:
            return
        try:
            self._writer.flush()
            self._writer.close()
        except Exception as error:
            self._writer_failed = True
            raise TensorBoardMonitoringError(
                f"Cannot close TensorBoard writer {self.log_directory}: {error}"
            ) from error
        finally:
            self._closed = True

    def fit(self, context: TrainingRunContext) -> TensorBoardFitMonitor:
        """Return one tag-scoped monitor for a single fit inside this study."""

        if context.study_parts() != self._study_key:
            raise TensorBoardMonitoringError(
                "Fit context does not belong to this study's stage, family, "
                "and outer fold"
            )
        return TensorBoardFitMonitor(self, context)


class TensorBoardFitMonitor:
    """Scope one fit's tagged events onto its study's shared writer.

    Every public method mirrors what a dedicated per-fit writer used to
    provide, so model adapters and Step 5/6 runners see no behavioral
    difference beyond the writer being shared with sibling fits.
    """

    def __init__(
        self,
        study: TensorBoardStudyMonitor,
        context: TrainingRunContext,
    ) -> None:
        self.study = study
        self.context = context
        self.log_directory = study.log_directory
        self._prefix = context.tag_prefix()
        self._completed = False
        self._logged_training_steps: set[int] = set()

    def __enter__(self) -> TensorBoardFitMonitor:
        """Return this monitor for a readable with statement in each runner."""

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        """Record a failed fit or require a completion record, never both."""

        pending_error: BaseException | None = None
        if exception is not None:
            try:
                self._write_text("progress/error", str(exception), step=0)
                self._write_scalars({"progress/status": -1.0}, step=0)
            except TensorBoardMonitoringError as error:
                pending_error = error
        elif not self._completed:
            pending_error = TensorBoardMonitoringError(
                f"Monitored fit {self.context.configuration_id!r} exited without "
                "a completion record"
            )
        if exception is None and pending_error is not None:
            raise pending_error
        return False

    def _tag(self, tag: str) -> str:
        """Namespace one scalar or text tag under this fit's study prefix."""

        return f"{self._prefix}/{tag}"

    def _write_text(self, tag: str, text: str, *, step: int) -> None:
        """Write one text event through the shared study writer."""

        full_tag = self._tag(tag)
        self.study._perform_write(
            lambda: self.study._writer.add_text(full_tag, text, step)
        )

    def _write_scalars(self, values: Mapping[str, Any], *, step: int) -> None:
        """Write all finite values as one flushed, tag-prefixed event group."""

        normalized = {
            self._tag(tag): numeric
            for tag, value in values.items()
            if (numeric := _finite_scalar(value)) is not None
        }

        def write() -> None:
            for tag, value in normalized.items():
                self.study._writer.add_scalar(tag, value, step)

        self.study._perform_write(write)

    def start_fit(
        self,
        *,
        hyperparameters: Mapping[str, Any],
        training_data: Any,
        validation_data: Any | None,
    ) -> None:
        """Write static configuration and data dimensions before model fitting."""

        run_details = asdict(self.context)
        self._write_text("configuration/run", _json_text(run_details), step=0)
        self._write_text(
            "configuration/hyperparameters",
            _json_text(hyperparameters),
            step=0,
        )
        self._write_scalars(
            {
                "data/training_rows": len(training_data),
                "data/validation_rows": (
                    0 if validation_data is None else len(validation_data)
                ),
                "data/input_features": _input_feature_count(
                    training_data,
                    self.context.representation,
                ),
                "progress/status": 0.0,
            },
            step=0,
        )

    def log_training_step(
        self,
        *,
        step: int,
        scalars: Mapping[str, Any],
        force: bool = False,
    ) -> bool:
        """Record one real optimization step when its configured interval is due."""

        if step <= 0:
            raise TensorBoardMonitoringError(
                "Training progress steps must use positive one-based values"
            )
        interval = (
            XGBOOST_LOG_INTERVAL
            if self.context.model_family == "xgboost"
            else NEURAL_LOG_INTERVAL
        )
        if not force and step % interval != 0:
            return False
        if step in self._logged_training_steps:
            return False
        self._write_scalars(scalars, step=step)
        self._logged_training_steps.add(step)
        return True

    def create_xgboost_callback(
        self,
        progress_reporter: Callable[..., bool],
    ) -> Any:
        """Build a fresh stateful callback for exactly one boosted-tree fit."""

        # Import lazily so opening Phase 2 status or monitoring non-XGBoost
        # models does not initialize the XGBoost callback library.
        from .xgboost_callback import XGBoostProgressCallback

        return XGBoostProgressCallback(progress_reporter)

    def complete_fit(
        self,
        *,
        training_summary: Mapping[str, Any],
        inference_seconds: float,
        evaluation_metrics: Mapping[str, Any] | None,
        prediction_rows: int,
    ) -> None:
        """Record final fit facts and permitted performance values."""

        values: dict[str, Any] = {
            "timing/training_seconds": training_summary.get("training_seconds"),
            "timing/inference_seconds": inference_seconds,
            "model/epochs_or_iterations": training_summary.get(
                "epochs_or_iterations"
            ),
            "model/best_epoch_or_iteration": training_summary.get(
                "best_epoch_or_iteration"
            ),
            "model/trainable_parameters": training_summary.get(
                "trainable_parameters"
            ),
            "data/prediction_rows": prediction_rows,
            "progress/status": 1.0,
        }
        if evaluation_metrics is not None:
            values.update(
                {
                    f"performance/{name}": value
                    for name, value in evaluation_metrics.items()
                }
            )
        self._write_scalars(values, step=0)
        self._completed = True


def create_study_monitor(
    *,
    stage: Literal["step_5", "step_6"],
    model_family: str,
    outer_fold: int,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> TensorBoardStudyMonitor:
    """Create the mandatory shared writer used by one Step 5 or Step 6 study."""

    ensure_tensorboard_available()
    return TensorBoardStudyMonitor(
        stage=stage,
        model_family=model_family,
        outer_fold=outer_fold,
        log_root=log_root,
    )


def _write_one_shot_run(
    run_directory: Path,
    *,
    log_root: Path,
    scalars: Mapping[str, Any],
    step: int,
    text_values: Mapping[str, Mapping[str, Any]] | None = None,
    replace: bool,
    purge_step: int | None = None,
) -> None:
    """Write a small summary run without leaking writer logic to a pipeline step."""

    ensure_tensorboard_available()
    if replace:
        _replace_run_directory(run_directory, log_root)
    else:
        run_directory.mkdir(parents=True, exist_ok=True)
    writer_class = _summary_writer_class()
    writer: Any | None = None
    try:
        writer = writer_class(
            log_dir=str(run_directory),
            max_queue=1,
            flush_secs=WRITER_FLUSH_SECONDS,
            purge_step=purge_step,
        )
        for tag, value in scalars.items():
            numeric = _finite_scalar(value)
            if numeric is not None:
                writer.add_scalar(tag, numeric, step)
        for tag, value in (text_values or {}).items():
            writer.add_text(tag, _json_text(value), step)
        writer.flush()
        writer.close()
    except Exception as error:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        raise TensorBoardMonitoringError(
            f"Cannot write TensorBoard summary at {run_directory}: {error}"
        ) from error


def log_step_5_candidate(
    *,
    model_family: str,
    outer_fold: int,
    candidate_number: int,
    candidate_record: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    log_root: Path = DEFAULT_LOG_ROOT,
) -> None:
    """Append one completed tuning candidate to its study-level live curve."""

    run_directory = _safe_run_directory(
        log_root,
        (
            "step_5",
            model_family,
            f"outer_fold_{outer_fold:02d}",
            "study_progress",
        ),
    )
    _write_one_shot_run(
        run_directory,
        log_root=log_root,
        scalars={
            "candidate/mean_inner_rmse": candidate_record.get("mean_inner_rmse"),
            "candidate/inner_rmse_standard_deviation": candidate_record.get(
                "inner_rmse_standard_deviation"
            ),
            "candidate/mean_training_seconds": candidate_record.get(
                "mean_training_seconds"
            ),
            "candidate/total_training_seconds": candidate_record.get(
                "total_training_seconds"
            ),
            "candidate/mean_inference_seconds": candidate_record.get(
                "mean_inference_seconds"
            ),
            "candidate/outer_retraining_iterations": candidate_record.get(
                "outer_retraining_iterations"
            ),
        },
        step=candidate_number,
        text_values={
            f"configuration/candidate_{candidate_number:03d}": hyperparameters,
        },
        replace=candidate_number == 1,
        purge_step=None if candidate_number == 1 else candidate_number,
    )


def log_step_5_selection(
    *,
    model_family: str,
    outer_fold: int,
    selected_record: Mapping[str, Any],
    log_root: Path = DEFAULT_LOG_ROOT,
) -> None:
    """Mark the automatically selected candidate inside one model family."""

    candidate_number = int(selected_record["candidate_number"])
    run_directory = _safe_run_directory(
        log_root,
        (
            "step_5",
            model_family,
            f"outer_fold_{outer_fold:02d}",
            "study_progress",
        ),
    )
    _write_one_shot_run(
        run_directory,
        log_root=log_root,
        scalars={
            "selection/selected_candidate_number": candidate_number,
            "selection/selected_mean_inner_rmse": selected_record.get(
                "mean_inner_rmse"
            ),
        },
        step=candidate_number,
        text_values={"selection/configuration": selected_record},
        replace=False,
    )


def publish_step_7_comparison(
    architecture_comparison: Any,
    efficiency_summary: Any,
    grouped_architecture_metrics: Any | None = None,
    *,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> None:
    """Publish final locked metrics only after Step 7 has passed its gate."""

    efficiency_lookup = efficiency_summary.set_index("model_family")
    for row in architecture_comparison.to_dict("records"):
        family = str(row["model_family"])
        efficiency = efficiency_lookup.loc[family].to_dict()
        run_directory = _safe_run_directory(
            log_root,
            ("step_7", "final_comparison", family),
        )
        scalars: dict[str, Any] = {"progress/status": 1.0}
        for metric in ("rmse", "mae", "r2", "bias"):
            scalars[f"locked_performance/{metric}_mean"] = row.get(
                f"{metric}_mean"
            )
            scalars[f"locked_performance/{metric}_seed_sd"] = row.get(
                f"{metric}_seed_sd"
            )
            scalars[f"locked_uncertainty/{metric}_ci_lower_95"] = row.get(
                f"{metric}_ci_lower_95"
            )
            scalars[f"locked_uncertainty/{metric}_ci_upper_95"] = row.get(
                f"{metric}_ci_upper_95"
            )
        for name in (
            "training_seconds_mean_per_run",
            "training_seconds_total",
            "inference_seconds_mean_per_run",
            "inference_milliseconds_per_endpoint",
            "trainable_parameters_mean",
            "serialized_model_bytes_mean",
        ):
            scalars[f"efficiency/{name}"] = efficiency.get(name)
        if grouped_architecture_metrics is not None:
            age_rows = grouped_architecture_metrics.loc[
                (grouped_architecture_metrics["model_family"] == family)
                & (grouped_architecture_metrics["group_type"] == "age_band")
            ]
            for age_row in age_rows.to_dict("records"):
                label = str(age_row["group_value"]).replace(">", "over_")
                label = label.replace("-", "_").replace(" ", "")
                for metric in ("rmse", "mae", "r2", "bias"):
                    scalars[f"locked_age_band/{label}/{metric}_mean"] = (
                        age_row.get(f"{metric}_mean")
                    )
                    scalars[f"locked_age_band/{label}/{metric}_seed_sd"] = (
                        age_row.get(f"{metric}_seed_sd")
                    )
        _write_one_shot_run(
            run_directory,
            log_root=log_root,
            scalars=scalars,
            step=0,
            text_values={
                "comparison/architecture": {
                    "model_family": family,
                }
            },
            replace=True,
        )
