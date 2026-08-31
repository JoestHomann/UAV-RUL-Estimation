"""Keep all TensorBoard-specific behavior outside the numbered Phase 2 steps.

The training and evaluation runners send small dictionaries of scalar values
to this module. They never construct TensorBoard writers directly. This keeps
the experiment code readable while giving every fitted model a stable,
human-readable run path in the dashboard.

Generated event files are a live monitoring aid only. The CSV and JSON
artifacts written by Steps 5 through 7 remain the authoritative scientific
results, so this module deliberately publishes the smallest set of values that
answers "is this run alive and is it going anywhere":

- "train/loss" and "val/rmse" per optimization step, for one fit
- "search/candidate_rmse" per completed Step 5 candidate, for one study

Every other quantity that used to be written here -- final performance
metrics, age-band breakdowns, timings, row counts, parameter counts, progress
flags, and the whole Step 7 comparison -- is a terminal value that renders as
a single point and already exists in the authoritative artifacts. Reading it
from the CSV is better than reading it from an event file.

One writer is shared by every fit inside a study (one model family evaluated
on one outer fold) instead of opening a fresh writer per candidate or inner
fold. This keeps the generated directory count fixed at one study instead of
growing with the candidate budget and inner-fold count, while distinct tag
prefixes still let every individual fit's curves be selected separately in
TensorBoard.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Literal, Mapping

import numpy as np


MONITORING_DIR = Path(__file__).resolve().parent
PHASE_DIR = MONITORING_DIR.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from run_layout import (  # noqa: E402
    tensorboard_log_root_for_specification,
)

# Neural epochs are inexpensive to record and are the natural optimization
# unit. XGBoost can execute thousands of rounds per fit, so recording every
# tenth round prevents excessive event files while retaining a useful curve.
NEURAL_LOG_INTERVAL = 1
XGBOOST_LOG_INTERVAL = 10

# SummaryWriter performs asynchronous disk writes. A short flush interval keeps
# the browser current, while explicit flush calls surface write failures while
# the fit that caused them is still running.
WRITER_FLUSH_SECONDS = 5

# Step 5 evaluates (candidate budget x inner fold count) fits per study, so
# writing a curve for each one produces hundreds of tag prefixes in a single
# scalar panel. During a normal search the study-level candidate curve is the
# useful view, so per-fit curves are opt-in for debugging one architecture.
# Step 6 retrains are few and each one matters, so they always write curves.
FIT_CURVE_ENVIRONMENT_VARIABLE = "PHASE2_TENSORBOARD_FIT_CURVES"


class TensorBoardMonitoringError(RuntimeError):
    """Explain a missing dependency or an unsafe generated run path."""


def default_log_root() -> Path:
    """Return the current run's event directory, resolved when it is needed.

    Event files live inside the run folder they belong to, so run 2's curves
    can never overwrite run 1's and archiving a run takes its curves with it.
    This is a function rather than a module constant because the run number
    comes from Step 1's generated specification, which does not exist yet when
    this module is first imported.
    """

    return tensorboard_log_root_for_specification()


def step_5_fit_curves_enabled() -> bool:
    """Report whether Step 5 inner fits may write per-epoch curves."""

    return os.environ.get(FIT_CURVE_ENVIRONMENT_VARIABLE, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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

        Every scalar this fit writes is namespaced under this prefix inside
        the study's single shared writer, so TensorBoard's regex filter box
        (for example "^candidate_007/") recovers exactly what a dedicated
        per-fit directory used to provide.
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
            "2_architecture_experiments\2_model_architecture_study\\requirements.txt"
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
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _report_monitoring_failure(description: str, error: BaseException) -> None:
    """Warn about a monitoring failure without stopping the experiment.

    Monitoring is a convenience layer over the training that produces the real
    results. Losing an event file costs a live curve; aborting a multi-hour
    Step 6 retraining because a flush failed costs the run. The authoritative
    CSV and JSON artifacts are written by the pipeline steps themselves and are
    unaffected by anything that happens in this module.
    """

    print(
        f"TensorBoard monitoring disabled for {description}: {error}",
        file=sys.stderr,
        flush=True,
    )


def calculate_regression_metrics(
    targets: Any,
    predictions: Any,
) -> dict[str, float]:
    """Calculate shared accuracy and one-sided safety metrics.

    This is not a monitoring helper. Step 5 selects candidates on the RMSE
    this function returns, so it is a load-bearing calculation that happens to
    live beside the code that once displayed it.
    """

    observed = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if observed.shape != predicted.shape or not len(observed):
        raise TensorBoardMonitoringError(
            "Targets and predictions must be nonempty arrays with equal shapes"
        )
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise TensorBoardMonitoringError(
            "Cannot evaluate non-finite targets or predictions"
        )
    residual = predicted - observed
    overprediction = np.maximum(residual, 0.0)
    underprediction = np.maximum(-residual, 0.0)
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
        "overprediction_rate": float(np.mean(residual > 0.0)),
        "mean_overprediction": float(np.mean(overprediction)),
        "root_mean_squared_overprediction": float(
            np.sqrt(np.mean(np.square(overprediction)))
        ),
        "overprediction_q90": float(np.quantile(overprediction, 0.90)),
        "overprediction_q95": float(np.quantile(overprediction, 0.95)),
        "maximum_overprediction": float(np.max(overprediction)),
        "underprediction_rate": float(np.mean(residual < 0.0)),
        "mean_underprediction": float(np.mean(underprediction)),
        "root_mean_squared_underprediction": float(
            np.sqrt(np.mean(np.square(underprediction)))
        ),
    }


class TensorBoardStudyMonitor:
    """Own one shared SummaryWriter for every fit inside one study.

    A study is one model family evaluated on one outer fold: every candidate
    and inner fold in Step 5, or every retraining seed in Step 6. Sharing one
    writer across the whole study keeps the generated directory count fixed at
    one per study regardless of the candidate budget or inner-fold count, while
    ``TrainingRunContext.tag_prefix()`` keeps each fit's curves separately
    selectable inside that one run.

    A writer that cannot be created or written to disables monitoring for the
    remainder of the study instead of interrupting training.
    """

    def __init__(
        self,
        *,
        stage: Literal["step_5", "step_6"],
        model_family: str,
        outer_fold: int,
        log_root: Path | None = None,
    ) -> None:
        if log_root is None:
            log_root = default_log_root()
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
        # Step 6 always publishes curves. Step 5 does so only when an operator
        # is debugging one architecture, because a full search would otherwise
        # fill one scalar panel with a prefix per candidate and inner fold.
        self.fit_curves_enabled = stage == "step_6" or step_5_fit_curves_enabled()
        self._writer: Any | None = None
        self._closed = False

        if not self.fit_curves_enabled:
            return
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
            _report_monitoring_failure(str(self.log_directory), error)
            self._writer = None

    def __enter__(self) -> TensorBoardStudyMonitor:
        """Return this monitor for a readable with statement in each runner."""

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        """Close the shared writer without ever masking the study's own error."""

        self.close()
        return False

    def _perform_write(self, action: Callable[[], None]) -> None:
        """Execute and flush one event group, disabling monitoring on failure."""

        if self._writer is None or self._closed:
            return
        try:
            action()
            self._writer.flush()
        except Exception as error:
            _report_monitoring_failure(str(self.log_directory), error)
            self._writer = None

    def close(self) -> None:
        """Flush and close this writer exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._writer is None:
            return
        try:
            self._writer.flush()
            self._writer.close()
        except Exception as error:
            _report_monitoring_failure(str(self.log_directory), error)
        finally:
            self._writer = None

    def fit(self, context: TrainingRunContext) -> TensorBoardFitMonitor:
        """Return one tag-scoped monitor for a single fit inside this study."""

        if context.study_parts() != self._study_key:
            raise TensorBoardMonitoringError(
                "Fit context does not belong to this study's stage, family, "
                "and outer fold"
            )
        return TensorBoardFitMonitor(self, context)


class TensorBoardFitMonitor:
    """Scope one fit's optimization curves onto its study's shared writer.

    A fit publishes exactly two curves, "train/loss" and "val/rmse", and only
    while it is running. Everything a fit produced once it finished belongs to
    the owning step's own artifacts.
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
        """Leave the shared study writer open for the fits that follow."""

        return False

    def _write_scalars(self, values: Mapping[str, Any], *, step: int) -> None:
        """Write all finite values as one flushed, tag-prefixed event group."""

        writer = self.study._writer
        if writer is None:
            return
        normalized = {
            f"{self._prefix}/{tag}": numeric
            for tag, value in values.items()
            if (numeric := _finite_scalar(value)) is not None
        }
        if not normalized:
            return

        def write() -> None:
            for tag, value in normalized.items():
                writer.add_scalar(tag, value, step)

        self.study._perform_write(write)

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
        if not self.study.fit_curves_enabled or self.study._writer is None:
            return False
        interval = (
            XGBOOST_LOG_INTERVAL
            if self.context.model_family in {"xgboost", "catboost"}
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


def create_study_monitor(
    *,
    stage: Literal["step_5", "step_6"],
    model_family: str,
    outer_fold: int,
    log_root: Path | None = None,
) -> TensorBoardStudyMonitor:
    """Create the shared writer used by one Step 5 or Step 6 study."""

    ensure_tensorboard_available()
    if log_root is None:
        log_root = default_log_root()
    return TensorBoardStudyMonitor(
        stage=stage,
        model_family=model_family,
        outer_fold=outer_fold,
        log_root=log_root,
    )


def log_step_5_candidate(
    *,
    model_family: str,
    outer_fold: int,
    candidate_number: int,
    mean_inner_rmse: Any,
    hyperparameters: Mapping[str, Any],
    log_root: Path | None = None,
) -> None:
    """Append one completed tuning candidate to its study's search curve.

    This is the curve an operator actually watches during a Step 5 run: has
    the search improved, and has it plateaued. The accompanying text makes a
    point on that curve interpretable without opening the Step 5 CSV.
    """

    if log_root is None:
        log_root = default_log_root()
    run_directory = _safe_run_directory(
        log_root,
        (
            "step_5",
            model_family,
            f"outer_fold_{outer_fold:02d}",
            "study_progress",
        ),
    )
    try:
        ensure_tensorboard_available()
        if candidate_number == 1:
            _replace_run_directory(run_directory, log_root)
        else:
            run_directory.mkdir(parents=True, exist_ok=True)
        writer_class = _summary_writer_class()
        writer = writer_class(
            log_dir=str(run_directory),
            max_queue=1,
            flush_secs=WRITER_FLUSH_SECONDS,
            # A repeated candidate number replaces its earlier point instead
            # of appending a second one, so a resumed study stays readable.
            purge_step=None if candidate_number == 1 else candidate_number,
        )
    except Exception as error:
        _report_monitoring_failure(str(run_directory), error)
        return

    try:
        rmse = _finite_scalar(mean_inner_rmse)
        if rmse is not None:
            writer.add_scalar("search/candidate_rmse", rmse, candidate_number)
        writer.add_text(
            f"search/candidate_{candidate_number:03d}",
            _json_text(hyperparameters),
            candidate_number,
        )
        writer.flush()
    except Exception as error:
        _report_monitoring_failure(str(run_directory), error)
    finally:
        try:
            writer.close()
        except Exception:
            pass

import time

def log_global_progress(
    step_number: int,
    model_family: str,
    completed_count: int,
    log_root: Path | None = None,
) -> None:
    """Log the total completed outer folds across an entire step."""
    if log_root is None:
        log_root = default_log_root()
        
    run_directory = _safe_run_directory(
        log_root,
        ("global_progress", f"step_{step_number}", model_family),
    )
    
    try:
        ensure_tensorboard_available()
        run_directory.mkdir(parents=True, exist_ok=True)
        writer_class = _summary_writer_class()
        
        # Open and close the writer quickly to ensure it flushes
        writer = writer_class(
            log_dir=str(run_directory),
            max_queue=1,
            flush_secs=1,
        )
        
        # We use time.time() so the X-axis is absolute wall-clock time.
        writer.add_scalar(
            "completed_outer_folds", 
            completed_count, 
            global_step=int(time.time())
        )
        writer.close()
    except Exception as error:
        _report_monitoring_failure(str(run_directory), error)
