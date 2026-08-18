"""Expose the small public monitoring interface used by Phase 2.

TensorBoard itself is imported lazily inside "monitoring.py". Importing this
package therefore does not initialize a writer, create a log file, or start a
dashboard server. The numbered pipeline steps use only the names exported here.
"""

from .monitoring import (
    DEFAULT_LOG_ROOT,
    FIT_CURVE_ENVIRONMENT_VARIABLE,
    TensorBoardMonitoringError,
    TrainingRunContext,
    calculate_regression_metrics,
    create_study_monitor,
    ensure_tensorboard_available,
    log_step_5_candidate,
    step_5_fit_curves_enabled,
)


__all__ = [
    "DEFAULT_LOG_ROOT",
    "FIT_CURVE_ENVIRONMENT_VARIABLE",
    "TensorBoardMonitoringError",
    "TrainingRunContext",
    "calculate_regression_metrics",
    "create_study_monitor",
    "ensure_tensorboard_available",
    "log_step_5_candidate",
    "step_5_fit_curves_enabled",
]
