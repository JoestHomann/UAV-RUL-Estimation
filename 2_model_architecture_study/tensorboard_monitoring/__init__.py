"""Expose the small public monitoring interface used by Phase 2.

TensorBoard itself is imported lazily inside "monitoring.py". Importing this
package therefore does not initialize a writer, create a log file, or start a
dashboard server. The numbered pipeline steps use only the names exported here.
"""

from .monitoring import (
    FIT_CURVE_ENVIRONMENT_VARIABLE,
    TensorBoardMonitoringError,
    TrainingRunContext,
    calculate_regression_metrics,
    create_study_monitor,
    default_log_root,
    ensure_tensorboard_available,
    log_step_5_candidate,
    step_5_fit_curves_enabled,
    log_global_progress,
)


__all__ = [
    "FIT_CURVE_ENVIRONMENT_VARIABLE",
    "TensorBoardMonitoringError",
    "TrainingRunContext",
    "calculate_regression_metrics",
    "create_study_monitor",
    "default_log_root",
    "ensure_tensorboard_available",
    "log_step_5_candidate",
    "step_5_fit_curves_enabled",
    "log_global_progress",
]
