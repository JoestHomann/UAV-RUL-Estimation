"""Expose the small public monitoring interface used by Phase 2.

TensorBoard itself is imported lazily inside "monitoring.py". Importing this
package therefore does not initialize a writer, create a log file, or start a
dashboard server. The numbered pipeline steps use only the names exported here.
"""

from .monitoring import (
    DEFAULT_LOG_ROOT,
    TensorBoardMonitoringError,
    TrainingRunContext,
    calculate_age_band_regression_metrics,
    calculate_regression_metrics,
    create_training_monitor,
    ensure_tensorboard_available,
    log_step_5_candidate,
    log_step_5_selection,
    publish_step_7_comparison,
)


__all__ = [
    "DEFAULT_LOG_ROOT",
    "TensorBoardMonitoringError",
    "TrainingRunContext",
    "calculate_age_band_regression_metrics",
    "calculate_regression_metrics",
    "create_training_monitor",
    "ensure_tensorboard_available",
    "log_step_5_candidate",
    "log_step_5_selection",
    "publish_step_7_comparison",
]
