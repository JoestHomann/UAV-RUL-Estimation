from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPOSITORY_ROOT / "2_model_architecture_study" / "4_model_adapters"
TABULAR_ADAPTER_ROOT = (
    REPOSITORY_ROOT / "2_model_architecture_study" / "2_tabular_data_adapter"
)
MONITORING_ROOT = REPOSITORY_ROOT / "2_model_architecture_study" / "tensorboard_monitoring"
for path in (ADAPTER_ROOT, TABULAR_ADAPTER_ROOT, MONITORING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.tabular.fold_fitted_transforms import (  # noqa: E402
    FaultModeTransformer,
    SignalCompressionTransformer,
)
from models.tabular.extra_trees import ExtraTreesAdapter  # noqa: E402
from models.tabular.hist_gradient_boosting import (  # noqa: E402
    HistGradientBoostingAdapter,
)
from policies import PolicyError, PredictionPolicy, TargetPolicy  # noqa: E402
from tabular_data_adapter import TabularDataset  # noqa: E402
from models.tabular.xgboost import (  # noqa: E402
    SeverityAsymmetricObjective,
    XGBoostAdapter,
)
from xgboost_callback import XGBoostProgressCallback  # noqa: E402


class NullTrainingMonitor:
    def create_xgboost_callback(self, progress_reporter):
        return XGBoostProgressCallback(progress_reporter)

    def log_training_step(self, *, step, scalars, force=False):
        return False


def signal_feature_names() -> tuple[str, ...]:
    names = ["feature__flight_cycle", "feature__log1p_flight_cycle"]
    names.extend(f"feature__telemetry_{number:02d}__last" for number in range(1, 23))
    names.extend(
        f"feature__telemetry_{number:02d}__degradation_score"
        for number in (7, 13, 15, 16, 19, 21, 22, 23, 25, 28)
    )
    return tuple(names)


class TargetPolicyTests(unittest.TestCase):
    def test_failure_cycle_round_trip_restores_raw_rul(self) -> None:
        policy = TargetPolicy.from_settings({"mode": "failure_cycle"})
        rul = np.asarray([90.0, 30.0, 5.0])
        cutoffs = np.asarray([10.0, 70.0, 95.0])
        transformed = policy.transform(rul, cutoffs)
        np.testing.assert_allclose(transformed, [100.0, 100.0, 100.0])
        np.testing.assert_allclose(
            policy.inverse_predictions(transformed, cutoffs),
            rul,
        )

    def test_failure_cycle_requires_aligned_cutoffs(self) -> None:
        policy = TargetPolicy.from_settings({"mode": "failure_cycle"})
        with self.assertRaises(PolicyError):
            policy.transform(np.asarray([1.0, 2.0]), None)

    def test_severity_loss_penalty_accelerates_only_for_overprediction(self) -> None:
        policy = PredictionPolicy.from_settings(
            {
                "loss": "severity_asymmetric_mse",
                "overprediction_weight": 2.0,
                "quantile": 0.5,
                "severity_scale": 10.0,
                "calibration": "none",
                "safety_offset": 0.0,
                "non_overprediction_coverage": 0.5,
            }
        )
        losses = policy.numpy_losses(
            np.zeros(4),
            np.asarray([-10.0, -5.0, 5.0, 10.0]),
        )
        np.testing.assert_allclose(losses[:2], [100.0, 25.0])
        np.testing.assert_allclose(losses[2:], [37.5, 200.0])

    def test_severity_xgboost_objective_has_positive_hessian(self) -> None:
        objective = SeverityAsymmetricObjective(2.0, 10.0)
        gradient, hessian = objective(
            np.zeros(3),
            np.asarray([-5.0, 0.0, 10.0]),
        )
        np.testing.assert_allclose(gradient, [-10.0, 0.0, 50.0])
        np.testing.assert_allclose(hessian, [2.0, 2.0, 8.0])


class FoldFittedTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = signal_feature_names()
        rng = np.random.default_rng(13)
        self.values = rng.normal(size=(40, len(self.names)))
        score_start = len(self.names) - 10
        self.values[20:, score_start:] += 4.0
        self.metadata = pd.DataFrame(
            {
                "uav_id": [f"UAV_{index // 2:03d}" for index in range(40)],
                "cutoff": [10, 20] * 20,
            }
        )

    def test_signal_compression_pca_is_fitted_and_reused(self) -> None:
        transformer = SignalCompressionTransformer("pca_only")
        transformed, names = transformer.fit_transform(self.values, self.names)
        repeated, repeated_names = transformer.transform(self.values, self.names)
        self.assertEqual(transformed.shape[1], 28)
        self.assertEqual(names, repeated_names)
        np.testing.assert_allclose(transformed, repeated)

    def test_disabled_compression_accepts_any_tabular_columns(self) -> None:
        transformer = SignalCompressionTransformer("none")
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        names = ("feature__one", "feature__two")
        transformed, transformed_names = transformer.fit_transform(values, names)
        np.testing.assert_array_equal(transformed, values)
        self.assertEqual(transformed_names, names)

    def test_fault_modes_are_fitted_from_training_uavs(self) -> None:
        transformer = FaultModeTransformer("indicator", seed=13)
        assignments = transformer.fit(self.values, self.names, self.metadata)
        repeated = transformer.assign(self.values, self.names)
        self.assertEqual(set(assignments.modes), {0, 1})
        np.testing.assert_array_equal(assignments.modes, repeated.modes)
        augmented, names = transformer.append_indicator(
            self.values,
            self.names,
            assignments,
        )
        self.assertEqual(augmented.shape[1], self.values.shape[1] + 1)
        self.assertEqual(names[-1], "derived__fault_mode")

    def test_extra_trees_executes_experts_and_compression_strategies(self) -> None:
        dataset = TabularDataset(
            features=pd.DataFrame(self.values, columns=self.names),
            metadata=self.metadata,
            target=pd.Series(120.0 - self.metadata["cutoff"].to_numpy()),
            sample_weights=pd.Series(np.ones(len(self.values))),
        )
        base = {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_leaf": 1,
            "max_features": 0.67,
        }
        for fault_strategy, compression_strategy in (
            ("experts", "none"),
            ("none", "pca_only"),
        ):
            adapter = ExtraTreesAdapter(
                hyperparameters={
                    **base,
                    "fault_mode_strategy": fault_strategy,
                    "signal_compression_strategy": compression_strategy,
                },
                seed=13,
            )
            adapter.fit(dataset, None)
            predictions = adapter.predict(dataset)
            self.assertEqual(len(predictions), len(dataset))
            self.assertTrue(np.isfinite(predictions).all())

    def test_failure_cycle_policy_is_applied_by_adapter(self) -> None:
        dataset = TabularDataset(
            features=pd.DataFrame(self.values, columns=self.names),
            metadata=self.metadata,
            target=pd.Series(120.0 - self.metadata["cutoff"].to_numpy()),
            sample_weights=pd.Series(np.ones(len(self.values))),
        )
        adapter = ExtraTreesAdapter(
            hyperparameters={
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_leaf": 1,
                "max_features": 0.67,
                "fault_mode_strategy": "none",
                "signal_compression_strategy": "none",
            },
            seed=13,
        )
        adapter.configure_policies(
            TargetPolicy.from_settings({"mode": "failure_cycle"}),
            PredictionPolicy(),
        )
        adapter.fit(dataset, None)
        np.testing.assert_allclose(adapter.predict(dataset), dataset.target)

    def test_hist_gradient_boosting_fits_weighted_tabular_rows(self) -> None:
        dataset = TabularDataset(
            features=pd.DataFrame(self.values, columns=self.names),
            metadata=self.metadata,
            target=pd.Series(120.0 - self.metadata["cutoff"].to_numpy()),
            sample_weights=pd.Series(np.linspace(0.5, 1.5, len(self.values))),
        )
        adapter = HistGradientBoostingAdapter(
            hyperparameters={
                "max_iter": 20,
                "learning_rate": 0.1,
                "max_leaf_nodes": 7,
                "max_depth": 3,
                "min_samples_leaf": 5,
                "l2_regularization": 1.0,
            },
            seed=13,
        )
        summary = adapter.fit(dataset, dataset)
        predictions = adapter.predict(dataset)
        self.assertEqual(summary.model_family, "hist_gradient_boosting")
        self.assertEqual(summary.epochs_or_iterations, 20)
        self.assertEqual(len(predictions), len(dataset))
        self.assertTrue(np.isfinite(predictions).all())

    def test_xgboost_executes_fold_fitted_experts(self) -> None:
        dataset = TabularDataset(
            features=pd.DataFrame(self.values, columns=self.names),
            metadata=self.metadata,
            target=pd.Series(120.0 - self.metadata["cutoff"].to_numpy()),
            sample_weights=pd.Series(np.ones(len(self.values))),
        )
        adapter = XGBoostAdapter(
            hyperparameters={
                "maximum_trees": 3,
                "learning_rate": 0.1,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "fault_mode_strategy": "experts",
                "signal_compression_strategy": "none",
            },
            seed=13,
            training_monitor=NullTrainingMonitor(),
            device="cpu",
        )
        adapter.fit(dataset, None)
        predictions = adapter.predict(dataset)
        self.assertEqual(len(predictions), len(dataset))
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
