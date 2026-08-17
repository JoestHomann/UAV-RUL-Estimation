"""Register every architecture and construct adapters from resolved settings.

The registry connects settings family names to concrete Python classes. It does
not sample hyperparameters, run validation, compare architectures, or select a
winner. Later steps provide one resolved candidate configuration at a time.
"""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from base import ModelAdapter, ModelAdapterError
from models.baselines.cycle_only_baseline import CycleOnlyBaselineAdapter
from models.baselines.mean_baseline import MeanBaselineAdapter
from models.neural.lstm import LSTMAdapter
from models.neural.mlp import MLPAdapter
from models.neural.neural_base import NeuralTrainingConfig
from models.neural.tcn import TCNAdapter
from models.neural.transformer import TransformerAdapter
from models.tabular.random_forest import RandomForestAdapter
from models.tabular.rbf_svr import RBFSVRAdapter
from models.tabular.regularized_linear import RegularizedLinearAdapter
from models.tabular.xgboost import XGBoostAdapter


STEP_DIR = Path(__file__).resolve().parent
DEFAULT_SPECIFICATION_PATH = (
    STEP_DIR.parent
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)


ADAPTER_CLASSES: dict[str, type[ModelAdapter]] = {
    "mean_baseline": MeanBaselineAdapter,
    "cycle_only_baseline": CycleOnlyBaselineAdapter,
    "regularized_linear": RegularizedLinearAdapter,
    "random_forest": RandomForestAdapter,
    "xgboost": XGBoostAdapter,
    "mlp": MLPAdapter,
    "tcn": TCNAdapter,
    "lstm": LSTMAdapter,
    "transformer": TransformerAdapter,
    "rbf_svr": RBFSVRAdapter,
}


# Exact parameter-name checks catch misspelled or incomplete resolved candidates
# before a third-party estimator silently accepts an unintended configuration.
EXPECTED_HYPERPARAMETERS: dict[str, set[str]] = {
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
}


def load_experiment_specification(
    path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict[str, Any]:
    """Read Step 1's resolved settings for registry and factory settings."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelAdapterError(
            "Cannot read the Step 1 experiment specification. Run the Step 1 "
            f"builder first. Details: {error}"
        ) from error
    try:
        settings = payload["settings"]
        architectures = settings["architectures"]
        settings["neural_training"]
        settings["evaluation"]
    except (KeyError, TypeError) as error:
        raise ModelAdapterError(
            f"Experiment specification is missing required field {error}"
        ) from error
    if set(architectures) != set(ADAPTER_CLASSES):
        raise ModelAdapterError(
            "Settings architecture names do not match implemented adapter names"
        )
    return payload


class ModelAdapterFactory:
    """Create one configured adapter without embedding model-specific branches.

    The factory reads shared neural settings, prediction clipping, and XGBoost
    stopping patience from the resolved settings. The caller supplies the
    candidate's resolved hyperparameters and seed.
    """

    def __init__(
        self,
        specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    ) -> None:
        self.specification_path = specification_path.resolve()
        self.specification = load_experiment_specification(specification_path)
        self.settings = self.specification["settings"]

    def _validate_hyperparameters(
        self,
        family: str,
        hyperparameters: dict[str, Any],
    ) -> None:
        observed = set(hyperparameters)
        expected = EXPECTED_HYPERPARAMETERS[family]
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ModelAdapterError(
                f"Resolved hyperparameters for {family!r} are invalid: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def create(
        self,
        family: str,
        hyperparameters: dict[str, Any],
        *,
        seed: int,
        training_iterations: int | None = None,
        training_monitor: Any | None = None,
        allow_disabled: bool = False,
    ) -> ModelAdapter:
        """Construct one model family from a fully resolved candidate."""

        if family not in ADAPTER_CLASSES:
            raise ModelAdapterError(f"Unknown model family {family!r}")
        if training_iterations is not None and training_iterations <= 0:
            raise ModelAdapterError("Fixed training iterations must be positive")
        architecture = self.settings["architectures"][family]
        if not self.settings["study"]["enabled"][family] and not allow_disabled:
            raise ModelAdapterError(
                f"Model family {family!r} is disabled in the "
                "architecture study settings"
            )
        self._validate_hyperparameters(family, hyperparameters)

        common_arguments: dict[str, Any] = {
            "hyperparameters": hyperparameters,
            "seed": seed,
            "prediction_minimum": self.settings["evaluation"]["prediction_minimum"],
            "training_monitor": training_monitor,
        }
        adapter_class = ADAPTER_CLASSES[family]
        if family == "xgboost":
            return adapter_class(
                **common_arguments,
                early_stopping_patience=architecture["early_stopping_patience"],
                training_iterations=training_iterations,
            )
        if family in {"mlp", "tcn", "lstm", "transformer"}:
            shared = self.settings["neural_training"]
            training_config = NeuralTrainingConfig(
                batch_size=shared["batch_size"],
                maximum_epochs=shared["maximum_epochs"],
                early_stopping_patience=shared["early_stopping_patience"],
                gradient_clip_global_norm=shared["gradient_clip_global_norm"],
            )
            return adapter_class(
                **common_arguments,
                training_config=training_config,
                training_epochs=training_iterations,
            )
        if training_iterations is not None:
            raise ModelAdapterError(
                f"Model family {family!r} does not accept fixed training iterations"
            )
        return adapter_class(**common_arguments)


def build_registry_payload(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict[str, Any]:
    """Describe installed adapters and their settings roles for later steps."""

    specification = load_experiment_specification(specification_path)
    settings = specification["settings"]
    families: dict[str, dict[str, Any]] = {}
    for family, architecture in settings["architectures"].items():
        adapter_class = ADAPTER_CLASSES[family]
        families[family] = {
            "adapter_class": adapter_class.__name__,
            "adapter_module": adapter_class.__module__,
            "implemented": True,
            "enabled": settings["study"]["enabled"][family],
            "status": architecture["status"],
            "representation": architecture["representation"],
            "feature_sets": architecture["feature_sets"],
            "lookbacks": architecture["lookbacks"],
            "variants": architecture["variants"],
            "resolved_hyperparameters": sorted(EXPECTED_HYPERPARAMETERS[family]),
            "supports_sample_weights": True,
            "stochastic": adapter_class.stochastic,
            "prediction_minimum": settings["evaluation"]["prediction_minimum"],
            "persistence": "trusted local joblib artifact",
        }

    return {
        "registry_version": 1,
        "settings_version": settings["settings_version"],
        "architecture_selection": "manual",
        "automatic_architecture_ranking": False,
        "common_methods": ["fit", "predict", "save", "load"],
        "libraries": {
            "joblib": version("joblib"),
            "scikit-learn": version("scikit-learn"),
            "tensorboard": version("tensorboard"),
            "torch": version("torch"),
            "xgboost": version("xgboost"),
        },
        "training_monitoring": "mandatory TensorBoard event logging",
        "families": families,
    }
