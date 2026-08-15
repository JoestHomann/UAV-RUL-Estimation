"""Register every architecture and construct adapters from resolved settings.

The registry connects contract family names to concrete Python classes. It does
not sample hyperparameters, run validation, compare architectures, or select a
winner. Later steps provide one resolved candidate configuration at a time.
"""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from base import ModelAdapter, ModelAdapterError
from neural_models import (
    LSTMAdapter,
    MLPAdapter,
    NeuralTrainingConfig,
    TCNAdapter,
    TransformerAdapter,
)
from tabular_models import (
    CycleOnlyBaselineAdapter,
    MeanBaselineAdapter,
    RandomForestAdapter,
    RBFSVRAdapter,
    RegularizedLinearAdapter,
    XGBoostAdapter,
)


STEP_DIR = Path(__file__).resolve().parent
DEFAULT_SPECIFICATION_PATH = (
    STEP_DIR.parent
    / "1_experiment_contract"
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
    """Read Step 1's resolved contract for registry and factory settings."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelAdapterError(
            "Cannot read the Step 1 experiment specification. Run the Step 1 "
            f"builder first. Details: {error}"
        ) from error
    try:
        contract = payload["contract"]
        architectures = contract["architectures"]
        contract["neural_training"]
        contract["evaluation"]
    except (KeyError, TypeError) as error:
        raise ModelAdapterError(
            f"Experiment specification is missing required field {error}"
        ) from error
    if set(architectures) != set(ADAPTER_CLASSES):
        raise ModelAdapterError(
            "Contract architecture names do not match implemented adapter names"
        )
    return payload


class ModelAdapterFactory:
    """Create one configured adapter without embedding model-specific branches.

    The factory reads shared neural settings, prediction clipping, and XGBoost
    stopping patience from the resolved contract. The caller supplies the
    candidate's resolved hyperparameters and seed.
    """

    def __init__(
        self,
        specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    ) -> None:
        self.specification_path = specification_path.resolve()
        self.specification = load_experiment_specification(specification_path)
        self.contract = self.specification["contract"]

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
        allow_disabled: bool = False,
    ) -> ModelAdapter:
        """Construct one model family from a fully resolved candidate."""

        if family not in ADAPTER_CLASSES:
            raise ModelAdapterError(f"Unknown model family {family!r}")
        if training_iterations is not None and training_iterations <= 0:
            raise ModelAdapterError("Fixed training iterations must be positive")
        architecture = self.contract["architectures"][family]
        if not architecture["enabled"] and not allow_disabled:
            raise ModelAdapterError(
                f"Model family {family!r} is disabled in the experiment contract"
            )
        self._validate_hyperparameters(family, hyperparameters)

        common_arguments: dict[str, Any] = {
            "hyperparameters": hyperparameters,
            "seed": seed,
            "prediction_minimum": self.contract["evaluation"]["prediction_minimum"],
        }
        adapter_class = ADAPTER_CLASSES[family]
        if family == "xgboost":
            return adapter_class(
                **common_arguments,
                early_stopping_patience=architecture["early_stopping_patience"],
                training_iterations=training_iterations,
            )
        if family in {"mlp", "tcn", "lstm", "transformer"}:
            shared = self.contract["neural_training"]
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
    """Describe installed adapters and their contract roles for later steps."""

    specification = load_experiment_specification(specification_path)
    contract = specification["contract"]
    families: dict[str, dict[str, Any]] = {}
    for family, architecture in contract["architectures"].items():
        adapter_class = ADAPTER_CLASSES[family]
        families[family] = {
            "adapter_class": adapter_class.__name__,
            "adapter_module": adapter_class.__module__,
            "implemented": True,
            "enabled": architecture["enabled"],
            "status": architecture["status"],
            "representation": architecture["representation"],
            "feature_sets": architecture["feature_sets"],
            "lookbacks": architecture["lookbacks"],
            "variants": architecture["variants"],
            "resolved_hyperparameters": sorted(EXPECTED_HYPERPARAMETERS[family]),
            "supports_sample_weights": True,
            "prediction_minimum": contract["evaluation"]["prediction_minimum"],
            "persistence": "trusted local joblib artifact",
        }

    return {
        "registry_version": 1,
        "contract_version": contract["contract_version"],
        "architecture_selection": "manual",
        "automatic_architecture_ranking": False,
        "common_methods": ["fit", "predict", "save", "load"],
        "libraries": {
            "joblib": version("joblib"),
            "scikit-learn": version("scikit-learn"),
            "torch": version("torch"),
            "xgboost": version("xgboost"),
        },
        "families": families,
    }
