"""Resolve contract search definitions into model-ready Optuna candidates.

This module owns only candidate construction. It does not load data, fit a
model, calculate a score, or choose a candidate. Keeping these responsibilities
separate makes it possible to inspect how every contract field becomes an
actual adapter configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from optuna.trial import Trial


class CandidateSpaceError(ValueError):
    """Represent an unsupported or inconsistent contract search definition."""


@dataclass(frozen=True)
class ResolvedCandidate:
    """Store one complete representation and model configuration."""

    feature_set: str | None
    lookback: int | None
    hyperparameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration object."""

        return {
            "feature_set": self.feature_set,
            "lookback": self.lookback,
            "hyperparameters": self.hyperparameters,
        }

    def canonical_json(self) -> str:
        """Return stable text used to identify duplicate resolved candidates."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


class CandidateSpace:
    """Translate one architecture's contract entry into Optuna suggestions."""

    SUPPORTED_KINDS = {
        "fixed",
        "categorical",
        "categorical_integer_sequences",
        "uniform",
        "log_uniform",
    }

    def __init__(self, architectures: dict[str, dict[str, Any]]) -> None:
        self.architectures = architectures

    def candidate_budget(self, family: str, maximum_budget: int) -> int:
        """Return one trial for a fixed baseline and the contract cap otherwise."""

        architecture = self._architecture(family)
        alternatives = max(
            len(architecture["feature_sets"]),
            len(architecture["lookbacks"]),
            1,
        )
        has_choices = bool(architecture["search"]) or alternatives > 1
        return maximum_budget if has_choices else 1

    def resolve(self, family: str, trial: Trial) -> ResolvedCandidate:
        """Resolve representation choices and every adapter hyperparameter."""

        architecture = self._architecture(family)
        feature_set = self._representation_choice(
            trial,
            name="representation__feature_set",
            values=architecture["feature_sets"],
        )
        lookback_value = self._representation_choice(
            trial,
            name="representation__lookback",
            values=architecture["lookbacks"],
        )
        lookback = int(lookback_value) if lookback_value is not None else None

        if family == "regularized_linear":
            hyperparameters = self._regularized_linear_parameters(
                trial,
                architecture["search"],
            )
        else:
            hyperparameters = {
                name: self._suggest_value(trial, name, definition)
                for name, definition in architecture["search"].items()
            }
        return ResolvedCandidate(feature_set, lookback, hyperparameters)

    def _architecture(self, family: str) -> dict[str, Any]:
        """Return one architecture and reject incomplete registry names."""

        try:
            architecture = self.architectures[family]
        except KeyError as error:
            message = f"Unknown architecture family {family!r}"
            raise CandidateSpaceError(message) from error
        required = {"feature_sets", "lookbacks", "search", "representation"}
        missing = sorted(required - set(architecture))
        if missing:
            raise CandidateSpaceError(
                f"Architecture {family!r} is missing fields {missing}"
            )
        return architecture

    @staticmethod
    def _representation_choice(
        trial: Trial,
        *,
        name: str,
        values: list[Any],
    ) -> Any | None:
        """Return no choice, one fixed choice, or one sampled alternative."""

        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return trial.suggest_categorical(name, values)

    def _regularized_linear_parameters(
        self,
        trial: Trial,
        search: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Sample only parameters active for the chosen linear variant.

        The Step 4 factory expects all four declared keys. Inactive values are
        therefore filled with deterministic contract defaults, while Optuna
        spends its candidate budget only on parameters that affect the model.
        """

        if "variant" not in search:
            raise CandidateSpaceError(
                "Regularized linear search does not define its variant"
            )
        variant = self._suggest_value(trial, "variant", search["variant"])
        resolved: dict[str, Any] = {"variant": variant}
        for name, definition in search.items():
            if name == "variant":
                continue
            inactive = (
                variant == "ridge" and name.startswith("elastic_net_")
            ) or (
                variant == "elastic_net" and name == "ridge_alpha"
            )
            resolved[name] = (
                self._default_value(definition)
                if inactive
                else self._suggest_value(trial, name, definition)
            )
        return resolved

    def _suggest_value(
        self,
        trial: Trial,
        name: str,
        definition: dict[str, Any],
    ) -> Any:
        """Map one validated contract distribution to the current Optuna API."""

        kind = definition.get("kind")
        if kind not in self.SUPPORTED_KINDS:
            raise CandidateSpaceError(
                f"Parameter {name!r} has unsupported search kind {kind!r}"
            )
        if kind == "fixed":
            return definition["value"]
        if kind == "categorical":
            return trial.suggest_categorical(name, definition["values"])
        if kind == "categorical_integer_sequences":
            values = definition["values"]
            choice_index = trial.suggest_categorical(
                f"{name}__choice",
                list(range(len(values))),
            )
            return list(values[int(choice_index)])
        if kind == "uniform":
            return trial.suggest_float(
                name,
                float(definition["low"]),
                float(definition["high"]),
            )
        return trial.suggest_float(
            name,
            float(definition["low"]),
            float(definition["high"]),
            log=True,
        )

    @staticmethod
    def _default_value(definition: dict[str, Any]) -> Any:
        """Choose a deterministic valid value for an inactive parameter."""

        kind = definition.get("kind")
        if kind == "fixed":
            return definition["value"]
        if kind in {"categorical", "categorical_integer_sequences"}:
            value = definition["values"][0]
            return list(value) if isinstance(value, list) else value
        if kind in {"uniform", "log_uniform"}:
            return definition["low"]
        raise CandidateSpaceError(
            f"Cannot choose a default for unsupported search kind {kind!r}"
        )
