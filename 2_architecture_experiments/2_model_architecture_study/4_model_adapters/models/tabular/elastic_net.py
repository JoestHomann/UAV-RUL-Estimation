"""Construct the Elastic Net estimator used by the regularized-linear family."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import ElasticNet


def create_elastic_net(
    hyperparameters: dict[str, Any],
    *,
    seed: int,
) -> ElasticNet:
    """Build Elastic Net with its resolved penalties and deterministic order."""

    return ElasticNet(
        alpha=float(hyperparameters["elastic_net_alpha"]),
        l1_ratio=float(hyperparameters["elastic_net_l1_ratio"]),
        max_iter=20_000,
        random_state=seed,
        selection="cyclic",
    )
