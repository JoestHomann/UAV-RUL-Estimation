"""Construct the Ridge estimator used by the regularized-linear family."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import Ridge


def create_ridge(hyperparameters: dict[str, Any]) -> Ridge:
    """Build Ridge with the contract-resolved penalty and stable LSQR solver."""

    return Ridge(
        alpha=float(hyperparameters["ridge_alpha"]),
        # Engineered features contain strongly redundant columns. LSQR avoids
        # directly solving an ill-conditioned normal equation.
        solver="lsqr",
        tol=1e-6,
        max_iter=20_000,
    )
