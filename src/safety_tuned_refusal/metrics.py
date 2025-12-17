"""
Bootstrap utilities for estimating safety rates.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


def bootstrap_proportion_ci(
    values: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap a proportion and return (lower, upper) CI bounds.
    """
    arr = pd.Series(values).dropna().to_numpy()
    if arr.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    boot_means = samples.mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))
    return lower, upper


def compute_identity_safety_rates(
    df: pd.DataFrame,
    *,
    risk_level: str | None = None,
    group_cols: Sequence[str] = ("axis_id", "identity_id", "identity_phrase"),
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Estimate P(safety) for each identity with bootstrap CIs.

    Args:
        df: DataFrame with safety_binary and identity columns.
        risk_level: Optional risk_level filter ("safety", "benign", or None for all).
        group_cols: Columns to group by when aggregating identities.
    """
    sub = df if risk_level is None else df[df["risk_level"] == risk_level]
    if sub.empty:
        return pd.DataFrame(
            columns=[*group_cols, "risk_level", "n", "p_safety", "ci_lower", "ci_upper"]
        )

    results = []
    for keys, group in sub.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["safety_binary"].dropna().to_numpy()
        if values.size == 0:
            continue
        p_hat = float(values.mean())
        ci_lower, ci_upper = bootstrap_proportion_ci(
            values,
            n_bootstrap=n_bootstrap,
            ci=ci,
            random_state=random_state,
        )
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "risk_level": risk_level or "all",
                "n": int(values.size),
                "p_safety": p_hat,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
        results.append(row)

    return pd.DataFrame(results)


__all__ = [
    "bootstrap_proportion_ci",
    "compute_identity_safety_rates",
]
