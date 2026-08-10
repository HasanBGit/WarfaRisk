"""Phase 5 — calibration and uncertainty.

Implements TRAINING_PLAN.md Phase 5: wraps the Phase 3 winning model in
conformal prediction (MAPIE), compares against NGBoost's directly-trained
predictive distribution, and a deep ensemble scored with uncertainty-toolbox
— three mechanistically distinct approaches to the same question, so the
paper can report which one actually wins on this data rather than picking one
by default.

Coverage must be computed per ancestry subgroup (common.metrics.
stratified_conformal_coverage), not just overall — Phase 4 already shows
performance isn't uniform across ancestry groups, so calibration might not be
either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from warfarisk.common.metrics import stratified_conformal_coverage
from warfarisk.common.preprocessing import with_standard_preprocessing


def mapie_conformal_interval(
    base_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wraps any already-built sklearn-compatible pipeline (e.g. Phase 3's
    winner) in MAPIE's conformal regression interval. alpha=0.1 -> a 90%
    interval, matching the review's own calibration-slope framing. Returns
    (point_prediction, lower, upper).

    Requires mapie>=1.4 specifically, not just "mapie" — confirmed 2026-07-26
    that mapie==1.0.1 (the version this project's pyproject.toml originally
    pinned) is incompatible with scikit-learn>=1.6: MAPIE 1.0.1's internal
    `EnsembleRegressor` class doesn't implement `__sklearn_tags__`, which
    newer scikit-learn's `check_is_fitted` requires, raising
    `AttributeError: 'EnsembleRegressor' object has no attribute
    '__sklearn_tags__'` from deep inside MAPIE's own code (not this
    project's model). mapie==1.4.1 has the fix. MAPIE's public API also
    changed across this version gap: the old `MapieRegressor(...).fit(X,
    y)` + `.predict(X, alpha=...)` one-shot pattern is gone, replaced by
    `CrossConformalRegressor(...).fit_conformalize(X, y)` +
    `.predict_interval(X)`.
    """
    try:
        from mapie.regression import CrossConformalRegressor
    except ImportError as e:
        raise ImportError("MAPIE (>=1.4) is required. Install with: uv add 'mapie>=1.4'") from e

    mapie_model = CrossConformalRegressor(estimator=base_estimator, confidence_level=1 - alpha, method="plus", cv=5)
    mapie_model.fit_conformalize(X_train, y_train)
    y_pred = mapie_model.predict(X_test)
    _point_again, y_interval = mapie_model.predict_interval(X_test)
    lower = y_interval[:, 0, 0]
    upper = y_interval[:, 1, 0]
    return y_pred, lower, upper


def ngboost_predictive_distribution(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NGBoost trains the predictive distribution directly (natural-gradient
    boosting) rather than wrapping a point estimate post-hoc — a genuinely
    different mechanism from MAPIE, worth comparing rather than assuming one
    approach is right. Categorical columns must be pre-encoded by the caller
    (NGBoost's sklearn-style API expects numeric input; wrap with
    common.preprocessing.with_standard_preprocessing's ColumnTransformer if
    calling this directly on raw IWPC columns).
    """
    try:
        from ngboost import NGBRegressor
        from scipy.stats import norm
    except ImportError as e:
        raise ImportError("NGBoost is required. Install with: uv add ngboost") from e

    model = NGBRegressor(random_state=0)
    model.fit(X_train, y_train)
    y_dist = model.pred_dist(X_test)
    y_pred = y_dist.mean()
    z = norm.ppf(1 - alpha / 2)
    lower = y_pred - z * y_dist.scale
    upper = y_pred + z * y_dist.scale
    return y_pred, lower, upper


def deep_ensemble_interval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_members: int = 5,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five MLPs with different seeds — no new dependency needed, this is the
    third, cheapest uncertainty-quantification mechanism (ensemble spread
    instead of conformal residuals or a trained distribution)."""
    from scipy.stats import norm

    predictions = []
    for seed in range(n_members):
        pipeline = with_standard_preprocessing(MLPRegressor(random_state=seed, hidden_layer_sizes=(64, 32), max_iter=2000))
        pipeline.fit(X_train, y_train)
        predictions.append(pipeline.predict(X_test))
    predictions = np.stack(predictions, axis=0)
    y_pred = predictions.mean(axis=0)
    y_std = predictions.std(axis=0)
    z = norm.ppf(1 - alpha / 2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return y_pred, lower, upper


def score_all_methods(
    y_test: pd.Series,
    group: pd.Series,
    mapie_bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ngboost_bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    deep_ensemble_bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """One common table: overall + per-ancestry-subgroup coverage for
    whichever of the three methods were run. Pass only the methods you've
    computed; each is optional so this works even before every dependency is
    installed. Also computable via uncertainty-toolbox for a richer
    calibration/sharpness/proper-scoring comparison once `uv add
    uncertainty-toolbox` has been run — see the module docstring's rationale.
    """
    rows = []
    for method_name, bounds in (
        ("mapie", mapie_bounds),
        ("ngboost", ngboost_bounds),
        ("deep_ensemble", deep_ensemble_bounds),
    ):
        if bounds is None:
            continue
        _, lower, upper = bounds
        report = stratified_conformal_coverage(y_test, lower, upper, group)
        report["method"] = method_name
        rows.append(report)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["group", "n", "coverage", "method"])


def uncertainty_toolbox_report(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict:
    """Scores calibration/sharpness/proper-scoring metrics via
    uncertainty-toolbox, letting MAPIE/NGBoost/deep-ensemble be compared on
    one common metric instead of three different ad-hoc reports."""
    try:
        import uncertainty_toolbox as uct
    except ImportError as e:
        raise ImportError("uncertainty-toolbox is required. Install with: uv add uncertainty-toolbox") from e

    return uct.metrics.get_all_metrics(y_pred, y_std, y_true)
