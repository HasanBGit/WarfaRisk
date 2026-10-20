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
    random_state: int = 20260725,
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

    mapie_model = CrossConformalRegressor(
        estimator=base_estimator, confidence_level=1 - alpha, method="plus", cv=5, random_state=random_state
    )
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


def mondrian_conformal_interval(
    base_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    group_train: pd.Series,
    group_test: pd.Series,
    alpha: float = 0.1,
    calib_fraction: float = 0.2,
    min_group_n: int = 10,
    random_state: int = 20260725,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Group-conditional (Mondrian) split conformal prediction, evaluated
    against reviewer feedback asking for a comparator to MAPIE's marginal-only
    guarantee. Unlike mapie_conformal_interval, this computes a SEPARATE
    nonconformity quantile per ancestry group rather than one pooled quantile,
    so it targets per-subgroup coverage directly rather than aggregate
    coverage.

    Procedure (standard split conformal, applied per group): hold out
    calib_fraction of the training fold as a calibration set, fit
    base_estimator on the remainder, compute absolute residuals on the
    calibration set, then for each group compute the finite-sample-corrected
    quantile ceil((n_g+1)(1-alpha))/n_g of that group's own residuals. Groups
    with fewer than min_group_n calibration points are pooled into the
    overall (marginal) calibration residuals instead of using an
    unstable/degenerate per-group quantile computed from too few points —
    this pooling threshold (min_group_n, default 10, matching roughly 1/alpha
    at alpha=0.1) is a stated design choice, not a property of Mondrian
    conformal prediction itself; smaller thresholds let more groups have a
    genuinely separate quantile at the cost of noisier per-group estimates.

    Returns (point_prediction, lower, upper, group_quantiles) where
    group_quantiles is a DataFrame reporting each group's calibration n,
    whether it was pooled, and its resulting quantile (in mg/week), so the
    pooling decision is auditable rather than hidden inside the function.
    """
    from sklearn.model_selection import train_test_split

    X_fit, X_calib, y_fit, y_calib, group_fit_ignored, group_calib = train_test_split(
        X_train, y_train, group_train, test_size=calib_fraction, random_state=random_state
    )
    base_estimator.fit(X_fit, y_fit)
    y_pred_calib = base_estimator.predict(X_calib)
    residuals_calib = np.abs(y_calib.to_numpy() - y_pred_calib)

    global_n = len(residuals_calib)
    global_level = min(1.0, np.ceil((global_n + 1) * (1 - alpha)) / global_n)
    global_q = float(np.quantile(residuals_calib, global_level, method="higher"))

    group_quantiles: dict[str, float] = {}
    quantile_rows = []
    for group_name in pd.unique(group_calib):
        group_mask = (group_calib.to_numpy() == group_name)
        n_g = int(group_mask.sum())
        pooled = n_g < min_group_n
        if pooled:
            q_g = global_q
        else:
            level_g = min(1.0, np.ceil((n_g + 1) * (1 - alpha)) / n_g)
            q_g = float(np.quantile(residuals_calib[group_mask], level_g, method="higher"))
        group_quantiles[group_name] = q_g
        quantile_rows.append({"group": group_name, "n_calib": n_g, "pooled_to_global": pooled, "quantile_mg_per_week": q_g})
    quantile_report = pd.DataFrame(quantile_rows).sort_values("n_calib", ascending=False).reset_index(drop=True)

    y_pred_test = base_estimator.predict(X_test)
    q_per_test_point = np.array([group_quantiles.get(g, global_q) for g in group_test.to_numpy()])
    lower = y_pred_test - q_per_test_point
    upper = y_pred_test + q_per_test_point
    return y_pred_test, lower, upper, quantile_report


def uncertainty_toolbox_report(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict:
    """Scores calibration/sharpness/proper-scoring metrics via
    uncertainty-toolbox, letting MAPIE/NGBoost/deep-ensemble be compared on
    one common metric instead of three different ad-hoc reports."""
    try:
        import uncertainty_toolbox as uct
    except ImportError as e:
        raise ImportError("uncertainty-toolbox is required. Install with: uv add uncertainty-toolbox") from e

    return uct.metrics.get_all_metrics(y_pred, y_std, y_true)
