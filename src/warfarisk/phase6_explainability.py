"""Phase 6 — explainability.

Implements TRAINING_PLAN.md Phase 6: SHAP on the Phase 3 winning model,
cross-checked against an independently trained InterpretML Explainable
Boosting Machine (EBM) — a glass-box model whose per-feature contributions
are exact, not a post-hoc approximation. Agreement between the two is real
evidence; disagreement is itself worth reporting, not a tiebreak to discard.

Both are computed overall AND per Phase-4 ancestry subgroup, extending rather
than just re-citing the review's own SHAP benchmark (58.1% genetic feature
importance) and Sridharan et al.'s finding that CYP2C9 contribution was
directionally inconsistent in misclassified cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def shap_feature_importance(fitted_pipeline, X: pd.DataFrame, sample_size: int | None = 200) -> pd.DataFrame:
    """Works for any fitted sklearn-compatible pipeline built via
    common.preprocessing.with_standard_preprocessing — uses shap.Explainer's
    model-agnostic path since the pipeline mixes tree and non-tree models
    across Phase 3's registry. Subsampled by default (sample_size) since
    KernelExplainer-style fallbacks are slow at this project's ~6,000-row
    scale; set sample_size=None to use every row.

    SHAP's default tabular masker computes np.isclose() between perturbed
    and original rows to skip invariant features — this hard-fails with
    `TypeError: unsupported operand type(s) for -: 'str' and 'str'` on raw
    string/object columns (confirmed 2026-07-26 running this on IWPC's raw
    genotype/clinical categoricals), a real SHAP limitation with mixed-dtype
    tabular data, not a data bug. Fix: SHAP only ever sees an integer-coded
    proxy of the categorical columns; predict_fn decodes back to the
    original string categories before calling the real fitted pipeline, so
    the explanation is of the model that was actually trained (one-hot via
    with_standard_preprocessing), not a different model refit on integer
    codes.
    """
    X_sample = X if sample_size is None else X.sample(min(sample_size, len(X)), random_state=0)

    categorical_cols = X_sample.select_dtypes(exclude="number").columns
    category_maps = {
        col: dict(enumerate(X_sample[col].astype("category").cat.categories)) for col in categorical_cols
    }
    X_encoded = X_sample.copy()
    for col in categorical_cols:
        X_encoded[col] = X_sample[col].astype("category").cat.codes.astype(float)
    X_encoded_values = X_encoded.to_numpy()

    def predict_fn(X_batch: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X_batch, columns=X_sample.columns)
        for col in categorical_cols:
            codes = X_df[col].round()
            X_df[col] = codes.map(lambda c: category_maps[col].get(int(c)) if c >= 0 else np.nan)
        return fitted_pipeline.predict(X_df)

    explainer = shap.Explainer(predict_fn, X_encoded_values)
    shap_values = explainer(X_encoded_values)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    mean_signed_shap = shap_values.values.mean(axis=0)
    return pd.DataFrame(
        {"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap, "mean_signed_shap": mean_signed_shap}
    ).sort_values("mean_abs_shap", ascending=False)


def shap_by_ancestry_subgroup(fitted_pipeline, X: pd.DataFrame, group: pd.Series, sample_size: int | None = 200) -> pd.DataFrame:
    """Same as shap_feature_importance but split by ancestry group — tests
    whether e.g. VKORC1 dominates uniformly across groups or attribution
    shifts, per TRAINING_PLAN.md Phase 6."""
    rows = []
    for group_name, idx in X.groupby(group).groups.items():
        sub_report = shap_feature_importance(fitted_pipeline, X.loc[idx], sample_size=sample_size)
        sub_report["group"] = group_name
        rows.append(sub_report)
    return pd.concat(rows, ignore_index=True)


def build_ebm_regressor():
    """InterpretML's Explainable Boosting Machine — a GAM-style glass-box
    model. Returns exact per-feature contributions natively via
    explain_global(), no post-hoc approximation."""
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
    except ImportError as e:
        raise ImportError("InterpretML is required. Install with: uv add interpret") from e
    return ExplainableBoostingRegressor(random_state=0)


def ebm_feature_importance(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Fits an EBM independently (not the Phase 3 winner — EBM is its own
    model class) and returns its native per-feature importance for
    cross-checking against shap_feature_importance()'s output on the same
    feature set.

    InterpretML doesn't recognize pandas 3.x's default `str` extension dtype
    for CSV-read string/categorical columns (confirmed 2026-07-26:
    `TypeError: <class 'pandas.StringDtype'> not supported` fitting on
    IWPC's raw genotype/clinical categoricals, e.g. Age here is genuinely a
    9-category age-band field in the raw IWPC export, not a numeric column —
    that's real data, not a bug) — only classic numpy `object` dtype. Cast
    non-numeric columns before fitting; this doesn't change what the model
    sees, just how pandas stores the same string values in memory.
    """
    X_train = X_train.copy()
    for col in X_train.select_dtypes(exclude="number").columns:
        X_train[col] = X_train[col].astype(object)

    ebm = build_ebm_regressor()
    ebm.fit(X_train, y_train)
    global_explanation = ebm.explain_global()
    data = global_explanation.data()
    return pd.DataFrame({"feature": data["names"], "importance": data["scores"]}).sort_values(
        "importance", ascending=False
    )


def compare_shap_vs_ebm(shap_report: pd.DataFrame, ebm_report: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank-correlates the two methods' top-N feature rankings — the
    agreement/disagreement check TRAINING_PLAN.md Phase 6 calls for, rather
    than trusting either method alone."""
    shap_ranked = shap_report.head(top_n).reset_index(drop=True)
    shap_ranked["shap_rank"] = shap_ranked.index + 1
    ebm_ranked = ebm_report.head(top_n).reset_index(drop=True)
    ebm_ranked["ebm_rank"] = ebm_ranked.index + 1
    merged = pd.merge(
        shap_ranked[["feature", "shap_rank"]],
        ebm_ranked[["feature", "ebm_rank"]],
        on="feature",
        how="outer",
    )
    merged["rank_agreement"] = merged["shap_rank"].notna() & merged["ebm_rank"].notna()
    return merged.sort_values(["shap_rank", "ebm_rank"])
