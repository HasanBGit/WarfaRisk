"""Shared sklearn preprocessing wrapper, used by every phase that fits a
plain scikit-learn-compatible regressor on IWPC data (Phases 1, 3, 4, 6).

Kept as a single function so the "fit imputation/encoding only inside the
training fold" rule (TRAINING_PLAN.md Phase 0/2) is enforced in exactly one
place: every caller wraps its estimator in this Pipeline instead of
transforming X before calling train_test_split.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def with_standard_preprocessing(estimator, dense: bool = False) -> Pipeline:
    """Median-impute numeric columns, most-frequent-impute + one-hot encode
    everything else, then hand off to `estimator`. Column selection is by
    dtype at fit time via `make_column_selector` (picklable, unlike a raw
    lambda — needed so a fitted pipeline can actually be joblib-dumped for
    reuse/Hub upload, not just used in-process), so this works unchanged
    across the clinical/genetic/combined feature sets without hardcoding
    column lists.

    dense=True forces OneHotEncoder to emit a dense array instead of its
    default sparse matrix — needed for estimators (e.g. TabPFNRegressor) that
    reject sparse input outright rather than densifying it themselves."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include="number")),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
                    ]
                ),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
