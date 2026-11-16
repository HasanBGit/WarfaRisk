"""
Shared sklearn preprocessing wrapper, used by every phase that fits a plain scikit-
learn-compatible regressor on IWPC data (Phases 1, 3, 4, 6)
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def with_standard_preprocessing(estimator, dense: bool = False) -> Pipeline:
    """
    Median-impute numeric columns, most-frequent-impute + one-hot encode everything
    else, then hand off to `estimator`
    """
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
