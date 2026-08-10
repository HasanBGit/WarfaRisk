import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.common.metrics import stratified_conformal_coverage
from warfarisk.phase4_ancestry_holdout import iwpc1780_ancestry_group
from warfarisk.phase5_calibration import ngboost_predictive_distribution


def encode_for_ngboost(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """NGBoost needs purely numeric input — one-hot encode categoricals the
    same way with_standard_preprocessing does, fit on train only."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), lambda df: df.select_dtypes(include="number").columns),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                lambda df: df.select_dtypes(exclude="number").columns,
            ),
        ]
    )
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    return X_train_enc, X_test_enc


def run_cohort(name, df, feature_set_fn, target_col, split_name, ancestry_col_or_fn):
    print(f"\n=== Phase 5: NGBoost predictive distribution, {name} ===", flush=True)
    split = splits.load_split(split_name)
    train_df, test_df = splits.apply_split(df, split)
    X_train = feature_set_fn(train_df, "combined")
    y_train = train_df[target_col]
    X_test = feature_set_fn(test_df, "combined")
    y_test = test_df[target_col]

    X_train_enc, X_test_enc = encode_for_ngboost(X_train, X_test)
    y_pred, lower, upper = ngboost_predictive_distribution(X_train_enc, y_train, X_test_enc, alpha=0.1)

    overall_coverage = ((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)).mean()
    print(f"Overall 90% interval empirical coverage: {overall_coverage:.4f} (n_test={len(y_test)})", flush=True)

    ancestry = ancestry_col_or_fn(test_df) if callable(ancestry_col_or_fn) else test_df[ancestry_col_or_fn]
    coverage_report = stratified_conformal_coverage(y_test, lower, upper, ancestry)
    print(coverage_report.to_string(), flush=True)
    coverage_report.to_csv(f"phase5_{name}_ngboost_coverage.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred, "lower": lower, "upper": upper}).to_csv(
        f"phase5_{name}_ngboost_predictions.csv", index=False
    )


df6256 = data_mod.load_iwpc_6256()
run_cohort("iwpc6256", df6256, data_mod.iwpc6256_feature_set, data_mod.IWPC6256_TARGET, "iwpc_6256", "Race (Reported)")

df1780 = data_mod.load_iwpc_1780()
run_cohort("iwpc1780", df1780, data_mod.iwpc1780_feature_set, data_mod.IWPC1780_TARGET, "iwpc_1780", iwpc1780_ancestry_group)

print(f"\nDone at {datetime.datetime.now().isoformat()}", flush=True)
print("NGBOOST_DONE", flush=True)
