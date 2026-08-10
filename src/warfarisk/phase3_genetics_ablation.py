"""Phase 3 — nested clinical/genetic ablation, plus comparator-sweep-1 models.

Implements TRAINING_PLAN.md Phase 3 and the first "Extended comparator sweep"
section: fits clinical-only -> genetic-only -> combined feature sets, across
Linear, ElasticNet/Lasso, RandomForest, XGBoost, CatBoost, MLP,
StackingRegressor, TabPFN-3, and AutoGluon, on IWPC-6256 and IWPC-1780
separately (keep 1780's pre-z-scored features as-is, don't pool with 6256's
raw units).

Every model builder below is guarded: this module always imports cleanly with
only numpy/pandas/scikit-learn installed. A model that needs an extra
dependency raises a clear `uv add <package>` error only when its build_*
function is actually called.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor

from warfarisk.common import data as data_mod
from warfarisk.common.metrics import regression_report
from warfarisk.common.preprocessing import with_standard_preprocessing
from warfarisk.common.runlog import append_run

# ---------------------------------------------------------------------------
# Model registry — sklearn-native models need no guard; everything else does.
# ---------------------------------------------------------------------------


def build_linear():
    return with_standard_preprocessing(LinearRegression())


def build_elastic_net():
    # Regularized linear comparator (matches Abdel-latif et al. 2026's LASSO
    # and the elastic-net component of Dryden et al. 2023's ensemble) — tests
    # whether genetic features survive regularization or are an overfit
    # artifact of a handful of rare genotypes, per the review's own finding
    # that some genotypes are represented by single-digit patient counts.
    return with_standard_preprocessing(ElasticNet(random_state=0))


def build_random_forest():
    return with_standard_preprocessing(RandomForestRegressor(random_state=0, n_estimators=300))


def build_mlp():
    # Matches Ma et al. 2021 and, more importantly, Jahmunah et al. 2023 —
    # the exact study whose IWPC-trained deep NN showed the ancestry-error
    # near-doubling (7.6 vs 14.2 mg/week) that Phase 4 is designed to test.
    # Using an MLP here (not just CatBoost/TabPFN) makes Phase 4's ancestry
    # hold-out an architecture-matched replication of Jahmunah's own finding.
    return with_standard_preprocessing(MLPRegressor(random_state=0, hidden_layer_sizes=(64, 32), max_iter=2000))


def build_stacking_ensemble():
    # Matches Wang et al. 2024's "heuristic stacking ensemble" — the study
    # reporting this literature's best-looking number (MAE 0.77 mg/week) from
    # a 64-patient test set. Replicated faithfully here so Phase 4 can show
    # what the same architecture does on a properly sized, ancestry-stratified
    # split — a direct test of the review's own accuracy-vs-rigor finding.
    base_estimators = [
        ("linear", LinearRegression()),
        ("rf", RandomForestRegressor(random_state=0, n_estimators=200)),
        ("mlp", MLPRegressor(random_state=0, hidden_layer_sizes=(32,), max_iter=2000)),
    ]
    stacked = StackingRegressor(estimators=base_estimators, final_estimator=LinearRegression())
    return with_standard_preprocessing(stacked)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def build_xgboost():
    # Matches Choi et al. 2023 and several other included studies.
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError("XGBoost is required. Install with: uv add xgboost") from e
    device = "cuda" if _cuda_available() else "cpu"
    return with_standard_preprocessing(XGBRegressor(random_state=0, n_estimators=300, tree_method="hist", device=device))


def build_catboost():
    # Natively handles star-allele category strings — no one-hot needed, so
    # this one is NOT wrapped in with_standard_preprocessing; CatBoost gets
    # the raw categorical columns directly (cat_features must be passed at
    # fit time — see run_catboost_ablation() below).
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise ImportError("CatBoost is required. Install with: uv add catboost") from e
    if _cuda_available():
        return CatBoostRegressor(random_state=0, verbose=False, task_type="GPU", devices="0")
    return CatBoostRegressor(random_state=0, verbose=False)


class _CatBoostAutoCategorical:
    """Wraps CatBoostRegressor so it's safe to use through the generic
    model_builder interface (Phase 4/5/6's `pipeline.fit(X, y)` /
    `pipeline.predict(X)` calls) instead of only run_catboost_ablation()'s
    explicit-Pool path. A bare CatBoostRegressor.fit(X, y) with no
    cat_features treats every column as numeric and crashes on the first
    real string value — confirmed 2026-07-26 running Phase 4's ancestry
    holdout: `CatBoostError: Bad value for num_feature[...]="female"` on the
    Gender column. Fix: auto-detect non-numeric dtype columns as categorical
    at fit time and fill their NaNs with the literal string "missing" (same
    fix already used in run_catboost_ablation() for the NaN-in-categorical
    bug, applied here too since this path bypasses that function).
    """

    def __init__(self, **kwargs):
        from catboost import CatBoostRegressor

        self._model = CatBoostRegressor(**kwargs)
        self._cat_features: list[str] = []

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._cat_features:
            X[col] = X[col].fillna("missing").astype(str)
        return X

    def fit(self, X: pd.DataFrame, y):
        self._cat_features = list(X.select_dtypes(exclude="number").columns)
        self._model.fit(self._prep(X), y, cat_features=self._cat_features)
        return self

    def predict(self, X: pd.DataFrame):
        return self._model.predict(self._prep(X))


def build_catboost_autocat():
    """Same underlying model as build_catboost(), but safe to pass as a
    model_builder to phase4_ancestry_holdout.leave_one_ancestry_out() or any
    other caller using the generic fit(X, y)/predict(X) interface — see
    _CatBoostAutoCategorical's docstring for why build_catboost() itself
    isn't safe there."""
    try:
        from catboost import CatBoostRegressor  # noqa: F401  (imported for the ImportError guard below)
    except ImportError as e:
        raise ImportError("CatBoost is required. Install with: uv add catboost") from e
    kwargs = {"random_state": 0, "verbose": False}
    if _cuda_available():
        kwargs.update(task_type="GPU", devices="0")
    return _CatBoostAutoCategorical(**kwargs)


def build_tabpfn():
    # TabPFN-3 (Prior-Labs/tabpfn_3 on HF) — zero-hyperparameter-search
    # tabular foundation model; a no-tuning ceiling check against the tuned
    # models above. Comfortably handles this project's ~6,000-row cohorts
    # (verified envelope: up to ~1M rows in the v3 release).
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as e:
        raise ImportError(
            "TabPFN is required. Install with: uv add tabpfn "
            "(checkpoint: huggingface.co/Prior-Labs/tabpfn_3)"
        ) from e
    device = "cuda" if _cuda_available() else "cpu"
    return with_standard_preprocessing(TabPFNRegressor(device=device), dense=True)


def run_autogluon_extreme_quality(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    time_limit: int = 1800,
    excluded_model_types: list[str] | None = None,
):
    """AutoGluon wraps TabPFN-3 (+ TabICL, TabDPT, and ~20 other models) in a
    zeroshot_2025_tabfm portfolio and auto-selects/stacks the winners on the
    exact data given — recommended as the Phase 3 ceiling benchmark to run
    FIRST, before the individual-model comparison above. Not wrapped in
    with_standard_preprocessing: AutoGluon does its own preprocessing
    internally from a raw DataFrame.

    time_limit (seconds, default 1800 = 30 min) caps `fit()` — "extreme_quality"
    has no built-in default time limit and can otherwise run for many hours on
    a single ~6,000-row cohort trying its full model portfolio; this is a
    deliberate, documented compromise for an interactive session, not a
    silently different number. Raise it for an unattended/overnight run.

    excluded_model_types: AutoGluon's time_limit is only checked *between*
    models, not during a single bagged fit — a model whose fixed cost is large
    (e.g. TabDPT/TabM/TabICL/Mitra running their own pretrained checkpoint
    across 8 CV folds) can massively overrun its allotted slice with no way to
    interrupt it mid-fit. Pass e.g. ["TABDPT", "TABICL", "TABM", "MITRA"] to
    exclude those for a bounded interactive run; leave None for the full
    portfolio on an unattended/overnight run.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as e:
        raise ImportError(
            "AutoGluon is required. Install with: uv add 'autogluon.tabular[tabarena]'"
        ) from e

    train_data = train_df[feature_cols + [target_col]]
    test_data = test_df[feature_cols + [target_col]]
    num_gpus = 1 if _cuda_available() else 0
    predictor = TabularPredictor(label=target_col).fit(
        train_data,
        presets="extreme_quality",
        time_limit=time_limit,
        num_gpus=num_gpus,
        excluded_model_types=excluded_model_types,
    )
    y_pred = predictor.predict(test_data.drop(columns=[target_col]))
    report = regression_report(test_data[target_col], y_pred)
    report["model"] = "autogluon_extreme_quality"
    report["time_limit_s"] = time_limit
    return report, predictor


MODEL_REGISTRY = {
    "linear": build_linear,
    "elastic_net": build_elastic_net,
    "random_forest": build_random_forest,
    "mlp": build_mlp,
    "stacking_ensemble": build_stacking_ensemble,
    "xgboost": build_xgboost,
    "tabpfn": build_tabpfn,
    # "catboost" deliberately excluded from this registry: it needs raw
    # categoricals (no one-hot), so it's fit via run_catboost_ablation()
    # instead of the generic run_nested_ablation() loop below.
}


# ---------------------------------------------------------------------------
# Nested ablation driver
# ---------------------------------------------------------------------------


def run_nested_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_fn,
    target_col: str,
    model_names: list[str] | None = None,
    dataset_name: str = "unknown",
) -> pd.DataFrame:
    """For each of clinical/genetic/combined feature sets, fit every model in
    model_names (defaults to every sklearn-native model that needs no extra
    dependency) and report MAE/R2/PW20. This is the table Phase 4 and Phase 6
    then slice further by ancestry subgroup / explainability.
    """
    model_names = model_names or ["linear", "elastic_net", "random_forest", "mlp", "stacking_ensemble"]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    rows = []
    for which in ("clinical", "genetic", "combined"):
        X_train = feature_set_fn(train_df, which)
        X_test = feature_set_fn(test_df, which)
        for model_name in model_names:
            pipeline = MODEL_REGISTRY[model_name]()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            report = regression_report(y_test, y_pred)
            report["model"] = model_name
            report["feature_set"] = which
            rows.append(report)
            append_run(
                phase="phase3",
                experiment=f"{model_name}[{which}]",
                dataset=dataset_name,
                metric="mae",
                value=report["mae"],
            )
    return pd.DataFrame(rows)


def run_catboost_ablation(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_set_fn, target_col: str, dataset_name: str = "unknown"
) -> pd.DataFrame:
    """Separate driver for CatBoost since it consumes raw categorical
    star-allele strings directly rather than going through
    with_standard_preprocessing's one-hot encoding."""
    catboost_model = build_catboost()  # raises with install instructions if catboost isn't installed
    from catboost import Pool

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    rows = []
    for which in ("clinical", "genetic", "combined"):
        X_train = feature_set_fn(train_df, which).copy()
        X_test = feature_set_fn(test_df, which).copy()
        cat_features = list(X_train.select_dtypes(exclude="number").columns)
        # CatBoost's native categorical handling rejects a raw float NaN
        # inside an otherwise-string column (missing genotype calls, e.g. a
        # patient with no CYP2C9 consensus result) — fill with an explicit
        # "missing" category rather than dropping rows or one-hot-encoding.
        X_train[cat_features] = X_train[cat_features].fillna("missing")
        X_test[cat_features] = X_test[cat_features].fillna("missing")
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)
        model = build_catboost()
        model.fit(train_pool)
        y_pred = model.predict(test_pool)
        report = regression_report(y_test, y_pred)
        report["model"] = "catboost"
        report["feature_set"] = which
        rows.append(report)
        append_run(phase="phase3", experiment=f"catboost[{which}]", dataset=dataset_name, metric="mae", value=report["mae"])
    return pd.DataFrame(rows)


def run_phase3_iwpc6256(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return run_nested_ablation(
        train_df, test_df, data_mod.iwpc6256_feature_set, data_mod.IWPC6256_TARGET, dataset_name="iwpc_6256"
    )


def run_phase3_iwpc1780(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return run_nested_ablation(
        train_df, test_df, data_mod.iwpc1780_feature_set, data_mod.IWPC1780_TARGET, dataset_name="iwpc_1780"
    )


def run_phase3_from_files(model_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Loads both IWPC cohorts, applies their saved Phase-0 splits, and runs
    the nested ablation on each. Defaults to the sklearn-native models that
    need no extra dependency; pass model_names (a subset of MODEL_REGISTRY,
    e.g. including 'xgboost') once that phase's uv extras are installed."""
    from common import splits

    results: dict[str, pd.DataFrame] = {}

    iwpc6256 = data_mod.load_iwpc_6256()
    split_6256 = splits.load_split("iwpc_6256")
    train_6256, test_6256 = splits.apply_split(iwpc6256, split_6256)
    results["iwpc_6256"] = run_nested_ablation(
        train_6256, test_6256, data_mod.iwpc6256_feature_set, data_mod.IWPC6256_TARGET,
        model_names=model_names, dataset_name="iwpc_6256",
    )

    iwpc1780 = data_mod.load_iwpc_1780()
    split_1780 = splits.load_split("iwpc_1780")
    train_1780, test_1780 = splits.apply_split(iwpc1780, split_1780)
    results["iwpc_1780"] = run_nested_ablation(
        train_1780, test_1780, data_mod.iwpc1780_feature_set, data_mod.IWPC1780_TARGET,
        model_names=model_names, dataset_name="iwpc_1780",
    )

    return results


if __name__ == "__main__":
    for dataset_name, df in run_phase3_from_files().items():
        print(f"\n=== {dataset_name} ===")
        print(df.to_string(index=False))
