# Known issues & dependency-compatibility notes

Real bugs found and fixed while building and running this pipeline, kept here
because anyone re-running the code on a newer dependency stack is likely to
hit the same ones. Each entry names the affected module and what changed;
see `docs/RESULTS.md` for the experimental results these fixes unblocked.

## Dependency / API version breaks

- **`mapie==1.0.1` is incompatible with `scikit-learn>=1.6`.** Its internal
  `EnsembleRegressor` doesn't implement `__sklearn_tags__`, which newer
  scikit-learn's `check_is_fitted` requires, raising
  `AttributeError: 'EnsembleRegressor' object has no attribute
  '__sklearn_tags__'` from inside MAPIE's own code (reproduced on a trivial
  synthetic example with plain `LinearRegression`, so it isn't specific to
  this project's model/data). Fixed by pinning `mapie>=1.4`
  (`pyproject.toml`) and rewriting `phase5_calibration.py`'s
  `mapie_conformal_interval()` for the new public API: the old one-shot
  `MapieRegressor(...).fit(X, y)` + `.predict(X, alpha=...)` pattern is gone,
  replaced by `CrossConformalRegressor(...).fit_conformalize(X, y)` +
  `.predict_interval(X)`.
- **`scikit-survival==0.24.1` is incompatible with `scikit-learn>=1.9`.**
  `sksurv.ensemble` imports the private Cython symbol `DTYPE` from
  `sklearn.tree._tree`, which no longer exists in current scikit-learn,
  breaking any import from that subpackage (including
  `RandomSurvivalForest`) even though `CoxPHSurvivalAnalysis` alone would be
  unaffected. Fixed by pinning `scikit-survival>=0.28`
  (`pyproject.toml`), which declares explicit
  `scikit-learn<1.10,>=1.9.0` support.
- **InterpretML doesn't recognize pandas 3.x's default `str` extension
  dtype.** Fitting an EBM on IWPC's raw categorical columns raises
  `TypeError: <class 'pandas.StringDtype'> not supported`. Fixed in
  `phase6_explainability.py`'s `ebm_feature_importance()` by casting
  non-numeric columns to classic `object` dtype before fitting — doesn't
  change what the model sees, just how pandas stores the same values.

## Model-library input-format quirks

- **TabPFN rejects sparse one-hot input.**
  `common/preprocessing.py`'s `with_standard_preprocessing()` defaults its
  `OneHotEncoder` to sparse output (fine for sklearn/XGBoost), but
  `TabPFNRegressor` raises `TypeError: Sparse data was passed for X, but
  dense data is required`. Fixed by adding a `dense: bool` parameter to
  `with_standard_preprocessing()`; `build_tabpfn()` passes `dense=True`,
  every other model is unaffected.
- **CatBoost rejects NaN inside categorical columns.** Raw genotype
  consensus columns (e.g. one VKORC1 column is 84% missing) mix a float
  `NaN` into an otherwise-string column; CatBoost's native categorical
  handling errors with `CatBoostError: bad object for id: nan` rather than
  treating missing as its own category. Fixed in
  `phase3_genetics_ablation.py`'s `run_catboost_ablation()` by filling
  categorical columns with the literal string `"missing"` before building
  the `Pool` — an explicit missing-genotype category, not a dropped row or
  an imputed guess.
- **`build_catboost()` is not safe to pass as a generic `model_builder`.**
  It's built around an explicit `Pool` + `cat_features` call
  (`run_catboost_ablation()`'s own path), but `phase4_ancestry_holdout.py`'s
  `leave_one_ancestry_out()` calls `pipeline.fit(X, y)` /
  `.predict(X)` directly on whatever builder it's given — a bare
  `CatBoostRegressor.fit(X, y)` with no `cat_features` treats every column
  as numeric and crashes on the first string value (`CatBoostError: Bad
  value for num_feature[...]="female"` on `Gender`). Fixed by adding
  `_CatBoostAutoCategorical` / `build_catboost_autocat()` to
  `phase3_genetics_ablation.py` — auto-detects non-numeric columns and
  fills their NaNs with `"missing"` at fit/predict time, same fix as above,
  applied to this different call path.
- **SHAP's default tabular masker fails on string categorical columns.**
  It calls `np.isclose()` between perturbed and original rows, which
  hard-fails (`TypeError: unsupported operand type(s) for -: 'str' and
  'str'`) on IWPC's raw string genotype/`Gender`/indication columns — a
  real mixed-dtype limitation, not an installation issue. Fixed in
  `phase6_explainability.py`'s `shap_feature_importance()` by wrapping the
  predict function so SHAP only ever sees an integer-coded proxy of
  categorical columns, decoded back to the original strings before calling
  the real fitted pipeline — so the explanation is of the actual trained
  model, not one refit on integer codes.
- **`ColumnTransformer` column selectors must be picklable.**
  `with_standard_preprocessing()` originally used raw lambdas for column
  selection, which `joblib.dump()` cannot pickle — this silently blocks
  saving/pushing *any* fitted pipeline's artifact, not just one phase.
  Fixed by switching to `sklearn.compose.make_column_selector` (identical
  behavior, verified same metrics before/after).

## Data-construction bugs (not library bugs)

- **eICU's raw `dosage` field is inconsistently formatted.** It's a
  `"NUMBER UNIT"` string where `UNIT` is sometimes literal text (`"5 MG"`)
  and sometimes a numeric lookup code (`"5 3"`, `"1 5001"`) — confirmed
  across all real rows in the cleaned eICU medication table. Any numeric
  operation on the raw field crashes. Fixed in `phase7_longitudinal_rl.py`'s
  `build_eicu_sequences()` by regex-extracting the leading numeric token.
- **`np.random.permutation` on a list of `(subject_id, hadm_id)` tuples
  silently converts them to unhashable lists.** Fixed by permuting indices
  instead of the tuples themselves.
- **Merging the survival/TTR table against admissions on `subject_id`
  alone causes a many-to-many join blowup.** `build_mimic3/4_sequences()`
  keys sequences by `(subject_id, hadm_id)`, and a patient can have
  multiple admissions — merging on `subject_id` alone inflated MIMIC-III's
  34-row TTR table to 250 rows and MIMIC-IV's 62 to 339 on the first
  attempt. Fixed by merging on the exact `(subject_id, hadm_id)` pair, with
  an assertion added (row count must never grow across the merge) so this
  class of bug fails loudly instead of silently inflating a downstream
  concordance number.

## AutoGluon runtime behavior

- **`extreme_quality`'s time budget isn't enforced mid-fit for every
  model.** AutoGluon's `time_limit` is only checked *between* models, not
  during a single bagged fit — a first attempt with the full model
  portfolio saw `TABDPT` alone overrun its ~460s allotted slice by more
  than 15 real minutes with no way to interrupt it. Re-run clean after
  excluding it and three other slow/unstable presets
  (`excluded_model_types=["TABDPT","TABICL","TABM","MITRA"]`, also
  recorded in `configs/model_hparams.yaml`). One `RealTabPFN-v2` preset
  config (`_r196`) also failed with `KeyError: 'kdi_alpha_1.0'` — a
  version-skew bug between AutoGluon's pinned preset and the installed
  `tabpfn` package, not this project's code; AutoGluon caught it
  internally and continued.
