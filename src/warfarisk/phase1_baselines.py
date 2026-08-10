"""Phase 1 — mandatory baselines, reported alongside every later result.

Implements TRAINING_PLAN.md Phase 1: naive median-dose predictor, clinical-only
linear regression, and warfit-learn-wrapped IWPC/Gage pharmacogenetic
equations. Every later phase's model is compared against these three, not
reported in isolation — this is what separates a real improvement from an
artifact of a favorable split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from warfarisk.common import data as data_mod
from warfarisk.common.metrics import regression_report
from warfarisk.common.preprocessing import with_standard_preprocessing
from warfarisk.common.runlog import append_run


def naive_median_dose(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Predict the training-set median dose for every test patient — the
    field's own reference floor (~34.1% PW20 per the review); any model that
    can't clear this isn't adding value."""
    median_dose = float(y_train.median())
    y_pred = np.full(len(y_test), median_dose)
    report = regression_report(y_test, y_pred)
    report["model"] = "naive_median_dose"
    report["median_dose"] = median_dose
    return report


def clinical_only_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    pipeline = with_standard_preprocessing(LinearRegression())
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = regression_report(y_test, y_pred)
    report["model"] = "clinical_only_linear_regression"
    return report


def warfit_learn_iwpc_baseline(df: pd.DataFrame) -> dict:
    """Was meant to wrap warfit-learn's published IWPC pharmacogenetic
    equation as a callable baseline. **Checked against a live install
    (warfit-learn==0.2.1) on 2026-07-25 and this isn't possible as originally
    planned:**

    - The package contains no hardcoded closed-form IWPC/Gage coefficients
      anywhere in its source (`estimators/`, `evaluation/`, `metrics/`,
      `preprocessing/` — all inspected directly). `preprocessing.prepare_iwpc()`
      only cleans/one-hot-encodes IWPC's raw columns into a fixed variable
      set; the "equation" is produced by fitting a plain `LinearRegression`
      on that output — i.e. it trains new coefficients on whatever data you
      feed it, it does not apply the paper's original published coefficients.
    - `prepare_iwpc()` is hard shape-locked to warfit-learn's own bundled raw
      PharmGKB pickle: `_verify_shape()` asserts `df.shape == (6256, 68)`
      and expects a column called `"Imputed VKORC1"` that doesn't exist in
      this project's independently-cleaned data. This project's own
      `iwpc_full_6256_clean.csv` is `(6037, 68)` — different row count
      (post-cleaning exclusions differ) and a different VKORC1 encoding
      (`*_consensus` columns, not one `Imputed VKORC1` column) — so calling
      `prepare_iwpc()` on it fails the assertion rather than silently
      producing a wrong number.

    Getting a real "published IWPC equation" baseline still requires the
    actual coefficients from the IWPC 2009 NEJM paper (Table 2), typed in by
    hand against the primary source — not fabricated from memory here, and
    not available from this dependency after all. Left unimplemented on
    purpose; do not call this function until that's done.
    """
    raise NotImplementedError(
        "warfit-learn 0.2.1 does not expose a reusable published-coefficient IWPC/Gage "
        "equation (verified by reading its installed source, see docstring above) and its "
        "IWPC data-prep path is shape-locked to a different raw file than this project's "
        "cleaned iwpc_full_6256_clean.csv. Implement the equation by hand from the primary "
        "source (IWPC, N Engl J Med 2009, Table 2) instead of importing it from this package."
    )


def _iwpc_age_decade(age_band: str) -> float:
    """IWPC's own cohort represents age as a decade band ("60 - 69", "90+"),
    matching the "age in decades" term the published equation expects
    directly — no conversion needed beyond parsing the band's leading digit
    into a decade index (e.g. "60 - 69" -> 6)."""
    if pd.isna(age_band):
        return np.nan
    if age_band.strip() == "90+":
        return 9.0
    return float(age_band.split()[0]) / 10.0


def iwpc_published_equation_predict(df: pd.DataFrame) -> np.ndarray:
    """Computes the actual published IWPC pharmacogenetic dosing equation
    (International Warfarin Pharmacogenetics Consortium, "Estimation of the
    Warfarin Dose with Clinical and Pharmacogenetic Data", N Engl J Med
    2009;360:753-64, Table 2) directly on raw iwpc_full_6256_clean.csv
    columns — a fixed, published formula with fixed published coefficients,
    not something to fit. Returns predicted weekly dose in mg.

    **Provenance caveat, stated plainly**: these coefficients are transcribed
    from the widely-published, widely-cited form of this equation as it
    commonly appears in the pharmacogenomics literature and clinical
    references (e.g. warfarindosing.org's public calculator implements the
    same published formula) — they were not re-derived by directly
    re-reading a copy of the original NEJM PDF's Table 2 in this session.
    Before this baseline's numbers are cited in the manuscript, cross-check
    the coefficients below against the primary source table directly.

    Known limitation, inherent to the equation itself, not this
    implementation: it only resolves CYP2C9 *1/*2/*3 and the core VKORC1
    -1639 SNP — patients with rarer alleles (*5, *6, *11, *13, *14, all
    present in this cohort in small numbers) are treated via the equation's
    own "genotype unknown" indicator, exactly as the original algorithm
    specifies for genotypes outside its 6 modeled combinations, not
    approximated or dropped.

    ONLY valid on raw-unit data (iwpc_full_6256_clean.csv) — iwpc_1780_clean.csv's
    Weight/Height/Age are pre-z-scored, which would silently produce garbage
    if pushed through these raw-unit coefficients, so this function
    deliberately isn't offered for that cohort.
    """
    age_decades = df["Age"].apply(_iwpc_age_decade)
    height_cm = df["Height (cm)"]
    weight_kg = df["Weight (kg)"]

    # IWPC's own published handling for missing height/weight: impute with
    # the cohort mean rather than drop the patient.
    height_cm = height_cm.fillna(height_cm.mean())
    weight_kg = weight_kg.fillna(weight_kg.mean())
    age_decades = age_decades.fillna(age_decades.mean())

    vkorc1 = df["VKORC1     -1639 consensus"]
    vkorc1_ag = (vkorc1 == "A/G").astype(float)
    vkorc1_aa = (vkorc1 == "A/A").astype(float)
    vkorc1_unknown = vkorc1.isna().astype(float)

    cyp2c9 = df["CYP2C9 consensus"]
    known_genotypes = {"*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"}
    cyp2c9_12 = (cyp2c9 == "*1/*2").astype(float)
    cyp2c9_13 = (cyp2c9 == "*1/*3").astype(float)
    cyp2c9_22 = (cyp2c9 == "*2/*2").astype(float)
    cyp2c9_23 = (cyp2c9 == "*2/*3").astype(float)
    cyp2c9_33 = (cyp2c9 == "*3/*3").astype(float)
    # "unknown" per the equation's own design: never tested, OR a genotype
    # combination outside the 6 it models (e.g. *1/*5, *1/*11) — not *1/*1.
    cyp2c9_unknown = (~cyp2c9.isin(known_genotypes | {"*1/*1"})).astype(float)

    race = df["Race (OMB)"]
    race_asian = (race == "Asian").astype(float)
    race_black = (race == "Black or African American").astype(float)
    race_missing_mixed = (race.isna() | (race == "Unknown")).astype(float)

    enzyme_inducer = (
        (df["Carbamazepine (Tegretol)"].fillna(0) == 1)
        | (df["Phenytoin (Dilantin)"].fillna(0) == 1)
        | (df["Rifampin or Rifampicin"].fillna(0) == 1)
    ).astype(float)
    amiodarone = (df["Amiodarone (Cordarone)"].fillna(0) == 1).astype(float)

    sqrt_dose = (
        5.6044
        - 0.2614 * age_decades
        + 0.0087 * height_cm
        + 0.0128 * weight_kg
        - 0.8677 * vkorc1_ag
        - 1.6974 * vkorc1_aa
        - 0.4854 * vkorc1_unknown
        - 0.5211 * cyp2c9_12
        - 0.9357 * cyp2c9_13
        - 1.0616 * cyp2c9_22
        - 1.9206 * cyp2c9_23
        - 2.3312 * cyp2c9_33
        - 0.2188 * cyp2c9_unknown
        - 0.1092 * race_asian
        - 0.2760 * race_black
        - 0.1032 * race_missing_mixed
        + 1.1816 * enzyme_inducer
        - 0.5503 * amiodarone
    )
    return (sqrt_dose**2).to_numpy()


def iwpc_published_equation_baseline(df_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates the fixed published equation on a held-out test set — no
    fitting step, since it's a fixed formula, but evaluated on the same
    Phase-0 test split as every other Phase 1 baseline for a fair, direct
    comparison."""
    y_pred = iwpc_published_equation_predict(df_test)
    report = regression_report(y_test, y_pred)
    report["model"] = "iwpc_published_equation"
    return report


def run_phase1_from_files() -> pd.DataFrame:
    """Loads iwpc_full_6256_clean.csv, applies the saved Phase-0 split, and
    runs the two baselines that don't need warfit-learn. This is the actual
    entry point used to produce real results — run_phase1() alone only
    operates on already-loaded/split DataFrames."""
    from common import splits

    df = data_mod.load_iwpc_6256()
    split = splits.load_split("iwpc_6256")
    train_df, test_df = splits.apply_split(df, split)
    return run_phase1(train_df, test_df)


def run_phase1(iwpc6256_train: pd.DataFrame, iwpc6256_test: pd.DataFrame) -> pd.DataFrame:
    """Runs all three Phase 1 baselines on the IWPC-6256 cohort: naive
    median, clinical-only linear regression, and the real published IWPC
    equation (warfit_learn_iwpc_baseline is a separate, deliberately-broken
    stub documenting why that package couldn't supply this instead)."""
    y_train = iwpc6256_train[data_mod.IWPC6256_TARGET]
    y_test = iwpc6256_test[data_mod.IWPC6256_TARGET]
    X_train = data_mod.iwpc6256_feature_set(iwpc6256_train, "clinical")
    X_test = data_mod.iwpc6256_feature_set(iwpc6256_test, "clinical")

    rows = [
        naive_median_dose(y_train, y_test),
        clinical_only_linear_regression(X_train, y_train, X_test, y_test),
        iwpc_published_equation_baseline(iwpc6256_test, y_test),
    ]
    for row in rows:
        append_run(phase="phase1", experiment=row["model"], dataset="iwpc_6256", metric="mae", value=row["mae"])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_phase1_from_files()
    print(results.to_string(index=False))
