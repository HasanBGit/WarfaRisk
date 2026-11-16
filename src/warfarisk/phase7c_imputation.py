"""Phase 7c — generative imputation for the ICU cohorts' real missingness"""

from __future__ import annotations

import pandas as pd

from warfarisk.common import data as data_mod


def build_mimic3_feature_matrix_with_missingness() -> pd.DataFrame:
    """
    Joins MIMIC-III admissions (demographics, admission type/location) with
    diagnoses (as a wide comorbidity indicator matrix) — real-world missingness
    lives in fields like ethnicity/marital_status/religion/language that aren't
    always recorded, matching the kind of gaps Wani & Abeer's own imputation methods
    were built to handle
    """
    admissions = data_mod.load_mimic3_admissions()
    diagnoses = data_mod.load_mimic3_diagnoses()

    top_codes = diagnoses["icd9_code"].value_counts().head(20).index
    diag_wide = (
        diagnoses[diagnoses["icd9_code"].isin(top_codes)]
        .assign(present=1)
        .pivot_table(
            index=data_mod.MIMIC3_PATIENT_ID_COL,
            columns="icd9_code",
            values="present",
            fill_value=0,
        )
    )

    feature_cols = [
        "age_years",
        "gender",
        "ethnicity",
        "marital_status",
        "religion",
        "language",
        "admission_type",
        "admission_location",
        "insurance",
    ]
    return admissions[[data_mod.MIMIC3_PATIENT_ID_COL] + feature_cols].merge(
        diag_wide, on=data_mod.MIMIC3_PATIENT_ID_COL, how="left"
    )


def impute_with_hyperimpute(df: pd.DataFrame, id_col: str, method: str = "gain") -> pd.DataFrame:
    """
    method in {'gain', 'miwae', 'mice', 'missforest'} — HyperImpute exposes all four
    under one API, letting this be a single call rather than four separate library
    integrations
    """
    try:
        from hyperimpute.plugins.imputers import Imputers
    except ImportError as e:
        raise ImportError("HyperImpute is required. Install with: uv add hyperimpute") from e

    feature_df = df.drop(columns=[id_col])
    categorical_cols = feature_df.select_dtypes(exclude="number").columns.tolist()
    numeric_df = feature_df.copy()
    for col in categorical_cols:
        numeric_df[col] = numeric_df[col].astype("category").cat.codes.replace(-1, pd.NA)

    imputer = Imputers().get(method)
    imputed = imputer.fit_transform(numeric_df)
    imputed.columns = numeric_df.columns
    imputed[id_col] = df[id_col].to_numpy()
    return imputed
