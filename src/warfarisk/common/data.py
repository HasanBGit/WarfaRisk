"""Loaders and column-name constants for the cleaned datasets.

Column names are copied verbatim from the real headers in
data/eda_notebooks/cleaned/*.csv (confirmed by direct inspection), not
guessed — including IWPC's original PharmGKB column names, whitespace and all.
Centralizing them here means a typo gets caught once, in one place, instead of
silently breaking a phase file that hand-typed the string.
"""

from __future__ import annotations

import pandas as pd

from . import paths

# ---------------------------------------------------------------------------
# IWPC-6256 (data/eda_notebooks/cleaned/iwpc_full_6256_clean.csv), n=6,037
# Raw units (Weight kg, Height cm, Age years). No CYP4F2 column exists.
# ---------------------------------------------------------------------------

IWPC6256_TARGET = "Therapeutic Dose of Warfarin"

IWPC6256_ANCESTRY_COLS = ["Race (Reported)", "Race (OMB)", "Ethnicity (Reported)", "Ethnicity (OMB)"]

# Columns that are downstream of / derived from the outcome and must never be
# used as model inputs for a dose-prediction target (see Phase 2 leakage audit).
IWPC6256_LEAKAGE_COLS = [
    "INR on Reported Therapeutic Dose of Warfarin",
    "Target INR",
    "Estimated Target INR Range Based on Indication",
    "Subject Reached Stable Dose of Warfarin",
]

IWPC6256_CLINICAL_COLS = [
    "Gender",
    "Age",
    "Height (cm)",
    "Weight (kg)",
    "Indication for Warfarin Treatment",
    # "Comorbidities" deliberately excluded: it's free-text, semicolon-joined
    # ("cancer; renal insufficiency", "no diabetes", ...), 1,595 unique values
    # across 6,037 rows. One-hot encoding it (as with_standard_preprocessing
    # does for every non-numeric column) explodes the feature space to ~1,595
    # near-singleton columns, badly slowing every model and adding pure noise
    # rather than signal. Its actually-useful structured content (diabetes,
    # CHF, valve replacement) is already captured by the dedicated flag
    # columns below (confirmed: Diabetes==1 co-occurs with "diabetes" in this
    # free-text field in 578/632 cases) — keep those, drop the free text.
    "Diabetes",
    "Congestive Heart Failure and/or Cardiomyopathy",
    "Valve Replacement",
    "Aspirin",
    "Acetaminophen or Paracetamol (Tylenol)",
    "Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day",
    "Simvastatin (Zocor)",
    "Atorvastatin (Lipitor)",
    "Fluvastatin (Lescol)",
    "Lovastatin (Mevacor)",
    "Pravastatin (Pravachol)",
    "Rosuvastatin (Crestor)",
    "Cerivastatin (Baycol)",
    "Amiodarone (Cordarone)",
    "Carbamazepine (Tegretol)",
    "Phenytoin (Dilantin)",
    "Rifampin or Rifampicin",
    "Sulfonamide Antibiotics",
    "Macrolide Antibiotics",
    "Anti-fungal Azoles",
    "Herbal Medications, Vitamins, Supplements",
    "Current Smoker",
]

# The "consensus" columns are PharmGKB's already-harmonized genotype calls
# (cleaner than pairing every raw genotype column with its QC column by hand).
IWPC6256_GENETIC_COLS = [
    "CYP2C9 consensus",
    "VKORC1     -1639 consensus",
    "VKORC1 497 consensus",
    "VKORC1 1173 consensus",
    "VKORC1 1542 consensus",
    "VKORC1 3730 consensus",
    "VKORC1 2255 consensus",
    "VKORC1     -4451 consensus",
]


def load_iwpc_6256() -> pd.DataFrame:
    return pd.read_csv(paths.IWPC_6256)


# ---------------------------------------------------------------------------
# IWPC-1780 (data/eda_notebooks/cleaned/iwpc_1780_clean.csv), n=1,780
# Weight/Height/Age are PRE-Z-SCORED in this file — do not pool raw with 6256.
# ---------------------------------------------------------------------------

IWPC1780_TARGET = "WeeklyDose_mg"
IWPC1780_ANCESTRY_COLS = ["Black", "Asian"]
IWPC1780_LEAKAGE_COLS = ["INR"]
IWPC1780_CLINICAL_COLS = ["Weight", "Height", "Age", "Enzyme", "Amiodarone", "Gender"]
IWPC1780_GENETIC_COLS = ["VKORC1.AG", "VKORC1.AA", "CYP2C9.12", "CYP2C9.13", "CYP2C9.other"]


def load_iwpc_1780() -> pd.DataFrame:
    return pd.read_csv(paths.IWPC_1780)


# ---------------------------------------------------------------------------
# eICU (clinical-only, longitudinal, no genotype)
# ---------------------------------------------------------------------------

EICU_STAY_ID_COL = "patientunitstayid"
EICU_INR_LABNAME_TARGET = "PT - INR"  # NOT "PT" (seconds) or "inr" substring match — see project memory on this bug

def load_eicu_inr_labs() -> pd.DataFrame:
    return pd.read_csv(paths.EICU_INR_LABS)


def load_eicu_demographics() -> pd.DataFrame:
    return pd.read_csv(paths.EICU_DEMOGRAPHICS)


def load_eicu_warfarin_medication() -> pd.DataFrame:
    return pd.read_csv(paths.EICU_WARFARIN_MEDICATION)


def load_eicu_warfarin_admissiondrug() -> pd.DataFrame:
    return pd.read_csv(paths.EICU_WARFARIN_ADMISSIONDRUG)


def load_eicu_pasthistory() -> pd.DataFrame:
    return pd.read_csv(paths.EICU_PASTHISTORY)


# ---------------------------------------------------------------------------
# MIMIC-III (clinical-only, longitudinal, no genotype)
# ---------------------------------------------------------------------------

MIMIC3_PATIENT_ID_COL = "subject_id"
MIMIC3_STAY_ID_COL = "hadm_id"


def load_mimic3_admissions() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC3_ADMISSIONS)


def load_mimic3_inr_trajectory() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC3_INR_TRAJECTORY)


def load_mimic3_warfarin_prescriptions() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC3_WARFARIN_PRESCRIPTIONS)


def load_mimic3_diagnoses() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC3_DIAGNOSES)


# ---------------------------------------------------------------------------
# MIMIC-IV (clinical-only, longitudinal, no genotype)
# ---------------------------------------------------------------------------

MIMIC4_PATIENT_ID_COL = "subject_id"
MIMIC4_STAY_ID_COL = "hadm_id"


def load_mimic4_admissions() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC4_ADMISSIONS)


def load_mimic4_inr_labs() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC4_INR_LABS)


def load_mimic4_warfarin_prescriptions() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC4_WARFARIN_PRESCRIPTIONS)


def load_mimic4_diagnoses() -> pd.DataFrame:
    return pd.read_csv(paths.MIMIC4_DIAGNOSES)


# ---------------------------------------------------------------------------
# Feature-set helpers shared by Phase 3 (ablation), Phase 4 (ancestry holdout),
# and Phase 6 (explainability) so the three nested feature sets are defined
# exactly once.
# ---------------------------------------------------------------------------


def iwpc6256_feature_set(df: pd.DataFrame, which: str) -> pd.DataFrame:
    """Return X for the 6256 cohort. which in {'clinical', 'genetic', 'combined'}."""
    if which == "clinical":
        cols = IWPC6256_CLINICAL_COLS
    elif which == "genetic":
        cols = IWPC6256_GENETIC_COLS
    elif which == "combined":
        cols = IWPC6256_CLINICAL_COLS + IWPC6256_GENETIC_COLS
    else:
        raise ValueError(f"which must be 'clinical', 'genetic', or 'combined', got {which!r}")
    return df[cols]


def iwpc1780_feature_set(df: pd.DataFrame, which: str) -> pd.DataFrame:
    """Return X for the 1780 cohort. which in {'clinical', 'genetic', 'combined'}."""
    if which == "clinical":
        cols = IWPC1780_CLINICAL_COLS
    elif which == "genetic":
        cols = IWPC1780_GENETIC_COLS
    elif which == "combined":
        cols = IWPC1780_CLINICAL_COLS + IWPC1780_GENETIC_COLS
    else:
        raise ValueError(f"which must be 'clinical', 'genetic', or 'combined', got {which!r}")
    return df[cols]
