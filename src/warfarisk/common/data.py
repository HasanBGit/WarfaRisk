"""Loaders and column-name constants for the cleaned datasets"""

from __future__ import annotations

import pandas as pd

from . import paths

IWPC6256_TARGET = "Therapeutic Dose of Warfarin"

IWPC6256_ANCESTRY_COLS = [
    "Race (Reported)",
    "Race (OMB)",
    "Ethnicity (Reported)",
    "Ethnicity (OMB)",
]

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


IWPC1780_TARGET = "WeeklyDose_mg"
IWPC1780_ANCESTRY_COLS = ["Black", "Asian"]
IWPC1780_LEAKAGE_COLS = ["INR"]
IWPC1780_CLINICAL_COLS = ["Weight", "Height", "Age", "Enzyme", "Amiodarone", "Gender"]
IWPC1780_GENETIC_COLS = ["VKORC1.AG", "VKORC1.AA", "CYP2C9.12", "CYP2C9.13", "CYP2C9.other"]


def load_iwpc_1780() -> pd.DataFrame:
    return pd.read_csv(paths.IWPC_1780)


EICU_STAY_ID_COL = "patientunitstayid"
EICU_INR_LABNAME_TARGET = "PT - INR"


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


def iwpc6256_feature_set(df: pd.DataFrame, which: str) -> pd.DataFrame:
    """Return X for the 6256 cohort. which in {'clinical', 'genetic', 'combined'}"""
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
    """Return X for the 1780 cohort. which in {'clinical', 'genetic', 'combined'}"""
    if which == "clinical":
        cols = IWPC1780_CLINICAL_COLS
    elif which == "genetic":
        cols = IWPC1780_GENETIC_COLS
    elif which == "combined":
        cols = IWPC1780_CLINICAL_COLS + IWPC1780_GENETIC_COLS
    else:
        raise ValueError(f"which must be 'clinical', 'genetic', or 'combined', got {which!r}")
    return df[cols]
