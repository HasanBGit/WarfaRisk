"""Single source of truth for where every dataset/split file lives on disk.

Resolved relative to this file, not hardcoded as an absolute string, so the
package still works if the project root is ever moved or checked out elsewhere.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent  # src/warfarisk/
REPO_ROOT = PACKAGE_DIR.parent.parent  # repo root
DATA_DIR = REPO_ROOT / "data"
CLEANED_DIR = DATA_DIR / "eda_notebooks" / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"
TRAINING_PLAN = REPO_ROOT / "docs" / "TRAINING_PLAN.md"

# --- IWPC cohorts (genotype + clinical, cross-sectional) ---
IWPC_6256 = CLEANED_DIR / "iwpc_full_6256_clean.csv"
IWPC_1780 = CLEANED_DIR / "iwpc_1780_clean.csv"

# --- eICU (clinical-only, longitudinal, no genotype) ---
EICU_INR_LABS = CLEANED_DIR / "eicu_inr_labs_clean.csv"
EICU_DEMOGRAPHICS = CLEANED_DIR / "eicu_patient_demographics_clean.csv"
EICU_WARFARIN_MEDICATION = CLEANED_DIR / "eicu_warfarin_medication_clean.csv"
EICU_WARFARIN_ADMISSIONDRUG = CLEANED_DIR / "eicu_warfarin_admissiondrug_clean.csv"
EICU_PASTHISTORY = CLEANED_DIR / "eicu_pasthistory_clean.csv"

# --- MIMIC-III (clinical-only, longitudinal, no genotype) ---
MIMIC3_ADMISSIONS = CLEANED_DIR / "mimic3_admissions_clean.csv"
MIMIC3_INR_TRAJECTORY = CLEANED_DIR / "mimic3_inr_trajectory_clean.csv"
MIMIC3_WARFARIN_PRESCRIPTIONS = CLEANED_DIR / "mimic3_warfarin_prescriptions_clean.csv"
MIMIC3_DIAGNOSES = CLEANED_DIR / "mimic3_diagnoses_clean.csv"

# --- MIMIC-IV (clinical-only, longitudinal, no genotype) ---
MIMIC4_ADMISSIONS = CLEANED_DIR / "mimic4_admissions_clean.csv"
MIMIC4_INR_LABS = CLEANED_DIR / "mimic4_inr_labs_clean.csv"
MIMIC4_WARFARIN_PRESCRIPTIONS = CLEANED_DIR / "mimic4_warfarin_prescriptions_clean.csv"
MIMIC4_DIAGNOSES = CLEANED_DIR / "mimic4_diagnoses_clean.csv"

# --- reference-only, not training rows ---
FAERS_WARFARIN = CLEANED_DIR / "faers_warfarin_deduped.csv"

# NOTE: raw source data (IWPC/PharmGKB/warfit-learn/MIMIC/eICU) is not shipped
# in this repo — see data/DATA.md for how to obtain and rebuild CLEANED_DIR
# locally. PHARMGKB_ALLELE_TABLES_DIR / WARFIT_LEARN_DIR intentionally removed;
# phase9_llm_explanation.py needs local PharmGKB allele tables per DATA.md
# before it can run.


def assert_data_dir_exists() -> None:
    """Fail loudly and early if the cleaned-data directory isn't where every phase expects it."""
    if not CLEANED_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected cleaned-data directory at {CLEANED_DIR!s}, not found. "
            "See data/DATA.md for how to obtain the source data and run the "
            "notebooks in data/eda_notebooks/ to rebuild it, or check the "
            "project layout hasn't moved."
        )
