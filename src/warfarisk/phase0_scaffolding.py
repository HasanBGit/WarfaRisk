"""Phase 0 — reproducibility scaffolding.

Implements TRAINING_PLAN.md Phase 0: generate and persist every dataset's
train/test split once (patient/stay-ID-level, seeded), and export a data
dictionary mapping raw column names to the constants defined in common/data.py.

Nothing here trains a model. Run generate_all_splits() once before any other
phase; every later phase loads splits via common.splits.load_split(), it
never re-randomizes.
"""

from __future__ import annotations

import json

from warfarisk.common import data as data_mod
from warfarisk.common import paths
from warfarisk.common import splits


def generate_all_splits(seed: int = splits.DEFAULT_SEED) -> dict[str, splits.Split]:
    """One row per patient in the IWPC cohorts, one row per stay/admission in
    the ICU cohorts — id_col chosen accordingly so a stay-level dataset is
    never accidentally split by lab-result row."""
    generated: dict[str, splits.Split] = {}

    iwpc6256 = data_mod.load_iwpc_6256()
    split = splits.make_split(iwpc6256, id_col="PharmGKB Subject ID", dataset_name="iwpc_6256", seed=seed)
    splits.save_split(split)
    generated["iwpc_6256"] = split

    iwpc1780 = data_mod.load_iwpc_1780()
    split = splits.make_split(iwpc1780, id_col="PatientID", dataset_name="iwpc_1780", seed=seed)
    splits.save_split(split)
    generated["iwpc_1780"] = split

    eicu_demo = data_mod.load_eicu_demographics()
    split = splits.make_split(eicu_demo, id_col=data_mod.EICU_STAY_ID_COL, dataset_name="eicu", seed=seed)
    splits.save_split(split)
    generated["eicu"] = split

    mimic3_adm = data_mod.load_mimic3_admissions()
    split = splits.make_split(mimic3_adm, id_col=data_mod.MIMIC3_PATIENT_ID_COL, dataset_name="mimic3", seed=seed)
    splits.save_split(split)
    generated["mimic3"] = split

    mimic4_adm = data_mod.load_mimic4_admissions()
    split = splits.make_split(mimic4_adm, id_col=data_mod.MIMIC4_PATIENT_ID_COL, dataset_name="mimic4", seed=seed)
    splits.save_split(split)
    generated["mimic4"] = split

    return generated


def export_data_dictionary() -> dict:
    """Maps every raw column name used by the pipeline to the common/data.py
    constant that references it and which cohort it belongs to — the
    human-readable artifact TRAINING_PLAN.md Phase 0 asks for."""
    dictionary = {
        "iwpc_6256": {
            "target": data_mod.IWPC6256_TARGET,
            "clinical_columns": data_mod.IWPC6256_CLINICAL_COLS,
            "genetic_columns": data_mod.IWPC6256_GENETIC_COLS,
            "ancestry_columns": data_mod.IWPC6256_ANCESTRY_COLS,
            "leakage_columns_excluded_from_X": data_mod.IWPC6256_LEAKAGE_COLS,
        },
        "iwpc_1780": {
            "target": data_mod.IWPC1780_TARGET,
            "clinical_columns": data_mod.IWPC1780_CLINICAL_COLS,
            "genetic_columns": data_mod.IWPC1780_GENETIC_COLS,
            "ancestry_columns": data_mod.IWPC1780_ANCESTRY_COLS,
            "leakage_columns_excluded_from_X": data_mod.IWPC1780_LEAKAGE_COLS,
            "note": "Weight/Height/Age are pre-z-scored in this file — do not pool raw with iwpc_6256",
        },
        "eicu": {
            "stay_id_col": data_mod.EICU_STAY_ID_COL,
            "inr_labname_filter": data_mod.EICU_INR_LABNAME_TARGET,
            "note": "labname == 'PT - INR' only, NOT 'PT' (seconds) or substring 'inr' match",
        },
        "mimic3": {"patient_id_col": data_mod.MIMIC3_PATIENT_ID_COL, "stay_id_col": data_mod.MIMIC3_STAY_ID_COL},
        "mimic4": {"patient_id_col": data_mod.MIMIC4_PATIENT_ID_COL, "stay_id_col": data_mod.MIMIC4_STAY_ID_COL},
    }
    return dictionary


def write_data_dictionary(out_path=None) -> None:
    out_path = out_path or (paths.REPO_ROOT / "data_dictionary.json")
    out_path.write_text(json.dumps(export_data_dictionary(), indent=2))


if __name__ == "__main__":
    # Explicit entry point — running this file is the one intentional
    # "prepare the pipeline" action; it does not fit or evaluate any model.
    paths.assert_data_dir_exists()
    generate_all_splits()
    write_data_dictionary()
    print(f"Splits written to {paths.SPLITS_DIR}")
    print(f"Data dictionary written to {paths.REPO_ROOT / 'data_dictionary.json'}")
