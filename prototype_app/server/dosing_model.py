"""Feature construction, inference and leave-one-out attribution for the CatBoost model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

MIN_WEEKLY_DOSE = 7.0
MAX_WEEKLY_DOSE = 150.0
MIN_INTERVAL_LOW = 5.0
MIN_IMPACT = 0.05

VKORC1_CONSENSUS = {"GG": "G/G", "AG": "A/G", "AA": "A/A"}
VKORC1_COL = "VKORC1     -1639 consensus"
CYP2C9_COL = "CYP2C9 consensus"

NOT_COLLECTED_BY_UI = frozenset(
    {
        "Gender",
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
        "Carbamazepine (Tegretol)",
        "Phenytoin (Dilantin)",
        "Rifampin or Rifampicin",
        "Sulfonamide Antibiotics",
        "Macrolide Antibiotics",
        "Anti-fungal Azoles",
        "Herbal Medications, Vitamins, Supplements",
        "Current Smoker",
        "VKORC1 497 consensus",
        "VKORC1 1173 consensus",
        "VKORC1 1542 consensus",
        "VKORC1 3730 consensus",
        "VKORC1 2255 consensus",
        "VKORC1     -4451 consensus",
    }
)


def age_to_bucket(age: float) -> str:
    """Convert an age to the decade-bucket text used by the IWPC-6256 codebook."""
    age = int(age)
    if age >= 90:
        return "90+"
    decade = (age // 10) * 10
    return f"{decade} - {decade + 9}"


def vkorc1_to_consensus(code: str) -> str:
    return VKORC1_CONSENSUS.get(code, "missing")


class DosingModel:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.metadata = json.loads(metadata_path.read_text())
        self.feature_cols = self.metadata["feature_cols"]
        self.cat_features = self.metadata["cat_features"]
        self.defaults = self.metadata["defaults"]
        self._model = CatBoostRegressor()
        self._model.load_model(str(model_path))

    @property
    def conformal_halfwidth(self) -> float:
        return self.metadata["conformal_halfwidth_90"]

    def build_feature_row(self, patient: dict) -> dict:
        row = dict(self.defaults)
        row["Age"] = age_to_bucket(patient["age"])
        row["Height (cm)"] = float(patient["height"])
        row["Weight (kg)"] = float(patient["weight"])

        vkorc1 = patient.get("vkorc1", "unknown")
        row[VKORC1_COL] = vkorc1_to_consensus(vkorc1) if vkorc1 != "unknown" else "missing"

        cyp2c9 = patient.get("cyp2c9", "unknown")
        row[CYP2C9_COL] = cyp2c9 if cyp2c9 != "unknown" else "missing"

        row["Amiodarone (Cordarone)"] = 1 if patient.get("amiodarone") else 0
        if patient.get("statin"):
            row["Atorvastatin (Lipitor)"] = 1
        if patient.get("enzymeInducer"):
            row["Rifampin or Rifampicin"] = 1
            row["Carbamazepine (Tegretol)"] = 1
        return row

    def predict_row(self, row: dict) -> float:
        features = pd.DataFrame([row], columns=self.feature_cols)
        features[self.cat_features] = features[self.cat_features].fillna("missing")
        pool = Pool(features, cat_features=self.cat_features)
        return float(self._model.predict(pool)[0])

    def predict_with_interval(self, row: dict) -> tuple[float, float, float]:
        prediction = max(MIN_WEEKLY_DOSE, min(MAX_WEEKLY_DOSE, self.predict_row(row)))
        low = max(MIN_INTERVAL_LOW, round(prediction - self.conformal_halfwidth, 1))
        high = round(prediction + self.conformal_halfwidth, 1)
        return prediction, low, high

    def _probe_fields(self, row: dict) -> list[tuple[str, str, str]]:
        return [
            (VKORC1_COL, f"VKORC1 -1639G>A ({row[VKORC1_COL]})", "genetics"),
            (CYP2C9_COL, f"CYP2C9 ({row[CYP2C9_COL]})", "genetics"),
            ("Age", f"Age bucket ({row['Age']})", "clinical"),
            ("Weight (kg)", f"Weight ({row['Weight (kg)']} kg)", "clinical"),
            ("Height (cm)", f"Height ({row['Height (cm)']} cm)", "clinical"),
            ("Amiodarone (Cordarone)", "Amiodarone Co-medication", "clinical"),
            ("Rifampin or Rifampicin", "Enzyme Inducer Therapy", "clinical"),
        ]

    def leave_one_out_impacts(self, row: dict) -> list[dict]:
        """Re-run the model with each collected field reset to its default."""
        full_prediction = self.predict_row(row)
        impacts = []
        for column, label, category in self._probe_fields(row):
            if row[column] == self.defaults[column]:
                continue
            probe_row = {**row, column: self.defaults[column]}
            impact = round(full_prediction - self.predict_row(probe_row), 1)
            if abs(impact) < MIN_IMPACT:
                continue
            impacts.append({"feature": label, "impact": impact, "category": category})
        impacts.sort(key=lambda item: -abs(item["impact"]))
        return impacts
