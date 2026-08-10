"""Phase 0 — reproducible, patient/stay-ID-level train/test splits.

Splits are computed once, saved to splits/*.json as ID lists, and re-loaded by
every later phase — never re-randomized. This is what makes "same split" a
guarantee instead of a hope: two runs of the same phase on the same dataset
name always evaluate on identical held-out patients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import paths

DEFAULT_SEED = 20260725  # today's date at the time this pipeline was built; fixed, not re-rolled
DEFAULT_TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Split:
    dataset_name: str
    id_col: str
    seed: int
    train_ids: list
    test_ids: list

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def make_split(
    df: pd.DataFrame,
    id_col: str,
    dataset_name: str,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> Split:
    """Split by unique ID (patient or stay), not by row — required whenever a
    dataset can have more than one row per patient (e.g. eICU/MIMIC longitudinal
    labs), and harmless when it's already one row per patient (IWPC)."""
    unique_ids = df[id_col].dropna().unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]
    return Split(
        dataset_name=dataset_name,
        id_col=id_col,
        seed=seed,
        train_ids=sorted(train_ids.tolist()),
        test_ids=sorted(test_ids.tolist()),
    )


def save_split(split: Split) -> None:
    paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = paths.SPLITS_DIR / f"{split.dataset_name}.json"
    out_path.write_text(split.to_json())


def load_split(dataset_name: str) -> Split:
    in_path = paths.SPLITS_DIR / f"{dataset_name}.json"
    if not in_path.exists():
        raise FileNotFoundError(
            f"No saved split for {dataset_name!r} at {in_path!s}. "
            "Run phase0_scaffolding.generate_all_splits() first."
        )
    data = json.loads(in_path.read_text())
    return Split(**data)


def apply_split(df: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) using a previously saved Split's ID lists."""
    train_ids = set(split.train_ids)
    test_ids = set(split.test_ids)
    train_df = df[df[split.id_col].isin(train_ids)].copy()
    test_df = df[df[split.id_col].isin(test_ids)].copy()
    assert train_ids.isdisjoint(test_ids), "train/test ID sets must be disjoint"
    return train_df, test_df
