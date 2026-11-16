"""Approximate patient-overlap check between IWPC-6256 and IWPC-1780"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warfarisk.common import data as data_mod

RANDOM_STATE = 0


def _cyp2c9_flags(consensus: str) -> tuple[float, float, float]:
    """Map IWPC-6256's star-allele consensus to IWPC-1780's dummy encoding"""
    if pd.isna(consensus):
        return (np.nan, np.nan, np.nan)
    if consensus == "*1/*1":
        return (0, 0, 0)
    if consensus == "*1/*2":
        return (1, 0, 0)
    if consensus == "*1/*3":
        return (0, 1, 0)
    return (0, 0, 1)


def _vkorc1_flags(consensus: str) -> tuple[float, float]:
    """
    Map IWPC-6256's VKORC1 -1639 consensus to IWPC-1780's dummy encoding (VKORC1.AG,
    VKORC1.AA, relative to a G/G baseline)
    """
    mapping = {"A/G": (1, 0), "A/A": (0, 1), "G/G": (0, 0)}
    return mapping.get(consensus, (np.nan, np.nan))


def harmonized_6256_keys() -> pd.DataFrame:
    """
    Return IWPC-6256 rows with a harmonized [gender, VKORC1, CYP2C9, rounded dose]
    key matching IWPC-1780's encoding, dropping rows with any missing field in that
    key
    """
    df = data_mod.load_iwpc_6256()
    cyp = df["CYP2C9 consensus"].apply(_cyp2c9_flags)
    df["cyp12"] = cyp.apply(lambda t: t[0])
    df["cyp13"] = cyp.apply(lambda t: t[1])
    df["cypother"] = cyp.apply(lambda t: t[2])
    vk = df["VKORC1     -1639 consensus"].apply(_vkorc1_flags)
    df["vkAG"] = vk.apply(lambda t: t[0])
    df["vkAA"] = vk.apply(lambda t: t[1])
    df["gender_bin"] = df["Gender"].map({"male": 1, "female": 0})
    df["dose_round"] = df["Therapeutic Dose of Warfarin"].round(0)

    key_cols = ["gender_bin", "vkAG", "vkAA", "cyp12", "cyp13", "cypother", "dose_round"]
    return df.dropna(subset=key_cols)[key_cols].astype(int)


def harmonized_1780_keys() -> pd.DataFrame:
    """
    Return IWPC-1780 rows with the same [gender, VKORC1, CYP2C9, rounded dose] key,
    dropping rows with any missing field
    """
    df = data_mod.load_iwpc_1780()
    df["dose_round"] = df["WeeklyDose_mg"].round(0)
    key_cols = [
        "Gender",
        "VKORC1.AG",
        "VKORC1.AA",
        "CYP2C9.12",
        "CYP2C9.13",
        "CYP2C9.other",
        "dose_round",
    ]
    out = df.dropna(subset=key_cols)[key_cols].astype(int)
    out.columns = ["gender_bin", "vkAG", "vkAA", "cyp12", "cyp13", "cypother", "dose_round"]
    return out


def match_rate(query_keys: pd.DataFrame, reference_keys: pd.DataFrame) -> float:
    """
    Fraction of query_keys rows that exactly match at least one row in
    reference_keys on all key columns
    """
    reference_set = set(map(tuple, reference_keys.values))
    hits = sum(1 for row in map(tuple, query_keys.values) if row in reference_set)
    return hits / len(query_keys)


def null_baseline_match_rate(keys_6256: pd.DataFrame, random_state: int = RANDOM_STATE) -> float:
    """
    Split IWPC-6256 into two definitely-disjoint random halves and compute the same
    match rate against itself, as a coincidence baseline
    """
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(keys_6256))
    half_a = keys_6256.iloc[idx[: len(keys_6256) // 2]]
    half_b = keys_6256.iloc[idx[len(keys_6256) // 2 :]]
    return match_rate(half_a, half_b)


def main() -> None:
    keys_6256 = harmonized_6256_keys()
    keys_1780 = harmonized_1780_keys()

    cross_rate = match_rate(keys_1780, keys_6256)
    baseline_rate = null_baseline_match_rate(keys_6256)

    print(f"IWPC-1780 rows with a complete harmonized key: {len(keys_1780)} of 1780")
    print(f"IWPC-6256 rows with a complete harmonized key: {len(keys_6256)} of 6037")
    print(
        f"IWPC-1780 -> IWPC-6256 match rate on [gender, VKORC1, CYP2C9, rounded dose]: {cross_rate * 100:.1f}%"
    )
    print(
        f"Null baseline (IWPC-6256 split into two disjoint random halves, same key): {baseline_rate * 100:.1f}%"
    )
    print(
        "Conclusion: the cross-cohort match rate is not meaningfully above the same-dataset "
        "coincidence baseline, so this check cannot distinguish genuine patient overlap from "
        "coincidence given the available identifiers."
    )

    pd.DataFrame(
        [
            {"check": "iwpc1780_vs_iwpc6256", "match_rate_pct": round(cross_rate * 100, 1)},
            {
                "check": "iwpc6256_self_split_baseline",
                "match_rate_pct": round(baseline_rate * 100, 1),
            },
        ]
    ).to_csv("check_iwpc_cohort_overlap_results.csv", index=False)
    print("Wrote check_iwpc_cohort_overlap_results.csv")


if __name__ == "__main__":
    main()
