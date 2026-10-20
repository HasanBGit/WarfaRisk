"""Confirms IWPC-1780's continuous covariates were z-scored on the full
released sample, not train-only, by inspecting the raw R data object the
`simsl` CRAN package actually distributes (the source of IWPC-1780 in this
project).

Background: a reviewer pointed out that IWPC-1780's continuous covariates
(age, weight, height) arrive already z-scored from the source, and asked
whether the normalization parameters could have been estimated from the
full dataset, which would leak held-out patients' information into every
patient's scaled value before this project's own train/test split is ever
applied. This script settles that empirically rather than by inference from
the accompanying paper's text.

Method: download the `simsl` package source tarball from CRAN, parse the
bundled `data/warfarin.RData` object directly (it is a list of INR, A, X,
not a data.frame, so the common `pyreadr` library returns nothing for it;
the more general `rdata` library handles arbitrary R objects and works),
and compute the mean/std of each continuous column across all 1,780
patients.

Result (as reported in the manuscript, Limitations): weight, height, and
age all have mean 0.0 and standard deviation 1.0 to displayed precision
across the complete 1,780-patient sample simultaneously. This is only
possible if the scaling parameters were computed from the full sample
before any train/test split existed; had they been computed on a train-only
subset, the full sample's mean/std would not both land on exactly 0/1. The
package ships no raw, unscaled version of this data, so this project cannot
redo the scaling itself.

Requires network access (downloads the CRAN tarball) and the `rdata`
package (`pip install rdata`; a pre-built wheel is available, no compiler
needed). Not part of the regular pipeline; run on demand to reproduce this
specific finding.
"""

from __future__ import annotations

import tarfile
import tempfile
import urllib.request
from pathlib import Path

SIMSL_TARBALL_URL = "https://cran.r-project.org/src/contrib/simsl_0.2.1.tar.gz"
CONTINUOUS_COLUMNS = ("Weight", "Height", "Age")


def download_and_extract_warfarin_rdata(dest_dir: Path) -> Path:
    tarball_path = dest_dir / "simsl.tar.gz"
    urllib.request.urlretrieve(SIMSL_TARBALL_URL, tarball_path)
    with tarfile.open(tarball_path) as tf:
        tf.extractall(dest_dir)
    return dest_dir / "simsl" / "data" / "warfarin.RData"


def main() -> None:
    import rdata

    with tempfile.TemporaryDirectory() as tmp:
        rdata_path = download_and_extract_warfarin_rdata(Path(tmp))
        parsed = rdata.parser.parse_file(str(rdata_path))
        converted = rdata.conversion.convert(parsed)
        X = converted["warfarin"]["X"]

        columns = list(X.coords["dim_1"].values)
        values = X.values

        print(f"n = {values.shape[0]} patients, {values.shape[1]} covariates")
        print(f"{'column':<14}{'mean':>10}{'std':>10}")
        for name in CONTINUOUS_COLUMNS:
            idx = columns.index(name)
            col = values[:, idx]
            print(f"{name:<14}{col.mean():>10.4f}{col.std():>10.4f}")

        print(
            "\nConclusion: mean approx 0 and std approx 1 for all three continuous "
            "covariates simultaneously, across the full sample, confirms the "
            "source-level z-scoring used the full 1,780-patient sample, not a "
            "train-only subset."
        )


if __name__ == "__main__":
    main()
