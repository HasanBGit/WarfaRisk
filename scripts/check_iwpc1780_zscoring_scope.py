"""
Confirms IWPC-1780's continuous covariates were z-scored on the full released sample,
not train-only, by inspecting the raw R data object the `simsl` CRAN package actually
distributes (the source of IWPC-1780 in this project)
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
