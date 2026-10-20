"""95% percentile-bootstrap CIs for the overall 90%-nominal interval coverage
(10,000 resamples of the test patients, seed 20260725) plus mean/median width.

Needs the per-patient prediction files written by run_mapie_local.py and
run_phase5_local.py (phase5_<cohort>_<method>_predictions.csv). Those files are
deliberately not committed (see .gitignore and data/DATA.md); regenerate them
locally first. Run from the directory that holds them.
"""

import sys

import numpy as np
import pandas as pd

SEED = 20260725
N_BOOT = 10_000

for path in sys.argv[1:] or ["phase5_iwpc6256_mapie_predictions.csv", "phase5_iwpc6256_deep_ensemble_predictions.csv"]:
    d = pd.read_csv(path)
    covered = ((d.y_true >= d.lower) & (d.y_true <= d.upper)).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = [covered[rng.integers(0, len(covered), len(covered))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    width = d.upper - d.lower
    print(f"{path}: coverage={covered.mean():.4f} CI=[{lo:.4f}, {hi:.4f}] width mean={width.mean():.2f} median={width.median():.2f}")
