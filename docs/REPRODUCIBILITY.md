# Reproducing the reported numbers

Seed `20260725` everywhere (splits, MAPIE, Mondrian, bootstraps). Per-patient
prediction files are not shipped (see `data/DATA.md`); regenerate them with the
`run_*` scripts, then run the checks from the directory that holds them.

| Reported item | Script | Output in `results/` |
|---|---|---|
| Baselines, nested ablation | `phase1_baselines`, `phase3_genetics_ablation` | `phase3_full_iwpc6256_summary.csv` |
| IWPC-1780 CatBoost | `run_catboost_local.py` | `phase3_catboost_iwpc1780_results.csv` |
| Ancestry hold-out (portability penalty) | `run_phase4_local.py`, `run_catboost_phase4_local.py` | `phase4_*_results.csv` |
| MAPIE / deep-ensemble / NGBoost coverage | `run_mapie_local.py`, `run_phase5_local.py`, `run_ngboost_local.py` | `phase5_*_coverage.csv` |
| Coverage bootstrap CIs, interval width | `bootstrap_coverage_ci.py` | stdout |
| Per-label binomial test (BH) | `check_mapie_ancestry_binomial_ci.py` | `check_mapie_ancestry_binomial_ci_results.csv` |
| Four-label widths | `check_mapie_ancestry_width.py` | `check_mapie_ancestry_width_results.csv` |
| Mondrian vs MAPIE | `run_mondrian_local.py` | `phase5_*_mondrian_vs_mapie_coverage.csv`, `*_mondrian_quantiles.csv` |
| Pooled-label sensitivity, Mondrian width | `check_pooled_black_sensitivity.py`, `check_mondrian_interval_width.py` | `check_pooled_black_sensitivity_results.csv`, `check_mondrian_interval_width_results.csv` |
| IWPC equation vs stacking (paired) | `check_iwpc_vs_stacking_paired.py` | `check_iwpc_vs_stacking_paired_results.csv` |
| Cohort overlap, z-scoring scope | `check_iwpc_cohort_overlap.py`, `check_iwpc1780_zscoring_scope.py` | `check_iwpc_cohort_overlap_results.csv` |
| SHAP importance | `run_phase6_local.py` | `phase6_*_shap.csv` |

Note: `mapie_conformal_interval` is seeded via `random_state` (default
`20260725`); without it, MAPIE cross-conformal coverage varies between runs
(e.g. Black subgroup 80.8% vs 82.2%). The published figures correspond to the
saved prediction files from the original run, so a fresh MAPIE run may differ
by about a percentage point.

The `method=mapie` rows in `results/phase5_*_mondrian_vs_mapie_coverage.csv` come
from the same-run MAPIE baseline of `run_mondrian_local.py` (overall 90.2%,
Black 80.8% on IWPC-6256). The supplement's MAPIE column (overall 89.8%, Black
82.2%) comes from the headline file `phase5_iwpc6256_mapie_predictions.csv`
(see `results/phase5_iwpc6256_mapie_coverage.csv` and
`check_mapie_ancestry_binomial_ci_results.csv`). The Mondrian columns are
identical in both.

Models (Hugging Face, account `HassanB4`): `warfarisk-baselines`,
`warfarisk-autogluon-6256`, `warfarisk-autogluon-1780`, `warfarisk-catboost-6256`.
Split files and SHAP/EBM outputs are in the AutoGluon repos under `splits/` and `shap/`.

## Fresh-clone verification (2026-10-20)

Clean clone of this repo, clean Python 3.11 environment (mapie 1.6.0,
scikit-learn 1.9.1), regenerated cleaned IWPC CSVs, then `scripts/run_mondrian_local.py`:

- Mondrian quantiles and Mondrian coverage reproduce exactly (identical
  quantile CSVs, zero coverage difference) on both cohorts.
- The MAPIE baseline does **not** reproduce exactly across environments even
  with `random_state=20260725`. Overall coverage is stable (IWPC-6256 90.2%,
  IWPC-1780 93.3%), but small-subgroup coverage moves: IWPC-6256 Black was 84.9%
  in this run, 80.8% in the earlier same-run baseline and 82.2% in the headline
  file. Subgroup MAPIE figures should therefore be read from the saved headline
  results, with a few percentage points of run-to-run variation for subgroups of
  about 50-75 patients.
- `check_*` scripts regenerate their results CSVs byte-identically from the
  saved predictions.
- Evidence logs under `results/logs/` cover the later phases; the Phase 1 and
  Phase 3 console logs cited in `docs/RESULTS.md` (`logs/phase1_*`,
  `logs/phase3_*`) were not preserved. Their numbers are reproduced by
  `results/phase3_full_iwpc6256_summary.csv` and the model `metrics.json` files
  on Hugging Face.
- CatBoost: `run_catboost_local.py` Phase 3 (both cohorts) and
  `run_catboost_phase4_local.py` (both cohorts) reproduce the committed CSVs
  byte-identically in the clean environment. The trailing Phase 4 section of
  `run_catboost_local.py` fails on string-valued categorical columns; use
  `run_catboost_phase4_local.py` for the Phase 4 results.
- Stacking ensemble (`run_phase4_local.py`, IWPC-6256) and NGBoost
  (`run_ngboost_local.py`, both cohorts) also reproduce the committed
  CSVs byte-identically in the clean environment.
- App latency (local, CatBoost + leave-one-out attributions, 30 calls to
  `/predict`): median 11 ms, 95th percentile 25 ms. LLM-explanation timings
  depend on the external inference provider and are not reproducible offline.
- Not re-run: AutoGluon `extreme_quality` (GPU, hours); its saved metrics are
  on the Hugging Face model cards.
- AutoGluon (IWPC-6256): the published `HassanB4/warfarisk-autogluon-6256`
  model, downloaded from the Hub and evaluated on the fixed test split on a
  different machine (macOS, CPU, AutoGluon 1.5.0), gives MAE 8.556, R2 0.475,
  PW20 0.465 against the stored 8.552 / 0.475 / 0.466. The 0.004 difference
  is platform-level floating-point variation in TabPFN inference.
