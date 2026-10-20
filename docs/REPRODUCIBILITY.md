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

Models: Hugging Face `HassanB4/warfarisk-baselines`, `warfarisk-autogluon-6256`,
`warfarisk-autogluon-1780`.
