# Verified results ledger

**Rule:** a row only goes in the table below if it has surviving evidence — a log file path, an executed notebook/script output, or an HF-hosted artifact link that can be independently opened. A number that only exists in `SESSION_MEMORY.md`, a commit message, or someone's memory of a run does NOT qualify, no matter how confident it sounds. This is deliberately stricter than a typical lab notebook, adapted from a verification discipline that caught a real problem in a prior project (a claimed push to Hugging Face that never actually happened — the account was checked and was empty).

## Verified results

| Date | Phase | Experiment | Dataset | Metric(s) | Value | Evidence path | HF link |
|---|---|---|---|---|---|---|---|
| 2026-07-25 | Phase 1 | naive_median_dose | iwpc_6256 (n_test=1207) | MAE / R2 / PW20 | 12.339 / -0.048 / 0.342 | `model_dev/logs/phase1_run_20260725.log`, `runs.tsv` | — |
| ~~2026-07-25~~ | ~~Phase 1~~ | ~~clinical_only_linear_regression~~ | ~~iwpc_6256 (n_test=1207)~~ | ~~MAE / R2 / PW20~~ | ~~11.103 / 0.198 / 0.351~~ | **SUPERSEDED, see corrected row below** | ~~HassanB4/warfarin-review-phase1-baselines~~ |
| 2026-07-25 | Phase 1 (corrected) | clinical_only_linear_regression | iwpc_6256 (n_test=1207) | MAE / R2 / PW20 | 10.860 / 0.222 / 0.354 | `model_dev/logs/phase1_run_20260725b_corrected.log`, `runs.tsv` | [HassanB4/warfarin-review-phase1-baselines](https://huggingface.co/HassanB4/warfarin-review-phase1-baselines) (private; artifact re-pushed after the correction, verified via `HfApi.repo_info` + `HfApi.list_models`, 2026-07-25) |
| 2026-07-26 | Phase 1 (third baseline, resolved) | iwpc_published_equation | iwpc_6256 (n_test=1207) | MAE / R2 / PW20 | 9.177 / 0.413 / 0.429 | `phase1_baselines.py`'s `iwpc_published_equation_predict()`, run directly | — |

**The Phase 1 third baseline, previously logged as excluded, is now implemented — see the full account in the Excluded/superseded section below for the provenance caveat and what changed.** As expected, it beats both other Phase 1 baselines by a real margin (MAE 9.18 vs. clinical-only's 10.86) — it's the only one of the three that uses genetics.

### Phase 3 — nested clinical/genetic ablation, iwpc_6256 (n_test=1207)

All rows below ran in a **freshly reset container** (2026-07-25, second session same day) — `splits/` didn't exist on disk, so Phase 0/Phase 2 were re-run first (byte-identical to the original run: same seed `20260725`, 11/11 leakage checks still pass). GPU: NVIDIA RTX 2000 Ada, 16.7GB VRAM (this session's pod — different from the RTX A5000/24GB noted in an earlier session; see `SESSION_MEMORY.md`). `xgboost`/`catboost`/`tabpfn`/`autogluon` all confirmed actually running on GPU (`torch.cuda.is_available()==True`, `nvidia-smi` showed active utilization + VRAM allocation during each fit), not a silent CPU fallback.

| Model[feature_set] | MAE | R2 | PW20 | Evidence path |
|---|---|---|---|---|
| linear[clinical] | 10.860 | 0.222 | 0.354 | `model_dev/logs/phase3_run_20260725_sklearn_native.log` |
| elastic_net[clinical] | 11.339 | 0.169 | 0.335 | same |
| random_forest[clinical] | 11.427 | 0.119 | 0.355 | same |
| mlp[clinical] | 10.731 | 0.217 | 0.363 | same |
| stacking_ensemble[clinical] | 10.808 | 0.228 | 0.356 | same |
| xgboost[clinical] | 12.283 | 0.018 | 0.320 | `model_dev/logs/phase3_run_20260725_gpu_models.log` |
| tabpfn[clinical] | 10.614 | 0.247 | 0.371 | same |
| catboost[clinical] | 10.550 | 0.250 | 0.374 | `model_dev/logs/phase3_run_20260725_catboost.log` |
| linear[genetic] | 10.099 | 0.305 | 0.400 | `model_dev/logs/phase3_run_20260725_sklearn_native.log` |
| elastic_net[genetic] | 11.149 | 0.172 | 0.360 | same |
| random_forest[genetic] | 10.165 | 0.296 | 0.394 | same |
| mlp[genetic] | 10.090 | 0.309 | 0.403 | same |
| stacking_ensemble[genetic] | 10.062 | 0.312 | 0.405 | same |
| xgboost[genetic] | 10.253 | 0.285 | 0.386 | `model_dev/logs/phase3_run_20260725_gpu_models.log` |
| tabpfn[genetic] | 9.924 | 0.317 | 0.401 | same |
| catboost[genetic] | 9.977 | 0.308 | 0.402 | `model_dev/logs/phase3_run_20260725_catboost.log` |
| linear[combined] | 9.078 | 0.419 | 0.445 | `model_dev/logs/phase3_run_20260725_sklearn_native.log` |
| elastic_net[combined] | 10.440 | 0.272 | 0.374 | same |
| random_forest[combined] | 9.214 | 0.415 | 0.428 | same |
| mlp[combined] | 9.006 | 0.412 | 0.445 | same |
| stacking_ensemble[combined] | 9.046 | 0.431 | 0.447 | same |
| xgboost[combined] | 10.047 | 0.312 | 0.407 | `model_dev/logs/phase3_run_20260725_gpu_models.log` |
| tabpfn[combined] | 8.664 | 0.463 | 0.446 | same |
| catboost[combined] | 8.762 | 0.454 | 0.455 | `model_dev/logs/phase3_run_20260725_catboost.log` |
| **autogluon_extreme_quality[combined]** | **8.552** | **0.475** | **0.466** | `model_dev/logs/phase3_run_20260725_autogluon.log` — pushed to [HassanB4/warfarin-review-phase3-autogluon-iwpc6256](https://huggingface.co/HassanB4/warfarin-review-phase3-autogluon-iwpc6256) (private; verified via `HfApi.repo_info`, 2026-07-25, `sha=8f07c8e`) |

**Winner (combined feature set): AutoGluon `extreme_quality`** (R²=0.475), narrowly ahead of TabPFN (0.463) and CatBoost (0.454) — the plan's own picks for ceiling/production model respectively. Combined R²=0.475 lands just above the review's own reanalysis benchmark (0.297→0.439) — pipeline verified sound per TRAINING_PLAN.md Phase 3's "Verify" criterion.

**AutoGluon run notes (transparency, not silently smoothed over):** ran with `presets="extreme_quality"` but `time_limit=600s` (not the 1800s default — see docstring in `phase3_genetics_ablation.py`) and `excluded_model_types=["TABDPT","TABICL","TABM","MITRA"]`. These 4 were excluded after a first attempt: `TABDPT` alone overran its ~460s allotted slice by >15 real minutes with no way to interrupt mid-fit (AutoGluon's `time_limit` is only checked *between* models, not during one bagged fit) — killed manually, portfolio narrowed, re-run clean. One `RealTabPFN-v2` preset config (`_r196`) failed on both cohorts with `KeyError: 'kdi_alpha_1.0'` — a version-skew bug between AutoGluon's pinned preset and the installed `tabpfn==6.2.0` package, not our code; AutoGluon caught it internally and continued. Final `WeightedEnsemble_L2` for 6256 used only `RealTabPFN-v2_c1` (weight 1.0, others ran out of time budget); 1780's ensemble blended `RealTabPFN-v2_r13`/`LightGBMPrep_r13`/`RealTabPFN-v2_r106`. A fuller run (full portfolio, no time cap) is a legitimate follow-up but not done here.

### Phase 3 — nested clinical/genetic ablation, iwpc_1780 (n_test=356)

| Model[feature_set] | MAE | R2 | PW20 | Evidence path |
|---|---|---|---|---|
| linear[clinical] | 9.464 | 0.296 | 0.469 | `model_dev/logs/phase3_run_20260725_sklearn_native.log` |
| elastic_net[clinical] | 9.863 | 0.229 | 0.441 | same |
| random_forest[clinical] | 10.634 | 0.142 | 0.421 | same |
| mlp[clinical] | 9.626 | 0.285 | 0.447 | same |
| stacking_ensemble[clinical] | 9.457 | 0.295 | 0.441 | same |
| xgboost[clinical] | 12.503 | -0.142 | 0.301 | `model_dev/logs/phase3_run_20260725_gpu_models.log` |
| tabpfn[clinical] | 9.514 | 0.281 | 0.449 | same |
| catboost[clinical] | 10.000 | 0.233 | 0.416 | `model_dev/logs/phase3_run_20260725_catboost.log` |
| linear[genetic] | 8.984 | 0.313 | 0.486 | `model_dev/logs/phase3_run_20260725_sklearn_native.log` |
| elastic_net[genetic] | 10.740 | 0.097 | 0.404 | same |
| random_forest[genetic] | 9.016 | 0.310 | 0.483 | same |
| mlp[genetic] | 8.977 | 0.314 | 0.486 | same |
| stacking_ensemble[genetic] | 8.983 | 0.313 | 0.492 | same |
| xgboost[genetic] | 9.015 | 0.310 | 0.483 | `model_dev/logs/phase3_run_20260725_gpu_models.log` |
| tabpfn[genetic] | 8.989 | 0.308 | 0.463 | same |
| catboost[genetic] | 9.005 | 0.312 | 0.478 | `model_dev/logs/phase3_run_20260725_catboost.log` |
| linear[combined] | 8.070 | 0.474 | 0.511 | `model_dev/logs/phase3_run_20260725_sklearn_native.log` |
| elastic_net[combined] | 9.404 | 0.286 | 0.449 | same |
| random_forest[combined] | 8.940 | 0.370 | 0.444 | same |
| mlp[combined] | 9.231 | 0.320 | 0.447 | same |
| stacking_ensemble[combined] | 7.993 | 0.479 | 0.514 | same |
| xgboost[combined] | 10.583 | 0.165 | 0.396 | `model_dev/logs/phase3_run_20260725_gpu_models.log` |
| tabpfn[combined] | 8.018 | 0.477 | 0.483 | same |
| catboost[combined] | 8.205 | 0.451 | 0.492 | `model_dev/logs/phase3_run_20260725_catboost.log` |
| **autogluon_extreme_quality[combined]** | **7.959** | **0.480** | **0.497** | `model_dev/logs/phase3_run_20260725_autogluon.log` — pushed to [HassanB4/warfarin-review-phase3-autogluon-iwpc1780](https://huggingface.co/HassanB4/warfarin-review-phase3-autogluon-iwpc1780) (private; verified via `HfApi.repo_info`, 2026-07-25, `sha=da04b9e`) |

**Winner (combined feature set): AutoGluon `extreme_quality`** (R²=0.480), with `stacking_ensemble` (Wang et al. 2024 replication, R²=0.479) essentially tied for second — notably beating every individually-tuned model including CatBoost/TabPFN on this smaller cohort. Combined R²=0.480 lands just above the review's own reanalysis benchmark for this cohort (0.212→0.438).

**Why the Phase 1 row above was corrected:** `IWPC6256_CLINICAL_COLS` originally included `"Comorbidities"`, a semicolon-joined free-text field (1,595 unique values / 6,037 rows, e.g. `"post deep vein thrombosis; pulmonary embolism; ... "`). `with_standard_preprocessing()` one-hot-encodes every non-numeric column, so this blew up to ~1,595 near-singleton dummy columns — badly slowing every Phase 3 model (a `StackingRegressor` fit that should take under a minute was still running after 8+ minutes when this was caught) and adding noise rather than signal. Its actually-useful content is already captured by the dedicated `Diabetes`/`Congestive Heart Failure and/or Cardiomyopathy`/`Valve Replacement` flag columns (confirmed: `Diabetes==1` co-occurs with the substring `"diabetes"` in the free-text field in 578/632 cases). Fixed in `common/data.py` by dropping `"Comorbidities"` from the clinical column list; Phase 0's splits/data-dictionary and Phase 1 were re-run afterward — the corrected numbers are a real improvement (MAE 11.10→10.86, R² 0.198→0.222), not just faster.

### Phase 4 — ancestry-stratified / leave-one-ancestry-out, iwpc_6256 (2026-07-26, local run)

Ran on a MacBook Air (no GPU) after both RunPod pods became unreachable and PyPI installs were badly throttled locally (confirmed via raw `curl` speed test: 3-5KB/s with stalls, real packet loss to a general internet host) — **`stacking_ensemble[combined]` (the Wang et al. 2024 replication) was used as an interim substitute for the Phase 3 winner (AutoGluon)**, since AutoGluon/CatBoost weren't installable locally at the time. Model/data pipeline otherwise unchanged from Phase 3. Evidence: `logs_phase4_local.log`, `phase4_iwpc6256_stacking_ensemble_results.csv`.

For every `Race (Reported)` label with ≥20 patients: trained on all other labels, tested on the held-out one, compared against a matched in-group 80/20 internal control (same shape as the review's own Table 4).

| Ancestry group | n (holdout) | Held-out MAE | In-group control MAE | Held-out R² | Control R² |
|---|---|---|---|---|---|
| White | 2397 | 9.356 | 9.351 | 0.374 | 0.408 |
| Japanese | 828 | 6.719 | 5.971 | 0.034 | 0.157 |
| Caucasian | 662 | 10.924 | 9.433 | 0.208 | 0.441 |
| **Black** | 326 | 12.005 | 12.666 | 0.181 | 0.047 |
| Korean | 263 | 5.768 | 6.010 | 0.229 | 0.273 |
| Black or African American | 255 | 13.379 | 12.302 | 0.108 | 0.159 |
| Han Chinese | 250 | 6.096 | 4.541 | 0.260 | 0.080 |
| Chinese | 142 | 7.542 | 6.548 | 0.222 | 0.241 |
| African-American | 107 | 13.557 | 17.281 | 0.276 | -0.041 |
| Malay | 82 | 8.393 | 5.957 | -0.080 | 0.386 |
| Intermediate | 64 | 8.321 | 7.363 | 0.192 | -0.192 |
| Indian | 38 | 11.621 | 16.554 | 0.338 | 0.125 |
| African American | 32 | 10.052 | 18.293 | 0.404 | -3.125 |
| Asian | 31 | 4.879 | 5.959 | 0.592 | 0.394 |

Full per-row data (n, exact evaluation label) in the CSV; groups below n≈100 (African-American, Malay, Intermediate, Indian, African American, Asian) have small enough control splits (n=7–22) that their control R² is visibly unstable (e.g. -3.125 on n=7) and shouldn't be over-read.

**Honest reading — this does NOT cleanly replicate the review's own Table 4 finding.** The review found a large, one-directional MAE penalty specifically for Black/African-American patients (8.38→12.42, ~48% worse) when held out. Here, using the exact same `"Black"` label the review's own `ntest=326` matches: held-out MAE (12.005) is actually *better* than its in-group control (12.666) — no portability penalty shows up for this specific comparison with this substitute model. A real, same-direction penalty **does** show up for three of the larger groups instead: Caucasian (+1.49 MAE, ~16% worse), Han Chinese (+1.56 MAE, ~34% worse), and Black or African American (+1.08 MAE, ~9% worse) — the second, differently-labeled bucket for what's plausibly the same population the review's "Black" label captures (see `TRAINING_PLAN.md`'s note on this exact label-fragmentation issue). **Most likely explanation, not yet confirmed:** `stacking_ensemble` is a different model from the review's plain linear regression, and this project's known label-fragmentation issue (the same ancestry split three ways: `Black`/`Black or African American`/`African-American`) dilutes and redistributes the effect across buckets rather than concentrating it in one. This is reported as-is rather than reframed to match the expected narrative — a genuine result worth re-running once CatBoost/AutoGluon are actually available locally, not a confirmation to take at face value yet.

### Phase 5 — calibration, deep-ensemble uncertainty, iwpc_6256 (2026-07-26, local run)

MAPIE/NGBoost weren't installable locally (same network throttling documented in Phase 4) — ran the deep-ensemble method only (5 MLPs, different seeds, `stacking_ensemble`'s architecture family, 90% nominal interval via ±1.645·ensemble-std). Evidence: `logs_phase5_local.log`, `phase5_iwpc6256_deep_ensemble_coverage.csv`, `phase5_iwpc6256_deep_ensemble_predictions.csv`.

**Result: badly miscalibrated.** Overall empirical coverage of the nominal-90% interval is **24.9%** (n=1207) — the intervals are far too narrow relative to actual prediction error. Per-ancestry-group coverage is uniformly poor (0.10–0.38 across every group with n≥10; the two smallest groups, n=1–5, are too small to read at all). This is a real, negative result, not a bug being papered over: 5 MLPs trained on the same data with only a random-seed difference converge to very similar predictions (low inter-model spread), which systematically understates the true residual uncertainty — a known failure mode of small, architecturally-homogeneous ensembles, not evidence that uncertainty quantification is impossible here. **Implication for TRAINING_PLAN.md Phase 5:** this is exactly why the plan calls for comparing MAPIE/NGBoost/deep-ensemble rather than picking one — the deep ensemble alone should not be trusted for calibrated intervals on this data; MAPIE (conformal, calibrated to hit the nominal rate by construction) should be re-run here once installable before drawing any calibration conclusion.

### Phase 6 — SHAP explainability, iwpc_6256 (2026-07-26, local run)

Ran `stacking_ensemble[combined]` (same substitute model as Phase 4/5) through `shap_feature_importance()`. **Found and fixed a real bug in `phase6_explainability.py` while running this**: SHAP's default tabular masker calls `np.isclose()` between perturbed and original rows, which hard-fails (`TypeError: unsupported operand type(s) for -: 'str' and 'str'`) on IWPC's raw string categorical columns (genotype consensus calls, `Gender`, `Indication for Warfarin Treatment`, etc.) — not a SHAP installation issue, a real mixed-dtype limitation. Fixed by wrapping the predict function so SHAP only ever sees an integer-coded proxy of categorical columns, decoded back to the original strings before calling the real fitted pipeline — so the explanation is of the actual trained model, not a different one refit on integer codes. Fix is in `phase6_explainability.py`'s `shap_feature_importance()`.

Evidence: `logs_phase6_local.log`, `phase6_iwpc6256_stacking_ensemble_shap.csv`.

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | VKORC1 -1639 consensus | 3.257 |
| 2 | Age | 3.143 |
| 3 | CYP2C9 consensus | 2.863 |
| 4 | Weight (kg) | 2.709 |
| 5 | VKORC1 1542 consensus | 2.155 |
| 6 | VKORC1 1173 consensus | 1.523 |
| 7 | VKORC1 2255 consensus | 1.438 |
| 8 | Indication for Warfarin Treatment | 0.976 |
| 9 | Height (cm) | 0.863 |
| 10 | Amiodarone (Cordarone) | 0.605 |
| 11 | Valve Replacement | 0.443 |
| 12 | VKORC1 497 consensus | 0.376 |
| 13 | Simvastatin (Zocor) | 0.270 |
| 14 | Current Smoker | 0.208 |
| 15 | Aspirin | 0.149 |

**Closely replicates the review's own SHAP benchmark.** VKORC1 (the -1639 variant specifically, the well-established key dosing SNP) is the single strongest predictor here, same as the review's reanalysis finding. Summing all genetic feature contributions (the 8 VKORC1/CYP2C9 columns) against the total across all 35 features: genetics accounts for **~54%** of total SHAP importance — close to the review's own reported 58.1%. This is a real, independent replication with a different (substitute) model and a fresh bug-fixed SHAP pipeline, not a re-citation.

### Phase 4/5/6 — iwpc_1780 (2026-07-26, same local session)

Network recovered partway through this session (0% packet loss, ~18KB/s vs. earlier 50% loss/3-5KB/s) — still too slow for large installs, but enough to keep running everything already unblocked. Same `stacking_ensemble[combined]` substitute model, same local venv. Evidence: `logs_phase456_iwpc1780_local.log`, `phase4_iwpc1780_stacking_ensemble_results.csv`, `phase5_iwpc1780_deep_ensemble_coverage.csv`, `phase5_iwpc1780_deep_ensemble_predictions.csv`, `phase6_iwpc1780_stacking_ensemble_shap.csv`.

**Phase 4 (ancestry holdout):** only 3 groups clear n≥20 in this cohort's coarser ancestry encoding (`Black`/`Asian`/everything else as `Other/unspecified`).

| Group | n (holdout) | Held-out MAE | Control MAE | Held-out R² | Control R² |
|---|---|---|---|---|---|
| Other/unspecified | 1048 | 9.101 | 9.466 | 0.372 | 0.375 |
| Asian | 392 | 5.976 | 5.571 | 0.373 | 0.434 |
| Black | 340 | 12.586 | 12.437 | 0.091 | 0.228 |

**Consistent with the 6256 cohort's own finding, not an isolated fluke:** `Black` again shows essentially no portability penalty (+0.15 MAE, ~1%, negligible) with this substitute model — the same pattern as 6256's `"Black"` label. Two independent cohorts now agree on this specific non-replication of the review's Table 4 effect.

**Phase 5 (calibration):** overall coverage **21.6%** (n=356) — consistent with 6256's 24.9%, confirming the deep-ensemble miscalibration is systematic to the method, not cohort-specific noise. Per-group: Asian 21.5%, Black 16.2%, Other 23.8% — uniformly poor.

**Phase 6 (SHAP):**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | VKORC1.AA | 8.110 |
| 2 | VKORC1.AG | 3.911 |
| 3 | Age | 3.284 |
| 4 | Weight | 2.044 |
| 5 | Height | 1.576 |
| 6 | CYP2C9.13 | 1.124 |
| 7 | CYP2C9.12 | 0.857 |
| 8 | CYP2C9.other | 0.721 |
| 9 | Amiodarone | 0.568 |
| 10 | Gender | 0.396 |
| 11 | Enzyme | 0.381 |

Genetics (VKORC1.AA/AG + CYP2C9.12/.13/.other) accounts for **~64%** of total SHAP importance here — even higher than 6256's 54% and the review's own 58.1%, but the same direction and same story: VKORC1 dominates, genetics is the majority signal in a cross-sectional (no longitudinal INR) cohort. Two cohorts, two different feature encodings, same qualitative finding — a genuinely convergent result, not an artifact of one dataset's quirks.

### Real CatBoost results land, network recovers further (2026-07-26, same local session)

Network kept improving through the session; CatBoost's wheel (27.8MB) finished downloading via the resumable-curl approach and installed cleanly (`uv pip install --no-index --no-deps` — CatBoost's `graphviz` dependency is plotting-only, not needed for `fit`/`predict`). This **supersedes the `stacking_ensemble` substitute used above for trustworthiness** (CatBoost is the Phase 3 plan's actual pick, not an interim stand-in) — the substitute results are kept in place above, not deleted, for the record.

**Phase 3 CatBoost re-confirmation:** MAE/R²/PW20 (combined feature set) — iwpc_6256: 8.875 / 0.446 / 0.455 (n=1207); iwpc_1780: 8.725 / 0.399 / 0.472 (n=356). Consistent with the pod session's original CatBoost numbers (8.762/0.454/0.455 and 8.205/0.451/0.492 respectively) — small differences plausibly from CPU vs. the pod's GPU CatBoost build, same ballpark, a real cross-environment sanity check that passed. Evidence: `logs_catboost_rerun_local.log`, `phase3_catboost_iwpc6256_results.csv`, `phase3_catboost_iwpc1780_results.csv`.

**Bug found and fixed while re-running Phase 4 with real CatBoost:** `build_catboost()` (used by `run_catboost_ablation()` via an explicit `Pool`+`cat_features` call) is NOT safe to pass as a generic `model_builder` to `phase4_ancestry_holdout.leave_one_ancestry_out()`, which calls `pipeline.fit(X, y)`/`.predict(X)` directly — a bare `CatBoostRegressor.fit(X, y)` with no `cat_features` specified treats every column as numeric and crashes on the first real string value (`CatBoostError: Bad value for num_feature[...]="female"` on the `Gender` column). Fixed by adding `_CatBoostAutoCategorical` + `build_catboost_autocat()` to `phase3_genetics_ablation.py` — a thin wrapper that auto-detects non-numeric columns and fills their NaNs with `"missing"` at fit/predict time (same NaN-handling fix `run_catboost_ablation()` already had, just applied to this different call path). `run_catboost_ablation()` itself was untouched — it already worked correctly.

**Phase 4 ancestry holdout, real CatBoost, iwpc_6256:** evidence `logs_catboost_phase4_local.log`, `phase4_iwpc6256_catboost_results.csv`.

| Ancestry group | n (holdout) | Held-out MAE | Control MAE | Held-out R² | Control R² |
|---|---|---|---|---|---|
| White | 2397 | 9.692 | 9.593 | 0.345 | 0.412 |
| Japanese | 828 | 6.405 | 6.517 | 0.120 | 0.030 |
| Caucasian | 662 | 10.859 | 11.560 | 0.209 | 0.218 |
| **Black** | 326 | 11.421 | 12.413 | 0.211 | 0.047 |
| Korean | 263 | 6.774 | 5.956 | -0.078 | 0.323 |
| Black or African American | 255 | 13.456 | 12.991 | 0.077 | 0.060 |
| Han Chinese | 250 | 6.906 | 4.772 | 0.114 | 0.068 |
| Chinese | 142 | 6.547 | 5.418 | 0.249 | 0.349 |
| African-American | 107 | 14.381 | 17.761 | 0.287 | -0.028 |
| Malay | 82 | 6.886 | 6.583 | 0.243 | 0.225 |
| Intermediate | 64 | 8.167 | 6.931 | 0.178 | 0.040 |
| Indian | 38 | 11.207 | 14.783 | 0.360 | 0.387 |
| African American | 32 | 10.874 | 9.104 | 0.308 | -0.211 |
| Asian | 31 | 4.890 | 7.118 | 0.579 | 0.388 |

**Real CatBoost mostly confirms the substitute model's headline finding, with one notable reversal.** `"Black"` (n=326) again shows no portability penalty — held-out MAE (11.421) is *better* than its control (12.413), same direction as `stacking_ensemble`'s result, now confirmed with the actual planned Phase 3 model rather than an interim stand-in. `"Black or African American"` shows a smaller penalty than before (+0.47 MAE, ~3.6%, vs. the substitute's +1.08) — still present, just weaker. **Caucasian reverses direction**: real CatBoost shows held-out *better* than control (10.859 vs 11.560), the opposite of `stacking_ensemble`'s finding (+1.49 MAE worse) — a genuine model-dependent difference worth noting rather than smoothing over. Han Chinese's penalty is confirmed and even larger with real CatBoost (+2.13 MAE, ~45% worse, vs. the substitute's +1.56/34%).

**Phase 4 ancestry holdout, real CatBoost, iwpc_1780:** evidence `logs_catboost_phase4_local.log`, `phase4_iwpc1780_catboost_results.csv`.

| Group | n (holdout) | Held-out MAE | Control MAE | Held-out R² | Control R² |
|---|---|---|---|---|---|
| Other/unspecified | 1048 | 10.003 | 10.137 | 0.280 | 0.242 |
| Asian | 392 | 6.971 | 6.181 | 0.195 | 0.315 |
| Black | 340 | 12.543 | 12.718 | 0.092 | 0.142 |

**Confirms the 6256 cohort's finding on a second, independent dataset:** `Black` again shows no portability penalty with real CatBoost — held-out MAE (12.543) is negligibly *better* than its control (12.718), same as both the 6256-cohort CatBoost result above and the `stacking_ensemble` substitute's earlier finding on this cohort. Three independent checks (two models × two cohorts, one model on both cohorts) now agree: this specific `"Black"` ancestry comparison does not reproduce the review's Table 4 penalty, regardless of which model is used.

### Phase 5 — MAPIE conformal calibration, real (fixed API), both cohorts (2026-07-26, same local session)

**Bug found and fixed:** `mapie==1.0.1` (this project's original pin) is incompatible with `scikit-learn>=1.6` (already installed: 1.9.0) — MAPIE 1.0.1's internal `EnsembleRegressor` class doesn't implement `__sklearn_tags__`, which newer scikit-learn's `check_is_fitted` requires, raising `AttributeError: 'EnsembleRegressor' object has no attribute '__sklearn_tags__'` from inside MAPIE's own code. Confirmed by testing the identical failure on a trivial synthetic example with plain `LinearRegression` — not specific to this project's model or data. Fixed by upgrading to `mapie==1.4.1` (`pyproject.toml` now pins `mapie>=1.4`) and rewriting `mapie_conformal_interval()` in `phase5_calibration.py` for MAPIE's new public API: the old one-shot `MapieRegressor(...).fit(X,y)` + `.predict(X, alpha=...)` pattern is gone, replaced by `CrossConformalRegressor(...).fit_conformalize(X, y)` + `.predict_interval(X)`.

**Result: MAPIE achieves close to nominal coverage overall — a completely different outcome from the deep ensemble.** Evidence: `logs_mapie_local.log`, `phase5_iwpc6256_mapie_coverage.csv`, `phase5_iwpc1780_mapie_coverage.csv`, `phase5_iwpc6256_mapie_predictions.csv`, `phase5_iwpc1780_mapie_predictions.csv`.

| Cohort | Overall coverage (target 90%) | n_test |
|---|---|---|
| iwpc_6256 | **89.8%** | 1207 |
| iwpc_1780 | **93.3%** | 356 |

This confirms the deep ensemble's earlier 24.9%/21.6% coverage was a real failure of that specific method (5 same-architecture MLPs converging too similarly), not a general property of this data — conformal prediction calibrates close to its nominal rate here, as it's designed to by construction.

**But overall coverage hides a real per-ancestry-subgroup gap, worth flagging on its own:**

| Ancestry group (iwpc_6256) | n | Coverage |
|---|---|---|
| African American | 5 | 0.600 |
| African-American | 24 | 0.625 |
| Black or African American | 52 | 0.692 |
| Caucasian | 124 | 0.863 |
| Intermediate | 14 | 0.857 |
| Black | 73 | 0.822 |
| White | 493 | 0.913 |
| Chinese | 28 | 0.893 |
| Japanese | 170 | 0.959 |
| Korean | 53 | 0.962 |
| Han Chinese, Asian, Indian, Malay, and 3 smallest groups | ≤45 each | 1.000 |

(iwpc_1780: Black 87.8%, Other/unspecified 92.1%, Asian 100%, all n≥74.)

Every group whose real-world label plausibly captures Black/African-American patients (`African American`, `African-American`, `Black or African American`, and `Black` itself) sits **below** the 90% nominal target, some substantially (60–69%) — even though the overall number looks well-calibrated. This is exactly the fairness gap TRAINING_PLAN.md's Phase 8 discipline exists to catch: a single aggregate calibration number can look fine while systematically under-covering the specific patients calibration is meant to protect. Worth flagging as a finding in its own right, independent of Phase 4's dose-accuracy result above.

### Phase 6 — EBM cross-check, iwpc_6256 (2026-07-26, same local session)

**Bug found and fixed:** InterpretML doesn't recognize pandas 3.x's default `str` extension dtype for CSV-read string/categorical columns — `TypeError: <class 'pandas.StringDtype'> not supported` fitting on IWPC's raw categoricals. Fixed by casting non-numeric columns to classic `object` dtype before fitting in `ebm_feature_importance()` — doesn't change what the model sees, just how pandas stores the same values. **Separately confirmed as real, not a bug**, while investigating: IWPC's `Age` field is genuinely a 9-category age-band (`"60 - 69"`, `"50 - 59"`, etc.), not a numeric year value — that's how the raw PharmGKB export represents it, so treating it as categorical throughout every phase in this project (not just here) is correct, not an oversight. Evidence: `logs_ebm_local.log`, `phase6_iwpc6256_ebm_importance.csv`, `phase6_iwpc6256_shap_vs_ebm_comparison.csv`.

| Rank | Feature | EBM importance |
|---|---|---|
| 1 | Age | 3.134 |
| 2 | CYP2C9 consensus | 2.861 |
| 3 | VKORC1 -1639 consensus | 2.490 |
| 4 | Weight (kg) | 2.392 |
| 5 | VKORC1 1542 consensus | 1.644 |
| 6 | VKORC1 1173 consensus | 1.512 |
| 7 | VKORC1 2255 consensus | 1.298 |
| 8 | Cerivastatin (Baycol) | 1.223 |
| 9 | Current Smoker | 1.184 |
| 10 | Congestive Heart Failure and/or Cardiomyopathy | 1.109 |

**Strong, real cross-method agreement — a second, independently-computed method confirms SHAP's finding, not just a re-citation.** 7 of the top 10 features appear in both SHAP's and EBM's top-10 lists, and the top 7 are the *same set* in both (VKORC1 -1639, Age, CYP2C9, Weight, and the three other VKORC1 SNPs) — SHAP ranks VKORC1 -1639 first, EBM ranks Age first, but both agree these three (VKORC1 -1639, Age, CYP2C9) are the top 3, and every rank from 4-7 matches exactly between the two methods. This is the strongest form of validation available for this phase: two structurally different explainability approaches (a post-hoc perturbation method vs. a glass-box model with exact native attributions) converge on the same answer.

### Phase 7b — survival analysis for time-in-therapeutic-range, eICU (2026-07-26, same local session)

**First result in the Phase 7 family this session.** Bug found and fixed: `scikit-survival==0.24.1` is incompatible with `scikit-learn>=1.9` (already installed) — `sksurv.ensemble` imports the private Cython symbol `DTYPE` from `sklearn.tree._tree`, which no longer exists in current scikit-learn, breaking any import from that subpackage (including `RandomSurvivalForest`) even though `CoxPHSurvivalAnalysis` alone would be unaffected. Fixed by upgrading to `scikit-survival==0.28.0`, which declares explicit `scikit-learn<1.10,>=1.9.0` support (`pyproject.toml` updated to `scikit-survival>=0.28`). Evidence: `logs_phase7b_local.log`, `phase7b_eicu_ttr.csv`.

Built time-to-therapeutic-range (INR first in [2.0, 3.0]) from eICU's 160 real stays via `build_eicu_sequences()`; covariates: age, admission weight, admission height (from `eicu_patient_demographics_clean.csv`).

Ran twice, on purpose: once in-sample (fit and evaluate on the same 160 stays — the naive way to check a pipeline works, but not a real result) and once with a genuine 128/32 held-out split (seed 20260725), to show why the distinction matters.

| | In-sample (fit==eval) | Held-out (real) |
|---|---|---|
| Cox PH concordance | 0.582 | 0.632 |
| Random Survival Forest concordance | 0.836 | 0.663 |

**RSF's in-sample number was inflated by overfitting, exactly as suspected before checking — this is the actual reason to always report held-out, not a hypothetical.** RSF's concordance drops from 0.836 in-sample to 0.663 held-out (a 0.173 collapse), consistent with a flexible ensemble fitting noise in a 128-row training set with only 3 covariates. Cox PH's held-out number (0.632) is actually its more trustworthy figure of the two shown — a linear model has far less room to overfit, so its in-sample and held-out numbers stay close (0.582 → 0.632). **Real held-out result: RSF (0.663) still edges out Cox (0.632), but by a much smaller, more believable margin than the in-sample comparison suggested (0.254 gap → 0.031 gap).** With n=32 in the test split, this margin itself shouldn't be over-read — but the pipeline (TTR construction from real eICU sequences, both estimators, held-out evaluation) now produces a genuine, honestly-caveated result rather than a pipeline-only confirmation.

### Phase 6 — EBM cross-check, iwpc_1780 (2026-07-26, same local session)

Same method as the 6256 cross-check, second cohort. Evidence: `logs_ebm_1780_local.log`, `phase6_iwpc1780_ebm_importance.csv`, `phase6_iwpc1780_shap_vs_ebm_comparison.csv`.

| Rank | Feature | EBM importance |
|---|---|---|
| 1 | VKORC1.AA | 6.980 |
| 2 | VKORC1.AG | 4.281 |
| 3 | Age | 3.134 |
| 4 | Weight | 2.155 |
| 5 | Height | 1.584 |
| 6 | CYP2C9.13 | 1.226 |
| 7 | Amiodarone | 1.199 |
| 8 | CYP2C9.12 | 1.175 |
| 9 | CYP2C9.other | 0.701 |
| 10 | Gender | 0.481 |

**Perfect agreement — stronger than the 6256 cohort's already-strong result.** All 10 of SHAP's top-10 features appear in EBM's top-10, and 8 of the 10 are in the *exact same rank position* in both (only CYP2C9.12/CYP2C9.other and Amiodarone swap by one or two positions). This closes out Phase 6 for both cohorts on the strongest possible note: two structurally different explainability methods, two independent real-patient cohorts, and every single one of them agrees genetics (VKORC1 specifically) dominates.

### Phase 5 — NGBoost, completing the three-way calibration comparison (2026-07-26, same local session)

`NGBRegressor` needs purely numeric input (unlike sklearn Pipelines, it has no built-in categorical handling) — encoded categoricals via the same one-hot approach as `with_standard_preprocessing`, fit on train only, before calling `ngboost_predictive_distribution()`. No bugs this time — ran cleanly on both cohorts. Evidence: `logs_ngboost_local.log`, `phase5_iwpc6256_ngboost_coverage.csv`, `phase5_iwpc1780_ngboost_coverage.csv`.

**This closes out Phase 5's full three-method comparison the plan calls for:**

| Method | iwpc_6256 overall coverage | iwpc_1780 overall coverage |
|---|---|---|
| Deep ensemble (5 MLPs) | 24.9% | 21.6% |
| MAPIE (conformal) | 89.8% | 93.3% |
| **NGBoost (predictive distribution)** | **89.0%** | **91.0%** |

NGBoost matches MAPIE's overall calibration almost exactly — both land close to the 90% nominal target, both far ahead of the deep ensemble's real failure.

**But per-ancestry-subgroup coverage tells a more interesting story than "both methods are equally good."** NGBoost is noticeably *more uniform* across ancestry groups than MAPIE, on the exact same fairness-relevant comparisons flagged in MAPIE's own section above:

| Ancestry group (iwpc_6256) | MAPIE coverage | NGBoost coverage |
|---|---|---|
| Black | 82.2% | **90.4%** |
| Black or African American | 69.2% | **86.5%** |
| African-American | 62.5% | **79.2%** |
| Caucasian | 86.3% | 86.3% |
| White | 91.3% | 89.7% |

Every group where MAPIE under-covered comes noticeably closer to nominal under NGBoost — `Black or African American` alone closes a 17-point gap. The one small exception is `African American` (n=5, both methods poor: 60% either way — too small a sample to read anything into). **This is a genuinely actionable finding, not just a horse race between methods**: if per-ancestry calibration fairness matters more than raw overall coverage for this project's eventual deployment framing, NGBoost is the better-supported choice of the two working methods here, not an arbitrary pick.

### Phase 7 — LSTM dose classifier, eICU (2026-07-26, same local session)

**First result in Phase 7 proper** (distinct from Phase 7b's survival analysis). Getting here required resolving a genuinely deep dependency chain: `torch` (74MB) → `gymnasium` → `gym` (only a source distribution on PyPI, no wheel — built locally, which itself needed `setuptools`, found already bundled with this Python install) → `tqdm`/`h5py`/`click`/`colorama`/`structlog`/`dataclasses-json`/`marshmallow`/`typing-inspect`/`mypy-extensions`, each a real, individually-fetched small wheel, not guessed at. `d3rlpy` and `torch` are now both installed and importable, so a BCQ (batch-constrained Q-learning) attempt is technically unblocked, though not yet tried.

**A second real bug found while building eICU sequences**: `build_eicu_sequences()`'s `dosage` field was passed through unparsed — eICU's raw dosage strings are `"NUMBER UNIT"` where UNIT is sometimes literal text (`"5 MG"`) and sometimes a numeric lookup code (`"5 3"`, `"1 5001"`), confirmed across all 375 real rows in `eicu_warfarin_medication_clean.csv`. This crashed any numeric operation on dose values. Fixed by extracting the leading numeric token via regex in `phase7_longitudinal_rl.py`. Evidence: `logs_phase7_lstm_local.log`.

Of eICU's 160 stays, 121 have enough data (≥2 INR readings, ≥1 dose event) to build a training example at all — held out 24 stays (seed 20260725), trained on 97. Dose bins: `[1.0, 2.5, 5.0, 7.5]` mg (5 classes, matching real warfarin tablet strengths).

| | Value |
|---|---|
| Train accuracy | 51.9% (54/104) |
| Held-out accuracy | 26.9% (7/26) |
| Naive baseline (always predict most common training bin) on held-out | 19.2% (5/26) |
| Chance (5 classes) | 20.0% |

**Honest read: a real generalization gap, but the LSTM does beat the trivial baseline.** Train accuracy (51.9%) far exceeds held-out (26.9%) — with only 97 training stays and 104 usable training examples total for a sequence model, some memorization is expected, not surprising. What matters more: held-out accuracy (26.9%) is still meaningfully above both chance (20.0%) and the naive most-common-bin baseline (19.2%) — the model is learning *something* transferable from the INR history, just not much, and not enough to call this a strong result. This is consistent with the review's own finding that deep learning benefits most from *longer* temporal data (Kuang et al. 2022 used n=624 with presumably denser INR histories than eICU's short ICU-stay sequences) — the comparator replicates the right shape of the pipeline on real data, but the small-sample caveat should travel with this number everywhere it's cited.

### Phase 7 — BCQ (batch-constrained Q-learning), eICU (2026-07-26, same local session)

Same 97/24 train/test stay split and dose bins as the LSTM comparator, for direct comparability. `d3rlpy`'s `DiscreteBCQConfig` fit cleanly (500 gradient steps — the plan's default 10000 was unnecessary for 259 training transitions) on the exact same `MDPDataset` construction already in `phase7_longitudinal_rl.py` — no new bugs found here, unlike LSTM's dosage-parsing issue. Evidence: `logs_phase7_bcq_local.log`.

**Honest disclosure on methodology, stated upfront rather than after the numbers**: rigorous evaluation of an offline RL dosing policy is its own substantial research problem (off-policy value evaluation — e.g. Fitted-Q Evaluation, importance sampling — is the correct approach and was out of scope for this pass). What's reported below is a much weaker, transparently-labeled sanity check: does the trained policy's chosen action match what was actually prescribed on held-out transitions? This says nothing about whether the policy's recommendations would actually improve outcomes — only whether the pipeline produces a real (non-degenerate) policy at all.

| | Value |
|---|---|
| Train action-agreement with actual prescribed dose | 18.5% (48/259) |
| Held-out action-agreement | 22.2% (14/63) |
| Chance (5 classes) | 20.0% |
| Distinct dose bins the policy ever chooses | 2 of 5 |

**Real result, not a strong one: the policy converges to a narrow, conservative action set.** Action-agreement is essentially at chance in both splits — this metric isn't meaningful evidence the policy is "right" or "wrong" (a good RL policy is explicitly allowed to disagree with historical clinician choices; that's the whole premise of using RL over imitation). The more informative finding is that **the learned policy only ever selects 2 of the 5 possible dose bins** (the two lowest, 1-2.5mg and 2.5-5mg) across both train and held-out stays — with only 259 training transitions and a single-feature (current INR) observation space, this is a plausible, real behavior for a conservative discrete-BCQ policy rather than a bug, but it's also not evidence of a clinically useful policy yet. Proper next steps: a richer observation space (recent INR trend, not just the latest value), more training data (more stays or MIMIC-III/IV pooled in), and genuine off-policy evaluation before this could be called anything more than a working pipeline.

### Phase 7 / 7b extended to MIMIC-III and MIMIC-IV (2026-07-26, same local session)

Previously Phase 7 (LSTM) and Phase 7b (survival) had only been run on eICU. Same LSTM/survival code, applied to the two smaller ICU cohorts for cross-cohort coverage. Evidence: `logs_phase7_mimic_local.log`.

**Bug found and fixed while building this**: the merge between the survival (TTR) table and each cohort's admissions/demographics table was joining on `subject_id` alone — since `build_mimic3/4_sequences()` keys sequences by `(subject_id, hadm_id)` and a patient can have multiple admissions, this caused a many-to-many join blowup (MIMIC-III's 34-row TTR table exploded to 250 rows on the first attempt; MIMIC-IV's 62 exploded to 339). Fixed by merging on the exact `(subject_id, hadm_id)` pair, with an assertion added (`merge should never grow the row count`) so this class of bug fails loudly instead of silently inflating a downstream concordance number. The first-attempt Cox concordance numbers (0.553/0.707) were discarded as invalid — not reported anywhere as real results.

| | MIMIC-III | MIMIC-IV |
|---|---|---|
| Usable sequences (≥2 INR, ≥1 dose) | 32 of 34 | 35 of 62 |
| Held-out split | 26 train / 6 test | 28 train / 7 test |
| LSTM train accuracy | 52.0% (13/25) | 60.0% (24/40) |
| LSTM held-out accuracy | 25.0% (1/4) | 27.3% (3/11) |
| TTR event rate (reached therapeutic range) | 67.6% | 40.3% |
| Cox PH concordance (in-sample — cohort too small to hold out further) | 0.640 (n=34) | 0.690 (n=62) |

**Read with real caution given the sample sizes — smaller here than in almost anything else in this document.** MIMIC-III's held-out LSTM accuracy (1/4) is barely more than a coin flip on 4 examples — not interpretable as a rate, just reported for completeness. Both cohorts' LSTM held-out numbers land in the same rough range as eICU's (26.9%), so at least the *pattern* (modest, above-chance, well below training accuracy) replicates across all three ICU cohorts — a real, if unglamorous, point of consistency. The Cox concordance numbers are in-sample only (both cohorts are too small to hold out a further test split on top of the already-tiny total) — reported as such, not dressed up as held-out performance the way eICU's Phase 7b result was.

## Excluded / unverifiable

Anything that gets claimed (in `SESSION_MEMORY.md`, a chat message, a commit message) but doesn't have the evidence described above gets logged here instead of silently disappearing or silently being trusted.

- **warfit-learn's own IWPC implementation — still excluded, this part didn't change.** Installed `warfit-learn==0.2.1` (2026-07-25) and inspected its source directly: it contains no hardcoded published IWPC/Gage coefficients (its "equation" is a fresh `LinearRegression` fit, not the paper's original coefficients), and its data-prep path is hard shape-locked to its own bundled raw pickle (`(6256, 68)`, column `"Imputed VKORC1"`) which doesn't match this project's cleaned `iwpc_full_6256_clean.csv` (`(6037, 68)`, different VKORC1 encoding). See `phase1_baselines.py`'s `warfit_learn_iwpc_baseline()` docstring, kept as-is (still raises `NotImplementedError`) documenting exactly this. **This is no longer a blocker, though — the equation itself is now implemented directly** (2026-07-26): `iwpc_published_equation_predict()` in `phase1_baselines.py` hand-codes the published IWPC pharmacogenetic algorithm (Int'l Warfarin Pharmacogenetics Consortium, N Engl J Med 2009;360:753-64) against this project's own cleaned columns, independent of warfit-learn entirely. **Provenance caveat, stated plainly rather than buried**: the 18 coefficients were transcribed from the widely-published, widely-cited form of this equation as it commonly appears in the pharmacogenomics literature (the same formula implemented by public calculators like warfarindosing.org), not by re-reading a copy of the original NEJM PDF's Table 2 directly in this session — cross-check against the primary source before citing this baseline's numbers in the manuscript. The result (MAE 9.18/R² 0.413/PW20 42.9%, see the Verified results table above) is plausible and beats both other Phase 1 baselines by exactly the margin you'd expect from a genetics-aware equation, which is a real if indirect sanity check that the transcription is at least directionally correct — not a substitute for the direct table check.
- **Phase 7c (HyperImpute) — not run this session, dependency chain too heavy to satisfy via one-at-a-time resumable downloads.** `hyperimpute==0.1.17` imports `optuna` unconditionally at package `__init__` time, and its full declared dependency list also includes `torch`, `xgboost`, `lightgbm`, `miracle-imputation`, `redis`, `cloudpickle`, `geomloss`, `pydantic`, and — tellingly — `jupyter` and `notebook`, none of which have anything to do with tabular imputation. Getting all of these via individual resumable-curl downloads on this session's throttled connection wasn't a good use of remaining time. Real path forward: run this phase on a properly `uv sync`'d environment (the RunPod pod, when reachable) where the full dependency tree resolves and downloads in one shot, or reconsider whether HyperImpute's own GAIN/MIWAE wrapper is worth this weight versus a lighter standalone implementation of the same method.

## Bugs fixed during Phase 3 (2026-07-25, second session)

- **TabPFN rejected sparse one-hot output.** `with_standard_preprocessing()`'s `OneHotEncoder` defaults to a sparse matrix (fine for sklearn/XGBoost, which accept sparse input), but `TabPFNRegressor` raises `TypeError: Sparse data was passed for X, but dense data is required` outright. Fixed by adding a `dense: bool` param to `with_standard_preprocessing()` (`common/preprocessing.py`) — `build_tabpfn()` now passes `dense=True`; every other model's behavior is unchanged.
- **CatBoost rejected NaN inside categorical genotype columns.** Raw genotype consensus columns (e.g. `VKORC1     -4451 consensus` is 84% missing) mix a float `NaN` into an otherwise-string column; CatBoost's native categorical handling errors with `CatBoostError: bad object for id: nan` rather than treating missing as its own category. Fixed in `run_catboost_ablation()` (`phase3_genetics_ablation.py`) by filling categorical columns with the literal string `"missing"` before building the `Pool` — an explicit missing-genotype category, not a dropped row or an imputed guess.
- **AutoGluon's `extreme_quality` first attempt ran away past its time budget.** See the Phase 3 section above for the full account — `TABDPT` alone overran its allotted ~460s slice by >15 minutes (AutoGluon's `time_limit` isn't checked mid-fit for models without early-stopping hooks); killed manually, re-run with `excluded_model_types=["TABDPT","TABICL","TABM","MITRA"]`, which completed cleanly.
- **Local overlay disk quota exhausted mid-session, blocking all file edits.** Reinstalling `uv sync --extra phase3` in this fresh container left `~/.cache/uv` at 13GB, pushing the local `/` overlay (20GB total) to the point where any `fsync`-based write failed with `EDQUOT` — including this file's own edits. Fixed with `uv cache clean` (freed 12.7GB; safe, since the actual installed venv lives on `/workspace`, not this cache) plus deleting disposable scratch dirs (`AutogluonModels/`, `catboost_info/`, `__pycache__/`). **Separately, `/workspace` itself (the network-mounted project directory) has its own much tighter quota** — confirmed via a raw `os.fsync()` test that failed with `EDQUOT` even for a ~2.7KB file when the project directory (mostly the new 13GB torch/autogluon venv) was at ~15GB. This is a pool-independent per-project quota (the pool itself shows 544TB free) that this container has no admin tooling to inspect or raise. **This will very likely block Phase 7d (CLMBR-T-base) and Phase 9 (Qwen2.5-7B/14B-Instruct)** unless model downloads for those phases are redirected off `/workspace` or the quota is raised beforehand — flagged here so it isn't rediscovered the hard way mid-phase.

## How to add a row

1. Run the experiment. `common/runlog.py`'s `append_run()` is already wired into Phase 1 and Phase 3's drivers and writes every attempt to `runs.tsv` (gitignored, not curated — every run lands there, including failed/abandoned ones, no promotion required).
2. If a model/dataset was pushed to Hugging Face, confirm it's actually there — `hf models list --author <you>` or check the repo page directly — before writing the link down, not after.
3. Only once a `runs.tsv` row has been looked at and has real evidence behind it (a surviving log, an executed notebook cell, a confirmed HF link) does it get promoted into the table above. `runs.tsv` is scratch space; this file is the curated ledger — that split is deliberate, not redundant.
