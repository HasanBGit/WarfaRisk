# WarfaRisk: Ancestry-Stratified, Reproducible Machine Learning for Warfarin Dose Prediction

<p align="center">
<img src="https://placehold.co/800x200/dbeafe/1e40af?text=WarfaRisk" alt="WarfaRisk — Warfarin Dose Prediction Pipeline">
</p>

> **Note:** this repository is hosted at `github.com/HasanBGit/PRSgene` — the
> name is a placeholder from when the repo was created and doesn't reflect
> the project (WarfaRisk). Renaming it is a suggested follow-up, not done in
> this release.

This repository contains the official code, splits, and results for **WarfaRisk**, a nine-phase, fully reproducible machine learning pipeline for warfarin maintenance-dose prediction on the public IWPC pharmacogenomic cohorts. Our best model (AutoGluon `extreme_quality`, combined clinical+genetic features) reaches **MAE 8.552 / R² 0.475 / PW20 0.466** on IWPC-6256 and **MAE 7.959 / R² 0.480 / PW20 0.497** on IWPC-1780 — and our ancestry-stratified calibration analysis surfaces a per-subgroup coverage gap that the aggregate number alone hides entirely.

#### By: Hassan Barmandah, Omar Abdullah Bawazir, Siraj Aldeen Marghalani, Moath Shaat, and Mariam M. AlEissa (corresponding author) — AI Center (AIC), Alfaisal University, Riyadh (with Umm Al-Qura University, Saudi Electronic University, Ministry of Health, King Khaled Eye Specialist Hospital Research Center)

[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/HasanBGit/PRSgene)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-3%20Models-F9D371?logo=huggingface&logoColor=black)](https://huggingface.co/HassanB4)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

> **This is a research artifact, not a validated clinical tool.** It has not been evaluated prospectively, carries no regulatory clearance, and should not be used to make an actual dosing decision.

---

## Model Description

Warfarin has one of the narrowest therapeutic indices in clinical medicine — stable dosing varies roughly ten-fold across patients, and the gap between an under-dose (thromboembolism) and an over-dose (major hemorrhage) is clinically decisive. WarfaRisk is a nine-phase pipeline that predicts stable weekly warfarin dose from clinical and pharmacogenomic features on the International Warfarin Pharmacogenetics Consortium (IWPC) cohorts (**IWPC-6256**, n=6,037; **IWPC-1780**, n=1,780), with a longitudinal-dosing extension track on three ICU demo cohorts (MIMIC-III, MIMIC-IV, eICU — clinical-only, no genotype).

The pipeline moves from mandatory baselines, through a nested clinical/genetic/combined feature ablation across nine model architectures, to ancestry-stratified leave-one-group-out fairness evaluation, a three-way calibration comparison (MAPIE conformal prediction, NGBoost, deep ensemble), and SHAP-vs-EBM cross-validated explainability — all on fixed, patient-level train/test splits under one documented random seed (`20260725`).

### Key Contributions

* **Nested Feature Ablation**: clinical-only, genetic-only, and combined feature sets across 9 model architectures (linear, elastic net, random forest, MLP, stacking ensemble, XGBoost, CatBoost, TabPFN, AutoGluon), on both IWPC cohorts, all on identical fixed splits
* **Ancestry-Stratified Fairness Evaluation**: leave-one-ancestry-group-out portability testing (not just aggregate accuracy), cross-checked between two independent model families so agreement or disagreement is itself informative
* **Calibration Gap Finding**: MAPIE and NGBoost both reach near-nominal overall coverage (89.8–93.3%) while a same-architecture deep ensemble collapses to 21.6–24.9%; per-ancestry-subgroup coverage then reveals every label plausibly capturing Black/African American patients under-covered even when the aggregate number looks fine
* **Cross-Validated Explainability**: SHAP and an independently-fit EBM (glass-box model) agree on the same dominant features — VKORC1 genotype, age, weight — a real cross-method replication, not a re-citation
* **Shipped Reproducibility**: the exact patient/stay-ID train/test splits this project's results were computed on ([`data/splits/*.json`](data/splits/)), and every number in [`docs/RESULTS.md`](docs/RESULTS.md) is gated on surviving log/artifact evidence, not asserted

---

## 🚀 How to Use

### Option A — load a published model directly

Three fitted models from this pipeline are published on the Hugging Face Hub:

```python
from autogluon.tabular import TabularPredictor
from huggingface_hub import snapshot_download

local_dir = snapshot_download("HassanB4/warfarin-review-phase3-autogluon-iwpc6256")
predictor = TabularPredictor.load(local_dir)
prediction = predictor.predict(your_dataframe)  # combined clinical+genetic features
```

See the model cards for [`phase1-baselines`](https://huggingface.co/HassanB4/warfarin-review-phase1-baselines), [`phase3-autogluon-iwpc6256`](https://huggingface.co/HassanB4/warfarin-review-phase3-autogluon-iwpc6256), and [`phase3-autogluon-iwpc1780`](https://huggingface.co/HassanB4/warfarin-review-phase3-autogluon-iwpc1780) for exact feature schemas.

### Option B — reproduce the pipeline

WarfaRisk has no single inference entrypoint — it's nine phases, each invoked explicitly:

```bash
uv sync                                        # base deps only
uv run python -m warfarisk.phase0_scaffolding  # regenerate splits, or use the shipped data/splits/*.json
uv run python -m warfarisk.phase1_baselines
uv sync --extra phase3
uv run python scripts/run_phase4_local.py      # etc. — see scripts/ and docs/RESULTS.md for the rest
```

Every phase past Phase 0 needs `data/eda_notebooks/cleaned/*.csv`, which is **not shipped** in this repository — see [`data/DATA.md`](data/DATA.md) for exactly how to obtain the source data and rebuild it locally.

---

## ⚙️ Training Procedure

The system fits and compares nine model architectures per cohort rather than fine-tuning a single deep model — the pipeline's central question is how much clinical vs. genetic information contributes to accuracy, ancestry fairness, and calibration, not chasing a single architecture.

### Training Data

Both IWPC cohorts are split 80/20 by patient ID with a fixed seed, distributed with this repository as ID lists (no clinical values):

| Cohort | n | Split (train/test) | Description |
| :--- | :---: | :---: | :--- |
| **IWPC-6256** | 6,037 | 4,830 / 1,207 | Clinical + CYP2C9/VKORC1 genotype consensus, 14 ancestry labels |
| **IWPC-1780** | 1,780 | 1,424 / 356 | Same features, binary ancestry flags, pre-z-scored continuous features |
| MIMIC-III demo | 17 patients | — | Clinical-only, longitudinal INR, no genotype |
| MIMIC-IV demo | 23 patients | — | Clinical-only, longitudinal INR, no genotype |
| eICU-CRD demo | 160 stays | — | Clinical-only, longitudinal INR, no genotype |

### Hyperparameters

| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Split Seed** | 20260725 | **Split Level** | Patient/stay ID (not row) |
| **Test Fraction** | 0.20 | **Feature Sets** | clinical / genetic / combined |
| **Best Model (both cohorts)** | AutoGluon `extreme_quality` | **AutoGluon Time Budget** | 600s |
| **AutoGluon Excluded Types** | TABDPT, TABICL, TABM, MITRA | **Conformal Target Coverage** | 90% (α=0.10) |
| **PW20 Acceptability Threshold** | ≥0.50 (IWPC-established) | **Calibration Methods Compared** | MAPIE, NGBoost, Deep Ensemble |

Full, file-referenced constants in [`configs/pipeline.yaml`](configs/pipeline.yaml) and [`configs/model_hparams.yaml`](configs/model_hparams.yaml).

### Model Selection

Nine architectures are trained on each of the three feature sets, on identical fixed splits, and compared on MAE/R²/PW20:

| Model (combined features, IWPC-6256) | MAE | R² | PW20 |
| :--- | :---: | :---: | :---: |
| **AutoGluon `extreme_quality`** | **8.552** | **0.475** | **0.466** |
| TabPFN | 8.664 | 0.463 | 0.446 |
| CatBoost | 8.762 | 0.454 | 0.455 |
| Stacking Ensemble | 9.046 | 0.431 | 0.447 |
| MLP | 9.006 | 0.412 | 0.445 |

Full 9-architecture × 3-feature-set tables for both cohorts are in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## 📊 Evaluation Results

Every number below is from [`docs/RESULTS.md`](docs/RESULTS.md), a curated, evidence-gated ledger — a row only qualifies with a surviving log file, executed notebook output, or independently-verified Hugging Face link.

### Our Results

| Phase | Cohort | Model | MAE | R² | PW20 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Phase 1 | IWPC-6256 | IWPC published equation (baseline) | 9.177 | 0.413 | 0.429 |
| **Phase 3** | **IWPC-6256** | **AutoGluon `extreme_quality`** | **8.552** | **0.475** | **0.466** |
| **Phase 3** | **IWPC-1780** | **AutoGluon `extreme_quality`** | **7.959** | **0.480** | **0.497** |

### Phase 1 — Mandatory Baselines (IWPC-6256, n_test=1207)

| Baseline | MAE | R² | PW20 |
| :--- | :---: | :---: | :---: |
| Naive median dose | 12.339 | -0.048 | 0.342 |
| Clinical-only linear regression | 10.860 | 0.222 | 0.354 |
| **IWPC published pharmacogenetic equation** | **9.177** | **0.413** | **0.429** |

### Phase 4 — Ancestry-Stratified Leave-One-Group-Out (IWPC-6256, real CatBoost)

| Ancestry Group | n (holdout) | Held-out MAE | Control MAE | Held-out R² | Control R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| White | 2397 | 9.692 | 9.593 | 0.345 | 0.412 |
| Caucasian | 662 | 10.859 | 11.560 | 0.209 | 0.218 |
| Black | 326 | 11.421 | 12.413 | 0.211 | 0.047 |
| Black or African American | 255 | 13.456 | 12.991 | 0.077 | 0.060 |
| Han Chinese | 250 | 6.906 | 4.772 | 0.114 | 0.068 |
| African-American | 107 | 14.381 | 17.761 | 0.287 | -0.028 |

**Honest finding:** this does not cleanly replicate the uniform, one-directional penalty for Black/African American patients reported in prior literature. `"Black"` shows no portability penalty here (held-out MAE better than its control), while Han Chinese and `"Black or African American"` do show a real one — part of the original pattern is model- and label-fragmentation-dependent, not fixed. Full 14-group table in `docs/RESULTS.md`.

### Phase 5 — Calibration (MAPIE Conformal Prediction)

| Cohort | Overall Coverage (target 90%) |
| :--- | :---: |
| IWPC-6256 | 89.8% |
| IWPC-1780 | 93.3% |

| Ancestry Group (IWPC-6256) | n | Coverage |
| :--- | :---: | :---: |
| African American | 5 | 60.0% |
| African-American | 24 | 62.5% |
| Black or African American | 52 | 69.2% |
| Black | 73 | 82.2% |
| Caucasian | 124 | 86.3% |
| White | 493 | 91.3% |

Every group whose label plausibly captures Black/African American patients sits below the 90% nominal target — a fairness gap invisible in the aggregate coverage number alone. A deep-ensemble uncertainty method tested in the same framework was far worse overall (24.9%/21.6% coverage) — see `docs/RESULTS.md` for the full comparison, including a finding that NGBoost narrows this per-subgroup gap further than MAPIE does.

---

## ⚠️ Limitations

* **Research-Only**: no prospective clinical validation, no regulatory status
* **Fragmented Ancestry Labels**: IWPC-6256 records Black/African-American patients under three separate, overlapping labels (`Black`, `Black or African American`, `African-American`), which fragments statistical power for subgroup analysis
* **IWPC-Equation Baseline Caveat**: its coefficients were transcribed from the widely-cited public form of the equation, not verified directly against the original paper
* **Small ICU Cohorts**: the longitudinal, clinical-only track uses demo-scale cohorts (n=17/23/160) — read as a working-pipeline demonstration, not a well-powered clinical finding
* **No Raw Data Shipped**: third parties must independently obtain IWPC access (and, optionally, the PhysioNet demo databases) to fully reproduce the pipeline from raw data — see [`data/DATA.md`](data/DATA.md)

---

## 🙏 Acknowledgements

We thank the PharmGKB / International Warfarin Pharmacogenetics Consortium for the IWPC dataset, PhysioNet for the MIMIC-III/IV and eICU-CRD demo databases, and the CPIC/PharmCAT projects for pharmacogenomic guideline data referenced in this pipeline's explainability layer.

### Related Links

* [PharmGKB](https://www.pharmgkb.org/)
* [PhysioNet](https://physionet.org/)
* [Hugging Face Models](https://huggingface.co/HassanB4)

---

## 📜 Citation

A paper describing this work is in preparation. This section will be updated with a full citation once it is published.

```bibtex
@misc{warfarisk,
    title={WarfaRisk: A Reproducible, Ancestry-Stratified Machine Learning Pipeline for Warfarin Dose Prediction},
    author={Barmandah, Hassan and Bawazir, Omar Abdullah and Marghalani, Siraj Aldeen and Shaat, Moath and AlEissa, Mariam M.},
    year={2026},
    note={Manuscript in preparation — update this citation once published}
}
```

---

## 📄 License

This project is licensed under the Apache 2.0 License. No data is redistributed by this repository; see [`data/DATA.md`](data/DATA.md) for each source dataset's own access terms.
