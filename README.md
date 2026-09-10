# Beyond Mean Absolute Error: Ancestry-Stratified Calibration and Explainability for Warfarin Dosing Models

A nine-phase, fully reproducible machine learning pipeline for warfarin maintenance-dose prediction on the public IWPC pharmacogenomic cohorts. Best model (AutoGluon `extreme_quality`, combined clinical+genetic features): **MAE 8.55 / R² 0.475 / PW20 0.466** on IWPC-6256, **MAE 7.96 / R² 0.480 / PW20 0.497** on IWPC-1780. An ancestry-stratified calibration analysis surfaces a per-subgroup coverage gap the aggregate number alone hides.

#### By: Hassan Barmandah, Omar Abdullah Bawazir, Siraj Aldeen Marghalani, Moath Shaat, Abdullah N. Alkattan, and Mariam M. AlEissa (corresponding author), Alfaisal University, Riyadh

[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/HasanBGit/WarfaRisk)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-3%20Models-F9D371?logo=huggingface&logoColor=black)](https://huggingface.co/collections/HassanB4/beyond-mae-ancestry-stratified-warfarin-dose-prediction-6a79b7f715e8533e72db4b4e)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

> **This is a research artifact, not a validated clinical tool.** It has not been evaluated prospectively, carries no regulatory clearance, and should not be used to make an actual dosing decision.

---

## What this is

The pipeline moves from mandatory baselines, through a nested clinical/genetic/combined feature ablation across nine model architectures, to ancestry-stratified leave-one-group-out fairness evaluation, a three-way calibration comparison (MAPIE conformal prediction, NGBoost, deep ensemble), and SHAP-vs-EBM cross-validated explainability — all on fixed, patient-level train/test splits under one documented random seed (`20260725`).

* **Nested feature ablation** across 9 architectures × 3 feature sets (clinical / genetic / combined), both IWPC cohorts
* **Ancestry-stratified fairness evaluation**, cross-checked between two independent model families
* **Calibration gap finding**: MAPIE and NGBoost reach near-nominal overall coverage (89.8–93.3%) while a same-architecture deep ensemble collapses to 21.6–24.9%; per-ancestry coverage then shows every label plausibly capturing Black/African American patients under-covered even when the aggregate looks fine
* **Cross-validated explainability**: SHAP and an independently-fit EBM agree on the same dominant features (VKORC1 genotype, age, weight)
* **Deployed clinical interface** ([`prototype_app/`](prototype_app/)) serving this pipeline's real trained model, real SHAP attributions, and real MAPIE conformal intervals, with a SHAP-conditioned RAG explanation layer

Full results, tables, and evidence trail: [`docs/RESULTS.md`](docs/RESULTS.md). Known limitations: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

---

## 🩺 Clinical Interface

`prototype_app/server/app_server.py` loads the fitted CatBoost model directly and serves real predictions, real leave-one-out feature attributions, and a real 90th-percentile conformal interval — no hardcoded numbers. A retrieval-augmented generation (RAG) layer grounds explanations in PharmGKB/CPIC guideline text and this project's own SHAP/ancestry findings, conditioned on each patient's top-ranked features, with every citation checked against retrieved evidence before display. Generative backbone: **MedGemma-27b-text-it** via Hugging Face Inference Providers (falls back to a deterministic, citation-verified generator with no key configured). See [`prototype_app/.env.example`](prototype_app/.env.example).

```bash
cd prototype_app
uv run --with flask,flask-cors,catboost,pandas python3 server/app_server.py   # serves on :8787
python3 -m http.server 8765                                                   # serves the static frontend
```

---

## 🚀 How to Use

### Option A: load a published model directly

```python
from autogluon.tabular import TabularPredictor
from huggingface_hub import snapshot_download

local_dir = snapshot_download("HassanB4/warfarisk-autogluon-6256")
predictor = TabularPredictor.load(local_dir)
prediction = predictor.predict(your_dataframe)  # combined clinical+genetic features
```

Model cards: [`warfarisk-baselines`](https://huggingface.co/HassanB4/warfarisk-baselines), [`warfarisk-autogluon-6256`](https://huggingface.co/HassanB4/warfarisk-autogluon-6256), [`warfarisk-autogluon-1780`](https://huggingface.co/HassanB4/warfarisk-autogluon-1780).

### Option B: reproduce the pipeline

```bash
uv sync                                        # base deps only
uv run python -m warfarisk.phase0_scaffolding  # regenerate splits, or use the shipped data/splits/*.json
uv run python -m warfarisk.phase1_baselines
uv sync --extra phase3
uv run python scripts/run_phase4_local.py      # etc.; see scripts/ and docs/RESULTS.md for the rest
```

Every phase past Phase 0 needs `data/eda_notebooks/cleaned/*.csv`, **not shipped** in this repository; see [`data/DATA.md`](data/DATA.md) for how to obtain the source data and rebuild it locally. Training config: [`configs/pipeline.yaml`](configs/pipeline.yaml), [`configs/model_hparams.yaml`](configs/model_hparams.yaml).

---

## 📊 Headline Results

| Phase | Cohort | Model | MAE | R² | PW20 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Phase 1 | IWPC-6256 | IWPC published equation (baseline) | 9.177 | 0.413 | 0.429 |
| **Phase 3** | **IWPC-6256** | **AutoGluon `extreme_quality`** | **8.552** | **0.475** | **0.466** |
| **Phase 3** | **IWPC-1780** | **AutoGluon `extreme_quality`** | **7.959** | **0.480** | **0.497** |

| Cohort | MAPIE Coverage (target 90%) |
| :--- | :---: |
| IWPC-6256 | 89.8% |
| IWPC-1780 | 93.3% |

Every ancestry group whose label plausibly captures Black/African American patients sits below the 90% nominal target, a fairness gap invisible in the aggregate number alone. Full per-architecture, per-ancestry, and per-subgroup calibration tables (including the honest finding that the Black/African-American penalty doesn't replicate uniformly across labels): [`docs/RESULTS.md`](docs/RESULTS.md).

---

## ⚠️ Limitations

* **Research-only**: no prospective clinical validation, no regulatory status
* **Fragmented ancestry labels**: IWPC-6256 splits Black/African-American patients across three overlapping labels, fragmenting subgroup statistical power
* **No raw data shipped**: third parties must independently obtain IWPC access (and, optionally, PhysioNet) to reproduce from raw data; see [`data/DATA.md`](data/DATA.md)

Full list: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

---

## 🙏 Acknowledgements

PharmGKB / the International Warfarin Pharmacogenetics Consortium for the IWPC dataset, PhysioNet for the MIMIC-III/IV and eICU-CRD demo databases, and the CPIC/PharmCAT projects for the guideline data referenced in the explainability layer.

* [PharmGKB](https://www.pharmgkb.org/) · [PhysioNet](https://physionet.org/) · [Hugging Face Model Collection](https://huggingface.co/collections/HassanB4/beyond-mae-ancestry-stratified-warfarin-dose-prediction-6a79b7f715e8533e72db4b4e)

---

## 📜 Citation

This work is described in the following manuscript, currently in preparation.

> Barmandah, H.; Bawazir, O.A.; Marghalani, S.A.; Shaat, M.; Alkattan, A.N.; AlEissa, M.M. Beyond Mean Absolute Error: Ancestry-Stratified Calibration and Explainability for Warfarin Dosing Models. Manuscript in preparation, 2026.

```bibtex
@unpublished{barmandah2026beyond,
    title={Beyond Mean Absolute Error: Ancestry-Stratified Calibration and Explainability for Warfarin Dosing Models},
    author={Barmandah, Hassan and Bawazir, Omar Abdullah and Marghalani, Siraj Aldeen and Shaat, Moath and Alkattan, Abdullah N. and AlEissa, Mariam M.},
    year={2026},
    note={Manuscript in preparation. Cite the published version once assigned.}
}
```

---

## 📄 License

Apache 2.0. No data is redistributed by this repository; see [`data/DATA.md`](data/DATA.md) for each source dataset's own access terms.
