# WarfaRisk

A nine-phase machine learning pipeline for predicting the weekly maintenance dose of warfarin, built on the public IWPC pharmacogenomic cohorts. It reports accuracy, calibration and explainability on the same fixed splits, and breaks calibration down by ancestry group.

[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/HasanBGit/WarfaRisk)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-4%20Models-F9D371?logo=huggingface&logoColor=black)](https://huggingface.co/collections/HassanB4/beyond-mae-ancestry-stratified-warfarin-dose-prediction-6a79b7f715e8533e72db4b4e)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

> This is a research artifact, not a validated clinical tool. It has not been evaluated prospectively, has no regulatory clearance, and must not be used to make a dosing decision.

Authors: Hassan Barmandah, Omar Abdullah Bawazir, Siraj Aldeen Marghalani, Moath Shaat, Abdullah N. Alkattan and Mariam M. AlEissa (corresponding author), Alfaisal University, Riyadh.

## Contents

- [Overview](#overview)
- [Headline results](#headline-results)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Development](#development)
- [Limitations](#limitations)
- [Data and license](#data-and-license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Overview

The pipeline starts with baseline models and ends with explanations. Every phase runs on fixed, patient-level train and test splits generated from one documented random seed (`20260725`).

| Phase | Name | What it does |
| :--- | :--- | :--- |
| 0 | Scaffolding | Creates the seeded splits and the data dictionary |
| 1 | Baselines | Median dose, clinical-only linear model, published IWPC equation |
| 2 | Leakage audit | Checks that no outcome-related column reaches the inputs and that train and test patients do not overlap |
| 3 | Nested ablation | Compares clinical, genetic and combined features across nine model architectures |
| 4 | Ancestry hold-out | Leaves one ancestry group out, checked with two model families |
| 5 | Calibration | Compares MAPIE conformal prediction, NGBoost and a deep ensemble |
| 6 | Explainability | Compares SHAP with an independently fitted EBM |
| 7 | Longitudinal track | Reinforcement-learning and LSTM dosing, time to therapeutic INR, imputation, clinical embeddings |
| 8 | Fairness reporting | Splits every results table by subgroup |
| 9 | Explanation layer | Retrieval-based explanations with citation checks |

The main findings:

- AutoGluon with combined clinical and genetic features gives the best accuracy on both cohorts.
- MAPIE and NGBoost reach close to the nominal 90% coverage overall (89.0 to 93.3%). A deep ensemble of the same architecture covers only 21.6 to 24.9%.
- Split by ancestry, every label that plausibly captures Black or African American patients falls below 90% coverage, even when the overall figure looks fine.
- SHAP and the EBM agree on the main predictors: VKORC1 genotype, age and weight.

Full tables and the evidence trail are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Headline results

| Phase | Cohort | Model | MAE | R² | PW20 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | IWPC-6256 | IWPC published equation (baseline) | 9.177 | 0.413 | 0.429 |
| 3 | IWPC-6256 | AutoGluon `extreme_quality` | 8.552 | 0.475 | 0.466 |
| 3 | IWPC-1780 | AutoGluon `extreme_quality` | 7.959 | 0.480 | 0.497 |

MAE is in mg/week. PW20 is the share of predictions within 20% of the true dose.

| Cohort | MAPIE coverage (target 90%) |
| :--- | :---: |
| IWPC-6256 | 89.8% |
| IWPC-1780 | 93.3% |

The per-architecture, per-ancestry and per-subgroup calibration tables are in [`docs/RESULTS.md`](docs/RESULTS.md). They also cover the leave-one-ancestry-out error results. These do not fully reproduce an earlier analysis: the Black label shows no error penalty under either model, while Han Chinese and Caucasian show penalties that depend on the model.

## Repository layout

| Path | Contents |
| :--- | :--- |
| `src/warfarisk/` | The pipeline, one module per phase, with shared helpers in `common/` |
| `scripts/` | Run scripts, statistical checks and plotting scripts |
| `prototype_app/` | Web interface and the Flask server behind it |
| `configs/` | Pipeline and model hyperparameter settings |
| `data/` | Split ID files, exploration notebooks and [`DATA.md`](data/DATA.md). No patient data. |
| `results/` | Result tables and logs |
| `docs/` | Results, known issues, reproducibility map and training plan |
| `tests/` | Tests for the prototype server |

## Quick start

### Load a published model

```python
from autogluon.tabular import TabularPredictor
from huggingface_hub import snapshot_download

local_dir = snapshot_download("HassanB4/warfarisk-autogluon-6256")
predictor = TabularPredictor.load(local_dir)
prediction = predictor.predict(your_dataframe)  # combined clinical and genetic features
```

Four models are on the Hub: [`warfarisk-baselines`](https://huggingface.co/HassanB4/warfarisk-baselines), [`warfarisk-autogluon-6256`](https://huggingface.co/HassanB4/warfarisk-autogluon-6256), [`warfarisk-autogluon-1780`](https://huggingface.co/HassanB4/warfarisk-autogluon-1780) and [`warfarisk-catboost-6256`](https://huggingface.co/HassanB4/warfarisk-catboost-6256). The other architectures in the nine-model comparison can be reproduced from the code and the result CSVs in `results/`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

### Reproduce the pipeline

```bash
uv sync                                        # base dependencies only
uv run python -m warfarisk.phase0_scaffolding  # regenerate splits, or use the shipped data/splits/*.json
uv run python -m warfarisk.phase1_baselines
uv sync --extra phase3
uv run python scripts/run_phase4_local.py      # continue with the scripts in scripts/
```

Every phase after Phase 0 needs `data/eda_notebooks/cleaned/*.csv`. These files are not in the repository. [`data/DATA.md`](data/DATA.md) explains how to get the source data and rebuild them. Training settings are in [`configs/pipeline.yaml`](configs/pipeline.yaml) and [`configs/model_hparams.yaml`](configs/model_hparams.yaml).

### Run the prototype interface

The server in `prototype_app/server/app_server.py` loads the fitted CatBoost model and returns a dose prediction, leave-one-out feature attributions and a 90% prediction interval. The interval is a fixed band whose half-width is the 90th percentile of the model's absolute errors on its held-out test set. It is not the MAPIE interval used in the per-ancestry calibration analysis. No numbers are hardcoded.

An explanation layer grounds each explanation in PharmGKB and CPIC guideline text and in this project's own SHAP and ancestry findings. Every citation is checked against the retrieved evidence before it is shown. The language model is MedGemma-27b-text-it through Hugging Face Inference Providers. With no key configured, the server falls back to a deterministic, citation-verified generator. Copy [`prototype_app/.env.example`](prototype_app/.env.example) to set a key.

A separate React and Django demo with fixed example data is in [WarfaRisk-Prototype-UI](https://github.com/HasanBGit/WarfaRisk-Prototype-UI). The interface connected to the trained model is the one in `prototype_app/`.

```bash
cd prototype_app
uv run --with flask,flask-cors,catboost,pandas python3 server/app_server.py   # API on port 8787
python3 -m http.server 8765                                                   # static front end on port 8765
```

## Development

```bash
uvx ruff check . && uvx ruff format --check .                  # lint and formatting
uv run --with pytest,flask,flask-cors,catboost,pandas pytest   # prototype server tests
```

The server is split into four modules:

| Module | Role |
| :--- | :--- |
| `settings.py` | Configuration from the environment and `.env` |
| `dosing_model.py` | Features, inference and attribution |
| `explain_service.py` | Proxy to the language model |
| `app_server.py` | Flask app factory and routes |

## Limitations

- The pipeline is for research only. It has no prospective clinical validation and no regulatory status.
- IWPC-6256 splits Black and African American patients across three overlapping labels, which reduces the statistical power of each subgroup.
- The prototype interface has not been evaluated prospectively. Its prediction interval comes from test-set residuals and carries no per-ancestry coverage guarantee.
- No raw data is included. To reproduce from raw data you need your own access to IWPC and, optionally, PhysioNet. See [`data/DATA.md`](data/DATA.md).

The full list is in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

## Data and license

The code is released under the Apache License 2.0. See [`LICENSE`](LICENSE). This repository does not redistribute any data. [`data/DATA.md`](data/DATA.md) lists the access terms for each source dataset.

## Acknowledgements

PharmGKB and the International Warfarin Pharmacogenetics Consortium provided the IWPC dataset. PhysioNet provided the MIMIC-III, MIMIC-IV and eICU-CRD demo databases. The CPIC and PharmCAT projects provided the guideline data used by the explanation layer.

Links: [PharmGKB](https://www.pharmgkb.org/), [PhysioNet](https://physionet.org/), [Hugging Face model collection](https://huggingface.co/collections/HassanB4/beyond-mae-ancestry-stratified-warfarin-dose-prediction-6a79b7f715e8533e72db4b4e).

## Citation

The pipeline and its results are published in the *Journal of Personalized Medicine*.

> Barmandah, H.; Bawazir, O.A.; Marghalani, S.A.; Shaat, M.; Alkattan, A.N.; AlEissa, M.M. Beyond Mean Absolute Error: Ancestry-Stratified Calibration and Explainability for Warfarin Dosing Models. *J. Pers. Med.* **2026**, *16* (10), 531. https://doi.org/10.3390/jpm16100531

```bibtex
@Article{jpm16100531,
    author={Barmandah, Hassan and Bawazir, Omar A. and Marghalani, Siraj Aldeen and Shaat, Muath and Alkattan, Abdullah N. and AlEissa, Mariam M.},
    title={Beyond Mean Absolute Error: Ancestry-Stratified Calibration and Explainability for Warfarin Dosing Models},
    journal={Journal of Personalized Medicine},
    volume={16},
    year={2026},
    number={10},
    article-number={531},
    url={https://www.mdpi.com/2075-4426/16/10/531},
    issn={2075-4426},
    doi={10.3390/jpm16100531}
}
```

The prototype system (pipeline, web interface and explanation layer) is described in a companion paper in MDPI *AI*.

> Barmandah, H.; Marghalani, S.A.; Bawazir, O.A.; Shaat, M.; Alkattan, A.N.; AlEissa, M.M. WarfaRisk: A Reproducible, Fairness-Aware Machine Learning Pipeline for Warfarin Dose Prediction (Prototype). *AI* **2026**, *7* (10), 431. https://doi.org/10.3390/ai7100431

```bibtex
@article{barmandah2026warfarisk,
  author  = {Barmandah, Hassan and Marghalani, Siraj Aldeen and Bawazir, Omar Abdullah and Shaat, Muath and Alkattan, Abdullah N. and AlEissa, Mariam M.},
  title   = {{WarfaRisk}: A Reproducible, Fairness-Aware Machine Learning Pipeline for Warfarin Dose Prediction (Prototype)},
  journal = {AI},
  publisher = {MDPI},
  volume  = {7},
  number  = {10},
  pages   = {431},
  year    = {2026},
  doi     = {10.3390/ai7100431},
}
```

A machine-readable version is in [`CITATION.cff`](CITATION.cff).
