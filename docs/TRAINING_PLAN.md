# Warfarin Dose/INR Model — Training Plan

Grounded in the full 32-study systematic review (`MDPI_Article/main.tex`) and the actual files in `data/eda_notebooks/cleaned/`. The review's own Section 4.6 ("Implications for Future Research") is a 10-point checklist for exactly the model this project is building — this plan operationalizes each point against data already on hand, and says explicitly where the data can't yet support a point rather than skipping it silently.

## What the review found that this plan has to answer to

1. **Reported accuracy is inversely related to evaluation rigor** — the smallest, least-validated cohorts report the best numbers. So every result below must ship with test-set size and validation design, not just a metric.
2. **Genetics is nominal, not substantive** — present in 65.6% of studies but usually 1–2 common variants in a cohort dominated by one genotype; ancestry-specific alleles (CYP2C9\*5/\*6/\*8/\*11, rs12777823) are absent field-wide. **Our own data has this exact gap** (see Phase 4).
3. **Ancestry mismatch, not algorithm choice, drives the biggest failures.** The review's own IWPC ancestry hold-out (Table 4) found linear-regression MAE jump from 8.38 (overall CV) to 12.42 mg/week when trained without Black/African-American patients and tested only on them. This is the single most important experiment to replicate on our own pipeline before trusting anything else.
4. **Calibration is reported by only 1 of 32 studies.** Point estimates are not enough.
5. **Explainability is reported by only 3 of 32 studies (9.4%).**
6. **Model complexity hasn't paid off** anywhere a fair comparison was run — linear regression matches RF/GBM/NN repeatedly. Don't assume a fancier model is progress; prove it on identical splits.

## Data inventory (what each phase actually trains on)

| File | n | Genotype? | Longitudinal? | Ancestry field | Role |
|---|---|---|---|---|---|
| `iwpc_full_6256_clean.csv` | 6,037 | CYP2C9 (\*1/\*2/\*3 only), VKORC1 (no CYP4F2) | No | `Race (Reported)`, `Race (OMB)`, `Ethnicity (Reported/OMB)` | Primary genetics cohort + ancestry hold-out |
| `iwpc_1780_clean.csv` | 1,780 | `CYP2C9.12`/`.13`/`.other`, `VKORC1.AG`/`.AA` | No | `Black`, `Asian` (binary flags) | Secondary genetics cohort; **Weight/Height/Age are pre-z-scored** — don't pool raw with 6256 |
| `mimic3_inr_trajectory_clean.csv` + `mimic3_warfarin_prescriptions_clean.csv` + `mimic3_admissions_clean.csv` | 17 pts / 450 INR readings | No | No usable ancestry granularity | Longitudinal, clinical-only |
| `mimic4_inr_labs_clean.csv` + `mimic4_warfarin_prescriptions_clean.csv` + `mimic4_admissions_clean.csv` | 23 pts / 593 INR readings | No | No | Longitudinal, clinical-only |
| `eicu_inr_labs_clean.csv` + `eicu_warfarin_medication_clean.csv` + `eicu_patient_demographics_clean.csv` | 160 stays / 1,020 INR readings | No | `ethnicity` field present | Largest longitudinal clinical-only cohort |
| `pharmgkb_allele_tables/` (raw, `data/usable/`) | — | star-allele lookup | — | — | Reference only, not training rows |

**No cohort has both genotype and longitudinal INR.** This is the same gap the review found across the whole literature (RL performs best but skips genetics; genetics work is cross-sectional). Phases below run genetics and longitudinal modeling as two separate tracks rather than pretending they can be merged with current data.

---

## Phase 0 — Reproducibility scaffolding (before any model is trained)

- Fixed random seeds; every split saved to disk (`splits/*.json`, patient IDs not row indices) so every later phase reuses the *same* train/test partition.
- All imputation, scaling, and feature selection fit **only** inside the training fold (`sklearn.Pipeline` + `cross_val_predict`, never fit-on-full-data-then-split) — this was the review's single most common risk-of-bias finding (87.5% of studies high-risk on the analysis domain).
- A data dictionary mapping every raw column above to its cleaned/model-ready name.
- **Verify:** re-running the pipeline twice with the same seed produces byte-identical splits and metrics.

## Phase 1 — Mandatory baselines (run before, and reported alongside, every later result)

Per patient row: naive median-dose predictor, clinical-only linear regression, and the published IWPC pharmacogenetic equation, all evaluated on the identical Phase-0 split. Every subsequent model's result is reported next to these three, not in isolation — this is what makes a claimed improvement real instead of an artifact of a favorable split (the review's central diagnostic).
- **Model/tool — pick this: [`warfit-learn`](https://github.com/gianlucatruda/warfit-learn)** (Truda & Marais, review ref [44], GPLv3). Implements fixed-dose, clinical, and IWPC pharmacogenetic algorithms as callable baselines, with the IWPC cohort already bundled — this project's own data pipeline already touches its bundled `iwpc.pkl`. Don't hand-roll the IWPC equation when a maintained, cited implementation exists.
- **Verify:** naive median-dose MAE roughly matches the review's own reference point (~34.1% PW20 baseline order of magnitude) as a sanity check that the pipeline isn't broken.

## Phase 2 — Leakage audit

Confirm on `iwpc_full_6256_clean.csv`/`iwpc_1780_clean.csv`: no row has `INR on Reported Therapeutic Dose of Warfarin` or `Target INR` used as a model *input* for a *dose* target (outcome leakage, flagged in the dataset-selection memory already). Confirm eICU/MIMIC longitudinal splits are **by patient/stay**, not by row, since multiple INR readings per stay would otherwise leak the same patient across train/test.
- **Verify:** patient/stay ID sets in train and test are disjoint; grep confirms no INR-derived column reaches `X` in the genetics-cohort models.

## Phase 3 — Nested incremental-value analysis for genetics

On `iwpc_full_6256_clean.csv` (n=6,037) and separately `iwpc_1780_clean.csv` (n=1,780, keep z-scored features as-is, don't re-harmonize with 6256), fit four nested models on identical splits: clinical-only → genetic-only → clinical+genetic → (6256 only) clinical+genetic+`INR on Reported Therapeutic Dose of Warfarin`-as-covariate-not-leakage-check. Report ΔMAE, ΔR², ΔPW20 at each step.
- **Model/tool — pick this: CatBoost.** Natively encodes categorical genotype strings (`CYP2C9 consensus`, `VKORC1 * consensus`) without manual one-hot, and this project's cohort size (n≈6,000, ~60 features) sits well inside its comfort zone.
- **Run alongside as a no-tuning ceiling estimate: [TabPFN-3](https://huggingface.co/Prior-Labs/tabpfn_3)** (`Prior-Labs/tabpfn_3` on HF, May 2026, supersedes TabPFN-v2 — scales to 1M rows with test-time compute scaling, SOTA on the TabArena benchmark). Your 6,037-row cohort is trivial for it.
- **Advanced option — run this first as the ceiling benchmark: [AutoGluon-Tabular](https://github.com/autogluon/autogluon) with `presets="extreme_quality"`.** Confirmed AutoGluon now natively wraps TabPFN (and TabICL/TabDPT) inside a `zeroshot_2025_tabfm` portfolio of ~22 models, auto-selects and stacks the winners on your exact data with one call (`TabularPredictor(...).fit(train_data, presets="extreme_quality")`) — no manual CatBoost-vs-TabPFN comparison needed as a first pass. Use its output as the accuracy ceiling, then decide whether CatBoost alone (for Phase 6's interpretability) is worth the accuracy gap.
- **Genotype-encoding upgrade to try alongside the above: [AlphaMissense](https://huggingface.co/datasets/katielink/dm_alphamissense)** precomputed pathogenicity scores for CYP2C9\*2 (p.Arg144Cys) and \*3 (p.Ile359Leu) as an engineered feature replacing/joining the star-allele label. **Real limitation, not a fix-all: VKORC1's key dosing variant (rs9923231, -1639G>A) is a promoter SNP, not missense — AlphaMissense structurally cannot score it**, and VKORC1 is often the *stronger* of the two predictors per this project's own SHAP benchmark. Treat this as a partial CYP2C9-only enhancement, not a genetics overhaul.
- Per the review's own repeated finding that model complexity rarely pays off (linear ≈ RF ≈ GBM ≈ NN across at least four of its included studies): if none of CatBoost/TabPFN-3/AutoGluon clearly beat the Phase 1 linear-regression baseline, report that honestly rather than picking the fancier model anyway.
- **Verify:** clinical+genetic result should land near the review's own reanalysis (R² 0.297→0.439 on 6256-derived data, 0.212→0.438 on 1780) — if it doesn't, something in the pipeline differs from the paper's and needs explaining before trusting anything downstream.

## Phase 4 — Ancestry-stratified / leave-one-ancestry-out validation (highest priority experiment)

Direct replication, then extension, of the review's own Table 4 using `Race (Reported)`/`Race (OMB)` in the 6256 cohort and `Black`/`Asian` flags in the 1780 cohort: train on all-but-one ancestry group, test only on the held-out group, for every group with enough patients to be meaningful. Report MAE/R² per group next to the matched in-group internal control (as the paper does with its White 80/20 control).
- **Model/tool:** none new — reuse whichever model wins Phase 3. This phase is a validation *protocol* (retrain on N-1 ancestry groups, test on the held-out one), not a new architecture choice.
- **Explicit known gap, not fixed by modeling:** the genotype columns in both files only resolve CYP2C9 \*1/\*2/\*3 and VKORC1's core SNPs — CYP2C9\*5/\*6/\*8/\*11 and rs12777823 (the variants specifically relevant to African-ancestry patients, per the review) are not present in either extract. This phase will very likely reproduce the review's own finding that error concentrates in the least-represented, worst-covered ancestry group — that's expected, and the fix is new genotyping data, not a better model. State this plainly in any writeup rather than tuning around it.
- **Verify:** held-out-group MAE compared against matched in-group control MAE, same shape as the review's Table 4.

## Phase 5 — Calibration and uncertainty

Wrap whichever model wins Phase 3 in conformal prediction (weighted conformal for continuous dose output, per the earlier HF/arXiv research memo — arXiv:2409.20412) so output is an interval, not a point estimate. Compute calibration **per ancestry subgroup**, not just overall — the review's own Sridharan et al. calibration slopes (0.94 RF vs 0.06 logistic regression) show algorithm choice matters here, and Phase 4 already shows performance isn't uniform across groups, so calibration might not be either.
- **Model/tool — pick this: [MAPIE](https://github.com/scikit-learn-contrib/MAPIE)** (scikit-learn-contrib). Model-agnostic conformal regression intervals wrapped around the Phase 3 winner — the direct implementation vehicle for arXiv:2409.20412, not just a citation.
- **Worth comparing, not committing to blind: [NGBoost](https://github.com/stanfordmlgroup/ngboost)** (Stanford). Trains a full predictive distribution directly rather than wrapping a point estimate post-hoc — a genuinely different approach to the same problem, cheap to run both and compare rather than assuming one is right.
- **Verify:** empirical coverage of the prediction interval matches nominal (e.g., 90% intervals contain the true dose ~90% of the time) overall *and* within each ancestry subgroup from Phase 4.

## Phase 6 — Explainability

SHAP on the Phase 3 winning tabular model, computed overall and split by the Phase-4 ancestry subgroups (does VKORC1 dominate uniformly across groups, or does attribution shift the way the review's own genetics-vs-INR-history hypothesis suggests it might). This directly extends, rather than just re-cites, the review's own SHAP benchmark (58.1% genetic feature importance) and Sridharan et al.'s finding that CYP2C9 contribution was directionally inconsistent in misclassified cases.
- **Model/tool — pick this: SHAP on the Phase 3 winner, cross-checked against [InterpretML's Explainable Boosting Machine](https://github.com/interpretml/interpret)** trained independently (regression supported natively). EBM is a glass-box GAM that returns exact per-feature contributions with no post-hoc approximation. Agreement between EBM's native attributions and the Phase-3 model's SHAP values is real evidence; disagreement is itself a finding worth reporting (given the review's own methodology-dependent SHAP results), not a tiebreak to ignore.
- **Verify:** SHAP summary plot generated per subgroup; any claim about "genetics matters most" is checked against whether it holds in every ancestry subgroup or just the majority one.

## Phase 7 — Longitudinal / dynamic-dosing track (separate from Phases 3–6, clinical-only)

On the pooled MIMIC-III/MIMIC-IV/eICU longitudinal INR + prescription data (no genotype available — stated explicitly, not worked around): a dose-adjustment model over the INR trajectory, following the review's own recommendation for dynamic rather than static targets. Candidate approach from the earlier model/paper research: offline contextual bandit (arXiv:2402.11123), with the temporal-resampling caution from arXiv:2602.06603 applied before trusting any offline eval built on binned timestamps.
- **Model/tool — pick this: [d3rlpy](https://github.com/takuseno/d3rlpy)**, using its built-in Batch-Constrained Q-learning (BCQ) implementation — the same algorithm class Petch et al. 2024 (the review's strongest real-patient RL study) used. Confirmed: **neither Petch et al. 2024 nor Zeng et al. 2022 released official code**, so this is a from-scratch reimplementation on a maintained library, not an adaptation of existing code. One unofficial, unmaintained reference exists — [`hlycharles/WarfarinDoseRL`](https://github.com/hlycharles/WarfarinDoseRL) (LinUCB/Lasso-bandit baselines on IWPC-style features) — worth reading for ideas, not worth depending on.
- **Verify:** offline eval protocol uses a temporally honest holdout (train on earlier stays/admissions, test on later ones, not a random shuffle) given the resampling-risk paper's warning.

## Phase 8 — Safety/fairness reporting, by design

Every phase above reports its headline metric stratified by ancestry (Phase 4's groups), sex, and dose category alongside the overall number — not computed once at the end. This is a formatting requirement on Phases 3–7's output tables, not a separate modeling step.

## Phase 9 — LLM explanation/citation layer

Per the earlier research memo (artifact: "Closing the Warfarin Trilemma Gap"): ground explanations in the Phase 6 SHAP output plus retrieved CPIC/PharmGKB guideline text and this project's own `pharmgkb_allele_tables/`. **Revised retrieval-corpus source:** rather than compiling CPIC text from scratch (confirmed not pre-packaged on HF), point retrieval at **[PharmCAT](https://github.com/PharmGKB/PharmCAT)** — the PharmGKB-maintained tool that turns genotype calls into CPIC/DPWG/FDA-label prescribing recommendations, with warfarin as a flagship supported drug — and its companion **[`cpicpgx/cpic-data`](https://github.com/cpicpgx/cpic-data)** repo, which is machine-readable (not PDF) and confirmed queryable down to CYP4F2/warfarin RxNorm granularity. Notable for the writeup: the current CPIC warfarin guideline already incorporates CYP4F2 and rs12777823 — the exact ancestry-relevant variants Phase 4 will show this project's own IWPC extracts lack — so the clinical guideline literature is ahead of the ML literature here, a citable point distinct from "no data exists." **New requirement from this review:** every generated explanation must surface a Phase-4-derived caveat when the patient's ancestry group was underrepresented or poorly covered in training, not just cite the SHAP value — otherwise the explanation layer would be reproducing the same ancestry-blindness the review criticizes in the underlying literature.

- **Retrieval stack — pick this: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)** for embedding guideline/allele-table passages (a corpus of a few thousand short passages doesn't need a bigger embedding model) **+ [FAISS](https://github.com/facebookresearch/faiss)** for local vector search — still the standard, unglamorous choice at this scale; no hosted vector DB needed.
- **Generator model — revised recommendation, held on re-check.** `BioMistral/BioMistral-7B` and `epfl-llm/meditron-7b` still exist, but a citation-faithfulness study surfaced a specific risk that matters more here than raw medical knowledge: heavy medical fine-tuning can cause **"knowledge conflict"**, where the model overrides retrieved evidence with its own memorized medical priors instead of citing what was actually retrieved. A second search pass (mid-2026) for a medical model with a *published* citation-faithfulness eval — checking `google/medgemma-27b-text-it` (now the strongest open medical model on aggregate benchmarks, upgraded from the 4B checkpoint) and HuatuoGPT-o1 (beats OpenBioLLM on reasoning) — found **neither has a retrieval-faithfulness-specific evaluation published**, i.e. no medical-tuned model has direct evidence it's good at the thing Phase 9 actually needs. That absence of evidence reinforces rather than overturns the original call: **default to a general-purpose instruction model — Qwen2.5-7B-Instruct or 14B-Instruct — under a strict-citation prompt**, since the retrieval layer (PharmCAT/cpic-data + SHAP output) already supplies the medical facts and the model's only remaining job is faithful citation, which is an instruction-following property, not a medical-knowledge one. `google/medgemma-27b-text-it` (gated, HAI-DEF terms) is the benchmark-strongest fallback to A/B against.
- **Verify:** before picking a final generator, run the same held-out set of SHAP+retrieval inputs through both the general-instruct and medical-tuned candidates and score citation faithfulness (does every claim trace to an actually-retrieved passage, or does the model insert unretrieved medical claims) — decide empirically on this project's own corpus, not from general benchmarks.

## Explicitly out of scope for this phase

Item 10 of the review's own recommendations — a prospective, randomized, three-arm clinical trial — is the eventual validation this whole pipeline should earn its way toward, not something buildable now. Nothing in Phases 0–9 should be described as "clinically validated"; at most, "internally and cross-ancestry validated on retrospective data," per the review's own standard for what separates a research pipeline from a deployable tool.

---

**Order of execution:** Phases 0–2 are one-time setup. Phase 3 and Phase 4 should run before anything else modeling-related — Phase 4 in particular is cheap (data already on hand) and directly tests whether this project's own pipeline reproduces the review's central finding, which needs to be known before any later phase's results can be trusted. Phases 5–6 depend on Phase 3's winning model. Phase 7 is independent and can run in parallel. Phase 9 depends on Phase 6.

---

## Extended comparator sweep — 5 more experiments/models to check

These aren't new phases — they're additional rows for Phase 3's (and Phase 7's) results table, chosen because each is directly traceable to a specific algorithm class an included study in the review actually used (Table 3), so the paper can say "we replicated the field's own top approaches on a leakage-free, ancestry-stratified pipeline" rather than just comparing against generic ML. None of these duplicate what's already picked (CatBoost, TabPFN-3, AutoGluon, EBM, warfit-learn, d3rlpy).

1. **XGBoost** — matches Choi et al. 2023 and several other included studies' gradient-boosting comparator of choice. Distinct library/implementation from CatBoost; standard reviewer expectation for a tabular ML paper, cheap to add.
2. **Elastic Net / Lasso regression** — matches Abdel-latif et al. 2026 (LASSO) and the elastic-net component of Dryden et al. 2023's ensemble. Tests whether the genetic features survive regularization, i.e. whether genetics is a robust signal or an artifact of an unregularized model overfitting a few rare genotypes — a real, checkable question given the review's own finding that rare genotypes are represented by single-digit patient counts in several cohorts.
3. **Feedforward MLP** — matches Ma et al. 2021 and, more importantly, **Jahmunah et al. 2023**, the exact study whose IWPC-trained deep neural network showed the ancestry-error near-doubling (7.6 vs 14.2 mg/week) this project's Phase 4 is designed to test. Running an MLP specifically (not just CatBoost/TabPFN) makes Phase 4's ancestry hold-out a direct architecture-matched replication of Jahmunah's own finding, not just a generic re-test.
4. **Stacking ensemble** (meta-learner over a few base models) — matches Wang et al. 2024's "heuristic stacking ensemble," the study that reported this literature's best-looking number (MAE 0.77 mg/week) from a 64-patient test set. This is the sharpest experiment of the five: replicate their approach faithfully, then show what the same architecture does on a properly sized, leakage-checked, ancestry-stratified split. It's a direct empirical test of the review's own central claim (reported accuracy is inversely related to evaluation rigor) using the literature's own best-scoring method, not a strawman.
5. **LSTM over the ICU longitudinal INR trajectories** — matches Kuang et al. 2022, who raised dose-classification accuracy from 51.7% to 70.0% by incorporating time-series INR data. Runs on the same MIMIC-III/IV/eICU data as Phase 7, as a direct sequence-model comparator to Phase 7's contextual-bandit/d3rlpy approach — tests whether the review's finding that deep learning benefits most from temporal data replicates at this project's much smaller cohort sizes (17–160 vs. Kuang's 624).

**Verify (all five):** same Phase-0 splits, same Phase-1 baselines reported alongside, same Phase-4 ancestry stratification applied where the model is trained on the IWPC cohorts (items 1–4). Item 4 in particular should be reported with its 64-patient-equivalent internal test-set size *and* its performance on the full ancestry-stratified split side by side — that contrast is the whole point of including it.

---

## Second comparator sweep — 4 more experiments (a 5th was checked and rejected)

Deliberately different experiment *types* from both sweeps above, not more tabular-regression variants.

1. **CLMBR-T-base as a fixed embedding extractor for the ICU cohorts** — [`StanfordShahLab/clmbr-t-base`](https://huggingface.co/StanfordShahLab/clmbr-t-base) (141M params, gated, pretrained on 2.57M Stanford patients, MEDS/OMOP-CDM schema). A ready-made **[MIMIC-IV demo in MEDS format](https://www.physionet.org/content/mimic-iv-demo-meds/0.0.1/)** already exists on PhysioNet, so the schema-conversion step is already done for this project's cohort. **Use it only as a frozen embedding extractor, not a fine-tuning target** — at n=23 (MIMIC-IV) or n=17 (MIMIC-III), fine-tuning a 141M-parameter model would badly overfit; the model card itself warns it may not generalize beyond Stanford's own hospital, so treat any gain here as exploratory, not a claimed generalizable result. Addresses Phase 7's real small-n problem more directly than training an LSTM/RL model from raw features alone.
2. **HyperImpute / GAIN for the ICU cohorts' real missingness** — [`vanderschaarlab/hyperimpute`](https://github.com/vanderschaarlab/hyperimpute) bundles GAIN, MIWAE (VAE-based), MICE, and MissForest in one library, from the same lab that wrote the original GAIN paper. This is a **direct, faithful replication of Wani & Abeer 2025's own method** (an included study in the review, Table 3: "RF + DAE/GAIN/VAE/MICE/EM imputation" on MIMIC-III) — same data source, same technique, properly leakage-checked per Phase 2. Strongest of the four new additions because it's a like-for-like comparison against a study already in the review's corpus.
3. **Survival analysis for time-in-therapeutic-range (TTR)** — framing "time until INR first reaches therapeutic range" as a time-to-event problem with censoring (patients discharged before reaching range, which is real in this data) instead of only predicting a dose or INR point value. `scikit-survival` (Cox regression, Random Survival Forest) is the right starting point at this project's scale; `pycox`'s deep survival models (DeepSurv, DeepHit) are real and maintained but likely to overfit below the eICU cohort's n=160 — worth trying only on that slice, not MIMIC's 17–23. This is a genuinely novel angle nothing else in the plan covers, and TTR is the clinical outcome the review itself treats as more meaningful than a raw dose MAE.
4. **Deep ensemble + `uncertainty-toolbox`** as a second, mechanistically distinct uncertainty-quantification method alongside Phase 5's MAPIE (conformal, post-hoc) and NGBoost (natural-gradient distributional). A deep ensemble needs no new dependency — five MLPs with different seeds is sufficient — and [`uncertainty-toolbox`](https://github.com/uncertainty-toolbox/uncertainty-toolbox) (maintained, regression-focused) scores calibration/sharpness across all three methods on one common metric, so the paper can report which UQ approach actually wins rather than picking one by default.

**Checked and rejected — DDI-aware comedication embeddings.** IWPC data encodes comedications (amiodarone, statins, antibiotics, antifungals) as flat binary flags; a learned drug-drug-interaction embedding was checked as a possible upgrade. Direct HF search turned up nothing credible — every "DDI" hit was a near-zero-download hobbyist model or an unrelated diffusion-model name collision, and academic DDI-GNN approaches exist only as one-off research code, not reusable pretrained embeddings. Documenting this here so it isn't re-investigated later: **keep the flat binary comedication flags as-is**, this isn't a gap a pretrained model currently fills.

**Verify (items 1–3, 5):** same Phase-0 splits and Phase-1 baselines as everything else. Item 1's embeddings feed into whichever Phase-3 winner is chosen (as additional input features, not a replacement model). Item 3 reports its own metrics (concordance index, time-dependent AUC) since MAE/R²/PW20 don't apply to a survival target — don't force it into the same table as the dose-regression results, keep it as its own clearly-labeled result.

---

## Publication strategy (target: IEEE Access)

**Core submission — Phases 3–6 bundled**, evaluated together on the same pipeline: nested clinical/genetic ablation, ancestry-stratified/leave-one-ancestry-out validation, calibration, and explainability. This is the strongest publishable unit on the idea alone (execution/results still pending) because it directly answers the exact gap the 32-study systematic review identified, uses real public data so it's independently checkable, and matches IEEE Access's actual selection criteria — broad interdisciplinary relevance (health AI + fairness + genomics), thorough validation, practical significance — over pure theoretical novelty, which the venue doesn't select for.

**Estimated acceptance likelihood, idea-only: 7/10.**
- *Caps it below 8–9:* (1) it's a rigorous follow-up validation study to the review itself, not a new method, so a reviewer could read it as incremental; (2) it cannot actually close the ancestry gap it studies — CYP2C9\*5/\*6/\*8/\*11, rs12777823, and CYP4F2 are absent from both IWPC extracts — so the likely headline finding is "the gap persists, now quantified more rigorously," a more modest claim than "we solved it"; (3) some ancestry subgroups are small enough (per the review's own Table 4 pattern) that a reviewer may reasonably question statistical power.
- *Keeps it from falling lower:* reproducible pipeline on real data, directly extends a documented gap across 32 studies, and covers calibration + explainability + fairness together — a combination almost absent from the review's own corpus.

**Phase 9 (LLM explanation layer) — bonus, not load-bearing.** Include it only if it ships with an actual citation-faithfulness metric (does every generated sentence trace to an actually-retrieved PharmCAT/cpic-data passage or SHAP value) rather than a qualitative demo — reviewers in 2026 will expect that bar for any "LLM explains X" claim. Done well, this moves the estimate to roughly 7.5–8/10.

**Phase 7 (RL on longitudinal ICU cohorts) — exploratory/future work, not a core result.** The largest available cohort is 160 eICU stays, the smallest is 17 MIMIC-III patients — a reviewer will reasonably question whether that supports an RL claim at all. Leading the paper with this would be the one choice most likely to sink an otherwise solid submission; keep it clearly scoped as future work if mentioned.

**Open structural decision:** one comprehensive paper (Phases 3–6, optionally +9) vs. two papers (a focused ancestry+calibration+explainability paper first, a separate LLM-explanation paper second). IEEE Access tends to reward a focused, thorough single contribution over one paper claiming five things at once — worth deciding before writing begins, not after.
