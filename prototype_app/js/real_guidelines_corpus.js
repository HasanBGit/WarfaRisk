/**
 * WarfaRisk RAG Corpus — grounded in real dataset-derived content: PharmGKB
 * allele definition tables, gnomAD population allele frequencies, CPIC/IWPC
 * guideline formulas, and this project's own aggregate SHAP/calibration
 * findings computed directly on the IWPC cohorts. Per the project's own
 * data-handling rule (see PRSgene/data/DATA.md), no per-patient free-text
 * or row-level value from the IWPC dataset is embedded here — IWPC data is
 * not redistributable, and every passage below is either a public reference
 * table (PharmGKB/gnomAD), a published guideline formula, or an aggregate
 * statistic already computed and documented in this project's own results
 * (never an individual patient's record).
 */
window.REAL_GUIDELINES_CORPUS = [
  // ---- PharmGKB allele tables (public reference/lookup tables, not patient rows) ----
  {
    "id": "pharmgkb-cyp2c9-alleles",
    "source": "PharmGKB Official CYP2C9 Allele Definition Table",
    "title": "CYP2C9 Pharmacogenomic Allele Variants & rsIDs",
    "text": "PharmGKB CYP2C9 allele definition table indexes 78 verified single nucleotide polymorphisms (SNPs) including rsIDs: rs114071557, rs67807361, rs142240658, rs1364419386, rs2031308986, rs564813580, rs371055887, rs1216169538, rs72558187, rs2493006942, rs761033063, rs762239445, rs771237265, rs1304490498, rs774607211... Known to modulate warfarin enzymatic activity and maintenance dosing.",
    "category": "genetics",
    "gene": "CYP2C9"
  },
  {
    "id": "pharmgkb-cyp4f2-alleles",
    "source": "PharmGKB Official CYP4F2 Allele Definition Table",
    "title": "CYP4F2 Pharmacogenomic Allele Variants & rsIDs",
    "text": "PharmGKB CYP4F2 allele definition table indexes 20 verified single nucleotide polymorphisms (SNPs) including rsIDs: rs3093200, rs3952537, rs4020346, rs138971789, rs142113670, rs2108622, rs145174239, rs144233412, rs3093153, rs200629062, rs556151888, rs145875499, rs148396222, rs114396708, rs150579280... Known to modulate warfarin enzymatic activity and maintenance dosing.",
    "category": "genetics",
    "gene": "CYP4F2"
  },
  {
    "id": "pharmgkb-vkorc1-alleles",
    "source": "PharmGKB Official VKORC1 Allele Definition Table",
    "title": "VKORC1 Pharmacogenomic Allele Variants & rsIDs",
    "text": "PharmGKB VKORC1 allele definition table indexes 1 verified single nucleotide polymorphisms (SNPs) including rsIDs: rs9923231... Known to modulate warfarin enzymatic activity and maintenance dosing.",
    "category": "genetics",
    "gene": "VKORC1"
  },
  {
    "id": "cpic-2017-primary-guideline",
    "source": "CPIC Guidelines for Pharmacogenetics-Guided Warfarin Dosing (2017 Update)",
    "title": "CPIC 2017 Dosing Algorithm & Clinical Recommendations",
    "text": "Clinical Pharmacogenetics Implementation Consortium (CPIC) 2017 update for warfarin dosing recommends integrating CYP2C9 (*2, *3, *5, *6, *8, *11), VKORC1 (-1639G>A / rs9923231), and CYP4F2 (rs2108622 / V433M) genotype data to calculate starting dose. VKORC1 A allele carriers exhibit 30-50% dose sensitivity. CYP2C9 poor metabolizers require up to 80% dose reduction.",
    "category": "guidelines",
    "gene": "CYP2C9/VKORC1/CYP4F2"
  },
  {
    "id": "iwpc-2009-consensus",
    "source": "International Warfarin Pharmacogenetics Consortium (IWPC, NEJM 2009)",
    "title": "IWPC 2009 Pharmacogenetic Dosing Algorithm",
    "text": "The IWPC baseline formula predicts sqrt(weekly dose) based on age, height, weight, VKORC1 G>A genotype, CYP2C9 star alleles (*1/*1, *1/*2, *1/*3, *2/*2, *2/*3, *3/*3), amiodarone use, statin use, enzyme inducer therapy, and Asian/Black ancestry flags. MAE in test cohort: 9.18 mg/week.",
    "category": "clinical",
    "gene": "Clinical+Genetics"
  },

  // ---- gnomAD population allele frequencies (public reference data, 1000 Genomes) ----
  {
    "id": "gnomad-vkorc1-population-freq",
    "source": "gnomAD population frequencies (1000 Genomes dataset, this project's own data pipeline)",
    "title": "Why VKORC1 -1639G>A Sensitivity Differs by Ancestry: Real Allele Frequencies",
    "text": "The warfarin-sensitizing VKORC1 -1639G>A allele (rs9923231) is carried by an estimated 89.2% of East Asian, 37.8% of non-Finnish European, 39.1% of Finnish, 39.1% of Admixed American, 17.8% of South Asian, and only 10.0% of African gnomAD samples. This population gradient is the direct mechanistic reason lower average doses are required in East Asian patients and higher average doses in African-ancestry patients: the allele itself, not just the modeling approach, differs in prevalence by ancestry.",
    "category": "genetics",
    "gene": "VKORC1"
  },
  {
    "id": "gnomad-cyp2c9-star2-population-freq",
    "source": "gnomAD population frequencies (1000 Genomes dataset, this project's own data pipeline)",
    "title": "Why CYP2C9*2 Dose Reduction Is Population-Specific: Real Allele Frequencies",
    "text": "The CYP2C9*2 loss-of-function allele (rs1799853, R144C) is carried by an estimated 12.7% of non-Finnish European, 9.8% of Admixed American, 3.9% of South Asian, 2.4% of African, and only 0.04% of East Asian gnomAD samples, essentially absent in that population. A CYP2C9*2-based dose reduction therefore applies almost exclusively to European-descent and, to a lesser extent, admixed American patients; it is not a meaningful factor for most East Asian patients.",
    "category": "genetics",
    "gene": "CYP2C9"
  },
  {
    "id": "gnomad-cyp2c9-star3-population-freq",
    "source": "gnomAD population frequencies (1000 Genomes dataset, this project's own data pipeline)",
    "title": "Why CYP2C9*3 Dose Reduction Is Population-Specific: Real Allele Frequencies",
    "text": "The CYP2C9*3 loss-of-function allele (rs1057910, I359L) is carried by an estimated 11.4% of South Asian, 6.6% of non-Finnish European, 5.6% of Finnish, 4.9% of Admixed American, 3.1% of East Asian, and essentially 0% of African gnomAD samples. Unlike CYP2C9*2, this variant is most common in South Asian populations, so its clearance-reducing effect is most clinically relevant there, not in Europeans, where CPIC recommendations are most often validated.",
    "category": "genetics",
    "gene": "CYP2C9"
  },
  {
    "id": "gnomad-cyp4f2-population-freq",
    "source": "gnomAD population frequencies (1000 Genomes dataset, this project's own data pipeline)",
    "title": "Why CYP4F2 V433M Dose Increase Is Population-Specific: Real Allele Frequencies",
    "text": "The CYP4F2*3 variant (rs2108622, V433M), which reduces vitamin K1 metabolism and requires a higher warfarin dose, is carried by an estimated 40.1% of South Asian, 28.6% of non-Finnish European, 25.8% of Admixed American, 24.6% of East Asian, 19.9% of Finnish, and only 9.9% of African gnomAD samples. This is the opposite direction of the VKORC1 gradient above (CYP4F2 raises dose requirements where it is common), which is part of why single-gene explanations of ancestry-stratified dosing are unreliable: the genes pull in different directions with different population gradients, and only the combined genotype captures the net effect.",
    "category": "genetics",
    "gene": "CYP4F2"
  },

  // ---- This project's own aggregate findings, computed directly on the IWPC cohorts.
  // Aggregate/summary statistics only -- never a per-patient row, per DATA.md. ----
  {
    "id": "warfarisk-shap-feature-ranking",
    "source": "WarfaRisk Phase 6 SHAP Explainability (this project, computed on IWPC-6256)",
    "title": "Which Real Dataset Fields Actually Drive the Dose Prediction",
    "text": "Ranking mean absolute SHAP contribution across all 35 modeled IWPC-6256 fields: VKORC1 -1639 consensus ranks first (3.257), followed by Age (3.143), CYP2C9 consensus (2.863), Weight in kg (2.709), and three rarer VKORC1 SNP columns (1173/1542/2255 consensus, combined ~5.1). Indication for Warfarin Treatment, Height, Amiodarone use, Valve Replacement, and current smoking status rank lower but still contribute. Summing all genetic-column contributions against the full 35-field total, genetics accounts for approximately 54% of total SHAP importance on this cohort.",
    "category": "clinical",
    "gene": "Model"
  },
  {
    "id": "african-ancestry-genotype-gap",
    "source": "WarfaRisk System Paper -- Ancestry-Stratified Evaluation (this project)",
    "title": "African Ancestry Decreased-Function Alleles (CYP2C9*5, *6, *8, *11 & rs12777823)",
    "text": "Standard clinical panels assessing only CYP2C9*2 and *3 miss African-specific variants (CYP2C9*5, *6, *8, *11 and rs12777823). This is a documented gap in the IWPC genotyping panel used to train most published warfarin dosing models, including WarfaRisk itself. CPIC guidelines recommend specialized panels for African ancestry patients.",
    "category": "fairness",
    "gene": "CYP2C9/VKORC1"
  },
  {
    "id": "han-chinese-portability-penalty",
    "source": "WarfaRisk System Paper -- Phase 4 Ancestry Portability Validation (this project)",
    "title": "Han Chinese Leave-One-Group-Out Transferability Penalty",
    "text": "Phase 4 out-of-distribution evaluation on IWPC-6256 demonstrates a +2.13 mg/week MAE portability penalty (CatBoost; +1.56 mg/week with a stacking ensemble) for Han Chinese ancestry when models are trained without adequate Asian sub-population representation, highlighting the importance of population-specific calibration.",
    "category": "fairness",
    "gene": "Model"
  },
];

// Note: a MAPIE calibration passage (id "mapie-conformal-calibration") is
// already defined as a core passage in vector_db.js's initCorpus() -- not
// duplicated here to avoid a colliding id between the two arrays.
