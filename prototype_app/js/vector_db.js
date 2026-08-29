/**
 * WarfaRisk Lightweight Vector DB (In-Memory TF-IDF + Cosine Similarity)
 * Grounded in PharmGKB Excel Allele Tables, CPIC 2017 Guidelines, IWPC
 * Literature, and real extracted findings from the warfarin ML/dosing
 * systematic and narrative reviews this project cites (Fulop et al. 2025,
 * Asiimwe et al. 2021, Fahmi et al. 2022 -- see js/real_guidelines_corpus.js)
 */

class LightweightVectorDB {
    constructor() {
        this.passages = [];
        this.vocabulary = new Set();
        this.idf = {};
        this.vectors = [];
    }

    /**
     * Initialize corpus with real PharmGKB + CPIC 2017 guideline data
     */
    initCorpus() {
        const corePassages = [
            {
                id: "cpic-vkorc1-1639",
                source: "CPIC 2017 Warfarin Guideline (rs9923231)",
                title: "VKORC1 -1639G>A Dosing Recommendation",
                text: "The VKORC1 -1639G>A polymorphism (rs9923231) significantly increases warfarin sensitivity. Patients carrying the A allele (AG or AA genotype) produce less VKORC1 subunit 1 protein, requiring lower maintenance doses (typically 20-50% reduction for AA homozygous variant). CPIC 2017 algorithms recommend genotype-guided dose reduction for AG and AA carriers.",
                category: "genetics",
                gene: "VKORC1"
            },
            {
                id: "cpic-cyp2c9-metabolizer",
                source: "CPIC 2017 Warfarin Guideline (CYP2C9)",
                title: "CYP2C9 Star Alleles & Metabolizer Status",
                text: "CYP2C9*2 (rs1799853) and CYP2C9*3 (rs1057910) reduce S-warfarin clearance by approximately 30-40% and 80-90% per allele copy, respectively. Intermediate metabolizers (*1/*2, *1/*3) and poor metabolizers (*2/*2, *2/*3, *3/*3) exhibit prolonged S-warfarin half-life, markedly increasing bleeding risk. Dosing algorithms reduce predicted weekly dose accordingly.",
                category: "genetics",
                gene: "CYP2C9"
            },
            {
                id: "cpic-african-ancestry-caveat",
                source: "CPIC 2017 Guideline - African Ancestry Special Considerations",
                title: "African Ancestry Genotyping Coverage Deficit",
                text: "Standard clinical pharmacogenetic panels evaluating only CYP2C9*2 and *3 fail to capture African-specific decreased-function variants, including CYP2C9*5, *6, *8, *11, and rs9332131/rs7900194 in CYP2C9/VKORC1 region. In patients of African descent, standard IWPC dosing algorithms may overestimate dose or suffer reduced prediction interval coverage (measured subgroup coverage 68.4% vs 90% target).",
                category: "fairness",
                gene: "CYP2C9/VKORC1"
            },
            {
                id: "iwpc-clinical-formula",
                source: "IWPC Pharmacogenetic Dosing Consortium (2009)",
                title: "IWPC Baseline Dose Equation Factors",
                text: "The published IWPC clinical equation estimates square root of weekly dose based on: Age (decades), Height (cm), Weight (kg), CYP2C9 genotype status, VKORC1 genotype status, Amiodarone use (reduces dose by ~30%), Statins, Enzyme Inducers (rifampin, carbamazepine, phenytoin increase clearance requiring higher dose), and Target INR.",
                category: "clinical",
                gene: "Clinical"
            },
            {
                id: "mapie-conformal-calibration",
                source: "WarfaRisk Phase 5 Calibration Analysis",
                title: "MAPIE Split Conformal Uncertainty Quantification",
                text: "MAPIE split conformal prediction provides non-parametric 90% prediction intervals [Lower, Upper mg/week]. While aggregate population coverage achieves nominal 89.8% bounds, per-ancestry subgroup analysis surfaces significant calibration gaps: African American subgroups exhibit empirical coverage of 68.4% (amber/red alert state), requiring clinical vigilance.",
                category: "uncertainty",
                gene: "Model"
            },
            {
                id: "ancestry-portability-penalty",
                source: "WarfaRisk Phase 4 Portability Validation",
                title: "Ancestry Leave-One-Group-Out Transferability",
                text: "Evaluating model transferability across ancestry groups reveals distinct out-of-distribution penalties. Specifically, Han Chinese patients demonstrate a statistically significant portability error jump (+2.13 mg/week MAE penalty with CatBoost, +1.56 mg/week with a stacking ensemble) when trained on multi-ethnic cohorts lacking specific Asian sub-population representations.",
                category: "fairness",
                gene: "Model"
            }
        ];

        const realPassages = (window.REAL_GUIDELINES_CORPUS && Array.isArray(window.REAL_GUIDELINES_CORPUS)) 
            ? window.REAL_GUIDELINES_CORPUS 
            : [];

        this.passages = [...realPassages, ...corePassages];
        this.buildIndex();
    }

    /**
     * Tokenize text into normalized lowercase words
     */
    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^a-z0-9\*\-]/g, ' ')
            .split(/\s+/)
            .filter(t => t.length > 1);
    }

    /**
     * Build TF-IDF index over corpus
     */
    buildIndex() {
        this.vocabulary.clear();
        this.passages.forEach(p => {
            const tokens = this.tokenize(p.title + " " + p.text);
            p.tokens = tokens;
            p.termFreq = {};
            tokens.forEach(t => {
                this.vocabulary.add(t);
                p.termFreq[t] = (p.termFreq[t] || 0) + 1;
            });
        });

        const N = this.passages.length;
        this.vocabulary.forEach(term => {
            const docCount = this.passages.filter(p => p.termFreq[term] > 0).length;
            this.idf[term] = Math.log((N + 1) / (docCount + 1)) + 1;
        });

        this.vectors = this.passages.map(p => this.computeVector(p.termFreq, p.tokens.length));
    }

    /**
     * Compute TF-IDF vector for document or query
     */
    computeVector(termFreq, totalTokens) {
        const vec = {};
        for (const term in termFreq) {
            if (this.idf[term]) {
                const tf = termFreq[term] / Math.max(1, totalTokens);
                vec[term] = tf * this.idf[term];
            }
        }
        return vec;
    }

    /**
     * Calculate Cosine Similarity between two term vectors
     */
    cosineSimilarity(vecA, vecB) {
        let dot = 0;
        let normA = 0;
        let normB = 0;

        for (const term in vecA) {
            const valA = vecA[term];
            normA += valA * valA;
            if (vecB[term]) {
                dot += valA * vecB[term];
            }
        }

        for (const term in vecB) {
            normB += vecB[term] * vecB[term];
        }

        if (normA === 0 || normB === 0) return 0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Query vector DB for top-K matching passages
     */
    search(queryText, topK = 4) {
        const tokens = this.tokenize(queryText);
        const termFreq = {};
        tokens.forEach(t => termFreq[t] = (termFreq[t] || 0) + 1);

        const queryVec = this.computeVector(termFreq, tokens.length);

        const results = this.passages.map((p, idx) => {
            const sim = this.cosineSimilarity(queryVec, this.vectors[idx]);
            return { passage: p, score: sim };
        });

        results.sort((a, b) => b.score - a.score);
        return results.slice(0, topK).map(r => r.passage);
    }
}

// Global instance export
window.vectorDB = new LightweightVectorDB();
window.vectorDB.initCorpus();
