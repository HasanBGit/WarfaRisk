/**
 * WarfaRisk Enterprise Application Controller
 * Handles SPA navigation, form state, trained model presets, RAG execution & settings
 */

class WarfaRiskApp {
    constructor() {
        this.currentScreen = 1;
        this.activePrediction = null;
        
        // Verified presets matching real trained dataset cohorts
        this.presets = {
            han_chinese: {
                age: 68, weight: 72, height: 168, ancestry: "Han Chinese",
                indication: "Atrial Fibrillation", targetINR: "2.5",
                vkorc1: "AG", cyp2c9: "*1/*1",
                amiodarone: false, statin: false, enzymeInducer: false
            },
            african_american: {
                age: 54, weight: 88, height: 175, ancestry: "Black or African American",
                indication: "Deep Vein Thrombosis / PE", targetINR: "2.5",
                vkorc1: "GG", cyp2c9: "*1/*3",
                amiodarone: false, statin: true, enzymeInducer: false
            },
            caucasian: {
                age: 72, weight: 80, height: 178, ancestry: "Caucasian",
                indication: "Mechanical Heart Valve", targetINR: "3.0",
                vkorc1: "AA", cyp2c9: "*2/*2",
                amiodarone: true, statin: true, enzymeInducer: false
            },
            clinical_only: {
                age: 62, weight: 75, height: 170, ancestry: "Other / Unspecified",
                indication: "Atrial Fibrillation", targetINR: "2.5",
                vkorc1: "unknown", cyp2c9: "unknown",
                amiodarone: false, statin: false, enzymeInducer: false
            }
        };
    }

    async init() {
        this.loadSettingsToModal();
        this.updatePatientSummary();
        await this.runPrediction();
    }

    getPatientFromForm() {
        return {
            age: parseFloat(document.getElementById("input-age").value) || 60,
            weight: parseFloat(document.getElementById("input-weight").value) || 70,
            height: parseFloat(document.getElementById("input-height").value) || 170,
            ancestry: document.getElementById("input-ancestry").value,
            vkorc1: document.getElementById("input-vkorc1").value,
            cyp2c9: document.getElementById("input-cyp2c9").value,
            amiodarone: document.getElementById("check-amiodarone").checked,
            statin: document.getElementById("check-statin").checked,
            enzymeInducer: document.getElementById("check-inducer").checked
        };
    }

    setFormFromPatient(patient) {
        document.getElementById("input-age").value = patient.age;
        document.getElementById("input-weight").value = patient.weight;
        document.getElementById("input-height").value = patient.height;
        document.getElementById("input-ancestry").value = patient.ancestry;
        document.getElementById("input-vkorc1").value = patient.vkorc1;
        document.getElementById("input-cyp2c9").value = patient.cyp2c9;
        document.getElementById("check-amiodarone").checked = !!patient.amiodarone;
        document.getElementById("check-statin").checked = !!patient.statin;
        document.getElementById("check-inducer").checked = !!patient.enzymeInducer;

        this.updatePatientSummary();
    }

    async loadPreset(key) {
        if (this.presets[key]) {
            this.setFormFromPatient(this.presets[key]);
            await this.runPrediction();
        }
    }

    updatePatientSummary() {
        const p = this.getPatientFromForm();
        const vkorcText = p.vkorc1 !== "unknown" ? p.vkorc1 : "Not Genotyped";
        const cypText = p.cyp2c9 !== "unknown" ? p.cyp2c9 : "Not Genotyped";

        const desc = `MRN-884920 • ${p.age}y Male, ${p.weight}kg, ${p.height}cm, ${p.ancestry} | VKORC1: ${vkorcText}, CYP2C9: ${cypText}`;
        document.getElementById("summary-patient-desc").textContent = desc;

        const hasGenetics = p.vkorc1 !== "unknown" && p.cyp2c9 !== "unknown";
        document.getElementById("summary-model-type").textContent = hasGenetics 
            ? "Model: Combined Pharmacogenomic (CatBoost/Stacking)" 
            : "Model: Clinical-Only Fallback";
    }

    switchScreen(num) {
        this.currentScreen = num;

        document.querySelectorAll(".nav-tab-btn").forEach((btn, idx) => {
            btn.classList.toggle("active", idx + 1 === num);
        });

        document.querySelectorAll(".screen-pane").forEach((pane, idx) => {
            pane.classList.toggle("active", idx + 1 === num);
        });

        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    async runPrediction() {
        const patient = this.getPatientFromForm();
        const result = await window.dosingEngine.predictDose(patient);
        this.activePrediction = result;

        this.renderScreen2(result);
        this.renderScreen3(result);
    }

    async runPredictionAndNavigate() {
        await this.runPrediction();
        this.switchScreen(2);
    }

    renderScreen2(result) {
        document.getElementById("res-weekly-dose").textContent = result.predictedWeeklyDose.toFixed(1);
        document.getElementById("res-daily-dose").textContent = `Equivalent to ${result.dailyDose.toFixed(2)} mg / day`;
        document.getElementById("res-iwpc-comp").textContent = `Baseline Published IWPC Formula (2009): ${result.iwpcWeeklyDose.toFixed(1)} mg/week`;

        const [low, high] = result.interval;
        document.getElementById("res-interval-text").textContent = `[${low.toFixed(1)} — ${high.toFixed(1)} mg/week]`;

        const minVal = 5.0, maxVal = 95.0;
        const leftPct = Math.max(0, Math.min(100, ((low - minVal) / (maxVal - minVal)) * 100));
        const widthPct = Math.max(5, Math.min(100 - leftPct, ((high - low) / (maxVal - minVal)) * 100));
        const markerPct = Math.max(0, Math.min(100, ((result.predictedWeeklyDose - minVal) / (maxVal - minVal)) * 100));

        document.getElementById("interval-fill-bar").style.left = `${leftPct}%`;
        document.getElementById("interval-fill-bar").style.width = `${widthPct}%`;
        document.getElementById("interval-marker-dot").style.left = `${markerPct}%`;

        const info = result.ancestryInfo;
        const calBadge = document.getElementById("badge-calibration-status");
        calBadge.textContent = `${info.coverage}% Coverage (${info.status.toUpperCase()})`;
        calBadge.className = `ui-badge ${info.status === 'green' ? 'emerald' : (info.status === 'red' ? 'red' : 'amber')}`;

        const cardNotice = document.getElementById("card-calibration-notice");
        cardNotice.className = `clinical-alert-box ${info.status}`;

        document.getElementById("text-calibration-detail").textContent = info.warning || 
            `Subgroup empirical coverage is ${info.coverage}%, meeting target 90% conformal calibration standards.`;

        const portNotice = document.getElementById("card-portability-notice");
        if (info.penalty) {
            portNotice.style.display = "flex";
            document.getElementById("text-portability-detail").textContent = 
                `Phase 4 validation flags a +2.13 mg/week MAE out-of-distribution transferability penalty (CatBoost) for ${this.getPatientFromForm().ancestry} ancestry.`;
        } else {
            portNotice.style.display = "none";
        }
    }

    renderScreen3(result) {
        const listEl = document.getElementById("shap-bar-list");
        listEl.innerHTML = "";

        const maxImpact = Math.max(...result.shapAttributions.map(s => Math.abs(s.impact)), 1.0);

        result.shapAttributions.forEach(item => {
            const isPos = item.impact > 0;
            const pct = Math.min(100, (Math.abs(item.impact) / maxImpact) * 100);

            const row = document.createElement("div");
            row.className = "shap-item-row";
            row.innerHTML = `
                <div class="shap-item-label" title="${item.feature}">${item.feature}</div>
                <div class="shap-bar-track">
                    <div class="shap-bar-fill ${isPos ? 'pos' : 'neg'}" style="width: ${pct}%;"></div>
                </div>
                <div class="shap-item-val ${isPos ? 'pos' : 'neg'}">
                    ${isPos ? '+' : ''}${item.impact.toFixed(1)} mg
                </div>
            `;
            listEl.appendChild(row);
        });

        this.generateRAGExplanation();
    }

    /**
     * SHAP-conditioned retrieval query construction.
     *
     * Rather than retrieving passages from a fixed "ancestry + genotype"
     * template, the query is built from this specific patient's own
     * top-ranked SHAP feature-attribution names — whatever the trained
     * model actually says is driving *this* prediction, not a generic
     * lookup. Retrieval therefore adapts per patient: a patient whose
     * dose is driven mostly by clinical factors (age/weight/amiodarone)
     * pulls different evidence than one whose dose is driven by VKORC1/
     * CYP2C9 genotype. When the patient's ancestry subgroup has measured
     * sub-nominal MAPIE coverage, the calibration status text is appended
     * to the query too, so the retrieval step surfaces the population
     * allele-frequency evidence that mechanistically explains the gap
     * (Section on gnomAD passages in js/real_guidelines_corpus.js),
     * rather than only a bare warning number with nothing behind it.
     */
    buildShapConditionedQuery(prediction) {
        const topFeatures = (prediction.shapAttributions || [])
            .slice(0, 3)
            .map(s => s.feature)
            .join(" ");

        const calibrationContext = (prediction.ancestryInfo && prediction.ancestryInfo.status !== "green")
            ? `${prediction.ancestryInfo.statusText || ""} calibration coverage`
            : "";

        return `warfarin dose explanation ${topFeatures} ${calibrationContext}`.trim();
    }

    async generateRAGExplanation() {
        if (!this.activePrediction) return;

        const outputEl = document.getElementById("rag-output-text");
        outputEl.innerHTML = `<span style="color: var(--text-muted); font-style: italic;">Retrieving guideline passages & invoking external API (${window.apiService.modelName})...</span>`;

        const query = this.buildShapConditionedQuery(this.activePrediction);
        const ragRes = await window.apiService.generateExplanation(this.activePrediction, query);

        let formattedText = ragRes.text.replace(/\[([^\]]+)\]/g, '<span class="citation-chip">[$1]</span>');
        outputEl.innerHTML = formattedText;

        document.getElementById("rag-provider-badge").textContent = ragRes.provider;

        const auditEl = document.getElementById("rag-citation-count");
        const audit = ragRes.verification;
        auditEl.textContent = `${audit.verifiedCount}/${audit.totalCitations} Verified (${audit.isFullyGrounded ? 'Fully Grounded' : 'Unverified Citations Present'})`;
        auditEl.style.color = audit.isFullyGrounded ? "#1B8A4B" : "#93690F";
    }

    regenerateRAG() {
        this.generateRAGExplanation();
    }

    openSettingsModal() {
        document.getElementById("modal-settings").classList.add("active");
    }

    closeSettingsModal() {
        document.getElementById("modal-settings").classList.remove("active");
    }

    loadSettingsToModal() {
        document.getElementById("setting-provider").value = window.apiService.apiProvider;
        document.getElementById("setting-key").value = window.apiService.apiKey;
        document.getElementById("setting-model").value = window.apiService.modelName;
        document.getElementById("setting-endpoint").value = window.apiService.customEndpoint;
    }

    onProviderChange() {
        const provider = document.getElementById("setting-provider").value;
        const modelInput = document.getElementById("setting-model");
        if (provider === "openrouter") {
            modelInput.value = "google/gemini-2.5-flash";
        } else if (provider === "gemini") {
            modelInput.value = "gemini-2.5-flash";
        } else {
            modelInput.value = "gpt-4o-mini";
        }
    }

    saveSettings() {
        const provider = document.getElementById("setting-provider").value;
        const key = document.getElementById("setting-key").value;
        const model = document.getElementById("setting-model").value;
        const endpoint = document.getElementById("setting-endpoint").value;

        window.apiService.saveSettings(provider, key, model, endpoint);
        this.closeSettingsModal();
        this.generateRAGExplanation();
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.app = new WarfaRiskApp();
    window.app.init();
});
