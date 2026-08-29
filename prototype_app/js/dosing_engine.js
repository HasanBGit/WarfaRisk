/**
 * WarfaRisk Dosing Engine
 * Implements:
 * 1. IWPC Pharmacogenetic Equation (Baseline)
 * 2. Real CatBoost model inference (Phase 3 combined feature set, iwpc_6256)
 *    via the local inference server (server/app_server.py) — falls back to
 *    a labeled local heuristic only if that server is unreachable.
 * 3. 90th-percentile-residual interval, computed from the real model's own
 *    held-out test set (see server/artifacts/*.json)
 * 4. Subgroup Ancestry Calibration & Portability Warning Engine (grounded in
 *    Phase 5 MAPIE coverage results, see RESULTS.md)
 * 5. Real per-feature marginal impact (leave-one-out against the actual
 *    trained model), or a labeled local approximation on fallback
 */

const INFERENCE_API_URL = "http://127.0.0.1:8787/predict";

class WarfaRiskEngine {
    constructor() {
        // Ancestry subgroup calibration coverage database (From Trained MAPIE Results: phase5_iwpc6256_mapie_coverage.csv)
        this.ancestryCalibrationMap = {
            "Caucasian": {
                coverage: 86.3,
                status: "amber",
                statusText: "Sub-nominal Coverage (86.3%)",
                penalty: false,
                warning: "Prediction interval reliability for this ancestry group is below the intended 90%. Dose selection for this patient should incorporate additional clinical judgment."
            },
            "Black or African American": { 
                coverage: 69.2, 
                status: "red", 
                statusText: "Subgroup Calibration Deficit (69.2% < 90% Target)", 
                penalty: false,
                warning: "Prediction interval reliability for this ancestry group is below the intended 90%. Dose selection for this patient should incorporate additional clinical judgment."
            },
            "African American": { 
                coverage: 60.0, 
                status: "red", 
                statusText: "Subgroup Calibration Deficit (60.0% < 90% Target)", 
                penalty: false,
                warning: "Prediction interval reliability for this ancestry group is below the intended 90%. Dose selection for this patient should incorporate additional clinical judgment."
            },
            "Han Chinese": {
                coverage: 100.0,
                status: "green",
                statusText: "Nominal Coverage (100.0%)",
                penalty: true,
                warning: "Calibration coverage is nominal for this ancestry group, but Phase 4 Leave-One-Group-Out Transferability reveals a separate +2.13 mg/week MAE portability penalty (CatBoost) for Han Chinese ancestry, so dose selection should still incorporate additional clinical judgment."
            },
            "Japanese": { coverage: 95.9, status: "green", statusText: "Nominal Coverage (95.9%)", penalty: false },
            "Hispanic": { coverage: 100.0, status: "green", statusText: "Nominal Coverage (100.0%)", penalty: false },
            "Other / Unspecified": { coverage: 89.8, status: "green", statusText: "Nominal Aggregate Coverage (89.8%)", penalty: false }
        };
    }

    /**
     * Validate patient inputs against physiological convex hull
     */
    validateInputs(patient) {
        const errors = [];
        const warnings = [];

        if (!patient.age || patient.age < 18 || patient.age > 100) {
            errors.push("Age must be between 18 and 100 years.");
        }
        if (!patient.weight || patient.weight < 30 || patient.weight > 200) {
            errors.push("Weight must be between 30 kg and 200 kg.");
        }
        if (!patient.height || patient.height < 100 || patient.height > 220) {
            errors.push("Height must be between 100 cm and 220 cm.");
        }

        const hasGenetics = patient.vkorc1 && patient.cyp2c9 && patient.vkorc1 !== "unknown" && patient.cyp2c9 !== "unknown";
        if (!hasGenetics) {
            warnings.push("Pharmacogenomic data (VKORC1 / CYP2C9) missing. System falling back to Clinical-Only model (MAE ~10.86 mg/week).");
        }

        return { isValid: errors.length === 0, errors, warnings, hasGenetics };
    }

    /**
     * Compute Published IWPC Baseline Equation (2009)
     */
    computeIWPC(patient) {
        const ageDecades = (patient.age || 60) / 10.0;
        const height = patient.height || 170;
        const weight = patient.weight || 70;

        let sqrtDose = 5.6413 - (0.0087 * patient.age) + (0.0128 * height) - (0.0028 * weight);

        // VKORC1 rs9923231 (-1639 G>A)
        if (patient.vkorc1 === "AG") sqrtDose -= 0.8674;
        else if (patient.vkorc1 === "AA") sqrtDose -= 1.6974;

        // CYP2C9
        if (patient.cyp2c9 === "*1/*2") sqrtDose -= 0.5211;
        else if (patient.cyp2c9 === "*1/*3") sqrtDose -= 0.9357;
        else if (patient.cyp2c9 === "*2/*2") sqrtDose -= 1.0616;
        else if (patient.cyp2c9 === "*2/*3") sqrtDose -= 1.9206;
        else if (patient.cyp2c9 === "*3/*3") sqrtDose -= 2.3312;

        // Comedications
        if (patient.amiodarone) sqrtDose -= 0.5503;
        if (patient.enzymeInducer) sqrtDose += 0.0129;

        // Race (IWPC formulation)
        if (patient.ancestry === "Han Chinese" || patient.ancestry === "Japanese") sqrtDose += 0.216;
        else if (patient.ancestry.includes("Black")) sqrtDose += 0.406;

        const weeklyDose = Math.max(7.0, Math.min(100.0, Math.pow(sqrtDose, 2)));
        return Math.round(weeklyDose * 10) / 10;
    }

    /**
     * Compute Primary Prediction via the real trained CatBoost model,
     * served locally by server/app_server.py. Throws if the backend is
     * unreachable so the caller can fall back explicitly and label the
     * result as a fallback rather than silently passing off a heuristic as
     * model output.
     */
    async predictDose(patient) {
        const validation = this.validateInputs(patient);
        const iwpcDose = this.computeIWPC(patient);
        const ancestryInfo = this.ancestryCalibrationMap[patient.ancestry] || this.ancestryCalibrationMap["Other / Unspecified"];

        try {
            const response = await fetch(INFERENCE_API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(patient)
            });
            if (!response.ok) throw new Error(`Inference server HTTP ${response.status}`);
            const data = await response.json();

            return {
                predictedWeeklyDose: data.predictedWeeklyDose,
                dailyDose: data.dailyDose,
                iwpcWeeklyDose: iwpcDose,
                interval: data.interval,
                intervalMethod: data.intervalMethod,
                ancestryInfo: ancestryInfo,
                validation: validation,
                shapAttributions: data.featureImpacts,
                modelType: data.modelType,
                modelTestMetrics: data.modelTestMetrics,
                fieldsDefaultedNotCollectedByUI: data.fieldsDefaultedNotCollectedByUI,
                isRealModel: true
            };
        } catch (err) {
            console.warn("Real inference server unreachable, using local fallback heuristic:", err);
            const fallback = this.predictDoseFallbackHeuristic(patient, validation, iwpcDose, ancestryInfo);
            fallback.inferenceError = err.message;
            return fallback;
        }
    }

    /**
     * Local hardcoded heuristic — NOT the trained model. Used only when
     * server/app_server.py (the real CatBoost inference backend) can't be
     * reached, so the UI never silently shows a fake number labeled as if
     * it came from the trained model.
     */
    predictDoseFallbackHeuristic(patient, validation, iwpcDose, ancestryInfo) {
        const hasGenetics = validation.hasGenetics;

        let baseDose = 35.0; // Mean population weekly dose

        // Age factor (-0.25 mg/week per year over 50)
        baseDose -= (patient.age - 50) * 0.26;

        // Weight factor (+0.22 mg/week per kg over 70)
        baseDose += (patient.weight - 70) * 0.22;

        // Height factor (+0.10 mg/week per cm over 170)
        baseDose += (patient.height - 170) * 0.10;

        // Amiodarone (-32% reduction)
        if (patient.amiodarone) baseDose *= 0.68;

        // Statins (-5% subtle interaction)
        if (patient.statin) baseDose *= 0.95;

        // Enzyme Inducers (+45% increase)
        if (patient.enzymeInducer) baseDose *= 1.45;

        // Genetics (Dominant drivers per SHAP analysis)
        if (hasGenetics) {
            if (patient.vkorc1 === "AG") baseDose -= 7.8;
            else if (patient.vkorc1 === "AA") baseDose -= 14.5;

            if (patient.cyp2c9 === "*1/*2") baseDose -= 4.2;
            else if (patient.cyp2c9 === "*1/*3") baseDose -= 8.1;
            else if (patient.cyp2c9 === "*2/*2") baseDose -= 9.5;
            else if (patient.cyp2c9 === "*2/*3") baseDose -= 12.8;
            else if (patient.cyp2c9 === "*3/*3") baseDose -= 15.6;
        }

        const predictedWeekly = Math.max(7.0, Math.min(95.0, Math.round(baseDose * 10) / 10));
        const dailyDose = Math.round((predictedWeekly / 7.0) * 100) / 100;

        // Heuristic-only interval (NOT the real model's conformal residual
        // interval, which requires the backend)
        const halfWidth = hasGenetics ? 12.4 : 16.2;
        const intervalLower = Math.max(5.0, Math.round((predictedWeekly - halfWidth) * 10) / 10);
        const intervalUpper = Math.round((predictedWeekly + halfWidth) * 10) / 10;

        // Compute illustrative feature attributions (NOT real SHAP/leave-one-out from a trained model)
        const shapAttributions = this.computeSHAP(patient, predictedWeekly, hasGenetics);

        return {
            predictedWeeklyDose: predictedWeekly,
            dailyDose: dailyDose,
            iwpcWeeklyDose: iwpcDose,
            interval: [intervalLower, intervalUpper],
            ancestryInfo: ancestryInfo,
            validation: validation,
            shapAttributions: shapAttributions,
            modelType: "⚠ Local Heuristic Fallback (real model backend unreachable) — NOT trained-model output",
            isRealModel: false
        };
    }

    /**
     * Compute individual patient SHAP attributions (+/- mg/week impact)
     */
    computeSHAP(patient, predictedDose, hasGenetics) {
        const attributions = [];
        const populationMeanDose = 35.0;

        if (hasGenetics && patient.vkorc1 && patient.vkorc1 !== "unknown") {
            const vkorcVal = patient.vkorc1 === "AA" ? -14.5 : (patient.vkorc1 === "AG" ? -7.8 : +2.4);
            attributions.push({
                feature: `VKORC1 -1639G>A (${patient.vkorc1})`,
                impact: vkorcVal,
                category: "genetics",
                evidenceUrl: "https://www.pharmgkb.org/guidelineAnnotation/PA166104949"
            });
        }

        if (hasGenetics && patient.cyp2c9 && patient.cyp2c9 !== "unknown") {
            let cypVal = 0;
            if (patient.cyp2c9 === "*1/*2") cypVal = -4.2;
            else if (patient.cyp2c9 === "*1/*3") cypVal = -8.1;
            else if (patient.cyp2c9 === "*2/*2") cypVal = -9.5;
            else if (patient.cyp2c9 === "*2/*3") cypVal = -12.8;
            else if (patient.cyp2c9 === "*3/*3") cypVal = -15.6;
            else cypVal = +1.8;

            attributions.push({
                feature: `CYP2C9 Star Allele (${patient.cyp2c9})`,
                impact: cypVal,
                category: "genetics",
                evidenceUrl: "https://www.pharmgkb.org/gene/PA126"
            });
        }

        const ageImpact = Math.round(-(patient.age - 50) * 0.26 * 10) / 10;
        attributions.push({
            feature: `Age (${patient.age} yrs)`,
            impact: ageImpact,
            category: "clinical",
            evidenceUrl: "https://pubmed.ncbi.nlm.nih.gov/19228618/"
        });

        const weightImpact = Math.round((patient.weight - 70) * 0.22 * 10) / 10;
        attributions.push({
            feature: `Body Weight (${patient.weight} kg)`,
            impact: weightImpact,
            category: "clinical",
            evidenceUrl: "https://pubmed.ncbi.nlm.nih.gov/19228618/"
        });

        if (patient.amiodarone) {
            attributions.push({
                feature: "Amiodarone Co-medication",
                impact: Math.round(-predictedDose * 0.32 * 10) / 10,
                category: "clinical",
                evidenceUrl: "https://www.fda.gov/drugs"
            });
        }

        if (patient.enzymeInducer) {
            attributions.push({
                feature: "Enzyme Inducer Therapy",
                impact: Math.round(predictedDose * 0.45 * 10) / 10,
                category: "clinical",
                evidenceUrl: "https://www.fda.gov/drugs"
            });
        }

        attributions.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
        return attributions;
    }
}

// Global instance export
window.dosingEngine = new WarfaRiskEngine();
