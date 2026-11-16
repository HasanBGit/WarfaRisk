
class APIService {
    constructor() {
        this.BACKEND_EXPLAIN_URL = "http://127.0.0.1:8787/explain";

        const storedKey = localStorage.getItem("warfarisk_api_key");
        this.apiKey = (storedKey && storedKey.trim() !== "") ? storedKey.trim() : "";
        this.apiProvider = localStorage.getItem("warfarisk_api_provider") || "openrouter";
        this.modelName = localStorage.getItem("warfarisk_model_name") || "google/gemini-2.5-flash";
        this.customEndpoint = localStorage.getItem("warfarisk_custom_endpoint") || "";
    }

    saveSettings(provider, apiKey, modelName, customEndpoint = "") {
        this.apiProvider = provider;
        this.apiKey = apiKey.trim();
        this.modelName = modelName.trim() || "google/gemini-2.5-flash";
        this.customEndpoint = customEndpoint.trim();

        localStorage.setItem("warfarisk_api_provider", this.apiProvider);
        localStorage.setItem("warfarisk_api_key", this.apiKey);
        localStorage.setItem("warfarisk_model_name", this.modelName);
        localStorage.setItem("warfarisk_custom_endpoint", this.customEndpoint);
    }

    buildStrictPrompt(predictionResult, passages) {
        const shapLines = predictionResult.shapAttributions
            .map(s => `- ${s.feature}: ${s.impact > 0 ? '+' : ''}${s.impact} mg/week`)
            .join("\n");

        const passageLines = passages
            .map(p => `[${p.source}] ${p.text}`)
            .join("\n\n");

        const ancestryNote = predictionResult.ancestryInfo.warning || "";

        return `You are explaining a warfarin dose prediction to a clinician.
You MUST base every factual claim ONLY on the SHAP feature attributions and retrieved passages given below.
For every sentence that makes a factual claim, cite its source in brackets, e.g. [SHAP: VKORC1] or [CPIC 2017 Warfarin Guideline (rs9923231)].
Do NOT introduce any medical fact that is not present in the inputs below, even if you believe it to be true.
If the ancestry coverage note below is non-empty, include it verbatim as the final sentence.

Predicted Stable Weekly Dose: ${predictionResult.predictedWeeklyDose} mg/week (Daily: ${predictionResult.dailyDose} mg/day)
90% Prediction Interval (from held-out test residuals): (${predictionResult.interval[0]} - ${predictionResult.interval[1]}) mg/week

SHAP feature attributions (feature: contribution):
${shapLines}

Retrieved Guideline Passages:
${passageLines}

Ancestry coverage note (include verbatim as the final sentence if non-empty): ${ancestryNote}

Explanation:`;
    }

    async generateExplanation(predictionResult, queryText) {
        const passages = window.vectorDB.search(queryText || `warfarin dosing ${predictionResult.ancestryInfo.status}`, 4);
        const prompt = this.buildStrictPrompt(predictionResult, passages);

        const isOpenRouterKey = this.apiKey.startsWith("sk-or-v1-");

        try {
            let responseText;
            let displayModel;

            if (this.apiKey) {
                if (isOpenRouterKey || this.apiProvider === "openrouter") {
                    responseText = await this.callOpenRouter(prompt);
                } else if (this.apiProvider === "gemini") {
                    responseText = await this.callGemini(prompt);
                } else {
                    responseText = await this.callOpenAI(prompt);
                }
                displayModel = isOpenRouterKey ? "google/gemini-2.5-flash (OpenRouter)" : `${this.apiProvider} (${this.modelName})`;
            } else {
                const proxyResult = await this.callBackendProxy(prompt);
                responseText = proxyResult.text;
                const providerLabel = proxyResult.provider === "huggingface" ? "Hugging Face" : "OpenRouter";
                displayModel = `${proxyResult.model} (${providerLabel}, Managed Key)`;
            }

            const verification = this.verifyCitations(responseText, passages, predictionResult.shapAttributions);
            verification.accuracy = this.verifyAccuracy(responseText, predictionResult.shapAttributions);
            return { text: responseText, passages, provider: displayModel, verification };
        } catch (err) {
            console.warn("External LLM API call failed, falling back to Grounded Clinical Engine:", err);
            const fallbackText = this.generateDeterministicFallback(predictionResult, passages);
            const verification = this.verifyCitations(fallbackText, passages, predictionResult.shapAttributions);
            verification.accuracy = this.verifyAccuracy(fallbackText, predictionResult.shapAttributions);
            return { text: fallbackText, passages, provider: "Grounded Clinical Engine (Fallback)", verification, apiError: err.message };
        }
    }

    async callBackendProxy(prompt) {
        const response = await fetch(this.BACKEND_EXPLAIN_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, model: this.modelName })
        });

        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || `Backend proxy HTTP ${response.status}`);
        }
        return data;
    }

    async callOpenRouter(prompt) {
        const endpoint = this.customEndpoint || "https://openrouter.ai/api/v1/chat/completions";
        const model = (this.modelName && this.modelName.includes("/")) ? this.modelName : "google/gemini-2.5-flash";

        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${this.apiKey}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model: model,
                messages: [
                    { role: "user", content: prompt }
                ],
                temperature: 0.2
            })
        });

        if (!response.ok) {
            const errBody = await response.text();
            throw new Error(`OpenRouter HTTP ${response.status}: ${errBody}`);
        }

        const data = await response.json();
        if (!data.choices || !data.choices[0] || !data.choices[0].message) {
            throw new Error("Invalid response schema from OpenRouter API");
        }

        return data.choices[0].message.content.trim();
    }

    async callGemini(prompt) {
        const model = "gemini-1.5-flash";
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${this.apiKey}`;

        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }],
                generationConfig: { temperature: 0.2 }
            })
        });

        if (!response.ok) {
            const errBody = await response.text();
            throw new Error(`Gemini HTTP ${response.status}: ${errBody}`);
        }

        const data = await response.json();
        return data.candidates[0].content.parts[0].text.trim();
    }

    async callOpenAI(prompt) {
        const endpoint = this.customEndpoint || "https://api.openai.com/v1/chat/completions";
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${this.apiKey}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model: this.modelName || "gpt-4o-mini",
                messages: [{ role: "user", content: prompt }],
                temperature: 0.2
            })
        });

        if (!response.ok) {
            const errBody = await response.text();
            throw new Error(`OpenAI HTTP ${response.status}: ${errBody}`);
        }

        const data = await response.json();
        return data.choices[0].message.content.trim();
    }

    generateDeterministicFallback(prediction, passages) {
        let lines = [];
        lines.push(`The recommended stable weekly warfarin dose is estimated at ${prediction.predictedWeeklyDose} mg/week (${prediction.dailyDose} mg/day) with a 90% prediction interval of (${prediction.interval[0]} - ${prediction.interval[1]}) mg/week [MAPIE Split Conformal Uncertainty Quantification].`);

        const vkorc = prediction.shapAttributions.find(s => s.feature.includes("VKORC1"));
        if (vkorc) {
            lines.push(`Dose recommendation is primary driven by ${vkorc.feature} (${vkorc.impact > 0 ? '+' : ''}${vkorc.impact} mg/week impact) [VKORC1 -1639G>A Dosing Recommendation].`);
        }

        const cyp = prediction.shapAttributions.find(s => s.feature.includes("CYP2C9"));
        if (cyp) {
            lines.push(`Concomitant ${cyp.feature} reduces warfarin clearance (${cyp.impact} mg/week impact) [CYP2C9 Star Alleles & Metabolizer Status].`);
        }

        const amio = prediction.shapAttributions.find(s => s.feature.includes("Amiodarone"));
        if (amio) {
            lines.push(`Amiodarone co-administration requires significant dose reduction due to CYP2C9 inhibition (${amio.impact} mg/week impact) [SHAP: ${amio.feature}].`);
        }

        if (prediction.ancestryInfo.warning) {
            lines.push(prediction.ancestryInfo.warning);
        }

        return lines.join(" ");
    }

    verifyCitations(text, passages, shapAttributions) {
        const matches = text.match(/\[([^\]]+)\]/g) || [];
        const citations = matches.map(m => m.replace(/[\[\]]/g, ''));

        const validSources = new Set([
            ...passages.map(p => p.source),
            ...passages.map(p => p.title),
            ...shapAttributions.map(s => s.feature),
            ...shapAttributions.map(s => `SHAP: ${s.feature}`),
            "MAPIE Split Conformal Uncertainty Quantification",
            "VKORC1 -1639G>A Dosing Recommendation",
            "CYP2C9 Star Alleles & Metabolizer Status"
        ]);

        const verified = [];
        const unverified = [];

        citations.forEach(c => {
            let isValid = Array.from(validSources).some(s => s.toLowerCase().includes(c.toLowerCase()) || c.toLowerCase().includes(s.toLowerCase()));
            if (isValid) verified.push(c);
            else unverified.push(c);
        });

        return {
            totalCitations: citations.length,
            verifiedCount: verified.length,
            unverifiedCount: unverified.length,
            unverified: unverified,
            isFullyGrounded: unverified.length === 0
        };
    }

    verifyAccuracy(text, shapAttributions, tolerance = 0.3) {
        const sentenceBoundaries = [];
        const boundaryRe = /(?<!\d)\.(?!\d)/g;
        let boundaryMatch;
        while ((boundaryMatch = boundaryRe.exec(text)) !== null) {
            sentenceBoundaries.push(boundaryMatch.index);
        }

        const mismatches = [];
        let checked = 0;
        const citationRe = /\[([^\]]+)\]/g;
        let citationMatch;
        while ((citationMatch = citationRe.exec(text)) !== null) {
            const citation = citationMatch[1];
            const feature = shapAttributions.find(s =>
                citation.toLowerCase().includes(`shap: ${s.feature}`.toLowerCase()) ||
                s.feature.toLowerCase().includes(citation.toLowerCase()) ||
                citation.toLowerCase().includes(s.feature.toLowerCase())
            );
            if (!feature) continue;

            const before = sentenceBoundaries.filter(b => b < citationMatch.index);
            const after = sentenceBoundaries.filter(b => b >= citationMatch.index + citationMatch[0].length);
            const start = before.length ? before[before.length - 1] + 1 : 0;
            const end = after.length ? after[0] : text.length;
            const sentence = text.slice(start, end);

            const numberRe = /(?<![A-Za-z0-9])-?\d+\.?\d*(?![A-Za-z0-9])/g;
            const stated = [...sentence.matchAll(numberRe)].map(m => parseFloat(m[0]));
            if (!stated.length) continue;

            checked += 1;
            const actual = feature.impact;
            const isAccurate = stated.some(n => Math.abs(n - actual) <= tolerance);
            if (!isAccurate) {
                mismatches.push({ feature: feature.feature, stated, actual });
            }
        }

        return {
            citationsChecked: checked,
            inaccurateCount: mismatches.length,
            inaccurate: mismatches,
            isFullyAccurate: mismatches.length === 0
        };
    }
}

window.apiService = new APIService();
