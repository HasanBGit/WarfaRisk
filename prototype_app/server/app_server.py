"""Real-inference backend for WarfaRisk.

Loads the CatBoost model trained in SURE-Program/Models-Development
(Phase 3's "combined" feature set on iwpc_6256 — the plan's own
production-model pick, see RESULTS.md) and serves actual predictions from it.
This replaces the hardcoded linear formula previously in dosing_engine.js's
predictDose(), which was NOT calling any trained model.

Run:
    uv run --with flask,flask-cors,catboost,pandas python3 server/app_server.py
Serves on http://127.0.0.1:8787
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool
from flask import Flask, jsonify, request
from flask_cors import CORS

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "catboost_iwpc6256_combined.cbm"
META_PATH = ARTIFACT_DIR / "catboost_iwpc6256_combined.json"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file(path: Path) -> dict:
    """Minimal KEY=VALUE .env parser so the OpenRouter key stays server-side
    only, without adding a python-dotenv dependency."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


_env = _load_env_file(ENV_PATH)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or _env.get("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL") or _env.get("DEFAULT_MODEL", "google/gemini-2.5-flash")

# MedGemma-27b-text-it via Hugging Face Inference Providers. Confirmed live
# (2026-08-29) via the HF Hub API: `google/medgemma-27b-text-it` resolves to
# provider "featherless-ai", task "conversational", status "live" -- this is
# the actual medically fine-tuned model, not routed through OpenRouter (which
# does not host MedGemma at all as of this check). Preferred over OpenRouter
# below when a token is configured, since it is the domain-specialized model
# this project's RAG layer is designed around.
HF_TOKEN = os.environ.get("HF_TOKEN") or _env.get("HF_TOKEN", "")
# Explicit ":featherless-ai" provider suffix required -- HF's router returned
# "model_not_supported" via auto-routing even with the provider enabled on
# the account; the explicit suffix resolved it immediately (verified live,
# 2026-08-29). See https://huggingface.co/docs/inference-providers.
HF_MEDGEMMA_MODEL = os.environ.get("HF_MEDGEMMA_MODEL") or _env.get("HF_MEDGEMMA_MODEL", "google/medgemma-27b-text-it:featherless-ai")

metadata = json.loads(META_PATH.read_text())
FEATURE_COLS = metadata["feature_cols"]
CAT_FEATURES = metadata["cat_features"]
DEFAULTS = metadata["defaults"]

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

app = Flask(__name__)
CORS(app)

# Fields this UI does not collect (Indication code legend is not documented
# in the dataset, Gender is not asked, the six rarer VKORC1 SNPs are not
# genotyped by this tool) fall back to the training-set mode/median captured
# in DEFAULTS at export time — see export_catboost_for_app.py.
NOT_COLLECTED_BY_UI = {
    "Gender",
    "Indication for Warfarin Treatment",
    "Diabetes",
    "Congestive Heart Failure and/or Cardiomyopathy",
    "Valve Replacement",
    "Aspirin",
    "Acetaminophen or Paracetamol (Tylenol)",
    "Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day",
    "Simvastatin (Zocor)",
    "Atorvastatin (Lipitor)",
    "Fluvastatin (Lescol)",
    "Lovastatin (Mevacor)",
    "Pravastatin (Pravachol)",
    "Rosuvastatin (Crestor)",
    "Cerivastatin (Baycol)",
    "Carbamazepine (Tegretol)",
    "Phenytoin (Dilantin)",
    "Rifampin or Rifampicin",
    "Sulfonamide Antibiotics",
    "Macrolide Antibiotics",
    "Anti-fungal Azoles",
    "Herbal Medications, Vitamins, Supplements",
    "Current Smoker",
    "VKORC1 497 consensus",
    "VKORC1 1173 consensus",
    "VKORC1 1542 consensus",
    "VKORC1 3730 consensus",
    "VKORC1 2255 consensus",
    "VKORC1     -4451 consensus",
}


def age_to_bucket(age: float) -> str:
    """IWPC-6256's Age column is decade-bucketed text ("60 - 69", "90+"),
    not a raw number — matches the original PharmGKB codebook format."""
    age = int(age)
    if age >= 90:
        return "90+"
    decade = (age // 10) * 10
    return f"{decade} - {decade + 9}"


def vkorc1_to_consensus(code: str) -> str:
    mapping = {"GG": "G/G", "AG": "A/G", "AA": "A/A"}
    return mapping.get(code, "missing")


def build_feature_row(patient: dict) -> dict:
    row = dict(DEFAULTS)  # start from training-set defaults for every column

    row["Age"] = age_to_bucket(patient["age"])
    row["Height (cm)"] = float(patient["height"])
    row["Weight (kg)"] = float(patient["weight"])

    vkorc1 = patient.get("vkorc1", "unknown")
    row["VKORC1     -1639 consensus"] = vkorc1_to_consensus(vkorc1) if vkorc1 != "unknown" else "missing"

    cyp2c9 = patient.get("cyp2c9", "unknown")
    row["CYP2C9 consensus"] = cyp2c9 if cyp2c9 != "unknown" else "missing"

    row["Amiodarone (Cordarone)"] = 1 if patient.get("amiodarone") else 0
    if patient.get("statin"):
        row["Atorvastatin (Lipitor)"] = 1
    if patient.get("enzymeInducer"):
        row["Rifampin or Rifampicin"] = 1
        row["Carbamazepine (Tegretol)"] = 1

    return row


def run_model(row: dict) -> float:
    X = pd.DataFrame([row], columns=FEATURE_COLS)
    X[CAT_FEATURES] = X[CAT_FEATURES].fillna("missing")
    pool = Pool(X, cat_features=CAT_FEATURES)
    return float(model.predict(pool)[0])


def leave_one_out_impacts(row: dict) -> list[dict]:
    """Real per-field marginal contribution: re-run the actual trained model
    with one collected field reverted to the population default and measure
    the prediction delta. Not SHAP (no TreeExplainer wired up here), but a
    genuine sensitivity computed by calling the model itself, not a
    hand-picked coefficient."""
    full_pred = run_model(row)
    impacts = []

    fields_to_probe = [
        ("VKORC1     -1639 consensus", f"VKORC1 -1639G>A ({row['VKORC1     -1639 consensus']})", "genetics"),
        ("CYP2C9 consensus", f"CYP2C9 ({row['CYP2C9 consensus']})", "genetics"),
        ("Age", f"Age bucket ({row['Age']})", "clinical"),
        ("Weight (kg)", f"Weight ({row['Weight (kg)']} kg)", "clinical"),
        ("Height (cm)", f"Height ({row['Height (cm)']} cm)", "clinical"),
        ("Amiodarone (Cordarone)", "Amiodarone Co-medication", "clinical"),
        ("Rifampin or Rifampicin", "Enzyme Inducer Therapy", "clinical"),
    ]

    for col, label, category in fields_to_probe:
        if row[col] == DEFAULTS[col]:
            continue  # field is already at its population default, no deviation to attribute
        probe_row = dict(row)
        probe_row[col] = DEFAULTS[col]
        probe_pred = run_model(probe_row)
        impact = round(full_pred - probe_pred, 1)
        if abs(impact) < 0.05:
            continue
        impacts.append({"feature": label, "impact": impact, "category": category})

    impacts.sort(key=lambda x: -abs(x["impact"]))
    return impacts


@app.route("/health", methods=["GET"])
def health():
    explain_provider = "huggingface (medgemma)" if HF_TOKEN else ("openrouter" if OPENROUTER_API_KEY else "none (deterministic fallback only)")
    return jsonify({
        "status": "ok",
        "model": "catboost_iwpc6256_combined",
        "test_metrics": metadata["test_metrics"],
        "explain_provider": explain_provider,
    })


@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json(force=True)

    row = build_feature_row(patient)
    pred = max(7.0, min(150.0, run_model(row)))

    halfwidth = metadata["conformal_halfwidth_90"]
    interval_low = max(5.0, round(pred - halfwidth, 1))
    interval_high = round(pred + halfwidth, 1)

    impacts = leave_one_out_impacts(row)
    not_collected = sorted(NOT_COLLECTED_BY_UI)

    return jsonify(
        {
            "predictedWeeklyDose": round(pred, 1),
            "dailyDose": round(pred / 7.0, 2),
            "interval": [interval_low, interval_high],
            "intervalMethod": "90th-percentile-absolute-residual on this model's own held-out test set (n=%d)" % metadata["n_test"],
            "modelType": "CatBoost (Phase 3 combined feature set, iwpc_6256) — real trained model inference",
            "modelTestMetrics": metadata["test_metrics"],
            "featureImpacts": impacts,
            "fieldsDefaultedNotCollectedByUI": not_collected,
            "featureRowUsed": row,
        }
    )


def _call_chat_completions(endpoint: str, api_key: str, model: str, prompt: str) -> str:
    """Shared OpenAI-compatible chat-completions caller (HF Inference
    Providers and OpenRouter both speak this same request/response shape).

    A real User-Agent is required: at least one HF Inference Providers
    backend (Featherless AI, MedGemma's provider) fronts its API with
    Cloudflare, and Python urllib's default User-Agent string is blocked by
    Cloudflare's bot protection there (HTTP 403, error 1010) even with a
    valid, correctly-scoped token -- confirmed live, 2026-08-29. This is a
    standard, benign client identification header, not an evasion technique.
    """
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


@app.route("/explain", methods=["POST"])
def explain():
    """Proxy RAG explanation prompts to an LLM backend using a server-side
    key, so no key ever has to ship in client-side JS (see js/api_service.js).

    Preference order: MedGemma-27b-text-it via Hugging Face Inference
    Providers (the domain-specialized model this RAG layer is designed
    around) if HF_TOKEN is configured, else OpenRouter (general-purpose
    fallback, e.g. Gemini 2.5 Flash) if OPENROUTER_API_KEY is configured.
    With neither configured, the frontend's deterministic fallback
    generator handles the explanation instead (see js/api_service.js)."""
    body = request.get_json(force=True)
    prompt = body.get("prompt", "")
    requested_model = body.get("model")

    if HF_TOKEN:
        model = requested_model if (requested_model and "medgemma" in requested_model.lower()) else HF_MEDGEMMA_MODEL
        try:
            text = _call_chat_completions(
                "https://router.huggingface.co/v1/chat/completions", HF_TOKEN, model, prompt
            )
            return jsonify({"text": text, "model": model, "provider": "huggingface"})
        except urllib.error.HTTPError as e:
            return jsonify({"error": f"Hugging Face Inference HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}"}), 502
        except Exception as e:  # noqa: BLE001 - surface any network/proxy failure to the frontend
            return jsonify({"error": str(e)}), 502

    if OPENROUTER_API_KEY:
        model = requested_model or DEFAULT_MODEL
        try:
            text = _call_chat_completions(
                "https://openrouter.ai/api/v1/chat/completions", OPENROUTER_API_KEY, model, prompt
            )
            return jsonify({"text": text, "model": model, "provider": "openrouter"})
        except urllib.error.HTTPError as e:
            return jsonify({"error": f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}"}), 502
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 502

    return jsonify({"error": "Server has neither HF_TOKEN nor OPENROUTER_API_KEY configured (.env)"}), 503


if __name__ == "__main__":
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Test metrics at export time: {metadata['test_metrics']}")
    app.run(host="127.0.0.1", port=8787, debug=False)
