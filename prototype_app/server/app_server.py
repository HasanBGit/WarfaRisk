"""Real-inference backend for WarfaRisk."""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS

from dosing_model import NOT_COLLECTED_BY_UI, DosingModel
from explain_service import (
    ExplainBackendError,
    ExplainUnavailableError,
    complete,
    select_backend,
)
from settings import META_PATH, MODEL_PATH, Settings

MODEL_TYPE = "CatBoost (Phase 3 combined feature set, iwpc_6256) — real trained model inference"
INTERVAL_METHOD = "90th-percentile-absolute-residual on this model's own held-out test set"


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.load()
    dosing_model = DosingModel(MODEL_PATH, META_PATH)
    metadata = dosing_model.metadata

    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": "catboost_iwpc6256_combined",
                "test_metrics": metadata["test_metrics"],
                "explain_provider": settings.explain_provider,
            }
        )

    @app.post("/predict")
    def predict():
        patient = request.get_json(force=True)
        row = dosing_model.build_feature_row(patient)
        prediction, interval_low, interval_high = dosing_model.predict_with_interval(row)
        return jsonify(
            {
                "predictedWeeklyDose": round(prediction, 1),
                "dailyDose": round(prediction / 7.0, 2),
                "interval": [interval_low, interval_high],
                "intervalMethod": f"{INTERVAL_METHOD} (n={metadata['n_test']})",
                "modelType": MODEL_TYPE,
                "modelTestMetrics": metadata["test_metrics"],
                "featureImpacts": dosing_model.leave_one_out_impacts(row),
                "fieldsDefaultedNotCollectedByUI": sorted(NOT_COLLECTED_BY_UI),
                "featureRowUsed": row,
            }
        )

    @app.post("/explain")
    def explain():
        body = request.get_json(force=True)
        try:
            backend = select_backend(settings, body.get("model"))
        except ExplainUnavailableError as error:
            return jsonify({"error": str(error)}), 503
        try:
            text = complete(backend, body.get("prompt", ""))
        except ExplainBackendError as error:
            return jsonify({"error": str(error)}), 502
        return jsonify({"text": text, "model": backend.model, "provider": backend.provider})

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"Loaded model from {MODEL_PATH}")
    app.run(host="127.0.0.1", port=8787, debug=False)
