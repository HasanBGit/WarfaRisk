"""Phase 7d — CLMBR-T-base as a frozen embedding extractor for the ICU cohorts"""

from __future__ import annotations

from pathlib import Path


def load_clmbr_model():
    """Loads the gated CLMBR-T-base checkpoint"""
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required. Install with: uv add transformers torch huggingface_hub"
        ) from e

    try:
        return AutoModel.from_pretrained("StanfordShahLab/clmbr-t-base", trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to load StanfordShahLab/clmbr-t-base — most likely gated-access approval hasn't been "
            "granted yet for the logged-in HF account (check `hf auth whoami` and the model page's access "
            "request), or the local `hf` credentials aren't configured. This has not been verified against "
            "a live, access-approved install in this environment."
        ) from e


def extract_embeddings(meds_dataset_path: Path, model=None):
    """Placeholder for the actual embedding-extraction call"""
    raise NotImplementedError(
        "extract_embeddings() needs the CLMBR-T-base repo's own usage example (or the EHRSHOT benchmark "
        "codebase) consulted once gated access is approved and the MEDS-format MIMIC-IV demo "
        f"(expected at {meds_dataset_path}) is downloaded from "
        "https://www.physionet.org/content/mimic-iv-demo-meds/0.0.1/ — do not guess the call signature "
        "before then."
    )
