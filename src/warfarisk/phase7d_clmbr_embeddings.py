"""Phase 7d — CLMBR-T-base as a frozen embedding extractor for the ICU cohorts.

Implements the "second comparator sweep" EHR-foundation-model addition to
TRAINING_PLAN.md Phase 7: `StanfordShahLab/clmbr-t-base` (141M params, gated,
pretrained on 2.57M Stanford patients) used ONLY as a frozen embedding
extractor feeding into whichever Phase 3 winner is chosen — never fine-tuned.
At n=17-160 patients, fine-tuning a 141M-parameter model would badly overfit;
the model card itself warns it may not generalize beyond Stanford's own
hospital, so any gain here is exploratory, not a claimed generalizable
result.

Prerequisites, both external to this codebase:
  1. Gated HF access to StanfordShahLab/clmbr-t-base must be approved first
     (huggingface.co/StanfordShahLab/clmbr-t-base) — `hf auth login` with an
     approved token.
  2. Input must be in MEDS schema / OMOP-CDM coded format, not raw MIMIC
     CSVs. A ready-made MIMIC-IV demo in MEDS format already exists at
     https://www.physionet.org/content/mimic-iv-demo-meds/0.0.1/ — download
     that instead of hand-rolling a MEDS conversion of this project's own
     mimic4_*_clean.csv files.

The exact model-loading and inference call signature (CLMBR is an EHR
sequence model, not a standard AutoModel/AutoTokenizer text model — it
expects coded patient-event streams, not free text) has NOT been verified
against a live install in this environment. Per this project's own rule
(see phase1_baselines.warfit_learn_iwpc_baseline for the same pattern): do
not fabricate a plausible-looking call signature for a specialized model
that hasn't actually been run. The function below is scaffolded up to the
point that needs verification, then raises rather than guessing further.
"""

from __future__ import annotations

from pathlib import Path


def load_clmbr_model():
    """Loads the gated CLMBR-T-base checkpoint. Requires
    `uv add transformers torch huggingface_hub` and prior `hf auth login`
    with access to StanfordShahLab/clmbr-t-base approved."""
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
    """Placeholder for the actual embedding-extraction call.

    NOT YET IMPLEMENTED — deliberately. CLMBR-T-base consumes MEDS-formatted
    patient event sequences through its own (non-standard) inference path;
    the exact function/class names for that path need to be read from the
    model repo's own usage example (or the EHRSHOT benchmark codebase, which
    was built around this exact model family) once gated access is approved
    and the MEDS-format MIMIC-IV demo data has been downloaded. Fabricating a
    call here that hasn't been checked against the real repo would silently
    break later rather than failing loudly now — so this raises instead.
    """
    raise NotImplementedError(
        "extract_embeddings() needs the CLMBR-T-base repo's own usage example (or the EHRSHOT benchmark "
        "codebase) consulted once gated access is approved and the MEDS-format MIMIC-IV demo "
        f"(expected at {meds_dataset_path}) is downloaded from "
        "https://www.physionet.org/content/mimic-iv-demo-meds/0.0.1/ — do not guess the call signature "
        "before then."
    )
