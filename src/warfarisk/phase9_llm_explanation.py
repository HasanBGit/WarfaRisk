"""Phase 9 — LLM explanation/citation layer.

Implements TRAINING_PLAN.md Phase 9: ground a clinician-facing explanation of
a dose/INR prediction in (a) the Phase 6 SHAP output and (b) retrieved
PharmCAT/cpic-data guideline passages plus this project's own
pharmgkb_allele_tables, then have an LLM synthesize the explanation under a
strict-citation prompt — retrieval-augmented structured synthesis, not free
generation. Default generator is a general-purpose instruction model
(Qwen2.5-Instruct), not a medical-tuned one: a citation-faithfulness study
found heavy medical fine-tuning can cause "knowledge conflict" (the model
overrides retrieved evidence with its own memorized priors), and this
phase's job is faithful citation, not medical recall — the retrieval layer
already supplies the facts. MedGemma-27b-text-it is the benchmark-strongest
medical-tuned fallback, to be A/B tested against the default on this
project's own corpus before picking a final model (see verify() below).

Every generated explanation must also surface a Phase-4-derived caveat when
the patient's ancestry group was underrepresented/poorly covered in
training — passed in explicitly as `ancestry_coverage_note`, not inferred by
the LLM, so that caveat can't silently get dropped by a bad generation.

Retrieval corpus source: PharmCAT (github.com/PharmGKB/PharmCAT) and its
companion cpicpgx/cpic-data repo are NOT vendored into this project — they
must be cloned separately (see build_retrieval_corpus()'s docstring). This
project's own data/usable/pharmgkb_allele_tables/ IS available locally and is
included automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from warfarisk.common import paths

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MEDICAL_FALLBACK_MODEL_NAME = "google/medgemma-27b-text-it"  # gated, HAI-DEF terms


@dataclass(frozen=True)
class Passage:
    source: str  # e.g. "cpic-data/warfarin" or "pharmgkb_allele_tables/CYP2C9"
    text: str


def build_retrieval_corpus(cpic_data_dir: Path | None = None) -> list[Passage]:
    """Assembles the retrieval corpus from two sources:
      1. This project's own PharmGKB allele-definition tables (already on
         disk at data/usable/pharmgkb_allele_tables/) — always included.
      2. PharmCAT's cpicpgx/cpic-data repo, if a local clone path is given —
         NOT auto-downloaded here (this project prepares code, it doesn't
         reach out to the network on import). Clone it yourself first:
             git clone https://github.com/cpicpgx/cpic-data
         then pass that path as cpic_data_dir.
    """
    passages: list[Passage] = []

    # TODO: this repo doesn't ship PharmGKB allele tables or a
    # PHARMGKB_ALLELE_TABLES_DIR constant (see data/DATA.md) — obtain the
    # tables locally and point this at them before running Phase 9.
    allele_dir = paths.PHARMGKB_ALLELE_TABLES_DIR
    if allele_dir.is_dir():
        for f in sorted(allele_dir.glob("*")):
            if f.is_file():
                passages.append(Passage(source=f"pharmgkb_allele_tables/{f.name}", text=f"[allele definition table: {f.name}]"))

    if cpic_data_dir is not None:
        cpic_data_dir = Path(cpic_data_dir)
        if not cpic_data_dir.is_dir():
            raise FileNotFoundError(
                f"{cpic_data_dir} does not exist. Clone github.com/cpicpgx/cpic-data first, "
                "then pass its local path here."
            )
        for f in sorted(cpic_data_dir.rglob("*warfarin*")):
            if f.is_file() and f.suffix in {".csv", ".tsv", ".json", ".txt"}:
                passages.append(Passage(source=f"cpic-data/{f.relative_to(cpic_data_dir)}", text=f.read_text(errors="ignore")))

    return passages


def embed_passages(passages: list[Passage]):
    """Requires `uv add sentence-transformers`. Returns (embeddings, model)
    so the same model instance can embed the query at retrieval time."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("sentence-transformers is required. Install with: uv add sentence-transformers") from e

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode([p.text for p in passages], convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, model


def build_faiss_index(embeddings):
    """Requires `uv add faiss-cpu`. A few thousand short guideline passages
    doesn't need a hosted vector DB — FAISS's flat index is the right,
    unglamorous choice at this corpus size."""
    try:
        import faiss
    except ImportError as e:
        raise ImportError("faiss-cpu is required. Install with: uv add faiss-cpu") from e

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized embeddings == cosine similarity
    index.add(embeddings)
    return index


def retrieve(query: str, index, passages: list[Passage], embedding_model, k: int = 5) -> list[Passage]:
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _distances, indices = index.search(query_embedding, k)
    return [passages[i] for i in indices[0] if i < len(passages)]


STRICT_CITATION_PROMPT_TEMPLATE = """You are explaining a warfarin dose prediction to a clinician. \
You MUST base every factual claim ONLY on the SHAP feature attributions and retrieved passages given below. \
For every sentence that makes a factual claim, cite its source in brackets, e.g. [SHAP: VKORC1] or [cpic-data/warfarin]. \
Do NOT introduce any medical fact that is not present in the inputs below, even if you believe it to be true. \
If the ancestry coverage note below is non-empty, include it verbatim as the final sentence.

Predicted dose: {predicted_dose}

SHAP feature attributions (feature: contribution):
{shap_lines}

Retrieved passages:
{passage_lines}

Ancestry coverage note (include verbatim as the final sentence if non-empty): {ancestry_coverage_note}

Explanation:"""


def build_prompt(predicted_dose: float, shap_report, retrieved_passages: list[Passage], ancestry_coverage_note: str = "") -> str:
    shap_lines = "\n".join(f"- {row.feature}: {row.mean_abs_shap:.4f}" for row in shap_report.itertuples())
    passage_lines = "\n".join(f"[{p.source}] {p.text[:500]}" for p in retrieved_passages)
    return STRICT_CITATION_PROMPT_TEMPLATE.format(
        predicted_dose=predicted_dose,
        shap_lines=shap_lines,
        passage_lines=passage_lines,
        ancestry_coverage_note=ancestry_coverage_note,
    )


def generate_explanation(prompt: str, model_name: str = DEFAULT_GENERATOR_MODEL_NAME, max_new_tokens: int = 512) -> str:
    """Requires `uv add transformers torch huggingface_hub` and, for
    MEDICAL_FALLBACK_MODEL_NAME specifically, prior `hf auth login` with
    HAI-DEF gated access approved."""
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError("transformers and torch are required. Install with: uv add transformers torch huggingface_hub") from e

    generator = pipeline("text-generation", model=model_name)
    output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return output[0]["generated_text"][len(prompt):].strip()


def citation_faithfulness_check(explanation_text: str, retrieved_passages: list[Passage], shap_report) -> dict:
    """A cheap, deterministic pre-check (not a substitute for human review):
    does every bracketed citation in the generated text actually match a
    retrieved passage's source string or a SHAP feature name? Flags citations
    that don't — the empirical A/B check TRAINING_PLAN.md Phase 9 calls for
    before picking a final generator model between the default and the
    medical-tuned fallback.
    """
    import re

    valid_sources = {p.source for p in retrieved_passages} | {f"SHAP: {row.feature}" for row in shap_report.itertuples()}
    cited = set(re.findall(r"\[([^\]]+)\]", explanation_text))
    unverifiable = cited - valid_sources
    return {
        "n_citations": len(cited),
        "n_unverifiable_citations": len(unverifiable),
        "unverifiable_citations": sorted(unverifiable),
        "fully_grounded": len(unverifiable) == 0,
    }


def citation_accuracy_check(explanation_text: str, shap_report, tolerance: float = 0.05) -> dict:
    """Catches the gap citation_faithfulness_check cannot: a correct citation
    attached to a fabricated magnitude or direction, e.g. "[SHAP: VKORC1]"
    cited next to a contribution value that does not match the real SHAP
    output for that feature. For each `[SHAP: <feature>]` citation, reads the
    numbers stated in the same sentence and flags the citation if none of
    them is within `tolerance` (relative) of the feature's actual
    mean_abs_shap value. This only checks SHAP citations, since a passage
    citation's "value" is free text, not a single number to compare against.
    """
    import re

    shap_values = {row.feature: row.mean_abs_shap for row in shap_report.itertuples()}
    # A period only ends a sentence when it isn't sitting between two
    # digits, so "0.82" isn't mistaken for a sentence boundary.
    sentence_boundaries = [m.start() for m in re.finditer(r"(?<!\d)\.(?!\d)", explanation_text)]
    mismatches = []
    n_checked = 0
    for match in re.finditer(r"\[SHAP:\s*([^\]]+)\]", explanation_text):
        feature = match.group(1).strip()
        if feature not in shap_values:
            continue
        before = [b for b in sentence_boundaries if b < match.start()]
        after = [b for b in sentence_boundaries if b >= match.end()]
        sentence_start = before[-1] + 1 if before else 0
        sentence_end = after[0] if after else len(explanation_text)
        sentence = explanation_text[sentence_start:sentence_end]
        # Excludes a digit that's part of a feature name like "VKORC1" or
        # "CYP2C9" (not preceded/followed by a letter or digit), so a stray
        # "1" from the citation bracket can't coincidentally "match" a real
        # SHAP value and mask an actual mismatch.
        stated_numbers = [float(n) for n in re.findall(r"(?<![A-Za-z0-9])-?\d+\.?\d*(?![A-Za-z0-9])", sentence)]
        if not stated_numbers:
            continue
        n_checked += 1
        true_value = shap_values[feature]
        if not any(abs(abs(n) - abs(true_value)) <= tolerance * max(abs(true_value), 1e-9) for n in stated_numbers):
            mismatches.append({"feature": feature, "stated": stated_numbers, "actual": true_value})
    return {
        "n_shap_citations_checked": n_checked,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "fully_accurate": len(mismatches) == 0,
    }
