# DDI-aware comedication embeddings — investigated and rejected

IWPC data encodes comedications (amiodarone, statins, antibiotics, antifungals) as flat binary
flags in `common/data.py`'s `IWPC6256_CLINICAL_COLS`. A learned drug-drug-interaction (DDI)
embedding was investigated as a possible upgrade over these flat flags.

**Finding:** no credible, maintained pretrained DDI embedding model exists on Hugging Face.
Direct search returned only near-zero-download hobbyist fine-tunes and unrelated name collisions
(e.g. diffusion models matching the "DDIM" string). Academic DDI-GNN approaches (HyGNN and
similar) exist only as one-off research code, not reusable pretrained embeddings.

**Decision:** keep the flat binary comedication flags as-is. Documented here so this isn't
re-investigated later — see `TRAINING_PLAN.md`'s "Second comparator sweep" section for the same
conclusion in context with the four experiments that *were* added alongside it.
