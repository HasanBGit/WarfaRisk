"""Verifies phase9_llm_explanation.citation_accuracy_check catches a correct
citation attached to a fabricated magnitude/direction, and does not flag an
accurate one."""

import pandas as pd

from warfarisk.phase9_llm_explanation import citation_accuracy_check

shap_report = pd.DataFrame({"feature": ["VKORC1", "Age"], "mean_abs_shap": [0.82, 0.18]})

accurate_text = (
    "VKORC1 genotype is the strongest driver of this dose, contributing 0.82 [SHAP: VKORC1]. "
    "Age contributes 0.18 [SHAP: Age]."
)
fabricated_text = (
    "VKORC1 genotype is the strongest driver of this dose, contributing 2.50 [SHAP: VKORC1]. "
    "Age contributes 0.18 [SHAP: Age]."
)

accurate_result = citation_accuracy_check(accurate_text, shap_report)
fabricated_result = citation_accuracy_check(fabricated_text, shap_report)

assert accurate_result["fully_accurate"] is True, accurate_result
assert accurate_result["n_shap_citations_checked"] == 2, accurate_result
assert fabricated_result["fully_accurate"] is False, fabricated_result
assert fabricated_result["n_mismatches"] == 1, fabricated_result
assert fabricated_result["mismatches"][0]["feature"] == "VKORC1", fabricated_result
assert fabricated_result["mismatches"][0]["stated"] == [2.5], fabricated_result

# Adversarial case: the feature name "VKORC1" contains a trailing digit "1".
# True value is 1.00, stated value is wrong (5.00) -- the stray "1" inside
# the feature name must NOT be picked up as a stated number and coincidentally
# "match" the true value, masking the real mismatch.
adversarial_shap_report = pd.DataFrame({"feature": ["VKORC1"], "mean_abs_shap": [1.00]})
adversarial_text = "VKORC1 genotype contributes 5.00 [SHAP: VKORC1]."
adversarial_result = citation_accuracy_check(adversarial_text, adversarial_shap_report)
assert adversarial_result["fully_accurate"] is False, adversarial_result
assert adversarial_result["mismatches"][0]["stated"] == [5.0], adversarial_result

print("accurate_result:", accurate_result)
print("fabricated_result:", fabricated_result)
print("adversarial_result:", adversarial_result)
print("OK: citation_accuracy_check correctly passes accurate claims, flags the fabricated one, "
      "and isn't fooled by a digit embedded in the feature name itself.")
