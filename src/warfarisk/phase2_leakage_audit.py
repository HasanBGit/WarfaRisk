"""Phase 2 — leakage audit.

Implements TRAINING_PLAN.md Phase 2: runs common/leakage.py's checks over
every cohort's feature matrix and every generated split, and prints a
pass/fail report. This is meant to run once after Phase 0's splits exist and
before any model in Phase 3+ is fit — a fast, cheap gate, not a model.
"""

from __future__ import annotations

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.common.leakage import LeakageError, assert_disjoint_ids, audit_iwpc1780, audit_iwpc6256


def audit_all() -> list[tuple[str, bool, str]]:
    """Returns a list of (check_name, passed, detail) tuples instead of
    raising on the first failure, so a single run reports every problem
    found rather than stopping at the first one."""
    results: list[tuple[str, bool, str]] = []

    def _check(name: str, fn) -> None:
        try:
            fn()
            results.append((name, True, "ok"))
        except LeakageError as e:
            results.append((name, False, str(e)))

    iwpc6256 = data_mod.load_iwpc_6256()
    iwpc1780 = data_mod.load_iwpc_1780()

    for which in ("clinical", "genetic", "combined"):
        _check(
            f"iwpc_6256[{which}]_no_leakage_columns",
            lambda which=which: audit_iwpc6256(data_mod.iwpc6256_feature_set(iwpc6256, which)),
        )
        _check(
            f"iwpc_1780[{which}]_no_leakage_columns",
            lambda which=which: audit_iwpc1780(data_mod.iwpc1780_feature_set(iwpc1780, which)),
        )

    for dataset_name in ("iwpc_6256", "iwpc_1780", "eicu", "mimic3", "mimic4"):
        def _check_split(dataset_name=dataset_name):
            split = splits.load_split(dataset_name)
            assert_disjoint_ids(split.train_ids, split.test_ids, context=dataset_name)

        _check(f"{dataset_name}_train_test_ids_disjoint", _check_split)

    return results


def print_report(results: list[tuple[str, bool, str]]) -> None:
    n_failed = sum(1 for _, passed, _ in results if not passed)
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + ("" if passed else f" — {detail}"))
    print(f"\n{len(results) - n_failed}/{len(results)} checks passed.")
    if n_failed:
        raise SystemExit(f"{n_failed} leakage check(s) failed — fix before proceeding to Phase 3+.")


if __name__ == "__main__":
    print_report(audit_all())
