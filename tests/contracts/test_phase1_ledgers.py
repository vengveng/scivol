"""Permanent CI checks for declarations, not mathematical Phase1 completion."""

from pathlib import Path
import warnings

import pytest

from ledger_validation import (
    load_ledgers,
    validate_ledgers,
    validate_core_adaptations,
    validate_evidence_claims,
    validate_retained_test_sources,
    validate_source_snapshots,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ledgers():
    return load_ledgers(ROOT)


def test_phase1_ledger_structure_and_cross_references(ledgers):
    for limitation in validate_ledgers(ledgers, ROOT):
        if "V2 normalization pending" in limitation:
            warnings.warn(limitation, RuntimeWarning, stacklevel=2)


def test_phase1_claims_are_supported_by_typed_evidence(ledgers):
    for limitation in validate_evidence_claims(ledgers, ROOT):
        warnings.warn(limitation, RuntimeWarning, stacklevel=2)


def test_retained_test_and_fixture_snapshot_identity(ledgers):
    validate_retained_test_sources(ledgers["test_transfer"], ROOT)


def test_source_transfer_snapshot_integrity(ledgers):
    from provenance_validation import is_v2, validate_provenance
    from ledger_validation import provenance_documents

    if is_v2(ledgers["source_transfers"]):
        fragments, _ = provenance_documents(ledgers)
        validate_provenance(fragments, ROOT, ledgers["phase1_scope"]["inventory_id"])
        return
    limitations = validate_source_snapshots(ledgers["source_transfers"], ROOT)
    for limitation in limitations:
        warnings.warn(limitation, RuntimeWarning, stacklevel=2)


def test_derived_owner_provenance_paths_and_hashes(ledgers):
    from provenance_validation import is_v2, validate_provenance
    from ledger_validation import provenance_documents

    if is_v2(ledgers["core_adaptations"]):
        fragments, _ = provenance_documents(ledgers)
        validate_provenance(fragments, ROOT, ledgers["phase1_scope"]["inventory_id"])
        return
    validate_core_adaptations(ledgers["core_adaptations"], ROOT)
