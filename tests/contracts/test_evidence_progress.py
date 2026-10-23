"""Synthetic evidence exercises achievable rules, never promotes real ledgers."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import ledger_validation as validation
from ledger_validation import ContractError, load_ledgers, validate_ledgers


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inventory():
    return load_ledgers(ROOT)


@pytest.fixture
def evidence_workspace(inventory, monkeypatch):
    documents = deepcopy(inventory)
    documents["evidence_registry"] = {
        "schema_version": 1, "inventory_id": documents["phase1_scope"]["inventory_id"], "records": [],
    }
    contents = {}
    read_bytes, read_text = Path.read_bytes, Path.read_text

    def fixture_bytes(path):
        return contents[path] if path in contents else read_bytes(path)

    def fixture_text(path, *args, **kwargs):
        return contents[path].decode("utf-8") if path in contents else read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", fixture_bytes)
    monkeypatch.setattr(Path, "read_text", fixture_text)
    return documents, contents


def bind_report(record, details, contents):
    report = {
        "schema_version": 1,
        "record_sha256": validation.canonical_digest({key: value for key, value in record.items() if key != "report"}),
        "details": details,
    }
    path = f"tests/contracts/_evidence_fixture/{record['id']}.json"
    content = json.dumps(report, sort_keys=True).encode()
    contents[ROOT / path] = content
    record["report"] = {"path": path, "sha256": validation.digest(content)}


def review():
    return {"status": "passed", "reviewer": "synthetic-independent-reviewer",
            "independent": True, "rationale": "Synthetic control, not production acceptance evidence."}


def reviewed_transfer(workspace):
    documents, contents = workspace
    transfer = documents["test_transfer"]
    row = transfer["nodes"][0]
    new_path = "tests/contracts/_evidence_fixture/test_mapped.py"
    new_source = b"def test_mapped(value, reference):\n    assert value == reference\n"
    contents[ROOT / new_path] = new_source
    original_path = row["original_node_id"].partition("::")[0]
    original = next(item for item in transfer["files"] if item["path"] == original_path)
    row.update(disposition="transferred", case_ids=["fixture:case-1", "fixture:case-2"],
               new_test_id=new_path + "::test_mapped", evidence_ids=["fixture-transfer"])
    transfer["statistics"]["transferred"] += 1
    record = {
        "id": "fixture-transfer", "kind": "transfer_reviewed",
        "original_node_id": row["original_node_id"], "original_sha256": row["sha256"],
        "assertion_ids": row["assertion_ids"].copy(), "case_ids": row["case_ids"].copy(),
        "required_case_ids": row["case_ids"].copy(), "coverage": "complete",
        "new_test_id": row["new_test_id"], "new_test_sha256": validation.digest(new_source),
        "source_bindings": [{"path": original_path, "sha256": original["sha256"]},
                            {"path": new_path, "sha256": validation.digest(new_source)}],
        "review": review(), "report": None, "supersession_justification": None,
    }
    bind_report(record, {
        "assertion_mapping_review": "Synthetic complete assertion mapping review.",
        "case_coverage_review": "Synthetic reviewed two-case expansion.",
        "oracle_tolerance_review": "Synthetic preservation review, not oracle qualification.",
    }, contents)
    documents["evidence_registry"]["records"].append(record)
    return row, record


def mathematical_reports(workspace, subject_type="capability"):
    documents, contents = workspace
    scope = documents["phase1_scope"]
    row = scope["capabilities"][0] if subject_type == "capability" else documents["math_owners"]["owners"][0]
    source_path = "core/src/_evidence_fixture/equations.hpp"
    source = b"// Synthetic equation source for validator controls only.\n"
    contents[ROOT / source_path] = source
    binding = {"path": source_path, "symbol": "fixture::equations", "source_sha256": validation.digest(source)}
    affected = [row] if subject_type == "owner" else [
        owner for owner in documents["math_owners"]["owners"] if owner["id"] in row["owner_ids"]
    ]
    for owner in affected:
        owner["canonical_owner"].update(status="bound", implementation_binding={**binding, "symbol": "fixture::" + owner["id"]})
    documents["math_owners"]["statistics"]["applied_bindings"] = sum(
        owner["canonical_owner"]["implementation_binding"] is not None for owner in documents["math_owners"]["owners"]
    )
    row["status"] = "mathematically_qualified"
    documents["phase1_scope" if subject_type == "capability" else "math_owners"]["statistics"]["qualified"] += 1
    gates = sorted(validation.REQUIRED_GATES if subject_type == "capability" else {
        "single_owner", "objective", "derivatives", "range", "references",
    })
    records = []
    for gate in gates:
        record = {
            "id": "fixture-math-" + gate, "kind": "mathematical_gate", "subject_type": subject_type,
            "subject_id": row["id"], "subject_sha256": validation.subject_digest(row),
            "gate_id": gate, "outcome": "passed", "case_ids": ["fixture:binary64-case"],
            "check_ids": row.get("independent_checks", []),
            "source_bindings": [{"path": source_path, "sha256": validation.digest(source)}],
            "review": review(), "report": None,
        }
        details = {
            "id": record["id"], "capability_ids": [row["id"]] if subject_type == "capability" else [],
            "gate_ids": [gate], "case_ids": record["case_ids"], "limitations": [],
            "independent_review": record["review"],
            "schema_and_math_versions": {"schema": 1, "math": "synthetic-v1"},
            "source_and_build_hashes": {"source": record["source_bindings"], "build": "synthetic-build"},
            "compiler_linker_drivers_and_flags": {"compiler": "synthetic-compiler", "linker": "synthetic-linker"},
            "binary_backend_dispatch_identity": {"binary": validation.digest(source), "dispatch": "synthetic-fixed"},
            "original_coordinates_and_actual_theta": {"theta_hex": ["0x1.0p-1"], "original_z_hex": ["0x0.0p+0"]},
            "sample_histories_seed_rows_units": {"rows": [0, 1], "n_eff": 2, "seed_hex": "0x1.0p+0", "units": "identity"},
            "requested_and_valid_masks_with_reasons": {"requested": list(range(1, 8)), "valid": list(range(1, 8))},
            "oracle_provenance_precision": {"method": "synthetic-AD", "precision_bits": 128},
            "tolerances_and_raw_differences": {"F": [1e-12, 0.0], "G": [1e-12, 0.0], "H": [1e-12, 0.0]},
            "frames_and_allocations": {"max_frame_bytes": 1024, "request_heap_allocations": 0},
            "paired_request_timings_consumed_outputs": {"pairs": [[0.001, 0.002], [0.002, 0.001]], "consumed_masks": list(range(1, 8))},
        }
        bind_report(record, details, contents)
        records.append(record)
    row["evidence_ids"] = [record["id"] for record in records]
    documents["evidence_registry"]["records"].extend(records)
    return row, records


def test_incomplete_statuses_and_pending_case_mappings_can_advance(inventory):
    documents = deepcopy(inventory)
    for name in ("phase1_scope", "test_transfer", "math_owners"):
        documents[name]["status"] = "incomplete"
    documents["phase1_scope"]["capabilities"][0]["status"] = "incomplete"
    documents["math_owners"]["owners"][0]["status"] = "incomplete"
    documents["phase1_scope"]["debt"][0]["status"] = "incomplete"
    row = documents["test_transfer"]["nodes"][0]
    row.update(case_ids=["case:pending-review"], new_test_id="tests/proposed.py::test_case",
               new_adapter="proposed_garch_adapter", required_case_expansion="pending_review")
    validate_ledgers(documents)
    assert row["disposition"] == "deferred"


def test_reviewed_partial_transfer_is_not_full_jet_qualification(evidence_workspace):
    documents, _ = evidence_workspace
    original_scope = deepcopy(documents["phase1_scope"])
    row, _ = reviewed_transfer(evidence_workspace)
    validate_ledgers(documents, ROOT)
    assert row["disposition"] == "transferred"
    assert documents["phase1_scope"] == original_scope


def test_pending_transfer_review_can_be_retained_without_claim(evidence_workspace):
    documents, _ = evidence_workspace
    row, record = reviewed_transfer(evidence_workspace)
    row["disposition"] = "deferred"
    documents["test_transfer"]["statistics"]["transferred"] -= 1
    record["review"]["status"] = "pending_review"
    record["coverage"] = "pending_review"
    record["report"] = None
    validate_ledgers(documents)


@pytest.mark.parametrize("subject_type", ["capability", "owner"])
def test_structured_mathematical_evidence_can_support_partial_qualification(evidence_workspace, subject_type):
    documents, _ = evidence_workspace
    original_phase_status = documents["phase1_scope"]["status"]
    row, _ = mathematical_reports(evidence_workspace, subject_type)
    validate_ledgers(documents, ROOT)
    assert row["status"] == "mathematically_qualified"
    assert documents["phase1_scope"]["status"] == original_phase_status


@pytest.mark.parametrize("field,value,match", [
    ("coverage", "pending_review", "coverage is still pending"),
    ("assertion_ids", [], "incomplete assertion coverage"),
    ("case_ids", [], "incomplete reviewed case coverage"),
    ("original_sha256", "0" * 64, "original node identity"),
    ("new_test_sha256", "0" * 64, "report/record identity"),
])
def test_unsubstantiated_transfer_rejected(evidence_workspace, field, value, match):
    documents, _ = evidence_workspace
    _, record = reviewed_transfer(evidence_workspace)
    record[field] = value
    with pytest.raises(ContractError, match=match):
        validate_ledgers(documents, ROOT)


@pytest.mark.parametrize("outcome", ["failed", "pending_review", "skipped", "xfail"])
def test_unsatisfied_mathematical_gate_rejected(evidence_workspace, outcome):
    documents, _ = evidence_workspace
    _, records = mathematical_reports(evidence_workspace)
    records[0]["outcome"] = outcome
    with pytest.raises(ContractError, match="cannot qualify"):
        validate_ledgers(documents, ROOT)


def test_missing_mathematical_gate_rejected(evidence_workspace):
    documents, _ = evidence_workspace
    row, _ = mathematical_reports(evidence_workspace)
    row["evidence_ids"].pop()
    with pytest.raises(ContractError, match="missing mathematical gates"):
        validate_ledgers(documents, ROOT)


def test_transfer_review_cannot_prove_mathematical_qualification(evidence_workspace):
    documents, _ = evidence_workspace
    _, record = reviewed_transfer(evidence_workspace)
    row = documents["phase1_scope"]["capabilities"][0]
    row.update(status="mathematically_qualified", evidence_ids=[record["id"]])
    documents["phase1_scope"]["statistics"]["qualified"] += 1
    with pytest.raises(ContractError, match="explicit mathematical_gate"):
        validate_ledgers(documents, ROOT)


def test_claim_requires_independent_review(evidence_workspace):
    documents, _ = evidence_workspace
    _, record = reviewed_transfer(evidence_workspace)
    record["review"]["independent"] = False
    with pytest.raises(ContractError, match="passed independent review"):
        validate_ledgers(documents, ROOT)


def test_current_evidence_source_drift_invalidates_claim(evidence_workspace):
    documents, contents = evidence_workspace
    _, records = mathematical_reports(evidence_workspace)
    contents[ROOT / records[0]["source_bindings"][0]["path"]] += b"// changed\n"
    with pytest.raises(ContractError, match="evidence artifact drift"):
        validate_ledgers(documents, ROOT)


def test_subject_contract_drift_invalidates_claim(evidence_workspace):
    documents, _ = evidence_workspace
    row, _ = mathematical_reports(evidence_workspace)
    row["debt_ids"] = row["debt_ids"][:-1]
    with pytest.raises(ContractError, match="subject contract drift"):
        validate_ledgers(documents, ROOT)


def test_bare_string_typed_registry_entry_rejected(evidence_workspace):
    documents, _ = evidence_workspace
    documents["evidence_registry"]["records"] = ["ad:passed"]
    with pytest.raises(ContractError, match="typed evidence registry schema violation"):
        validate_ledgers(documents)


def test_provisional_links_are_debt_not_acceptance_evidence(inventory):
    documents = deepcopy(inventory)
    row = documents["test_transfer"]["nodes"][0]
    row["evidence_ids"] = ["planned-review-not-yet-registered"]
    limitations = validate_ledgers(documents)
    assert limitations and "provisional evidence links unresolved" in limitations[0]
    row.update(disposition="transferred", case_ids=["case:proposed"], new_test_id="tests/proposed.py::test_case")
    documents["test_transfer"]["statistics"]["transferred"] += 1
    with pytest.raises(ContractError, match="unresolved evidence IDs"):
        validate_ledgers(documents, ROOT)


def test_unresolved_math_evidence_cannot_qualify(inventory):
    documents = deepcopy(inventory)
    row = documents["phase1_scope"]["capabilities"][0]
    row.update(status="mathematically_qualified", evidence_ids=["unsupported"])
    documents["phase1_scope"]["statistics"]["qualified"] += 1
    with pytest.raises(ContractError, match="unresolved evidence IDs"):
        validate_ledgers(documents, ROOT)


def test_transfer_review_cannot_establish_fixed_execution(evidence_workspace):
    documents, _ = evidence_workspace
    _, transfer_record = reviewed_transfer(evidence_workspace)
    row, _ = mathematical_reports(evidence_workspace)
    row["fixed_execution_evidence"] = [transfer_record["id"]]
    with pytest.raises(ContractError, match="fixed evidence must bind"):
        validate_ledgers(documents, ROOT)


def test_missing_owner_binding_blocks_mathematical_claim(evidence_workspace):
    documents, _ = evidence_workspace
    row, _ = mathematical_reports(evidence_workspace)
    owner = next(owner for owner in documents["math_owners"]["owners"] if owner["id"] in row["owner_ids"])
    owner["canonical_owner"].update(status="pending", implementation_binding=None)
    documents["math_owners"]["statistics"]["applied_bindings"] -= 1
    with pytest.raises(ContractError, match="needs owner implementation bindings"):
        validate_ledgers(documents, ROOT)


def test_supersession_requires_reviewed_justification(evidence_workspace):
    documents, contents = evidence_workspace
    row, record = reviewed_transfer(evidence_workspace)
    row["disposition"] = "superseded"
    documents["test_transfer"]["statistics"]["transferred"] -= 1
    documents["test_transfer"]["statistics"]["superseded"] += 1
    with pytest.raises(ContractError, match="supersession requires independent justification"):
        validate_ledgers(documents, ROOT)
    record["supersession_justification"] = "Synthetic independently reviewed stronger replacement of each mapped assertion."
    path = ROOT / record["report"]["path"]
    details = json.loads(contents[path])["details"]
    bind_report(record, details, contents)
    validate_ledgers(documents, ROOT)


def test_known_skip_cannot_establish_transfer_coverage(evidence_workspace):
    documents, contents = evidence_workspace
    _, record = reviewed_transfer(evidence_workspace)
    source_path = record["new_test_id"].partition("::")[0]
    source = b"import pytest\n\ndef test_mapped():\n    pytest.skip('required oracle missing')\n"
    contents[ROOT / source_path] = source
    record["new_test_sha256"] = validation.digest(source)
    record["source_bindings"][-1]["sha256"] = validation.digest(source)
    details = json.loads(contents[ROOT / record["report"]["path"]])["details"]
    bind_report(record, details, contents)
    with pytest.raises(ContractError, match="skip/xfail cannot establish"):
        validate_ledgers(documents, ROOT)
