"""Mutation controls prove that structural validation fails closed."""

from copy import deepcopy
import builtins
import json
from pathlib import Path

import pytest

import ledger_validation as validation
from ledger_validation import ContractError, load_ledgers, validate_ledgers


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inventory():
    return load_ledgers(ROOT)


@pytest.mark.parametrize("ledger,path,value,match", [
    ("phase1_scope", ("capabilities", 0, "id"), "unknown", "reverse consumer mismatch"),
    ("phase1_scope", ("capabilities", 0, "owner_ids"), ["missing"], "unknown or missing owner"),
    ("phase1_scope", ("capabilities", 0, "owner_ids"), ["layout", "layout"], "duplicate ID"),
    ("phase1_scope", ("capabilities", 0, "debt_ids"), ["missing"], "unknown debt"),
    ("phase1_scope", ("capabilities", 0, "required_gate_ids"), ["scope"], "missing required gates"),
    ("phase1_scope", ("capabilities", 0, "request_masks"), [1, 2, 3, 4, 5, 6], "schema violation"),
    ("phase1_scope", ("capabilities", 0, "request_masks"), [True, 2, 3, 4, 5, 6, 7], "schema violation"),
    ("phase1_scope", ("capabilities", 0, "request_masks"), [1, 2, 3, 4, 5, 6, 6], "schema violation"),
    ("phase1_scope", ("capabilities", 0, "order_axes"), [], "order axes mismatch"),
    ("phase1_scope", ("capabilities", 0, "order_axes", 0, "minimum"), 0, "order axes mismatch"),
    ("phase1_scope", ("capabilities", 0, "coordinates"), ["theta"], "coordinate promise"),
    ("phase1_scope", ("capabilities", 0, "order_requirement"), "11 only", "general-order requirement"),
    ("phase1_scope", ("capabilities", 0, "sample_contract"), "missing", "unknown sample contract"),
    ("phase1_scope", ("capabilities", 0, "density"), "studentt", "composition identity"),
    ("phase1_scope", ("capabilities", 0, "status"), "mathematically_qualified", "schema violation"),
    ("phase1_scope", ("status",), "complete", "Phase1 closure"),
    ("phase1_scope", ("closure_policy", "required_skip_or_xfail"), "pass", "closure policy weakened"),
    ("phase1_scope", ("closure_policy", "catalogue_status_inheritance"), "allowed", "inheritance forbidden"),
    ("phase1_scope", ("schema_version",), "2.0.0", "unsupported schema version"),
    ("phase1_scope", ("statistics", "qualified"), 1, "statistics mismatch"),
    ("math_owners", ("owners", 0, "id"), "unknown", "shared owner inventory"),
    ("math_owners", ("owners", 0, "phase1_consumer_ids"), [], "reverse consumer mismatch"),
    ("math_owners", ("owners", 0, "phase1_consumer_ids"), ["unknown"], "reverse consumer mismatch"),
    ("math_owners", ("owners", 0, "canonical_owner", "status"), "qualified", "unevidenced canonical owner"),
    ("math_owners", ("inventory_id",), "different", "inventory identity"),
    ("math_owners", ("authority", "promised_scope"), "other.json", "conflicting authorities"),
    ("test_transfer", ("nodes", 0, "candidate_capability_set_id"), "missing", "unknown candidate set"),
    ("test_transfer", ("candidate_capability_sets", 0, "capability_ids"), ["missing"], "unknown or empty candidate"),
    ("test_transfer", ("nodes", 0, "disposition"), "transferred", "schema violation"),
    ("test_transfer", ("support_units", 0, "disposition"), "xfail", "skip/xfail"),
    ("test_transfer", ("native_tests", 0, "disposition"), "skip", "skip/xfail"),
    ("test_transfer", ("nodes", 0, "blocking_phase"), 2, "Phase1 blocker"),
    ("test_transfer", ("nodes", 0, "original_node_id"), "tests/missing.py::test_x", "unknown original file"),
    ("test_transfer", ("nodes", 0, "line"), 0, "invalid source span"),
    ("test_transfer", ("files", 0, "sha256"), "not-a-hash", "invalid snapshot hash"),
    ("test_transfer", ("files", 0, "snapshot_kind"), "HEAD", "unsupported snapshot kind"),
    ("test_transfer", ("statistics", "transferred"), 1, "statistics mismatch"),
    ("source_transfers", ("schema_version",), True, "schema violation"),
    ("source_transfers", ("transfers", 0, "source_sha256"), "not-a-hash", "schema violation"),
    ("source_transfers", ("status",), "qualified", "schema violation"),
    ("source_transfers", ("transfers", 0, "destination"), "core/src/../../../outside", "unsafe repository path"),
    ("core_adaptations", ("schema_version",), "1", "schema violation"),
    ("core_adaptations", ("owners", 0, "sha256"), "not-a-hash", "schema violation"),
    ("core_adaptations", ("owners", 0, "status"), "qualified", "schema violation"),
    ("core_adaptations", ("owners", 0, "path"), "core/src/../../../outside", "unsafe repository path"),
    ("core_adaptations", ("owners", 0, "source_owners"), [], "schema violation"),
    ("core_adaptations", ("owners", 0, "consumer_sources"), ["tests/fake.cpp"], "invalid provenance path"),
    ("core_adaptations", ("owners", 0, "regression_tests"), ["localdev/evidence.py"], "ignored artifact"),
])
def test_mutated_contract_rejected(inventory, ledger, path, value, match):
    documents = deepcopy(inventory)
    target = documents[ledger]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ContractError, match=match):
        validate_ledgers(documents)


@pytest.mark.parametrize("ledger,collection", [
    ("phase1_scope", "capabilities"), ("phase1_scope", "debt"),
    ("math_owners", "owners"), ("test_transfer", "nodes"),
    ("test_transfer", "files"), ("test_transfer", "candidate_capability_sets"),
    ("source_transfers", "transfers"),
    ("core_adaptations", "owners"),
])
def test_duplicate_records_rejected(inventory, ledger, collection):
    documents = deepcopy(inventory)
    rows = documents[ledger][collection]
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ContractError, match="duplicate ID"):
        validate_ledgers(documents)


def test_schema_cannot_weaken_its_own_validation(inventory):
    documents = deepcopy(inventory)
    documents["phase1_scope"]["schema"] = {"type": "object"}
    with pytest.raises(ContractError, match="schema changed"):
        validate_ledgers(documents)


def test_missing_required_field_rejected_by_jsonschema(inventory):
    documents = deepcopy(inventory)
    del documents["phase1_scope"]["capabilities"][0]["required_gate_ids"]
    with pytest.raises(ContractError, match="schema violation"):
        validate_ledgers(documents)


def test_coherent_scope_deletion_still_rejected(inventory):
    documents = deepcopy(inventory)
    removed = documents["phase1_scope"]["capabilities"].pop()
    for owner in documents["math_owners"]["owners"]:
        if removed["id"] in owner["phase1_consumer_ids"]:
            owner["phase1_consumer_ids"].remove(removed["id"])
    for candidate in documents["test_transfer"]["candidate_capability_sets"]:
        if removed["id"] in candidate["capability_ids"]:
            candidate["capability_ids"].remove(removed["id"])
    with pytest.raises(ContractError, match="promised scope changed"):
        validate_ledgers(documents)


@pytest.mark.parametrize("identifier,axis", [
    ("mean:arma+vol:garch+density:normal", "q_ma"),
    ("mean:arx+vol:garch+density:normal", "design_columns"),
    ("mean:harx+vol:egarch+density:normal", "horizons"),
    ("math:normal_profile+mean:arma", "p_ar"),
    ("math:dcc+law:studentt", "series"),
])
def test_missing_joint_design_profile_or_correlation_axis_rejected(inventory, identifier, axis):
    documents = deepcopy(inventory)
    row = next(row for row in documents["phase1_scope"]["capabilities"] if row["id"] == identifier)
    row["order_axes"] = [item for item in row["order_axes"] if item["name"] != axis]
    with pytest.raises(ContractError, match="order axes mismatch"):
        validate_ledgers(documents)


@pytest.mark.parametrize("ledger,collection", [("phase1_scope", "capabilities"), ("math_owners", "owners")])
@pytest.mark.parametrize("evidence_id", ["catalogue:complete", "pytest:skipped", "pytest:xfail", "ad:passed"])
def test_evidence_label_does_not_establish_qualification(inventory, ledger, collection, evidence_id):
    documents = deepcopy(inventory)
    row = documents[ledger][collection][0]
    row["status"] = "mathematically_qualified"
    row["evidence_ids"] = [evidence_id]
    documents["phase1_scope"]["evidence_records"] = [evidence_id]
    documents[ledger]["statistics"]["qualified"] = 1
    with pytest.raises(ContractError, match="bare string evidence records"):
        validate_ledgers(documents)


def test_transferred_node_requires_verifiable_evidence(inventory):
    documents = deepcopy(inventory)
    row = documents["test_transfer"]["nodes"][0]
    row.update(disposition="transferred", case_ids=["case:1"], new_test_id="tests/new.py::test_x", evidence_ids=["pytest:xfail"])
    documents["test_transfer"]["statistics"]["transferred"] = 1
    documents["phase1_scope"]["evidence_records"] = ["pytest:xfail"]
    with pytest.raises(ContractError, match="bare string evidence records"):
        validate_ledgers(documents)


def test_implementation_binding_does_not_imply_qualification(inventory):
    documents = deepcopy(inventory)
    owner = documents["math_owners"]["owners"][0]
    owner["canonical_owner"].update(
        status="bound",
        implementation_binding={"path": "core/src/fixture_owner.hpp", "symbol": "fixture::owner", "source_sha256": "0" * 64},
    )
    documents["math_owners"]["statistics"]["applied_bindings"] = sum(
        row["canonical_owner"]["implementation_binding"] is not None
        for row in documents["math_owners"]["owners"]
    )
    validate_ledgers(documents)
    assert owner["status"] == "pending"
    assert owner["evidence_ids"] == []


@pytest.mark.parametrize("text", ['{"id":1,"id":2}', '{"nested":{"id":1,"id":2}}', '{"value":NaN}'])
def test_ambiguous_json_rejected(text):
    with pytest.raises(ContractError):
        json.loads(text, object_pairs_hook=validation.strict_object, parse_constant=validation.reject_nonfinite)


@pytest.mark.parametrize("field,value,match", [
    ("sha256", "0" * 64, "retained node hash drift"),
    ("line", 67, "source span drift"),
    ("parametrization", [], "decorator drift"),
    ("assertion_ids", [], "assertion inventory drift"),
])
def test_retained_node_mutations_rejected(inventory, field, value, match):
    transfer = deepcopy(inventory["test_transfer"])
    transfer["nodes"][0][field] = value
    with pytest.raises(ContractError, match=match):
        validation.validate_retained_test_sources(transfer, ROOT)


def test_retained_file_byte_mutation_rejected(inventory, monkeypatch):
    read = validation.retained_bytes
    path = inventory["test_transfer"]["files"][0]["path"]
    monkeypatch.setattr(validation, "retained_bytes", lambda root, relative: read(root, relative) + (b"\n" if relative == path else b""))
    with pytest.raises(ContractError, match="retained snapshot drift"):
        validation.validate_retained_test_sources(inventory["test_transfer"], ROOT)


@pytest.fixture
def source_snapshot(inventory, monkeypatch):
    # Synthetic bytes exercise snapshot lifetimes without pinning live implementations.
    root = ROOT / "tests/contracts/_synthetic_snapshot"
    original = b'#include "original.hpp"\nint fixture_value = 1;\n'
    relocated = b'#include "../math/original.hpp"\nint fixture_value = 1;\n'
    current = b'#include "../math/original.hpp"\nint fixture_value = 2;\n'
    row = {
        "source": "scivol/fixture.hpp", "destination": "core/src/fixture.hpp",
        "source_sha256": validation.digest(original),
        "relocated_sha256": validation.digest(relocated),
        "current_sha256": validation.digest(current),
        "adaptation": "Synthetic relocation followed by synthetic current adaptation.",
        "includes": [{"source": "original.hpp", "destination": "../math/original.hpp"}],
    }
    document = deepcopy(inventory["source_transfers"])
    document["transfers"] = [row]
    document["discovery"]["seeds"] = [row["source"]]
    contents = {root / row["source"]: original, root / row["destination"]: current}
    read_bytes, is_file = Path.read_bytes, Path.is_file

    def fixture_bytes(path):
        return contents[path] if path in contents else read_bytes(path)

    def fixture_exists(path):
        return True if path in contents else is_file(path)

    monkeypatch.setattr(Path, "read_bytes", fixture_bytes)
    monkeypatch.setattr(Path, "is_file", fixture_exists)
    return document, root


def test_original_relocation_and_current_adaptation_are_distinct(source_snapshot):
    document, root = source_snapshot
    row = document["transfers"][0]
    assert row["relocated_sha256"] != row["current_sha256"]
    assert validation.validate_source_snapshots(document, root, original_snapshot_root=root) == ()


@pytest.mark.parametrize("field,match", [
    ("source_sha256", "pinned original snapshot"),
    ("relocated_sha256", "original relocation hash mismatch"),
    ("current_sha256", "current snapshot drift"),
])
def test_snapshot_hash_mutations_rejected(source_snapshot, field, match):
    document, root = source_snapshot
    document["transfers"][0][field] = "0" * 64
    with pytest.raises(ContractError, match=match):
        validation.validate_source_snapshots(document, root, original_snapshot_root=root)


def test_include_relocation_mutation_rejected(source_snapshot):
    document, root = source_snapshot
    document["transfers"][0]["includes"][0]["destination"] = "wrong.hpp"
    with pytest.raises(ContractError, match="original relocation hash mismatch"):
        validation.validate_source_snapshots(document, root)


def test_conflicting_include_relocations_rejected(source_snapshot):
    document, root = source_snapshot
    document["transfers"][0]["includes"].append({"source": "original.hpp", "destination": "wrong.hpp"})
    with pytest.raises(ContractError, match="conflicting include relocation"):
        validation.validate_source_snapshots(document, root)


def test_unretained_original_is_reported_not_replaced_by_current_bytes(source_snapshot):
    document, root = source_snapshot
    document["transfers"][0]["source_sha256"] = "0" * 64
    limitations = validation.validate_source_snapshots(document, root)
    assert len(limitations) == 1
    assert "original pinned bytes unavailable" in limitations[0]
    assert "relocation integrity unverified" in limitations[0]


def test_explicit_original_snapshot_cannot_be_missing(source_snapshot, monkeypatch):
    document, root = source_snapshot
    source = root / document["transfers"][0]["source"]
    is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: False if path == source else is_file(path))
    with pytest.raises(ContractError, match="pinned original snapshot missing or corrupt"):
        validation.validate_source_snapshots(document, root, original_snapshot_root=root)


def test_current_destination_byte_mutation_rejected(source_snapshot, monkeypatch):
    document, root = source_snapshot
    destination = root / document["transfers"][0]["destination"]
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"corrupt" if path == destination else read_bytes(path))
    with pytest.raises(ContractError, match="current snapshot drift"):
        validation.validate_source_snapshots(document, root)


@pytest.mark.parametrize("field,value,match", [
    ("sha256", "0" * 64, "derived owner snapshot drift"),
    ("path", "core/src/missing_owner.hpp", "missing referenced file"),
    ("consumer_sources", ["core/src/missing_consumer.cpp"], "missing referenced file"),
    ("regression_tests", ["tests/missing_regression.py"], "missing referenced file"),
])
def test_derived_owner_snapshot_and_pointer_mutations_rejected(inventory, field, value, match):
    document = deepcopy(inventory["core_adaptations"])
    document["owners"][0][field] = value
    with pytest.raises(ContractError, match=match):
        validation.validate_core_adaptations(document, ROOT)


def test_duplicate_derived_owner_paths_rejected(inventory):
    documents = deepcopy(inventory)
    documents["core_adaptations"]["owners"][1]["path"] = documents["core_adaptations"]["owners"][0]["path"]
    with pytest.raises(ContractError, match="derived owner paths: duplicate ID"):
        validate_ledgers(documents)


def test_integration_status_does_not_promote_capability_qualification(inventory):
    documents = deepcopy(inventory)
    original_scope = deepcopy(documents["phase1_scope"])
    documents["core_adaptations"]["owners"][0]["status"] = "integrated"
    validate_ledgers(documents)
    assert documents["phase1_scope"] == original_scope


def test_ignored_inventory_artifacts_are_not_ci_inputs(inventory, monkeypatch):
    read_bytes, read_text = Path.read_bytes, Path.read_text

    def guarded_read_bytes(path):
        assert not any(part == "localdev" or part.startswith("localdev_") for part in path.parts)
        return read_bytes(path)

    def guarded_read_text(path, *args, **kwargs):
        assert not any(part == "localdev" or part.startswith("localdev_") for part in path.parts)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    validate_ledgers(load_ledgers(ROOT))
    validation.validate_retained_test_sources(inventory["test_transfer"], ROOT)
    validation.validate_core_adaptations(inventory["core_adaptations"], ROOT)


def test_validation_does_not_import_scivol_or_ad(inventory, source_snapshot, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in ("scivol", "jax", "jaxlib"), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    validate_ledgers(inventory)
    validation.validate_retained_test_sources(inventory["test_transfer"], ROOT)
    validation.validate_core_adaptations(inventory["core_adaptations"], ROOT)
    document, root = source_snapshot
    validation.validate_source_snapshots(document, root)


@pytest.mark.parametrize("relative", [".", "../outside", "/outside", "tests/../../outside", "localdev/cases.json", "localdev_data/cases.json"])
def test_unsafe_or_ignored_snapshot_path_rejected(relative):
    with pytest.raises(ContractError):
        validation.repository_path(ROOT, relative)
