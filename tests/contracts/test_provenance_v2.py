"""Shared-format examples are synthetic evidence, never ledger qualification."""

from copy import deepcopy
import json
from pathlib import Path
import warnings

import pytest

from ledger_validation import ContractError, digest, load_ledgers, validate_ledgers
from provenance_validation import context_digest, owner_digest, validate_provenance, validate_schema


ROOT = Path(__file__).resolve().parents[2]


def snapshot(value=None, retained_path=None, state="known"):
    return {"state": state, "sha256": digest(value) if state == "known" else None,
            "retained_path": retained_path, "reason": None if state == "known" else "No historical identity recorded"}


def source(path, value, selection=None, kind="source"):
    return {"kind": kind, "path": path, "snapshot": snapshot(value),
            "selection": selection or {"kind": "whole_file"}}


def ranges(value, start, end):
    return {"kind": "byte_ranges", "ranges": [{"start": start, "end": end, "sha256": digest(value[start:end])}]}


def owner(path, value, kind, origins, method):
    return {"destination": path, "kind": kind, "origins": origins, "materialization": method,
            "snapshots": {"relocated": snapshot(state="not_applicable") if kind == "authored" else snapshot(value),
                          "current": snapshot(value), "history": []},
            "changes": [], "context_refs": [], "review_ref": None, "status": "incomplete", "debt": []}


def fragment(owners=None, producer="fixture"):
    return {"ledger": "source_provenance", "schema_version": "2.0.0", "inventory_id": "fixture-inventory",
            "producer": producer, "owners": owners or [], "uses": [], "unassigned_origins": [],
            "context_refs": [], "debt": []}


def store(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)


def pin_review(root, document, row, report_changes=None):
    path = f"contracts/reviews/{Path(row['destination']).name}.json"
    row["review_ref"] = path
    row["status"] = "provenance_reviewed"
    contexts = {item["path"]: item for item in document["context_refs"]}
    vendors = {path: json.loads((root / path).read_bytes()) for path, item in contexts.items() if item["role"] == "vendor"}
    report = {"kind": "provenance_review", "destination": row["destination"], "record_sha256": owner_digest(row),
              "context_sha256": context_digest(row, contexts, vendors),
              "outcome": "passed", "reviewer": "independent fixture reviewer", "independent": True,
              "rationale": "Synthetic reviewed identities and source mapping"}
    report.update(report_changes or {})
    value = json.dumps(report).encode()
    store(root, path, value)
    context = {"path": path, "role": "provenance_review", "snapshot": snapshot(value)}
    document["context_refs"] = [item for item in document["context_refs"] if item["path"] != path] + [context]


@pytest.fixture
def examples(tmp_path):
    a, b, upstream = b"abcDEFghi\n", b"XYZ\n", b"// upstream\n"
    for path, value in [("legacy/a.hpp", a), ("legacy/b.hpp", b), ("vendor/upstream.hpp", upstream),
                        ("licenses/vendor.txt", b"Fixture license notice\n")]:
        store(tmp_path, path, value)
    copied = owner("core/copied.hpp", a, "copied", [source("legacy/a.hpp", a)], "verbatim")
    partitioned = owner("core/partitioned.hpp", a[3:6], "partitioned", [source("legacy/a.hpp", a, ranges(a, 3, 6))], "selected_bytes")
    composed = owner("core/composed.hpp", a[:3] + b, "composed",
                     [source("legacy/a.hpp", a, ranges(a, 0, 3)), source("legacy/b.hpp", b)], "selected_bytes")
    authored = owner("core/authored.hpp", b"// authored adapter\n", "authored", [], "authored")
    derived = owner("core/derived.hpp", b"// derived from captured parent\n", "derived",
                    [source(copied["destination"], a, kind="owner")], "reviewed_adaptation")
    vendor = owner("core/vendor.hpp", upstream, "vendor", [source("vendor/upstream.hpp", upstream)], "verbatim")
    vendor["vendor_ref"] = "contracts/vendors/fixture.json"
    first, second = fragment([copied, partitioned, composed]), fragment([authored, derived, vendor], "other producer")
    manifest = json.dumps({"package": "fixture vendor", "version": "1.0", "license_refs": ["licenses/vendor.txt"]}).encode()
    store(tmp_path, vendor["vendor_ref"], manifest)
    second["context_refs"].append({"path": vendor["vendor_ref"], "role": "vendor", "snapshot": snapshot(manifest)})
    second["context_refs"].append({"path": "licenses/vendor.txt", "role": "license", "snapshot": snapshot(b"Fixture license notice\n")})
    for document in (first, second):
        for row in document["owners"]:
            # Initial/current values in this fixture are retained at the destination.
            value = {copied["destination"]: a, partitioned["destination"]: a[3:6], composed["destination"]: a[:3] + b,
                     authored["destination"]: b"// authored adapter\n", derived["destination"]: b"// derived from captured parent\n",
                     vendor["destination"]: upstream}[row["destination"]]
            store(tmp_path, row["destination"], value)
            pin_review(tmp_path, document, row)
    second["uses"] = [{"consumer": authored["destination"], "target": {"kind": "owner", "path": copied["destination"]},
                       "role": "include", "include_spelling": "copied.hpp", "debt": []}]
    return tmp_path, {"contracts/first.json": first, "contracts/second.json": second}


def test_all_operations_and_cross_fragment_reuse_are_reviewable(examples):
    root, documents = examples
    assert validate_provenance(documents, root, "fixture-inventory") == ()


def test_many_vendor_files_reuse_one_package_and_license_authority(examples):
    root, documents = examples
    second = documents["contracts/second.json"]
    row = deepcopy(second["owners"][2])
    row["destination"] = "core/vendor_second.hpp"
    store(root, row["destination"], b"// upstream\n")
    second["owners"].append(row)
    pin_review(root, second, row)
    assert sum(item["role"] == "vendor" for item in second["context_refs"]) == 1
    assert validate_provenance(documents, root) == ()


def test_include_cycles_do_not_create_origin_ownership_cycles(examples):
    root, documents = examples
    documents["contracts/first.json"]["uses"].append({"consumer": "core/copied.hpp",
        "target": {"kind": "owner", "path": "core/authored.hpp"}, "role": "include", "debt": []})
    assert validate_provenance(documents, root) == ()


def test_incomplete_known_history_can_advance_without_phase_claim(examples):
    root, documents = examples
    row = documents["contracts/second.json"]["owners"][0]
    row["status"] = "incomplete"
    assert "provenance review pending" in " ".join(validate_provenance(documents, root))
    row["status"] = "provenance_reviewed"
    assert validate_provenance(documents, root) == ()


def test_missing_original_and_partition_are_explicit_valid_debt(tmp_path):
    row = owner("core/extracted.hpp", b"current\n", "partitioned", [
        {"kind": "source", "path": "legacy/missing.hpp", "snapshot": snapshot(state="not_recorded"),
         "selection": {"kind": "unresolved", "reason": "Original frozen bytes missing"}}
    ], "not_recorded")
    row["snapshots"]["relocated"] = snapshot(state="not_recorded")
    row["debt"] = ["Recover original frozen bytes and destination-specific partition"]
    store(tmp_path, row["destination"], b"current\n")
    document = fragment([row])
    document["unassigned_origins"] = [{"origin": deepcopy(row["origins"][0]), "reason": "Destination mapping pending"}]
    limitations = validate_provenance({"contracts/example.json": document}, tmp_path)
    assert any("original pinned bytes unavailable" in item for item in limitations)
    assert any("unassigned origin" in item for item in limitations)
    row["status"] = "provenance_reviewed"
    with pytest.raises(ContractError, match="unsupported provenance review"):
        validate_provenance({"contracts/example.json": document}, tmp_path)


def test_original_drift_is_not_rebound_to_current_bytes(examples):
    root, documents = examples
    for row in documents["contracts/first.json"]["owners"]:
        row["status"] = "incomplete"
        row["debt"] = ["Retain original pinned source snapshot"]
    documents["contracts/second.json"]["owners"][1]["status"] = "incomplete"
    documents["contracts/second.json"]["owners"][1]["debt"] = ["Parent review pending"]
    store(root, "legacy/a.hpp", b"new source bytes\n")
    before = deepcopy(documents)
    assert any("original pinned bytes unavailable" in item for item in validate_provenance(documents, root))
    assert documents == before


def test_unavailable_original_is_reported_not_required_for_incomplete_ci(tmp_path):
    row = owner("core/copied.hpp", b"old\n", "copied", [source("legacy/not-retained.hpp", b"old\n")], "verbatim")
    store(tmp_path, row["destination"], b"old\n")
    document = {"contracts/example.json": fragment([row])}
    assert any("original pinned bytes unavailable" in item for item in validate_provenance(document, tmp_path))
    assert any("current bytes not checked" in item for item in validate_provenance(document))
    row["status"] = "provenance_reviewed"
    with pytest.raises(ContractError, match="unsupported provenance review"):
        validate_provenance(document, tmp_path)


def test_ignored_historical_artifacts_are_not_ci_inputs(tmp_path):
    row = owner("core/copied.hpp", b"old\n", "copied", [source("legacy/a.hpp", b"old\n")], "verbatim")
    row["origins"][0]["snapshot"]["retained_path"] = "localdev/frozen/missing.hpp"
    row["debt"] = ["Archive source snapshot in a durable location"]
    store(tmp_path, row["destination"], b"old\n")
    document = fragment([row])
    document["context_refs"] = [{"path": "localdev/frozen/receipt.json", "role": "handoff", "snapshot": snapshot(b"ignored receipt") }]
    assert any("unavailable" in item for item in validate_provenance({"contracts/example.json": document}, tmp_path))


def test_historical_use_is_not_duplicate_authority_or_current_qualification(examples):
    root, documents = examples
    first, second = documents.values()
    row = first["owners"][0]
    original, new = b"abcDEFghi\n", b"new current bytes\n"
    store(root, "contracts/snapshots/copied.initial", original)
    row["snapshots"]["relocated"]["retained_path"] = "contracts/snapshots/copied.initial"
    row["snapshots"]["current"] = snapshot(new)
    row["changes"] = [{"from_sha256": digest(original), "to_sha256": digest(new), "description": "Reviewed later edit"}]
    store(root, row["destination"], new)
    pin_review(root, first, row)
    second["uses"][0]["observation"] = snapshot(original)
    # The derived origin resolves the parent's retained initial version, not its new bytes.
    limitations = validate_provenance(documents, root)
    assert limitations == ("core/authored.hpp: historical use of core/copied.hpp; not current qualification",)


def test_external_inputs_and_missing_owner_versions_are_explicit_debt(examples):
    root, documents = examples
    second = documents["contracts/second.json"]
    second["uses"].append({"consumer": "core/authored.hpp", "target": {"kind": "external", "name": "libm",
                            "version": "compiler-provided", "reason": "System dependency outside repository ownership"},
                           "role": "link_input", "debt": []})
    second["uses"][0]["observation"] = snapshot(b"unknown captured version")
    second["uses"][0]["debt"] = ["Recover captured owner version"]
    assert any("observed owner version unresolved" in item for item in validate_provenance(documents, root))


@pytest.mark.parametrize("path,value,match", [
    (("schema_version",), "3.0.0", "schema violation"),
    (("owners", 0, "status"), "mathematically_qualified", "schema violation"),
    (("owners", 0, "snapshots", "current", "sha256"), "x", "schema violation"),
    (("owners", 0, "destination"), "core/../outside", "unsafe repository path"),
    (("owners", 0, "destination"), "core//copied.hpp", "noncanonical repository path"),
    (("owners", 0, "destination"), ".", "noncanonical repository path"),
    (("owners", 0, "destination"), "/outside", "unsafe repository path"),
    (("owners", 0, "snapshots", "current", "state"), "not_recorded", "schema violation"),
    (("owners", 0, "origins", 0, "snapshot", "sha256"), "0" * 64, "unsupported provenance review"),
    (("owners", 1, "origins", 0, "selection", "ranges", 0, "start"), 6, "invalid or overlapping byte ranges"),
    (("owners", 1, "origins", 0, "selection", "ranges", 0, "end"), 99, "exceeds pinned original"),
    (("owners", 1, "origins", 0, "selection", "ranges", 0, "sha256"), "0" * 64, "fragment hash mismatch"),
    (("owners", 0, "snapshots", "relocated", "sha256"), "0" * 64, "materialization hash mismatch"),
    (("owners", 0, "review_ref"), None, "retained typed provenance review required"),
    (("owners", 0, "debt"), ["Review still missing"], "unsupported provenance review"),
])
def test_mutations_fail_closed(examples, path, value, match):
    root, documents = examples
    target = documents["contracts/first.json"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ContractError, match=match):
        validate_provenance(documents, root)


@pytest.mark.parametrize("report_changes,match", [
    ({"outcome": "xfail"}, "schema violation"),
    ({"independent": False}, "schema violation"),
    ({"record_sha256": "0" * 64}, "stale or misbound"),
    ({"destination": "core/other.hpp"}, "stale or misbound"),
])
def test_review_must_pass_and_bind_exact_owner_contract(examples, report_changes, match):
    root, documents = examples
    first = documents["contracts/first.json"]
    pin_review(root, first, first["owners"][0], report_changes)
    with pytest.raises(ContractError, match=match):
        validate_provenance(documents, root)


def test_updated_vendor_manifest_requires_new_provenance_review(examples):
    root, documents = examples
    second = documents["contracts/second.json"]
    context = next(item for item in second["context_refs"] if item["role"] == "vendor")
    manifest = json.loads((root / context["path"]).read_bytes())
    manifest["version"] = "2.0"
    value = json.dumps(manifest).encode()
    store(root, context["path"], value)
    context["snapshot"] = snapshot(value)
    with pytest.raises(ContractError, match="stale or misbound provenance review"):
        validate_provenance(documents, root)


@pytest.mark.parametrize("collection", ["owners", "context_refs", "uses"])
def test_duplicate_authority_or_use_rejected_even_identical(examples, collection):
    root, documents = examples
    first, second = documents.values()
    if collection == "uses":
        first[collection].append(deepcopy(second[collection][0]))
    else:
        second[collection].append(deepcopy(first[collection][0]))
    with pytest.raises(ContractError, match="duplicate"):
        validate_provenance(documents, root)


def test_use_cannot_restate_current_hash_authority(examples):
    root, documents = examples
    documents["contracts/second.json"]["uses"][0]["current_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="schema violation"):
        validate_provenance(documents, root)


def test_current_archive_cannot_hide_destination_drift(examples):
    root, documents = examples
    row = documents["contracts/first.json"]["owners"][0]
    store(root, "contracts/snapshots/current", b"abcDEFghi\n")
    row["snapshots"]["current"]["retained_path"] = "contracts/snapshots/current"
    store(root, row["destination"], b"changed destination\n")
    with pytest.raises(ContractError, match="current snapshot drift"):
        validate_provenance(documents, root)


def test_overlapping_partitions_and_unpinned_full_originals_rejected(examples):
    root, documents = examples
    origin = documents["contracts/first.json"]["owners"][1]["origins"][0]
    origin["selection"]["ranges"].append(deepcopy(origin["selection"]["ranges"][0]))
    with pytest.raises(ContractError, match="overlapping byte ranges"):
        validate_provenance(documents, root)
    origin["selection"]["ranges"].pop()
    origin["snapshot"] = snapshot(state="not_recorded")
    with pytest.raises(ContractError, match="require a pinned full original"):
        validate_provenance(documents, root)


def test_pending_review_paths_are_still_structurally_checked(examples):
    root, documents = examples
    row = documents["contracts/second.json"]["owners"][0]
    row.update(status="incomplete", review_ref="../outside")
    with pytest.raises(ContractError, match="unsafe repository path"):
        validate_provenance(documents, root)


@pytest.mark.parametrize("mutation,match", [
    ("current", "current snapshot drift"), ("retained", "pinned snapshot drift"),
    ("license", "current snapshot drift"), ("review", "schema violation"),
])
def test_changed_bytes_and_bare_review_labels_rejected(examples, mutation, match):
    root, documents = examples
    row = documents["contracts/first.json"]["owners"][0]
    if mutation == "current":
        store(root, row["destination"], b"drift\n")
    elif mutation == "retained":
        row["origins"][0]["snapshot"]["retained_path"] = "contracts/snapshots/bad.original"
        store(root, "contracts/snapshots/bad.original", b"wrong original\n")
    elif mutation == "license":
        store(root, "licenses/vendor.txt", b"new notice\n")
    else:
        value = b'"approved"'
        store(root, row["review_ref"], value)
        context = next(item for item in documents["contracts/first.json"]["context_refs"] if item["path"] == row["review_ref"])
        context["snapshot"] = snapshot(value)
    with pytest.raises(ContractError, match=match):
        validate_provenance(documents, root)


@pytest.mark.parametrize("mutation,match", [
    ("authored", "authored files cannot claim copied origins"),
    ("partition", "partitioned origin needs"),
    ("license", "vendor license must reference"),
    ("chain", "disconnected or reordered"),
    ("cycle", "cyclic origin ownership"),
    ("missing_parent", "unsupported provenance review"),
    ("incomplete_debt", "missing provenance needs explicit debt"),
])
def test_invalid_operations_or_claims_rejected(examples, mutation, match):
    root, documents = examples
    first, second = documents.values()
    if mutation == "authored":
        second["owners"][0]["origins"] = [deepcopy(first["owners"][0]["origins"][0])]
    elif mutation == "partition":
        first["owners"][1]["origins"][0]["selection"] = {"kind": "whole_file"}
    elif mutation == "license":
        value = json.dumps({"package": "fixture vendor", "version": "1.0", "license_refs": ["missing/license.txt"]}).encode()
        store(root, "contracts/vendors/fixture.json", value)
        next(item for item in second["context_refs"] if item["role"] == "vendor")["snapshot"] = snapshot(value)
    elif mutation == "chain":
        first["owners"][0]["changes"] = [{"from_sha256": "0" * 64, "to_sha256": "1" * 64, "description": "Wrong endpoints"}]
    elif mutation == "cycle":
        first["owners"][0]["origins"][0].update(kind="owner", path="core/derived.hpp")
    elif mutation == "missing_parent":
        second["owners"][1]["origins"][0]["snapshot"]["sha256"] = "0" * 64
    else:
        first["owners"][0]["status"] = "incomplete"
        first["owners"][0]["origins"][0].update(kind="owner", path="core/unregistered.hpp")
    with pytest.raises(ContractError, match=match):
        validate_provenance(documents, root)


def test_review_without_byte_verification_cannot_pass(examples):
    _, documents = examples
    with pytest.raises(ContractError, match="unsupported provenance review"):
        validate_provenance(documents)


def test_unknown_refs_need_explicit_debt(examples):
    root, documents = examples
    documents["contracts/second.json"]["uses"][0]["target"]["path"] = "core/unassigned.hpp"
    with pytest.raises(ContractError, match="unresolved use needs explicit debt"):
        validate_provenance(documents, root)
    documents["contracts/second.json"]["uses"][0]["debt"] = ["Chief to register shared owner"]
    assert any("target owner unresolved" in item for item in validate_provenance(documents, root))


def test_symlink_escape_rejected(examples, tmp_path):
    root, documents = examples
    link = root / "core/copied.hpp"
    link.unlink()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.hpp"
    outside.write_bytes(b"abcDEFghi\n")
    link.symlink_to(outside)
    with pytest.raises(ContractError, match="escapes repository"):
        validate_provenance(documents, root)


def test_shared_schema_is_valid_and_not_supplied_by_workers():
    document = fragment()
    validate_schema(document, "fixture")
    document["schema"] = {"type": "object"}
    with pytest.raises(ContractError, match="schema violation"):
        validate_schema(document, "fixture")


def test_inventory_mismatch_rejected(examples):
    root, documents = examples
    with pytest.raises(ContractError, match="inventory identity mismatch"):
        validate_provenance(documents, root, "other inventory")


def test_loader_supports_arbitrary_v2_shards_and_retains_v1_checks(monkeypatch):
    import ledger_validation as validation

    extra = ROOT / "contracts/future_fragment.json"
    original_read, original_glob = validation.read_json, Path.glob
    document = fragment()
    inventory = original_read(ROOT / "contracts/phase1_scope.json")["inventory_id"]
    document["inventory_id"] = inventory

    def glob(path, pattern):
        paths = list(original_glob(path, pattern))
        return iter(paths + [extra]) if path == ROOT / "contracts" and pattern == "*.json" else iter(paths)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(validation, "read_json", lambda path: deepcopy(document) if path == extra else original_read(path))
    documents = load_ledgers(ROOT)
    assert documents["provenance_fragments"]["contracts/future_fragment.json"]["ledger"] == "source_provenance"
    assert any("legacy producer format not structurally/hash validated" in item for item in validate_ledgers(documents))
    documents["source_transfers"]["transfers"][0]["source_sha256"] = "invalid"
    with pytest.raises(ContractError, match="schema violation"):
        validate_ledgers(documents)


def test_mixed_v1_v2_destination_cannot_have_two_owners(examples):
    root, documents = examples
    with pytest.raises(ContractError, match="duplicate destination authority with V1 owner"):
        validate_provenance(documents, root, legacy_destinations={"core/copied.hpp"})


def test_loader_routes_converted_base_fragments_without_mutating_v1_inventory():
    documents = load_ledgers(ROOT)
    inventory = documents["phase1_scope"]["inventory_id"]
    original_scope = deepcopy(documents["phase1_scope"])
    for key in ("source_transfers", "core_adaptations"):
        documents[key] = fragment()
        documents[key]["inventory_id"] = inventory
    limitations = validate_ledgers(documents)
    assert any("V2 normalization pending" in item for item in limitations)
    assert documents["phase1_scope"] == original_scope


def test_v2_missing_header_cannot_silently_fall_back_to_unvalidated_v1():
    documents = load_ledgers(ROOT)
    documents.setdefault("provenance_fragments", {})["contracts/example_source_transfers.json"] = fragment()
    del documents["provenance_fragments"]["contracts/example_source_transfers.json"]["ledger"]
    with pytest.raises(ContractError, match="V2 header cannot be downgraded"):
        validate_ledgers(documents)


def test_live_provenance_transition_reports_debt_without_decoding_family_formats():
    from ledger_validation import provenance_documents

    _, limitations = provenance_documents(load_ledgers(ROOT))
    for limitation in limitations:
        warnings.warn(limitation, RuntimeWarning, stacklevel=2)
