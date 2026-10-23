"""Read-only structural checks. A passing inventory is not qualification evidence."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


LEDGER_NAMES = ("phase1_scope", "test_transfer", "math_owners", "source_transfers", "core_adaptations")
EVIDENCE_REGISTRY_PATH = "contracts/evidence_registry.json"
SCHEMA_DIGESTS = {
    "phase1_scope": "e665b141f5184e70cd36b92a179f0a7e98fbbb67279d508de810b426749f6ab6",
    "test_transfer": "04d210bc460e67e87b8273bbd420dc99df4ac06ceaabaf163cdc24596bfed368",
    "math_owners": "835d70b1d273961ed24e7e390b42251332203d0c17540671af6fd6c8d7e712f2",
}
DENSITIES = (
    "normal", "studentt", "skewt", "ged", "generalizedhyperbolic",
    "normalinversegaussian", "hyperbolic", "variancegamma",
)
MEANS = ("none", "arma", "arx", "harx")
VOLATILITIES = ("garch", "gjr-garch", "egarch", "aparch")
REQUIRED_GATES = frozenset((
    "scope", "independent_build", "single_owner", "objective", "derivatives",
    "requests", "range", "preparation_memory", "fixed_runtime", "references",
    "production_binary", "request_performance",
))
SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
TEXT = {"type": "string", "minLength": 1}
QUOTED_INCLUDE = re.compile(rb'(^\s*#\s*include\s*")([^"\r\n]+)(")', re.MULTILINE)


class ContractError(ValueError):
    """Invalid structure, identity, or unsupported evidence claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> None:
    raise ContractError(f"non-finite JSON number: {value}")


def load_ledgers(root: Path) -> dict[str, dict]:
    documents = {}
    for name in LEDGER_NAMES:
        path = root / "contracts" / f"{name}.json"
        documents[name] = read_json(path)
    registry = root / EVIDENCE_REGISTRY_PATH
    if registry.is_file():
        documents["evidence_registry"] = read_json(registry)
    known_paths = {f"contracts/{name}.json" for name in LEDGER_NAMES} | {EVIDENCE_REGISTRY_PATH}
    fragments = {}
    for path in sorted((root / "contracts").glob("*.json")):
        relative = path.relative_to(root).as_posix()
        if relative not in known_paths:
            fragments[relative] = read_json(path)
    if fragments:
        documents["provenance_fragments"] = fragments
    return documents


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object,
                          parse_constant=reject_nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path}: {error}") from error


def validate_ledgers(documents: dict[str, dict], root: Path | None = None) -> tuple[str, ...]:
    """Validate reusable structure; optionally check retained bytes, never import tests."""
    validate_schemas(documents)
    validate_inventory_links(documents)
    validate_scope_requirements(documents["phase1_scope"])
    validate_test_transfers(documents["test_transfer"], documents["phase1_scope"])
    from provenance_validation import is_v2, validate_provenance

    fragments, transitional = provenance_documents(documents)
    legacy_destinations = set()
    if not is_v2(documents["source_transfers"]):
        validate_source_transfer_structure(documents["source_transfers"])
        legacy_destinations.update(row["destination"] for row in documents["source_transfers"]["transfers"])
    if not is_v2(documents["core_adaptations"]):
        validate_core_adaptations(documents["core_adaptations"], root)
        legacy_destinations.update(row["path"] for row in documents["core_adaptations"]["owners"])
    limitations = validate_evidence_claims(documents, root) + transitional
    limitations += validate_provenance(fragments, root, documents["phase1_scope"]["inventory_id"],
                                      legacy_destinations)
    if root is not None:
        validate_retained_test_sources(documents["test_transfer"], root)
        if not is_v2(documents["source_transfers"]):
            limitations += validate_source_snapshots(documents["source_transfers"], root)
    return limitations


def provenance_documents(documents: dict[str, dict]) -> tuple[dict, tuple[str, ...]]:
    """Route by shared format, never by the producer's model family."""
    from provenance_validation import is_v2

    candidates = {f"contracts/{name}.json": documents[name] for name in LEDGER_NAMES[3:]}
    additional = documents.get("provenance_fragments", {})
    require(not candidates.keys() & additional.keys(), "duplicate provenance fragment path")
    candidates.update(additional)
    fragments, limitations = {}, []
    for path, document in candidates.items():
        validate_repository_reference(path)
        require(isinstance(document, dict), f"{path}: ledger must be an object")
        if is_v2(document):
            fragments[path] = document
        else:
            require(document.get("ledger") is None and document.get("schema_version") != "2.0.0",
                    f"{path}: unknown ledger format; V2 header cannot be downgraded")
            require(path in {f"contracts/{name}.json" for name in LEDGER_NAMES[3:]} or
                    (path.startswith("contracts/") and path.endswith("_source_transfers.json")),
                    f"{path}: unknown ledger format")
            limitations.append(f"{path}: V2 normalization pending; " + (
                "existing V1 checks retained" if path in {f"contracts/{name}.json" for name in LEDGER_NAMES[3:]}
                else "legacy producer format not structurally/hash validated"
            ))
    return fragments, tuple(limitations)


def closed_object(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def source_transfer_schema() -> dict:
    """Rawls' ledger has no embedded schema; describe its existing format exactly."""
    include = closed_object({"source": TEXT, "destination": TEXT})
    transfer = closed_object({
        "source": TEXT, "destination": TEXT, "source_sha256": SHA256,
        "relocated_sha256": SHA256, "current_sha256": SHA256,
        "adaptation": TEXT, "includes": {"type": "array", "items": include},
    })
    return closed_object({
        "schema_version": {"const": 1, "type": "integer"},
        "status": {"const": "prototype/incomplete"},
        "source_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "source_state": TEXT, "license": TEXT,
        "discovery": closed_object({
            "compiler": TEXT, "method": TEXT,
            "seeds": {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True},
        }),
        "sample_contract": closed_object({
            "default_first_scored": {"type": "integer", "minimum": 0},
            "scored_count": TEXT, "stop": TEXT, "traversal": TEXT, "validation": TEXT,
        }),
        "build_contract": closed_object({
            "version_source": TEXT, "backend_flags": TEXT, "native_arch_option": TEXT,
            "frame_gate": TEXT, "solvers": TEXT,
        }),
        "transfers": {"type": "array", "items": transfer, "minItems": 1},
        "qualification_debt": {"type": "array", "items": TEXT, "minItems": 1},
    })


def core_adaptation_schema() -> dict:
    references = {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True}
    owner = closed_object({
        "id": TEXT, "path": TEXT, "sha256": SHA256, "derivation": TEXT,
        "source_owners": references, "consumer_sources": references,
        "regression_tests": references,
        "status": {"enum": ["integration_pending", "integrated", "incomplete"]},
    })
    return closed_object({
        "schema_version": {"const": 1, "type": "integer"}, "purpose": TEXT,
        "owners": {"type": "array", "items": owner, "minItems": 1},
        "qualification": TEXT,
    })


def validate_schemas(documents: dict[str, dict]) -> None:
    from provenance_validation import is_v2, validate_schema

    require(set(LEDGER_NAMES) <= documents.keys() and
            documents.keys() <= set(LEDGER_NAMES) | {"evidence_registry", "provenance_fragments"},
            "missing or unknown ledger")
    require(isinstance(documents.get("provenance_fragments", {}), dict), "invalid provenance fragment collection")
    for name in LEDGER_NAMES:
        document = documents[name]
        require(isinstance(document, dict), f"{name}: ledger must be an object")
        if name in LEDGER_NAMES[3:] and is_v2(document):
            validate_schema(document, name)
            continue
        if name == "source_transfers":
            schema = source_transfer_schema()
        elif name == "core_adaptations":
            schema = core_adaptation_schema()
        else:
            schema = document.get("schema")
            require(isinstance(schema, dict), f"{name}: missing embedded schema")
            require(canonical_digest(schema) == SCHEMA_DIGESTS[name],
                    f"{name}: schema changed; review the versioned validator contract")
            require(document.get("schema_version") == "1.0.0", f"{name}: unsupported schema version")
            require(document.get("ledger") == name, f"{name}: wrong ledger identity")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ContractError(f"{name}: invalid schema: {error.message}") from error
        errors = list(Draft202012Validator(schema).iter_errors(document))
        require(not errors, f"{name}: schema violation: " + "; ".join(
            f"{list(error.absolute_path)}: {error.message}" for error in errors[:3]
        ))


def canonical_digest(value: object) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def unique_values(values: list, label: str) -> set:
    require(all(isinstance(value, str) and value.strip() for value in values), f"{label}: empty or non-string ID")
    require(len(values) == len(set(values)), f"{label}: duplicate ID")
    return set(values)


def index_records(records: list[dict], key: str, label: str) -> dict[str, dict]:
    unique_values([record[key] for record in records], label)
    return {record[key]: record for record in records}


def validate_inventory_links(documents: dict[str, dict]) -> None:
    scope, transfer, owners = (documents[name] for name in LEDGER_NAMES[:3])
    require(len({document["inventory_id"] for document in (scope, transfer, owners)}) == 1,
            "inventory identity mismatch")
    require(len({document["as_of"] for document in (scope, transfer, owners)}) == 1,
            "inventory date mismatch")
    require(scope["authority"] == transfer["authority"] == owners["authority"], "conflicting authorities")
    authority_paths = {
        "implemented_capabilities": "core/src/catalog.hpp",
        "promised_scope": "contracts/phase1_scope.json",
        "test_disposition": "contracts/test_transfer.json",
        "equation_ownership_and_invalidation": "contracts/math_owners.json",
    }
    for key, path in authority_paths.items():
        require(scope["authority"][key] == path, f"wrong authority for {key}")
    capabilities = index_records(scope["capabilities"], "id", "capabilities")
    owner_index = index_records(owners["owners"], "id", "owners")
    debt = index_records(scope["debt"], "id", "debt")
    index_records(scope["densities"], "id", "densities")
    shared = unique_values(scope["shared_owner_ids"], "shared owners")
    require(shared == set(owner_index), "shared owner inventory mismatch")
    unique_values([owner["canonical_owner"]["logical_id"] for owner in owner_index.values()], "canonical owners")
    reverse = {identifier: set() for identifier in owner_index}
    for capability in capabilities.values():
        identifier = capability["id"]
        referenced = unique_values(capability["owner_ids"], f"{identifier}: owners")
        require(referenced and referenced <= owner_index.keys(), f"{identifier}: unknown or missing owner")
        for owner in referenced:
            reverse[owner].add(identifier)
        debts = unique_values(capability["debt_ids"], f"{identifier}: debt")
        require(debts <= debt.keys(), f"{identifier}: unknown debt")
    for owner in owner_index.values():
        consumers = unique_values(owner["phase1_consumer_ids"], f"{owner['id']}: consumers")
        require(consumers == reverse[owner["id"]], f"{owner['id']}: reverse consumer mismatch")
    for document in (transfer, owners):
        require(unique_values(document["debt_ids"], "ledger debt") <= debt.keys(), "unknown ledger debt")
    require(owners["statistics"] == {
        "logical_owners": len(owner_index),
        "applied_bindings": sum(owner["canonical_owner"]["implementation_binding"] is not None for owner in owner_index.values()),
        "qualified": sum(owner["status"] == "mathematically_qualified" for owner in owner_index.values()),
    }, "owner statistics mismatch")


def expected_capability_ids() -> set[str]:
    complete = {
        f"mean:{mean}+vol:{volatility}+density:{density}"
        for mean in MEANS for volatility in VOLATILITIES + (("none",) if mean != "none" else ())
        for density in DENSITIES
    }
    local = {f"math:{kind}+density:{density}" for kind in ("local_density", "constant_scale_intention") for density in DENSITIES}
    profiles = {f"math:normal_profile+mean:{mean}" for mean in MEANS if mean != "none"}
    return complete | local | profiles | {"math:dcc+law:normal", "math:dcc+law:studentt"}


def integer_axis(name: str, minimum: int, zero: str | None) -> dict:
    result = {"name": name, "type": "integer", "minimum": minimum}
    if zero is not None:
        result["zero"] = zero
    return result


def expected_order_axes(capability: dict) -> list[dict]:
    mean = capability.get("mean", "none")
    axes = []
    if mean == "arma":
        axes = [integer_axis(name, 0, "valid") for name in ("p_ar", "q_ma")]
    elif mean in ("arx", "harx"):
        lag = integer_axis("lags", 0, "valid") if mean == "arx" else {
            "name": "horizons", "type": "positive_integer_vector", "minimum": 1, "zero": "invalid",
        }
        expression = "int(constant) + lags + n_exog" if mean == "arx" else "int(constant) + len(horizons) + n_exog"
        axes = [lag, integer_axis("n_exog", 0, "valid"), {"name": "constant", "type": "boolean"},
                {"name": "design_columns", "type": "derived_integer", "expression": expression,
                 "zero": "test_declared_empty_design"}]
    if capability.get("volatility", "none") != "none":
        axes += [integer_axis(name, 1, "invalid") for name in ("p_vol", "q_vol")]
    if capability["kind"] == "fixed_input_dependence_jet":
        axes = [integer_axis(name, 1, "invalid") for name in ("p_corr", "q_corr")]
        axes.append(integer_axis("series", 2, None))
    return axes


def validate_scope_requirements(scope: dict) -> None:
    capabilities = scope["capabilities"]
    require({row["id"] for row in capabilities} == expected_capability_ids(), "promised scope changed")
    require({row["id"] for row in scope["densities"]} == set(DENSITIES), "density inventory changed")
    require(set(scope["gate_definitions"]) == REQUIRED_GATES, "required gate definitions changed")
    for row in capabilities:
        identifier = row["id"]
        require(row["density"] in DENSITIES, f"{identifier}: unknown density")
        require(unique_values(row["required_gate_ids"], f"{identifier}: gates") == REQUIRED_GATES,
                f"{identifier}: missing required gates")
        require(all(type(mask) is int for mask in row["request_masks"]) and
                len(row["request_masks"]) == 7 and set(row["request_masks"]) == set(range(1, 8)),
                f"{identifier}: all seven integer request masks required")
        if identifier.startswith("mean:"):
            require(row["kind"] == "complete_jet" and identifier ==
                    f"mean:{row.get('mean')}+vol:{row.get('volatility')}+density:{row['density']}",
                    f"{identifier}: composition identity mismatch")
        elif identifier.startswith("math:normal_profile"):
            require(row["kind"] == "exact_profile" and row["density"] == "normal" and
                    identifier == f"math:normal_profile+mean:{row.get('mean')}", f"{identifier}: profile identity mismatch")
        elif identifier.startswith("math:dcc"):
            require(row["kind"] == "fixed_input_dependence_jet" and identifier == f"math:dcc+law:{row['density']}",
                    f"{identifier}: DCC identity mismatch")
        else:
            require(row["kind"] in ("local_density", "constant_scale_intention") and identifier ==
                    f"math:{row['kind']}+density:{row['density']}", f"{identifier}: local identity mismatch")
        require(row.get("order_axes", []) == expected_order_axes(row), f"{identifier}: order axes mismatch")
        coordinates = ["theta"] if row["kind"] == "fixed_input_dependence_jet" else ["theta", "original_z"]
        if row["kind"] not in ("local_density", "constant_scale_intention"):
            require(row.get("coordinates") == coordinates, f"{identifier}: coordinate promise changed")
        if row["kind"] == "complete_jet":
            require(row.get("order_requirement") ==
                    "runtime arbitrary declared orders and build-time instantiation of the same policy for any valid tuple",
                    f"{identifier}: general-order requirement missing")
            require(row.get("sample_contract") in scope["sample_contracts"], f"{identifier}: unknown sample contract")
    counts = Counter(row["kind"] for row in capabilities)
    require(scope["statistics"] == {
        "univariate_compositions": counts["complete_jet"], "local_density_intentions": counts["local_density"],
        "constant_scale_intentions": counts["constant_scale_intention"], "normal_profile_objectives": counts["exact_profile"],
        "fixed_input_dcc_laws": counts["fixed_input_dependence_jet"], "capabilities": len(capabilities),
        "qualified": sum(row["status"] == "mathematically_qualified" for row in capabilities),
    }, "scope statistics mismatch")
    for key in ("missing_registration", "missing_evidence", "required_skip_or_xfail", "remaining_phase1_debt"):
        require(scope["closure_policy"][key] == "fail", f"closure policy weakened: {key}")
    require(scope["closure_policy"]["catalogue_status_inheritance"] == "forbidden", "catalogue qualification inheritance forbidden")


def validate_test_transfers(transfer: dict, scope: dict) -> None:
    candidates = index_records(transfer["candidate_capability_sets"], "id", "candidate sets")
    capabilities = {row["id"] for row in scope["capabilities"]}
    for row in candidates.values():
        ids = unique_values(row["capability_ids"], f"{row['id']}: capabilities")
        require(ids and ids <= capabilities, f"{row['id']}: unknown or empty candidate capabilities")
    files = index_records(transfer["files"], "path", "test files")
    native = index_records(transfer["native_tests"], "path", "native files")
    fixtures = index_records(transfer["fixtures"], "path", "fixtures")
    index_records(transfer["outside_phase1_files"], "path", "outside files")
    rows = transfer["nodes"] + transfer["support_units"] + transfer["native_tests"]
    identifiers = unique_values([row["original_node_id"] for row in rows], "test nodes")
    later = unique_values([row["original_node_id"] for row in transfer["mixed_file_later_phase_node_ids"]], "later-phase nodes")
    require(identifiers.isdisjoint(later), "conflicting Phase1/later-phase node disposition")
    all_paths = list(files) + list(native) + list(fixtures)
    unique_values(all_paths, "snapshot paths")
    for row in list(files.values()) + list(native.values()) + list(fixtures.values()):
        validate_snapshot_record(row)
    assertion_ids = []
    for row in rows:
        identifier = row["original_node_id"]
        path, separator, qualified = identifier.partition("::")
        require(separator and qualified and path in files.keys() | native.keys(), f"{identifier}: unknown original file")
        require(row["candidate_capability_set_id"] in candidates, f"{identifier}: unknown candidate set")
        require(row["disposition"] in ("deferred", "transferred", "superseded"), f"{identifier}: invalid disposition; skip/xfail is not transfer")
        unique_values(row["case_ids"], f"{identifier}: cases")
        unique_values(row["evidence_ids"], f"{identifier}: evidence")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", row["sha256"])), f"{identifier}: invalid SHA256")
        if row["disposition"] == "deferred":
            require(type(row["blocking_phase"]) is int and row["blocking_phase"] == 1 and row["reason"].strip(),
                    f"{identifier}: deferred Phase1 work needs its Phase1 blocker")
        else:
            require(row["case_ids"] and row["evidence_ids"] and row["new_test_id"],
                    f"{identifier}: transfer/supersession requires cases, new test and evidence")
        if "assertion_ids" in row:
            require(1 <= row["line"] <= row["end_line"], f"{identifier}: invalid source span")
            expected = [f"{identifier}::assertion:{number}" for number in range(1, len(row["assertion_ids"]) + 1)]
            require(row["assertion_ids"] == expected, f"{identifier}: assertion identity mismatch")
            assertion_ids.extend(row["assertion_ids"])
        else:
            require(identifier == path + "::main", f"{identifier}: native node identity mismatch")
            for anchor in row["assertion_anchors"]:
                require(anchor["line"] >= 1 and re.fullmatch(r"[0-9a-f]{64}", anchor["sha256"]), f"{identifier}: invalid native anchor")
    unique_values(assertion_ids, "assertion IDs")
    require(transfer["statistics"] == {
        "python_files": len(files), "phase1_test_definitions": len(transfer["nodes"]),
        "mixed_file_later_phase_definitions_not_transferred": len(later), "support_units": len(transfer["support_units"]),
        "python_assertion_anchors": len(assertion_ids), "native_files": len(native),
        "native_lexical_assertion_anchors": sum(len(row["assertion_anchors"]) for row in native.values()),
        "fixtures": len(fixtures), "transferred": sum(row["disposition"] == "transferred" for row in rows),
        "superseded": sum(row["disposition"] == "superseded" for row in rows),
    }, "test-transfer statistics mismatch")


def validate_snapshot_record(row: dict) -> None:
    require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), f"{row['path']}: invalid snapshot hash")
    require(type(row["bytes"]) is int and row["bytes"] >= 0, f"{row['path']}: invalid byte count")
    require(row["snapshot_kind"] == "working_tree_bytes", f"{row['path']}: unsupported snapshot kind")
    blob = row["index_blob"]
    require(blob is None or re.fullmatch(r"[0-9a-f]{40}", blob), f"{row['path']}: invalid index blob")


def evidence_registry_schema() -> dict:
    ids = {"type": "array", "items": TEXT, "uniqueItems": True}
    artifact = closed_object({"path": TEXT, "sha256": SHA256})
    review = closed_object({
        "status": {"enum": ["pending_review", "passed", "rejected"]},
        "reviewer": TEXT, "independent": {"type": "boolean"}, "rationale": TEXT,
    })
    common = {
        "id": TEXT, "source_bindings": {"type": "array", "items": artifact, "minItems": 1},
        "report": {"anyOf": [{"type": "null"}, artifact]}, "review": review,
    }
    transfer = closed_object({
        **common, "kind": {"const": "transfer_reviewed"}, "original_node_id": TEXT,
        "original_sha256": SHA256, "assertion_ids": ids, "case_ids": ids,
        "required_case_ids": ids, "coverage": {"enum": ["pending_review", "complete"]},
        "new_test_id": {"type": ["null", "string"]},
        "new_test_sha256": {"anyOf": [{"type": "null"}, SHA256]},
        "supersession_justification": {"type": ["null", "string"]},
    })
    mathematical = closed_object({
        **common, "kind": {"const": "mathematical_gate"},
        "subject_type": {"enum": ["capability", "owner", "debt"]}, "subject_id": TEXT,
        "subject_sha256": SHA256, "gate_id": {"enum": sorted(REQUIRED_GATES)},
        "outcome": {"enum": ["passed", "failed", "pending_review", "skipped", "xfail"]},
        "case_ids": ids, "check_ids": ids,
    })
    return closed_object({
        "schema_version": {"type": "integer", "const": 1}, "inventory_id": TEXT,
        "records": {"type": "array", "items": {"oneOf": [transfer, mathematical]}},
    })


def validated_evidence_registry(documents: dict[str, dict]) -> dict[str, dict]:
    scope = documents["phase1_scope"]
    require(scope["evidence_records"] == [],
            "bare string evidence records cannot prove claims; use the typed evidence_registry sidecar")
    document = documents.get("evidence_registry", {
        "schema_version": 1, "inventory_id": scope["inventory_id"], "records": [],
    })
    errors = list(Draft202012Validator(evidence_registry_schema()).iter_errors(document))
    require(not errors, "typed evidence registry schema violation: " + "; ".join(error.message for error in errors[:2]))
    require(document["inventory_id"] == scope["inventory_id"], "evidence inventory identity mismatch")
    registry = index_records(document["records"], "id", "typed evidence records")
    subjects = {
        "capability": {row["id"]: row for row in scope["capabilities"]},
        "owner": {row["id"]: row for row in documents["math_owners"]["owners"]},
        "debt": {row["id"]: row for row in scope["debt"]},
    }
    transfers = documents["test_transfer"]
    nodes = {row["original_node_id"]: row for row in transfers["nodes"] + transfers["support_units"] + transfers["native_tests"]}
    for record in registry.values():
        index_records(record["source_bindings"], "path", f"{record['id']}: source bindings")
        artifacts = record["source_bindings"] + ([record["report"]] if record["report"] else [])
        for artifact in artifacts:
            validate_repository_reference(artifact["path"])
        if record["kind"] == "transfer_reviewed":
            require(record["original_node_id"] in nodes, f"{record['id']}: unknown transfer subject")
        else:
            require(record["subject_id"] in subjects[record["subject_type"]], f"{record['id']}: unknown mathematical subject")
    return registry


def validate_evidence_claims(documents: dict[str, dict], root: Path | None = None) -> tuple[str, ...]:
    scope, transfer, owners = (documents[name] for name in LEDGER_NAMES[:3])
    registry = validated_evidence_registry(documents)
    provisional = set()
    subjects = [(row, "capability") for row in scope["capabilities"]]
    subjects += [(row, "owner") for row in owners["owners"]] + [(row, "debt") for row in scope["debt"]]
    for row, subject_type in subjects:
        claimed = row["status"] not in ("pending", "incomplete")
        evidence = resolve_evidence(row, registry, claimed, provisional)
        fixed = unique_values(row.get("fixed_execution_evidence", []), f"{row['id']}: fixed evidence")
        if claimed:
            require(fixed <= registry.keys(), f"{row['id']}: unresolved fixed execution evidence")
            for identifier in fixed:
                record = registry[identifier]
                require(record["kind"] == "mathematical_gate" and record["subject_type"] == subject_type
                        and record["subject_id"] == row["id"] and record["gate_id"] in ("fixed_runtime", "production_binary"),
                        f"{row['id']}: fixed evidence must bind a mathematical dispatch gate")
                if record not in evidence:
                    evidence.append(record)
        else:
            provisional.update(fixed - registry.keys())
        if row["status"] not in ("pending", "incomplete"):
            allowed = ("closed", "resolved", "complete") if subject_type == "debt" else ("mathematically_qualified",)
            require(row["status"] in allowed, f"{row['id']}: unsupported qualification status")
            validate_mathematical_claim(row, subject_type, evidence, scope, owners, root)
    for row in transfer["nodes"] + transfer["support_units"] + transfer["native_tests"]:
        evidence = resolve_evidence(row, registry, row["disposition"] != "deferred", provisional)
        if row["disposition"] in ("transferred", "superseded"):
            validate_transfer_claim(row, evidence, transfer, root)
    validate_owner_bindings(owners)
    validate_phase_closure(scope, transfer, owners)
    return (f"{len(provisional)} provisional evidence links unresolved; none establish transfer or qualification",) if provisional else ()


def resolve_evidence(row: dict, registry: dict[str, dict], claimed: bool, provisional: set[str]) -> list[dict]:
    identifier = row.get("id", row.get("original_node_id"))
    ids = unique_values(row["evidence_ids"], f"{identifier}: evidence")
    if claimed:
        require(ids <= registry.keys(), f"{identifier}: unresolved evidence IDs")
    else:
        provisional.update(ids - registry.keys())
    return [registry[identifier] for identifier in row["evidence_ids"] if identifier in registry]


def validate_owner_bindings(owners: dict) -> None:
    for owner in owners["owners"]:
        canonical = owner["canonical_owner"]
        require(canonical["status"] in ("pending", "incomplete", "declared", "implemented", "bound", "applied"),
                f"{owner['id']}: unevidenced canonical owner claim")
        binding = canonical["implementation_binding"]
        if canonical["status"] in ("implemented", "bound", "applied"):
            require(binding is not None, f"{owner['id']}: implementation status needs a binding")
        if binding is not None:
            validate_repository_reference(binding["path"])
            require(binding["symbol"].strip(), f"{owner['id']}: empty binding symbol")


def subject_digest(row: dict) -> str:
    return canonical_digest({key: value for key, value in row.items()
                             if key not in ("status", "evidence_ids", "fixed_execution_evidence")})


def verified_report(record: dict, root: Path | None) -> dict:
    require(root is not None, f"{record['id']}: a claim requires current source/report identity checks")
    require(record["review"]["status"] == "passed" and record["review"]["independent"],
            f"{record['id']}: claim requires passed independent review")
    require(record["report"] is not None, f"{record['id']}: claim requires a retained structured report")
    contents = {}
    for artifact in record["source_bindings"] + [record["report"]]:
        content = retained_bytes(root, artifact["path"])
        require(digest(content) == artifact["sha256"], f"{artifact['path']}: evidence artifact drift")
        contents[artifact["path"]] = content
    try:
        report = json.loads(contents[record["report"]["path"]].decode("utf-8"),
                            object_pairs_hook=strict_object, parse_constant=reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{record['id']}: invalid structured report") from error
    schema = closed_object({
        "schema_version": {"type": "integer", "const": 1}, "record_sha256": SHA256,
        "details": {"type": "object", "minProperties": 1},
    })
    require(not list(Draft202012Validator(schema).iter_errors(report)), f"{record['id']}: invalid structured report")
    payload = {key: value for key, value in record.items() if key != "report"}
    require(report["record_sha256"] == canonical_digest(payload), f"{record['id']}: report/record identity mismatch")
    return report["details"]


def transfer_assertion_ids(row: dict) -> list[str]:
    if "assertion_ids" in row:
        return row["assertion_ids"]
    return [f"{row['original_node_id']}::line:{anchor['line']}:{anchor['sha256']}" for anchor in row["assertion_anchors"]]


def validate_transfer_claim(row: dict, evidence: list[dict], transfer: dict, root: Path | None) -> None:
    records = [record for record in evidence if record["kind"] == "transfer_reviewed" and
               record["original_node_id"] == row["original_node_id"]]
    require(records, f"{row['original_node_id']}: transferred mapping requires transfer_reviewed evidence, not jet qualification")
    record = records[-1]
    require(record["original_sha256"] == row["sha256"], f"{record['id']}: original node identity mismatch")
    require(record["coverage"] == "complete", f"{record['id']}: assertion/case coverage is still pending review")
    require(set(record["assertion_ids"]) == set(transfer_assertion_ids(row)), f"{record['id']}: incomplete assertion coverage")
    require(set(record["case_ids"]) == set(record["required_case_ids"]) == set(row["case_ids"]) and row["case_ids"],
            f"{record['id']}: incomplete reviewed case coverage")
    require(record["new_test_id"] == row["new_test_id"] and record["new_test_sha256"] is not None,
            f"{record['id']}: new test identity mismatch")
    if row["disposition"] == "superseded":
        require(record["supersession_justification"] and record["supersession_justification"].strip(),
                f"{record['id']}: supersession requires independent justification")
    details = verified_report(record, root)
    required = ("assertion_mapping_review", "case_coverage_review", "oracle_tolerance_review")
    require(all(isinstance(details.get(key), str) and details[key].strip() for key in required),
            f"{record['id']}: missing transfer review details")
    snapshots = {item["path"]: item for item in transfer["files"] + transfer["native_tests"]}
    original_path = row["original_node_id"].partition("::")[0]
    bindings = {item["path"]: item["sha256"] for item in record["source_bindings"]}
    require(bindings.get(original_path) == snapshots[original_path]["sha256"], f"{record['id']}: original file binding missing")
    path, separator, qualified = row["new_test_id"].partition("::")
    require(separator and path.startswith("tests/"), f"{record['id']}: invalid new test node ID")
    require(bindings.get(path) == record["new_test_sha256"], f"{record['id']}: new test file binding missing")
    source = retained_bytes(root, path).decode("utf-8")
    if path.endswith(".py"):
        tree = ast.parse(source)
        functions = dict(qualified_functions(tree))
        require(qualified in functions, f"{record['id']}: new test node missing")
        required_code = [functions[qualified]] + [node for node in tree.body
                                                if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in required_code:
            names = [ast.unparse(item.func).split(".")[-1] for item in ast.walk(node) if isinstance(item, ast.Call)]
            require(not set(names) & {"skip", "skipif", "xfail", "importorskip"},
                    f"{record['id']}: skip/xfail cannot establish required assertion coverage")
    else:
        require(path.endswith(".cpp") and qualified == "main" and re.search(r"\bmain\s*\(", source),
                f"{record['id']}: unsupported or missing native test node")


def validate_mathematical_claim(row: dict, subject_type: str, evidence: list[dict], scope: dict, owners: dict, root: Path | None) -> None:
    records = [record for record in evidence if record["kind"] == "mathematical_gate" and
               record["subject_type"] == subject_type and record["subject_id"] == row["id"]]
    require(records, f"{row['id']}: qualification requires explicit mathematical_gate evidence")
    if subject_type == "capability":
        required_gates, required_checks = set(row["required_gate_ids"]), set()
    elif subject_type == "owner":
        required_gates = {"single_owner", "objective", "derivatives", "range", "references"}
        required_checks = set(row["independent_checks"])
    else:
        required_gates, required_checks = {"scope"}, set()
    require(required_gates <= {record["gate_id"] for record in records}, f"{row['id']}: missing mathematical gates")
    require(required_checks <= {check for record in records for check in record["check_ids"]},
            f"{row['id']}: missing independent checks")
    for record in records:
        require(record["subject_sha256"] == subject_digest(row), f"{record['id']}: subject contract drift")
        require(record["outcome"] == "passed", f"{record['id']}: failed/pending/skip/xfail cannot qualify")
        require(record["case_ids"], f"{record['id']}: mathematical evidence needs cases")
        details = verified_report(record, root)
        validate_mathematical_report(details, record, scope)
    affected = [row] if subject_type == "owner" else [
        owner for owner in owners["owners"] if owner["id"] in row.get("owner_ids", [])
    ]
    sources = {item["path"]: item["sha256"] for record in records for item in record["source_bindings"]}
    for owner in affected:
        binding = owner["canonical_owner"]["implementation_binding"]
        require(binding is not None, f"{row['id']}: mathematical claim needs owner implementation bindings")
        require(sources.get(binding["path"]) == binding["source_sha256"], f"{row['id']}: owner implementation evidence mismatch")


def validate_mathematical_report(details: dict, record: dict, scope: dict) -> None:
    require(set(scope["evidence_protocol"]["required_fields"]) <= details.keys(),
            f"{record['id']}: missing mathematical evidence protocol fields")
    require(details["id"] == record["id"] and details["gate_ids"] == [record["gate_id"]] and
            details["case_ids"] == record["case_ids"] and details["independent_review"] == record["review"],
            f"{record['id']}: mathematical report identity mismatch")
    require(isinstance(details["capability_ids"], list), f"{record['id']}: capability IDs must be structured")
    if record["subject_type"] == "capability":
        require(record["subject_id"] in details["capability_ids"], f"{record['id']}: capability report coverage mismatch")
    identity_fields = {"id", "capability_ids", "gate_ids", "case_ids", "limitations", "independent_review"}
    for field in set(scope["evidence_protocol"]["required_fields"]) - identity_fields:
        require(isinstance(details[field], (dict, list)) and bool(details[field]),
                f"{record['id']}: {field} needs structured data, not an acceptance label")
    require(isinstance(details["limitations"], list), f"{record['id']}: limitations must be explicit")


def validate_phase_closure(scope: dict, transfer: dict, owners: dict) -> None:
    incomplete = ("pending", "incomplete")
    for document in (scope, transfer, owners):
        require(document["status"] in incomplete + ("complete", "mathematically_qualified"),
                f"{document['ledger']}: unsupported ledger status")
    if any(document["status"] not in incomplete for document in (scope, transfer, owners)):
        require(all(row["status"] == "mathematically_qualified" for row in scope["capabilities"] + owners["owners"])
                and all(row["status"] in ("closed", "resolved", "complete") for row in scope["debt"])
                and all(row["disposition"] in ("transferred", "superseded")
                        for row in transfer["nodes"] + transfer["support_units"] + transfer["native_tests"]),
                "Phase1 closure has unsatisfied capability, owner, debt or transfer claims")


def validate_repository_reference(relative: str) -> None:
    parts = PurePosixPath(relative)
    require(relative and parts.parts and not parts.is_absolute() and ".." not in parts.parts and "\\" not in relative,
            f"unsafe repository path: {relative}")
    require(parts.parts[0] not in ("localdev",) and not parts.parts[0].startswith("localdev_"),
            f"ignored artifact cannot be a permanent CI input: {relative}")


def repository_path(root: Path, relative: str) -> Path:
    validate_repository_reference(relative)
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), f"path escapes repository: {relative}")
    return path


def retained_bytes(root: Path, relative: str) -> bytes:
    path = repository_path(root, relative)
    try:
        return path.read_bytes()
    except OSError as error:
        raise ContractError(f"missing retained source: {relative}") from error


def validate_retained_test_sources(transfer: dict, root: Path) -> None:
    snapshots = transfer["files"] + transfer["native_tests"] + transfer["fixtures"]
    contents = {}
    for row in snapshots:
        content = retained_bytes(root, row["path"])
        require(len(content) == row["bytes"] and digest(content) == row["sha256"], f"{row['path']}: retained snapshot drift")
        contents[row["path"]] = content
    parsed = {}
    for row in transfer["files"]:
        path = row["path"]
        source = contents[path].decode("utf-8")
        tree = ast.parse(source, filename=path)
        imports = [ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        require(imports == row["imports"], f"{path}: import inventory drift")
        parsed[path] = (source, dict(qualified_functions(tree)))
    for row in transfer["nodes"] + transfer["support_units"]:
        path, _, qualified = row["original_node_id"].partition("::")
        source, functions = parsed[path]
        require(qualified in functions, f"{row['original_node_id']}: missing retained node")
        node = functions[qualified]
        require((row["line"], row["end_line"]) == (node.lineno, node.end_lineno), f"{row['original_node_id']}: source span drift")
        require(digest(ast.get_source_segment(source, node).encode()) == row["sha256"], f"{row['original_node_id']}: retained node hash drift")
        require(row["parametrization"] == [ast.get_source_segment(source, decorator) for decorator in node.decorator_list],
                f"{row['original_node_id']}: decorator drift")
        require(len(row["assertion_ids"]) == len(assertion_nodes(node)), f"{row['original_node_id']}: assertion inventory drift")
    for row in transfer["native_tests"]:
        lines = contents[row["path"]].decode("utf-8").splitlines()
        for anchor in row["assertion_anchors"]:
            require(anchor["line"] <= len(lines), f"{row['path']}: missing native anchor")
            line = lines[anchor["line"] - 1]
            require(line.strip() == anchor["expression"] and digest(line.encode()) == anchor["sha256"],
                    f"{row['path']}: native anchor drift")


def qualified_functions(tree: ast.AST, prefix: tuple[str, ...] = ()):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield from qualified_functions(node, prefix + (node.name,))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield "::".join(prefix + (node.name,)), node


def assertion_nodes(node: ast.AST) -> list[ast.AST]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Assert) or (
        isinstance(item, ast.Call) and re.search(r"(?:^|\.)(?:assert\w*|raises|check\w*)$", ast.unparse(item.func))
    )]


def validate_source_snapshots(document: dict, root: Path, original_snapshot_root: Path | None = None) -> tuple[str, ...]:
    """Current hashes bind current bytes. Original hashes bind only original bytes."""
    validate_source_transfer_structure(document)
    limitations = []
    for row in document["transfers"]:
        source_path = repository_path(original_snapshot_root or root, row["source"])
        destination = retained_bytes(root, row["destination"])
        require(digest(destination) == row["current_sha256"], f"{row['destination']}: current snapshot drift")
        original = source_path.read_bytes() if source_path.is_file() else None
        if original is None or digest(original) != row["source_sha256"]:
            require(original_snapshot_root is None, f"{row['source']}: pinned original snapshot missing or corrupt")
            limitations.append(f"{row['source']}: original pinned bytes unavailable; relocation integrity unverified")
            continue
        relocated = reconstruct_include_relocation(original, row)
        require(digest(relocated) == row["relocated_sha256"], f"{row['destination']}: original relocation hash mismatch")
    return tuple(limitations)


def validate_source_transfer_structure(document: dict) -> None:
    index_records(document["transfers"], "source", "transfer sources")
    index_records(document["transfers"], "destination", "transfer destinations")
    sources = {row["source"] for row in document["transfers"]}
    require(set(document["discovery"]["seeds"]) <= sources, "discovery seed missing from transfer closure")
    for row in document["transfers"]:
        require(row["source"].startswith("scivol/") and row["destination"].startswith("core/src/"), "invalid transfer ownership path")
        validate_repository_reference(row["source"])
        validate_repository_reference(row["destination"])
        mapping = {}
        for include in row["includes"]:
            require(include["source"] not in mapping or mapping[include["source"]] == include["destination"],
                    f"{row['source']}: conflicting include relocation")
            mapping[include["source"]] = include["destination"]


def validate_core_adaptations(document: dict, root: Path | None = None) -> None:
    """Validate declared provenance pointers, not derivations or test execution."""
    owners = index_records(document["owners"], "id", "derived owner IDs")
    index_records(document["owners"], "path", "derived owner paths")
    for owner in owners.values():
        references = [(owner["path"], "core/src/")]
        references += [(path, "core/src/") for path in owner["source_owners"] + owner["consumer_sources"]]
        references += [(path, "tests/") for path in owner["regression_tests"]]
        for relative, prefix in references:
            validate_repository_reference(relative)
            require(relative.startswith(prefix), f"{owner['id']}: invalid provenance path: {relative}")
            if root is not None:
                require(repository_path(root, relative).is_file(), f"{owner['id']}: missing referenced file: {relative}")
        if root is not None:
            require(digest(retained_bytes(root, owner["path"])) == owner["sha256"],
                    f"{owner['path']}: derived owner snapshot drift")


def reconstruct_include_relocation(original: bytes, row: dict) -> bytes:
    mapping = {}
    for include in row["includes"]:
        before, after = include["source"].encode(), include["destination"].encode()
        require(before not in mapping or mapping[before] == after, f"{row['source']}: conflicting include relocation")
        mapping[before] = after
    occurrences = Counter(match[2] for match in QUOTED_INCLUDE.finditer(original))
    declared = Counter(include["source"].encode() for include in row["includes"])
    require(occurrences == declared, f"{row['source']}: include snapshot mismatch")
    return QUOTED_INCLUDE.sub(lambda match: match[1] + mapping[match[2]] + match[3], original)
