"""One file-provenance contract for every producer; no model-family dispatch."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator

from ledger_validation import (
    ContractError, canonical_digest, digest, read_json, reject_nonfinite,
    repository_path, require, retained_bytes, strict_object, validate_repository_reference,
)


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "contracts/schemas/source_provenance.schema.json"


def is_v2(document: dict) -> bool:
    return isinstance(document, dict) and document.get("ledger") == "source_provenance"


@lru_cache(maxsize=3)
def schema_validator(definition: str | None) -> Draft202012Validator:
    schema = read_json(SCHEMA_PATH)
    if definition is not None:
        schema = {"$defs": schema["$defs"], "$ref": f"#/$defs/{definition}"}
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_schema(document: dict, label: str, definition: str | None = None) -> None:
    errors = list(schema_validator(definition).iter_errors(document))
    require(not errors, f"{label}: provenance schema violation: " + "; ".join(
        f"{list(error.absolute_path)}: {error.message}" for error in errors[:3]
    ))


def validate_provenance(fragments: dict[str, dict], root: Path | None = None,
                        inventory_id: str | None = None,
                        legacy_destinations: set[str] = frozenset()) -> tuple[str, ...]:
    """Index authority, verify identities, then assess review. Never advance a status."""
    for path, document in fragments.items():
        check_path(path)
        validate_schema(document, path)
    identities = {document["inventory_id"] for document in fragments.values()}
    if inventory_id is not None:
        identities.add(inventory_id)
    require(len(identities) <= 1, "provenance inventory identity mismatch")
    owners = authority_index(fragments, "owners", "destination")
    contexts = authority_index(fragments, "context_refs", "path")
    require(not owners.keys() & legacy_destinations, "duplicate destination authority with V1 owner")
    reject_origin_cycles(owners)
    blobs = SnapshotReader(root)
    context_blobs, vendors, limitations = verify_contexts(contexts, blobs)
    for owner in owners.values():
        debt = verify_owner(owner, owners, contexts, vendors, blobs)
        if owner["status"] == "provenance_reviewed":
            require(not debt, f"{owner['destination']}: unsupported provenance review: {debt}")
            verify_review(owner, contexts, vendors, context_blobs)
        else:
            limitations.extend(f"{owner['destination']}: {item}" for item in debt)
            limitations.append(f"{owner['destination']}: provenance review pending")
    limitations.extend(verify_uses(fragments, owners, blobs))
    for path, document in fragments.items():
        limitations.extend(f"{path}: {item}" for item in document["debt"])
        for item in document["unassigned_origins"]:
            verify_origin(item["origin"], owners, blobs, [])
            limitations.append(f"{path}: unassigned origin {item['origin']['path']}: {item['reason']}")
    return tuple(dict.fromkeys(limitations))


def authority_index(fragments: dict[str, dict], collection: str, key: str) -> dict[str, dict]:
    index = {}
    for document in fragments.values():
        for record in document[collection]:
            path = record[key]
            check_path(path, allow_ignored=collection == "context_refs")
            require(path not in index, f"{path}: duplicate {collection} authority")
            index[path] = record
    return index


def ignored_path(path: str) -> bool:
    first = PurePosixPath(path).parts[0]
    return first == "localdev" or first.startswith("localdev_")


def check_path(path: str, allow_ignored: bool = False) -> None:
    require(isinstance(path, str) and path.strip() and path != "." and str(PurePosixPath(path)) == path,
            f"noncanonical repository path: {path}")
    require(not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts
            and "\\" not in path, f"unsafe repository path: {path}")
    if not allow_ignored:
        validate_repository_reference(path)


class SnapshotReader:
    """Cache retained bytes; old source drift is missing history, not a new original."""

    def __init__(self, root: Path | None):
        self.root = root
        self.cache: dict[str, bytes] = {}

    def read(self, path: str) -> bytes:
        require(self.root is not None, f"{path}: retained verification requires a repository root")
        if path not in self.cache:
            self.cache[path] = retained_bytes(self.root, path)
        return self.cache[path]

    def snapshot(self, snapshot: dict, fallback: str | None = None,
                 current: bool = False) -> bytes | None:
        retained = snapshot["retained_path"]
        if retained is not None:
            check_path(retained, allow_ignored=True)
        if snapshot["state"] != "known" or self.root is None:
            return None
        path = retained or fallback
        if path is None or ignored_path(path):
            return None
        check_path(path)
        if retained is None and not current and not repository_path(self.root, path).is_file():
            return None
        value = self.read(path)
        if digest(value) != snapshot["sha256"]:
            label = "current snapshot drift" if current else "pinned snapshot drift"
            require(retained is None and not current, f"{path}: {label}")
            return None
        return value


def verify_contexts(contexts: dict, blobs: SnapshotReader) -> tuple[dict, dict, list[str]]:
    verified, vendors, limitations = {}, {}, []
    for path, context in contexts.items():
        pinned = context["role"] in ("license", "vendor", "provenance_review")
        if pinned:
            check_path(path)
            require(context["snapshot"]["state"] == "known", f"{path}: context must be pinned")
        value = blobs.snapshot(context["snapshot"], path, current=pinned)
        if value is None:
            limitations.append(f"{path}: context bytes unavailable")
        else:
            verified[path] = value
            if context["role"] == "vendor":
                manifest = typed_artifact(value, path, "vendor_manifest")
                for reference in manifest["license_refs"]:
                    check_path(reference)
                    require(reference in contexts and contexts[reference]["role"] == "license",
                            f"{path}: vendor license must reference a pinned license context")
                vendors[path] = manifest
    return verified, vendors, limitations


def known_versions(owner: dict) -> set[str]:
    snapshots = owner["snapshots"]
    return {item["sha256"] for item in [snapshots["relocated"], *snapshots["history"], snapshots["current"]]
            if item["state"] == "known"}


def captured_owner_bytes(snapshot: dict, path: str, parent: dict | None,
                         blobs: SnapshotReader) -> bytes | None:
    value = blobs.snapshot(snapshot, path)
    if value is None and parent is not None and snapshot["state"] == "known":
        snapshots = parent["snapshots"]
        for version in [snapshots["relocated"], *snapshots["history"], snapshots["current"]]:
            if version["sha256"] == snapshot["sha256"]:
                value = blobs.snapshot(version, path)
                break
    return value


def verify_origin(origin: dict, owners: dict, blobs: SnapshotReader, debt: list[str]) -> bytes | None:
    path, snapshot, selection = origin["path"], origin["snapshot"], origin["selection"]
    check_path(path, allow_ignored=origin["kind"] == "source")
    require(snapshot["state"] != "not_applicable", f"{path}: declared origin cannot be inapplicable")
    require(selection["kind"] != "byte_ranges" or snapshot["state"] == "known",
            f"{path}: byte ranges require a pinned full original identity")
    parent = None
    if origin["kind"] == "owner":
        parent = owners.get(path)
        if parent is None or snapshot["sha256"] not in known_versions(parent):
            debt.append(f"origin owner version unresolved: {path}")
        elif parent["status"] != "provenance_reviewed":
            debt.append(f"origin owner provenance incomplete: {path}")
    original = captured_owner_bytes(snapshot, path, parent, blobs)
    if original is None:
        debt.append(f"original pinned bytes unavailable: {path}")
    if selection["kind"] == "unresolved":
        debt.append(f"source partition unresolved: {path}")
        return None
    if selection["kind"] == "whole_file":
        return original
    previous_end, pieces = 0, []
    for item in selection["ranges"]:
        start, end = item["start"], item["end"]
        require(type(start) is int and type(end) is int and previous_end <= start < end,
                f"{path}: invalid or overlapping byte ranges")
        previous_end = end
        if original is not None:
            require(end <= len(original), f"{path}: byte range exceeds pinned original")
            fragment = original[start:end]
            require(digest(fragment) == item["sha256"], f"{path}: selected fragment hash mismatch")
            pieces.append(fragment)
    return b"".join(pieces) if original is not None else None


def verify_owner(owner: dict, owners: dict, contexts: dict, vendors: dict, blobs: SnapshotReader) -> list[str]:
    path, kind, snapshots = owner["destination"], owner["kind"], owner["snapshots"]
    debt = list(owner["debt"])
    require(snapshots["current"]["state"] == "known", f"{path}: current identity must be known")
    current = blobs.snapshot(snapshots["current"], path, current=True)
    # An archived binding cannot substitute for the actual destination.
    if blobs.root is not None:
        require(digest(blobs.read(path)) == snapshots["current"]["sha256"], f"{path}: current snapshot drift")
    if current is None:
        debt.append("current bytes not checked")
    validate_owner_operation(owner)
    selected = [verify_origin(origin, owners, blobs, debt) for origin in owner["origins"]]
    relocated = blobs.snapshot(snapshots["relocated"], path)
    if kind != "authored" and relocated is None:
        debt.append("initial materialized bytes unavailable")
    for snapshot in snapshots["history"]:
        require(snapshot["state"] == "known", f"{path}: history entries must identify bytes")
        if blobs.snapshot(snapshot, path) is None:
            debt.append("historical destination bytes unavailable")
    validate_change_chain(owner, debt)
    if owner["materialization"] in ("verbatim", "selected_bytes") and all(item is not None for item in selected):
        require(snapshots["relocated"]["state"] == "known" and
                digest(b"".join(selected)) == snapshots["relocated"]["sha256"],
                f"{path}: materialization hash mismatch")
    if owner["materialization"] == "not_recorded":
        debt.append("materialization not recorded")
    missing_links = [origin for origin in owner["origins"] if origin["kind"] == "owner" and (
        origin["path"] not in owners or origin["snapshot"]["sha256"] not in known_versions(owners[origin["path"]])
    )]
    for reference in owner["context_refs"]:
        check_path(reference, allow_ignored=True)
        if reference not in contexts:
            debt.append(f"context reference unresolved: {reference}")
            missing_links.append(reference)
        elif blobs.snapshot(contexts[reference]["snapshot"], reference) is None:
            debt.append(f"context bytes unavailable: {reference}")
    if owner["review_ref"] is not None:
        reference = owner["review_ref"]
        check_path(reference)
        if reference not in contexts:
            debt.append(f"review reference unresolved: {reference}")
            missing_links.append(reference)
        else:
            require(contexts[reference]["role"] == "provenance_review", f"{path}: wrong review context role")
    if kind == "vendor":
        vendor_ref = owner["vendor_ref"]
        check_path(vendor_ref)
        require(vendor_ref in contexts and contexts[vendor_ref]["role"] == "vendor",
                f"{path}: vendor must reference a pinned manifest context")
        if vendor_ref not in vendors:
            debt.append(f"vendor manifest bytes unavailable: {vendor_ref}")
        for reference in vendors.get(vendor_ref, {}).get("license_refs", []):
            if blobs.snapshot(contexts[reference]["snapshot"], reference, current=True) is None:
                debt.append(f"vendor license bytes unavailable: {reference}")
    if owner["status"] == "incomplete" and (missing_links or owner["materialization"] == "not_recorded"):
        require(owner["debt"], f"{path}: missing provenance needs explicit debt")
    return debt


def validate_owner_operation(owner: dict) -> None:
    path, kind, origins = owner["destination"], owner["kind"], owner["origins"]
    method, relocated = owner["materialization"], owner["snapshots"]["relocated"]
    identities = [canonical_digest(origin) for origin in origins]
    require(len(identities) == len(set(identities)), f"{path}: duplicate source origin")
    if kind == "authored":
        require(not origins and method == "authored" and relocated["state"] == "not_applicable",
                f"{path}: authored files cannot claim copied origins or relocation")
    else:
        require(origins and method != "authored" and relocated["state"] != "not_applicable",
                f"{path}: extracted/derived files need explicit origins and initial identity")
    if kind in ("copied", "vendor"):
        require(len(origins) == 1 and origins[0]["selection"]["kind"] == "whole_file",
                f"{path}: copied/vendor origin must be one whole file")
    if kind == "partitioned":
        require(len(origins) == 1 and origins[0]["selection"]["kind"] != "whole_file",
                f"{path}: partitioned origin needs an exact or unresolved selection")
    if kind == "composed":
        require(len(origins) >= 2, f"{path}: composition needs multiple origins")
    if method == "verbatim":
        require(len(origins) == 1 and origins[0]["selection"]["kind"] == "whole_file",
                f"{path}: verbatim materialization needs one whole file")


def validate_change_chain(owner: dict, debt: list[str]) -> None:
    snapshots, changes = owner["snapshots"], owner["changes"]
    path = owner["destination"]
    history = snapshots["history"]
    values = [item["sha256"] for item in history]
    require(len(values) == len(set(values)), f"{path}: duplicate historical snapshot")
    initial = snapshots["relocated"]
    if owner["kind"] == "authored":
        initial = history[0] if history else snapshots["current"]
        history = history[1:]
    if initial["state"] != "known":
        require(not changes, f"{path}: changes cannot fabricate an unknown initial identity")
        debt.append("initial snapshot identity not recorded")
        return
    chain = [initial["sha256"], *[item["sha256"] for item in history]]
    if chain[-1] != snapshots["current"]["sha256"]:
        chain.append(snapshots["current"]["sha256"])
    require(len(chain) == len(set(chain)), f"{path}: repeated destination identity in history")
    expected = list(zip(chain, chain[1:]))
    actual = [(change["from_sha256"], change["to_sha256"]) for change in changes]
    if not actual and expected:
        debt.append("destination changes not recorded")
    else:
        require(actual == expected, f"{path}: disconnected or reordered change history")


def owner_digest(owner: dict) -> str:
    return canonical_digest({key: value for key, value in owner.items() if key != "status"})


def context_digest(owner: dict, contexts: dict, vendors: dict) -> str:
    references = set(owner["context_refs"])
    if owner["kind"] == "vendor":
        references.add(owner["vendor_ref"])
        references.update(vendors[owner["vendor_ref"]]["license_refs"])
    require(owner["review_ref"] not in references, "review cannot bind its own artifact hash")
    return canonical_digest({path: contexts[path] for path in sorted(references)})


def typed_artifact(value: bytes, path: str, definition: str) -> dict:
    try:
        document = json.loads(value.decode("utf-8"), object_pairs_hook=strict_object,
                              parse_constant=reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: invalid {definition}: {error}") from error
    validate_schema(document, path, definition)
    return document


def verify_review(owner: dict, contexts: dict, vendors: dict, context_blobs: dict) -> None:
    path, reference = owner["destination"], owner["review_ref"]
    require(reference in contexts and contexts[reference]["role"] == "provenance_review" and
            reference in context_blobs, f"{path}: retained typed provenance review required")
    report = typed_artifact(context_blobs[reference], reference, "provenance_review")
    require(report["destination"] == path and report["record_sha256"] == owner_digest(owner) and
            report["context_sha256"] == context_digest(owner, contexts, vendors),
            f"{path}: stale or misbound provenance review")


def verify_uses(fragments: dict, owners: dict, blobs: SnapshotReader) -> list[str]:
    seen, limitations = set(), []
    for document in fragments.values():
        for use in document["uses"]:
            consumer, target, role = use["consumer"], use["target"], use["role"]
            check_path(consumer)
            if "include_spelling" in use:
                require(role == "include", f"{consumer}: include spelling on non-include use")
            identity = (consumer, target["kind"], target.get("path", target.get("name")), role)
            require(identity not in seen, f"{consumer}: duplicate use reference")
            seen.add(identity)
            debt = list(use["debt"])
            if consumer not in owners:
                debt.append("consumer owner unresolved")
            if target["kind"] == "owner":
                path = target["path"]
                check_path(path)
                parent = owners.get(path)
                if parent is None:
                    debt.append(f"target owner unresolved: {path}")
                observation = use.get("observation")
                if observation is not None:
                    observed = captured_owner_bytes(observation, path, parent, blobs)
                    if observation["state"] != "known" or parent is None or observation["sha256"] not in known_versions(parent):
                        debt.append(f"observed owner version unresolved: {path}")
                    elif observation["sha256"] != parent["snapshots"]["current"]["sha256"]:
                        limitations.append(f"{consumer}: historical use of {path}; not current qualification")
                    if observed is None:
                        debt.append(f"observed pinned bytes unavailable: {path}")
            else:
                require("observation" not in use, f"{consumer}: external input cannot assert repository owner bytes")
                limitations.append(f"{consumer}: external input {target['name']}: {target['reason']}")
            require(not debt or use["debt"], f"{consumer}: unresolved use needs explicit debt")
            limitations.extend(f"{consumer}: {item}" for item in debt)
    return limitations


def reject_origin_cycles(owners: dict) -> None:
    visiting, visited = set(), set()

    def visit(path: str) -> None:
        require(path not in visiting, f"{path}: cyclic origin ownership")
        if path in visited or path not in owners:
            return
        visiting.add(path)
        for origin in owners[path]["origins"]:
            if origin["kind"] == "owner":
                visit(origin["path"])
        visiting.remove(path)
        visited.add(path)

    for path in owners:
        visit(path)
