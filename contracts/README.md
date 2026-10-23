# Phase1 ledger validation

Run from the repository root with Python, pytest and jsonschema installed:

```sh
python -m pytest -q tests/contracts
```

Missing jsonschema is a collection failure, never a skipped validation. These tests
do not import scivol, execute retained tests, build native code, or run AD.

`tests/contracts/ledger_validation.py` validates the three embedded Draft 2020-12
schemas with jsonschema. Schema digests pin the reviewed 1.0.0 contracts so a ledger
cannot weaken its own validator. Changing a schema requires an explicit validator
review. The helper supplies a closed schema for the existing version-1
`source_transfers.json` and `core_adaptations.json` formats, which have no embedded
schemas.

## Shared V2 provenance

`schemas/source_provenance.schema.json` and
`tests/contracts/provenance_validation.py` define one format and checker for every
producer. The loader discovers top-level `contracts/*.json` fragments by the
`ledger: source_provenance`, `schema_version: 2.0.0` header, regardless of filename.
The existing V1 source-transfer and core-adaptation checks still run until those
files are converted. Unknown headers and unsupported V2 versions fail.

Each shard declares its inventory identity and producer, then `owners`, `uses`,
`unassigned_origins`, `context_refs` and `debt`. An owner's canonical
repository-relative `destination` is its ID. Two shards cannot own the same path,
even with identical hashes. V2 cannot duplicate an owner in either checked V1
ledger. A `use` references an owner path or a declared external input; it cannot
restate authoritative current/original hashes. Reverse consumers are derived from
uses, not recorded as a second authority.

Owner kinds are `copied`, `partitioned`, `composed`, `authored`, `derived` and
`vendor`. Ordered origins each pin their own full-file original snapshot and
select `whole_file`, exact zero-based half-open `byte_ranges` with fragment hashes,
or an explicitly `unresolved` partition. Owner-origin edges capture a particular
version of another owner. They must be acyclic; include/use graphs need not be.
Include spelling is separate from the resolved owner path.

Every snapshot records `{state, sha256, retained_path, reason}`. `known` requires a
raw-byte SHA-256; `not_recorded` and `not_applicable` require null hashes and an
explanation. Only genuinely authored files have no origins and an inapplicable
relocated snapshot. Copied/derived files cannot use that classification to hide
missing history. `relocated` means the initial materialized destination, including
any extraction or initial edits, not necessarily an include-only relocation.
Keep it unchanged as `current` advances; preserve intermediate snapshots in
`history` and ordered `changes` with pinned from/to hashes and descriptions.

The validator checks actual current destination bytes, explicit retained snapshots,
range bounds/hashes and connected change history. `verbatim` and `selected_bytes`
materializations are replayed in memory when pinned originals are available.
`reviewed_adaptation` uses an independent review rather than executing conversion
scripts. A changed live source is never substituted for its historical original.
Historical use observations may differ from current hashes; they are not current
qualification. Missing versions, origins, selections or owners require explicit
debt and remain valid incomplete records. Known original hashes need not have
archived bytes for an incomplete record; unavailable bytes produce limitations,
not availability-dependent failures. Ignored historical/context pointers are
reported unavailable and are never read or required by permanent CI.

`provenance_reviewed` is achievable but separate from mathematical qualification.
It requires verified current/original/initial/history bytes, resolved reviewed
parent owners, no owner debt, and a retained typed review referenced by
`review_ref`. Its context has role `provenance_review` and a pinned snapshot. The
report fields are `{kind, destination, record_sha256, context_sha256, outcome, reviewer,
independent, rationale}`. `kind` is `provenance_review`, outcome is `passed`, and
independent is true. `record_sha256` is `owner_digest(owner)`, which excludes only
status. `context_sha256` uses `context_digest()` to pin the referenced context
records, including vendor/license bindings. A changed contract needs a new report.
Bare strings, failed/skip/xfail
outcomes, missing bytes and stale reports cannot support review. Reviewer
independence and the truth of adaptation narratives still need real review.

Vendor owners use `vendor_ref` to one pinned JSON manifest with
`{package, version, license_refs}` in a `vendor` context. Package/version and
license declarations are not duplicated in every file owner. Its license paths
resolve to pinned `license` contexts.
The checker verifies notice hashes; it does not certify license compliance.
Synthetic cross-shard examples and mutation controls are in
`test_provenance_v2.py`; they create no production evidence or ledger claims.

### Transitional debt

Worker fragments are not converted by this change. Additional legacy
`*_source_transfers.json` files are loaded and reported as unnormalized, not passed
through family-specific validators. Their structural/hash validation and complete
global ownership coverage remain debt. This includes mean, power, joint, profile,
joint-power, dependence and future causal-mean fragments until each uses V2. The
existing V1 checks are narrower and remain in force; passing CI does not mean all
provenance is normalized or Phase1 is qualified.

Conversion must wait for whole-worker handoff freeze. Preserve existing original
and relocated pins verbatim, and keep missing historical hashes explicit. Do not
fill them from current bytes or HEAD. Later shared polynomial-status changes in
joint/power backends and the profile jet, GH K1 curvature changes, and typed-helper
updates belong in destination change history with their review/evidence pointers,
not rewritten original transfer identities. Worker owners retain responsibility
for converting their fragments. No permanent per-family converter is shipped.

Checks cover unique identities, authority and inventory agreement, scope/owner
references in both directions, the full 173-capability promise, all twelve gates,
all seven request masks, declared order/design axes and coordinates, test-transfer
references, assertion identities, derived counts, and evidence references.
Negative tests mutate copies in memory; they never repair inventory data.

`core_adaptations.json` additionally checks unique derived IDs/paths, current owner
hashes, and existence of repository-confined source, consumer and regression-test
pointers. Integration status is separate from qualification. These derived IDs are
not automatically canonical bindings in `math_owners.json`. Regression test pointers
prove neither execution nor acceptance. Hashes for referenced consumers and tests
are not present in this format, so those checks establish path existence only.

## Snapshot identity

Retained Python, C++ and fixture files must match their working-tree byte hashes.
Python AST checks also verify definition hashes, source spans, imports, decorators
and assertion counts. C++ anchors check recorded lines and hashes only; they are
not semantic assertion or compiler checks. Index blob IDs are historical metadata,
not substitutes for the recorded dirty/untracked working-tree snapshot.

Source transfers distinguish three facts:

- `source_sha256` identifies original source bytes.
- `relocated_sha256` identifies those bytes after declared include relocation.
- `current_sha256` identifies the current destination after later adaptations.

The current destination must match `current_sha256`. Its hash need not equal the
original relocation hash. When retained original bytes match `source_sha256`, the
helper reconstructs the include-only relocation in memory and checks its hash.
When original bytes are absent or have evolved, it reports relocation integrity as
unverified. It does not substitute HEAD, an index blob, or newer bytes for the
original snapshot. An explicitly supplied `original_snapshot_root` must contain
the pinned bytes or validation fails.

The ignored `inventory_sources`, `source_inventory` and `assertion_artifact` paths
are provenance pointers, not permanent CI dependencies. Their absence cannot prove
historical integrity. Full frozen source coverage and individual Python assertion
hashes remain unverified without durable copies of those artifacts.

## Incremental progress and claims

There is no all-pending rule. Capabilities, owners and debts may remain pending or
become incomplete. Deferred test records may acquire case IDs, proposed tests and
adapters while coverage is under review. Availability and implementation bindings
do not automatically qualify anything. Validators never modify statuses.
Unresolved links on pending/incomplete/deferred records are reported as provisional
debt, never acceptance evidence. They do not force ignored files into CI. The same
links must resolve to typed records before any transfer/qualification claim passes.

A `transferred` assertion mapping requires `transfer_reviewed` evidence, not full
jet binary/performance qualification. It must cover every recorded assertion and
every independently reviewed required case, match the original node hash, bind the
original/new test file hashes, identify an existing new test node, and have passed
independent review. Supersession also needs an explicit independent justification.
Known skip/xfail/importorskip paths cannot establish required assertion coverage.
Goodall can keep dispositions deferred while staged case coverage remains pending.

A mathematically qualified capability separately requires passed evidence for all
its declared gates. Owner qualification requires equation/ownership, derivative,
range and reference gates plus its declared independent checks and implementation
binding. Resolved debt needs a reviewed scope gate. Mathematical reports include
every field in the declared evidence protocol. Subject-contract digests and current
source/report hashes prevent reuse after drift. Failed, pending, skipped or xfailed
gate outcomes cannot support qualification. Whole-phase closure additionally requires
all capabilities/owners qualified, all debts resolved and all required transfers
reviewed. Partial transfer or qualification never implies Phase1 completion.

These are structural checks on independently reviewed evidence, not a rerun of its
mathematics. Report contents, case expansion and reviewer independence still require
real human/development review; a hash or a synthetic fixture is not scientific proof.

## Required owner-side data

The existing `phase1_scope.json` schema admits only strings in `evidence_records`.
Keep that legacy array empty. Bare acceptance labels are rejected. To support a
claim, the ledger owner must create durable `contracts/evidence_registry.json` with
`schema_version: 1`, the matching `inventory_id`, and a `records` array of objects.
The loader discovers it when present; absence is allowed until a claim references it.
No registry or production evidence is created by these validation tests.

The closed record schema is `evidence_registry_schema()` in the helper. Both record
kinds require unique IDs, source bindings as `{path, sha256}`, a report binding, and
review metadata `{status, reviewer, independent, rationale}`. Pending reviews may
have a null report and need not read evidence artifacts. Accepted claims require a
passed independent review and retained non-ignored files.

- `transfer_reviewed` stores original node/hash, assertion IDs, covered and required
  case IDs, coverage state, new test ID/hash and optional supersession justification.
- `mathematical_gate` stores subject type/ID/contract digest, gate ID, outcome, case
  IDs and independent-check IDs. Capability, owner and debt subjects are distinct.

Reports are JSON objects with `schema_version: 1`, `record_sha256`, and `details`.
The record digest uses `canonical_digest()` on the record excluding `report`.
Transfer details contain `assertion_mapping_review`, `case_coverage_review` and
`oracle_tolerance_review`. Mathematical details contain the scope's complete evidence
protocol, matching record/gate/case/review identities, and structured numerical/build
data rather than acceptance labels. `subject_digest()` excludes status and evidence
links, so ordinary pending-to-qualified transitions do not invalidate evidence.
Synthetic positive/negative format examples are in `test_evidence_progress.py`;
they are never written as real ledger evidence.

Python node mappings can use the current schemas unchanged. Before transferring
support/native records, their owner must revise the existing null-only new-test/
adapter schema fields and review/version the schema change with the validator.
Ignored evidence is not read or required by permanent CI. Evidence used to support
an actual claim must first be retained in an allowed repository path.

No checks here certify expanded pytest cases, oracle independence, mathematical
correctness, request validity, runtime/fixed dispatch, compiler dependency closure,
backend ownership, allocations, frames, or performance. Detailed derivation review
and regression execution for core adaptations
remain pending evidence.
Include relocation proves only the original transfer, not later equation/domain
adaptations or newly derived owners. Reviewed changes update current hashes in the
source-transfer ledger without rewriting the original pins; no implementation
hash is hard-coded in the tests.
