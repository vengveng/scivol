from __future__ import annotations

from pathlib import Path

import scivol._core as _core
from scivol._devtools.surface_manifest import (
    ABSENT_SURFACE_IDS,
    AD_FULL,
    C_FUSED,
    CPP_JET,
    DENSITIES,
    NATIVE_ENGINE,
    NATIVE_JETS,
    PY_C,
    SURFACE_BY_ID,
    SURFACES,
    VERIFICATION_EXCEPTIONS,
    canonical_tensor_markdown,
    native_jet_markdown,
    verification_exceptions_markdown,
)
from scivol.roles import Role


def _generated_block(text: str, name: str) -> str:
    start = f"<!-- surface-manifest:{name}:start -->"
    end = f"<!-- surface-manifest:{name}:end -->"
    before, marker, after = text.partition(start)
    assert marker, f"missing {start}"
    block, marker, _ = after.partition(end)
    assert marker, f"missing {end}"
    return block.strip()


def test_manifest_declares_current_univariate_surface_tensor() -> None:
    surface_ids = [surface.surface_id for surface in SURFACES]

    assert len(SURFACES) == 152
    assert len(surface_ids) == len(set(surface_ids))
    assert len(SURFACES) + len(ABSENT_SURFACE_IDS) == 152
    assert set(ABSENT_SURFACE_IDS).isdisjoint(surface_ids)


def test_manifest_builds_public_specs() -> None:
    for surface in SURFACES:
        spec = surface.build()

        assert callable(spec.fit), surface.surface_id
        assert callable(spec.loglikelihood), surface.surface_id
        assert callable(spec.score), surface.surface_id
        assert callable(spec.hessian), surface.surface_id
        assert callable(spec.filter), surface.surface_id
        assert callable(spec.fix), surface.surface_id
        assert callable(spec.simulate), surface.surface_id
        assert spec.get_component(Role.DENSITY) is not None
        if surface.mean is not None:
            assert spec.get_component(Role.MEAN) is not None
        if surface.volatility is not None:
            assert spec.get_component(Role.VOLATILITY) is not None


def test_fused_log_claims_have_core_wrappers() -> None:
    for surface in SURFACES:
        if surface.log_nll != C_FUSED:
            continue

        assert surface.log_nll_wrapper is not None, surface.surface_id
        assert surface.log_grad_wrapper is not None, surface.surface_id
        assert hasattr(_core, surface.log_nll_wrapper), surface.surface_id
        assert hasattr(_core, surface.log_grad_wrapper), surface.surface_id


def test_specific_log_path_regressions_are_encoded() -> None:
    garch_normal = SURFACE_BY_ID["mean:none+vol:garch+density:normal"]
    garch_ged = SURFACE_BY_ID["mean:none+vol:garch+density:ged"]
    arma_garch_ged = SURFACE_BY_ID["mean:arma+vol:garch+density:ged"]
    gjr_normal = SURFACE_BY_ID["mean:none+vol:gjr-garch+density:normal"]
    gjr_studentt = SURFACE_BY_ID["mean:none+vol:gjr-garch+density:studentt"]
    gjr_ged = SURFACE_BY_ID["mean:none+vol:gjr-garch+density:ged"]
    egarch_normal = SURFACE_BY_ID["mean:none+vol:egarch+density:normal"]
    arma_egarch_ged = SURFACE_BY_ID["mean:arma+vol:egarch+density:ged"]

    assert garch_ged.log_nll == C_FUSED
    assert garch_ged.log_grad == C_FUSED
    assert gjr_ged.log_nll == PY_C
    assert gjr_ged.log_grad == PY_C
    assert garch_normal.default_log_mode is True
    assert garch_ged.default_log_mode is True
    assert garch_normal.default_engine == NATIVE_ENGINE
    assert garch_ged.default_engine == NATIVE_ENGINE
    assert arma_garch_ged.default_engine == NATIVE_ENGINE
    assert arma_garch_ged.default_log_mode is True
    assert gjr_normal.default_log_mode is True
    assert gjr_studentt.default_log_mode is True
    assert gjr_studentt.default_engine == NATIVE_ENGINE
    assert egarch_normal.default_log_mode is True
    assert egarch_normal.default_engine == NATIVE_ENGINE
    assert arma_egarch_ged.default_log_mode is True
    assert arma_egarch_ged.default_engine == NATIVE_ENGINE


def test_native_defaults_cover_the_promoted_fixed_mle_rows() -> None:
    native_defaults = {
        surface.surface_id
        for surface in SURFACES
        if surface.default_engine == NATIVE_ENGINE
    }
    expected = {
        surface.surface_id
        for surface in SURFACES
        if (
            surface.mean in (None, "ARMA")
            and surface.volatility in ("GARCH", "GJR-GARCH", "APARCH")
        ) or surface.volatility == "EGARCH" and surface.mean in (None, "ARMA")
        or surface.volatility is None and surface.mean in ("ARMA","ARX","HARX")
        or surface.mean in ("ARX", "HARX") and surface.volatility in ("GARCH", "GJR-GARCH", "EGARCH", "APARCH")
    }

    assert native_defaults == expected


def test_complete_native_jets_have_machine_checked_completion_evidence() -> None:
    expected = {
        surface.surface_id
        for surface in SURFACES
        if (
            surface.volatility in ("GARCH", "GJR-GARCH", "APARCH")
            and surface.mean in (None, "ARMA")
        ) or surface.volatility == "EGARCH" and surface.mean in (None, "ARMA")
        or surface.volatility is None and surface.mean in ("ARMA","ARX","HARX")
        or surface.mean in ("ARX", "HARX") and surface.volatility in ("GARCH", "GJR-GARCH", "EGARCH", "APARCH")
    }
    assert {contract.surface_id for contract in NATIVE_JETS} == expected

    for contract in NATIVE_JETS:
        surface = SURFACE_BY_ID[contract.surface_id]
        if (surface.density == "VarianceGamma" and surface.volatility != "APARCH" and not (
            surface.mean in ("ARX", "HARX") and surface.volatility == "EGARCH"
        )) or (
            surface.volatility == "EGARCH" and surface.mean is not None
        ):
            assert contract.status == "incomplete"
            assert any(contract.surface_id in item.surface_ids for item in VERIFICATION_EXCEPTIONS)
        else:
            assert contract.status == "complete"
        assert contract.order_axes
        assert contract.general_order
        assert contract.generated_fixed_orders
        assert contract.precompiled_orders
        assert contract.additive_objective
        assert contract.sparse_requests
        assert contract.transformed_hessian
        assert contract.model_domain
        assert contract.theta_ad
        assert contract.transformed_ad
        assert contract.sparse_projection
        assert contract.fixed_runtime_parity
        assert Path(contract.ad_evidence).is_file()
        assert Path(contract.dispatch_evidence).is_file()
        assert Path(contract.performance_evidence).is_file()


def test_verification_exceptions_cover_non_full_validation() -> None:
    surface_ids = set(SURFACE_BY_ID)
    non_full = {
        surface.surface_id
        for surface in SURFACES
        if surface.theta_validation != AD_FULL or surface.log_validation != AD_FULL
    }
    covered: set[str] = set()

    for exception in VERIFICATION_EXCEPTIONS:
        assert exception.surface_ids, exception.group
        assert set(exception.surface_ids) <= surface_ids
        covered.update(exception.surface_ids)

    assert non_full <= covered


def test_promoted_joint_aparch_contracts_cover_all_densities() -> None:
    contracts = {contract.surface_id: contract for contract in NATIVE_JETS}
    promoted = [surface for surface in SURFACES
                if surface.mean in ("ARX", "HARX") and surface.volatility == "APARCH"]
    assert len(promoted) == 16
    assert {(surface.mean, surface.density) for surface in promoted} == {
        (mean, density) for mean in ("ARX", "HARX") for density in DENSITIES
    }
    for surface in promoted:
        contract = contracts[surface.surface_id]
        assert contract.status == "complete"
        assert contract.replacement_default
        assert surface.default_engine == NATIVE_ENGINE
        assert surface.default_log_mode is False
        assert surface.constrained == surface.log_nll == surface.log_grad == surface.log_hess == CPP_JET
        assert surface.theta_validation == surface.log_validation == AD_FULL
        assert contract.order_axes == (
            "lags" if surface.mean == "ARX" else "horizons", "n_exog", "p_vol", "q_vol",
        )
        assert contract.precompiled_orders == ("columns=4,11", "columns=7,32")
        assert contract.ad_evidence == contract.dispatch_evidence == "tests/test_native_joint_mean.py"
        assert contract.performance_evidence == "benchmark_native_joint.py"
    assert len(NATIVE_JETS) == len(contracts) == 152
    for label in ("ARX + APARCH", "HARX + APARCH"):
        row = next(line for line in native_jet_markdown().splitlines()
                   if line.startswith(f"| {label} |"))
        assert "| complete |" in row
        assert row.endswith(f"| {', '.join(DENSITIES)} |")


def test_aparch_completion_covers_every_mean_and_density() -> None:
    from scivol._native_fit import _uses_native_by_default

    promoted = [surface for surface in SURFACES if surface.volatility == "APARCH"]
    affected = {surface.surface_id for surface in promoted}
    assert len(affected) == 4 * len(DENSITIES)
    assert {(surface.mean, surface.density) for surface in promoted} == {
        (mean, density) for mean in (None, "ARMA", "ARX", "HARX") for density in DENSITIES
    }
    assert all(contract.status == "complete" and contract.replacement_default
               for contract in NATIVE_JETS if contract.surface_id in affected)
    assert all(surface.default_engine == NATIVE_ENGINE for surface in promoted)
    assert all(affected.isdisjoint(item.surface_ids) for item in VERIFICATION_EXCEPTIONS)
    assert {item.group for item in VERIFICATION_EXCEPTIONS} == {
        "Joint EGARCH, fixed-layout code generation",
        "VarianceGamma, except APARCH and joint ARX/HARX-EGARCH",
    }
    for surface in promoted:
        if surface.mean in (None, "ARMA"):
            assert _uses_native_by_default(surface.build())


def test_promoted_joint_aparch_default_routing_preserves_explicit_controls() -> None:
    from scivol._joint_mean_fit import supports_joint_mean, uses_joint_native_by_default
    from scivol._mixins import FitsMixin

    for surface in SURFACES:
        if surface.mean not in ("ARX", "HARX") or surface.volatility != "APARCH":
            continue
        spec = surface.build()
        assert supports_joint_mean(spec)
        assert uses_joint_native_by_default(spec)
        for log_mode in (None, False, True):
            options = {} if log_mode is None else {"log_mode": log_mode}
            for solver, expected in ((None, True), ("native", True), ("legacy", False), ("slsqp", False)):
                kwargs = options if solver is None else dict(options, solver=solver)
                assert FitsMixin._should_use_native_engine(
                    spec, method="mle", has_auto_components=False,
                    has_exogenous_data=True, force_legacy_engine=False, kwargs=kwargs,
                ) is expected


def test_promoted_joint_egarch_contract_matches_public_dispatch() -> None:
    from scivol._joint_mean_fit import uses_joint_native_by_default
    from scivol._devtools.surface_manifest import CPP_JET

    contracts = {contract.surface_id: contract for contract in NATIVE_JETS}
    promoted = [surface for surface in SURFACES
                if surface.mean in ("ARX", "HARX") and surface.volatility == "EGARCH"]
    assert len(promoted) == 16
    for surface in promoted:
        assert uses_joint_native_by_default(surface.build())
        assert surface.default_engine == NATIVE_ENGINE
        assert surface.constrained == surface.log_nll == surface.log_grad == surface.log_hess == CPP_JET
        assert contracts[surface.surface_id].status == "incomplete"
        assert contracts[surface.surface_id].precompiled_orders == ("columns=4,11", "columns=7,32")
        exceptions = [item for item in VERIFICATION_EXCEPTIONS if surface.surface_id in item.surface_ids]
        assert [item.group for item in exceptions] == [
            "Joint EGARCH, fixed-layout code generation"
        ]


def test_fixed_layout_debt_cannot_be_reported_as_complete() -> None:
    affected = {
        surface.surface_id for surface in SURFACES
        if surface.volatility == "EGARCH" and surface.mean is not None
    }
    debt = next(item for item in VERIFICATION_EXCEPTIONS
                if item.group == "Joint EGARCH, fixed-layout code generation")
    assert set(debt.surface_ids) == affected
    for contract in NATIVE_JETS:
        if contract.surface_id in affected:
            assert contract.status == "incomplete"


def test_models_md_generated_blocks_match_manifest() -> None:
    text = Path("MODELS.md").read_text(encoding="utf-8")

    assert _generated_block(text, "tensor") == canonical_tensor_markdown()
    assert _generated_block(text, "verification-exceptions") == verification_exceptions_markdown()
    assert _generated_block(text, "native-jets") == native_jet_markdown()
