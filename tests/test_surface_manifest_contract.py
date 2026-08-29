from __future__ import annotations

from pathlib import Path

import scivol._core as _core
from scivol._devtools.surface_manifest import (
    ABSENT_SURFACE_IDS,
    AD_FULL,
    C_FUSED,
    LEGACY_ENGINE,
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

    assert len(SURFACES) == 134
    assert len(surface_ids) == len(set(surface_ids))
    assert len(SURFACES) + len(ABSENT_SURFACE_IDS) == 136
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
    }
    assert {contract.surface_id for contract in NATIVE_JETS} == expected

    for contract in NATIVE_JETS:
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


def test_models_md_generated_blocks_match_manifest() -> None:
    text = Path("MODELS.md").read_text(encoding="utf-8")

    assert _generated_block(text, "tensor") == canonical_tensor_markdown()
    assert _generated_block(text, "verification-exceptions") == verification_exceptions_markdown()
    assert _generated_block(text, "native-jets") == native_jet_markdown()
