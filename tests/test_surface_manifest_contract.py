from __future__ import annotations

from pathlib import Path

import scivol._core as _core
from scivol._devtools.surface_manifest import (
    ABSENT_SURFACE_IDS,
    AD_FULL,
    C_FUSED,
    PY_C,
    SURFACE_BY_ID,
    SURFACES,
    VERIFICATION_EXCEPTIONS,
    canonical_tensor_markdown,
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

    assert len(SURFACES) == 126
    assert len(surface_ids) == len(set(surface_ids))
    assert len(SURFACES) + len(ABSENT_SURFACE_IDS) == 128
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
    gjr_normal = SURFACE_BY_ID["mean:none+vol:gjr-garch+density:normal"]
    gjr_studentt = SURFACE_BY_ID["mean:none+vol:gjr-garch+density:studentt"]
    gjr_ged = SURFACE_BY_ID["mean:none+vol:gjr-garch+density:ged"]

    assert garch_ged.log_nll == C_FUSED
    assert garch_ged.log_grad == C_FUSED
    assert gjr_ged.log_nll == PY_C
    assert gjr_ged.log_grad == PY_C
    assert garch_normal.default_log_mode is True
    assert garch_ged.default_log_mode is False
    assert gjr_normal.default_log_mode is True
    assert gjr_studentt.default_log_mode is False


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
