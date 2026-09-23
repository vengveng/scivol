from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional


C = "C"
C_FUSED = "C-fused"
CPP_JET = "C++ jet"
PY_C = "Py+C"
AD_FULL = "AD full"
AD_PARTIAL = "AD partial"
RUNTIME = "Runtime"
NOT_APPLICABLE = "N/A"
NATIVE_ENGINE = "native"
LEGACY_ENGINE = "legacy"
DENSITIES = (
    "Normal",
    "StudentT",
    "SkewT",
    "GED",
    "GeneralizedHyperbolic",
    "Hyperbolic",
    "NormalInverseGaussian",
    "VarianceGamma",
)
FUSED_DENSITIES = ("Normal", "StudentT", "SkewT", "GED")
GH_DENSITIES = (
    "GeneralizedHyperbolic",
    "Hyperbolic",
    "NormalInverseGaussian",
    "VarianceGamma",
)
ABSENT_SURFACE_IDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceContract:
    surface_id: str
    mean: Optional[str]
    volatility: Optional[str]
    density: str
    kind: str
    constrained: str
    log_nll: str
    log_grad: str
    log_hess: str
    theta_validation: str
    log_validation: str
    default_engine: str
    default_log_mode: Optional[bool]
    log_nll_wrapper: Optional[str] = None
    log_grad_wrapper: Optional[str] = None

    @property
    def label(self) -> str:
        parts = [part for part in (self.mean, self.volatility, self.density) if part is not None]
        return " + ".join(parts)

    def build(self):
        from ..components import (
            APARCH,
            ARMA,
            ARX,
            EGARCH,
            GARCH,
            GED,
            GJRGARCH,
            GeneralizedHyperbolic,
            Hyperbolic,
            HARX,
            Normal,
            NormalInverseGaussian,
            SkewT,
            StudentT,
            VarianceGamma,
        )

        mean_factories = {
            "ARMA": lambda: ARMA(1, 1),
            "ARX": lambda: _with_exog(ARX(1)),
            "HARX": lambda: _with_exog(HARX((1, 5))),
        }
        vol_factories = {
            "APARCH": lambda: APARCH(1, 1),
            "GARCH": lambda: GARCH(1, 1),
            "GJR-GARCH": lambda: GJRGARCH(1, 1),
            "EGARCH": lambda: EGARCH(1, 1),
        }
        density_factories = {
            "Normal": Normal,
            "StudentT": StudentT,
            "SkewT": SkewT,
            "GED": GED,
            "GeneralizedHyperbolic": GeneralizedHyperbolic,
            "Hyperbolic": Hyperbolic,
            "NormalInverseGaussian": NormalInverseGaussian,
            "VarianceGamma": VarianceGamma,
        }

        components = []
        if self.mean is not None:
            components.append(mean_factories[self.mean]())
        if self.volatility is not None:
            components.append(vol_factories[self.volatility]())
        components.append(density_factories[self.density]())

        spec = components[0]
        for component in components[1:]:
            spec = spec + component
        return spec


@dataclass(frozen=True)
class VerificationException:
    group: str
    surface_ids: tuple[str, ...]
    missing: str
    current: str
    target: str


@dataclass(frozen=True)
class NativeJetContract:
    surface_id: str
    status: str
    order_axes: tuple[str, ...]
    general_order: bool
    generated_fixed_orders: bool
    precompiled_orders: tuple[str, ...]
    additive_objective: bool
    sparse_requests: bool
    transformed_hessian: bool
    model_domain: bool
    theta_ad: bool
    transformed_ad: bool
    sparse_projection: bool
    fixed_runtime_parity: bool
    ad_evidence: str
    dispatch_evidence: str
    performance_evidence: str
    replacement_default: bool


def _with_exog(mean):
    mean.set_n_exog(1)
    return mean


def _surface_id(mean: Optional[str], volatility: Optional[str], density: str) -> str:
    mean_part = (mean or "none").lower()
    vol_part = (volatility or "none").lower()
    density_part = density.lower()
    return f"mean:{mean_part}+vol:{vol_part}+density:{density_part}"


def _suffix(density: str) -> str:
    return {"Normal": "normal", "StudentT": "studentt", "SkewT": "skewt", "GED": "ged"}[density]


def _log_wrappers(
    mean: Optional[str],
    volatility: Optional[str],
    density: str,
    log_nll: str,
) -> tuple[Optional[str], Optional[str]]:
    if log_nll != C_FUSED:
        return None, None

    suffix = _suffix(density)
    if mean is None and volatility == "GARCH":
        stem = f"_log_garch_ll_pq_{suffix}"
    elif mean is None and volatility == "GJR-GARCH":
        stem = f"_log_gjr_garch_ll_pq_{suffix}"
    elif mean is None and volatility == "EGARCH":
        stem = f"_log_egarch_ll_pq_{suffix}"
    elif mean == "ARMA" and volatility is None:
        stem = f"_log_arma_nll_pq_{suffix}"
    elif mean == "ARMA" and volatility == "GARCH":
        stem = f"_log_arma_garch_nll_pq_{suffix}"
    elif mean == "ARMA" and volatility == "GJR-GARCH":
        stem = f"_log_arma_gjr_garch_nll_pq_{suffix}"
    elif mean == "ARMA" and volatility == "EGARCH":
        stem = f"_log_arma_egarch_nll_pq_{suffix}"
    else:
        return None, None

    return stem, stem.replace("_nll_pq_", "_nll_grad_pq_").replace("_ll_pq_", "_ll_grad_pq_")


def _make_surface(
    *,
    mean: Optional[str],
    volatility: Optional[str],
    density: str,
    kind: str,
    constrained: str = C,
    log_nll: str = PY_C,
    log_grad: str = PY_C,
    log_hess: str = PY_C,
    theta_validation: str,
    log_validation: str,
    default_engine: str = LEGACY_ENGINE,
    default_log_mode: Optional[bool] = False,
) -> SurfaceContract:
    log_nll_wrapper, log_grad_wrapper = _log_wrappers(mean, volatility, density, log_nll)
    return SurfaceContract(
        surface_id=_surface_id(mean, volatility, density),
        mean=mean,
        volatility=volatility,
        density=density,
        kind=kind,
        constrained=constrained,
        log_nll=log_nll,
        log_grad=log_grad,
        log_hess=log_hess,
        theta_validation=theta_validation,
        log_validation=log_validation,
        default_engine=default_engine,
        default_log_mode=default_log_mode,
        log_nll_wrapper=log_nll_wrapper,
        log_grad_wrapper=log_grad_wrapper,
    )


def _extend(
    surfaces: list[SurfaceContract],
    *,
    mean: Optional[str],
    volatility: Optional[str],
    kind: str,
    densities: Iterable[str],
    log_nll: str,
    log_grad: str,
    theta_validation: Callable[[str], str],
    log_validation: Callable[[str], str],
    default_engine: str | Callable[[str], str] = LEGACY_ENGINE,
    default_log_mode: Optional[bool] | Callable[[str], Optional[bool]] = False,
) -> None:
    for density in densities:
        default_engine_for_density = (
            default_engine(density) if callable(default_engine) else default_engine
        )
        default_for_density = (
            default_log_mode(density) if callable(default_log_mode) else default_log_mode
        )
        surfaces.append(
            _make_surface(
                mean=mean,
                volatility=volatility,
                density=density,
                kind=kind,
                log_nll=log_nll if density in FUSED_DENSITIES else PY_C,
                log_grad=log_grad if density in FUSED_DENSITIES else PY_C,
                theta_validation=theta_validation(density),
                log_validation=log_validation(density),
                default_engine=default_engine_for_density,
                default_log_mode=default_for_density,
            )
        )


def _egarch_theta_validation(density: str) -> str:
    return AD_FULL


def _egarch_log_validation(density: str) -> str:
    return AD_FULL


def _arma_egarch_validation(density: str) -> str:
    return AD_FULL


def _meanx_gjr_theta_validation(density: str) -> str:
    return AD_FULL


def _meanx_egarch_theta_validation(density: str) -> str:
    return AD_FULL


def _surfaces() -> tuple[SurfaceContract, ...]:
    surfaces: list[SurfaceContract] = []

    _extend(
        surfaces,
        mean=None,
        volatility="GARCH",
        kind="vol only",
        densities=DENSITIES,
        log_nll=C_FUSED,
        log_grad=C_FUSED,
        theta_validation=lambda _density: AD_FULL,
        log_validation=lambda _density: AD_FULL,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )
    _extend(
        surfaces,
        mean=None,
        volatility="GJR-GARCH",
        kind="vol only",
        densities=("Normal", "StudentT", "SkewT"),
        log_nll=C_FUSED,
        log_grad=C_FUSED,
        theta_validation=lambda _density: AD_FULL,
        log_validation=lambda _density: AD_FULL,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )
    surfaces.append(
        _make_surface(
            mean=None,
            volatility="GJR-GARCH",
            density="GED",
            kind="vol only",
            theta_validation=AD_FULL,
            log_validation=AD_FULL,
            default_engine=NATIVE_ENGINE,
            default_log_mode=True,
        )
    )
    for density in GH_DENSITIES:
        surfaces.append(
            _make_surface(
                mean=None,
                volatility="GJR-GARCH",
                density=density,
                kind="vol only",
                theta_validation=AD_FULL,
                log_validation=AD_FULL,
                default_engine=NATIVE_ENGINE,
                default_log_mode=True,
            )
        )
    _extend(
        surfaces,
        mean=None,
        volatility="EGARCH",
        kind="vol only",
        densities=DENSITIES,
        log_nll=C_FUSED,
        log_grad=C_FUSED,
        theta_validation=_egarch_theta_validation,
        log_validation=_egarch_log_validation,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )
    _extend(
        surfaces,
        mean=None,
        volatility="APARCH",
        kind="vol only",
        densities=DENSITIES,
        log_nll=PY_C,
        log_grad=PY_C,
        theta_validation=lambda _density: AD_FULL,
        log_validation=lambda _density: AD_FULL,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )

    for density in DENSITIES:
        surfaces.append(
            _make_surface(
                mean="ARMA",
                volatility=None,
                density=density,
                kind="standalone mean",
                constrained=CPP_JET,
                log_nll=CPP_JET,
                log_grad=CPP_JET,
                log_hess=CPP_JET,
                theta_validation=AD_FULL,
                log_validation=AD_FULL,
                default_engine=NATIVE_ENGINE,
                default_log_mode=density in (*FUSED_DENSITIES,"VarianceGamma"),
            )
        )

    _extend(
        surfaces,
        mean="ARMA",
        volatility="GARCH",
        kind="linked joint",
        densities=DENSITIES,
        log_nll=C_FUSED,
        log_grad=C_FUSED,
        theta_validation=lambda _density: AD_FULL,
        log_validation=lambda _density: AD_FULL,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )
    _extend(
        surfaces,
        mean="ARMA",
        volatility="GJR-GARCH",
        kind="linked joint",
        densities=("Normal", "StudentT", "SkewT"),
        log_nll=C_FUSED,
        log_grad=C_FUSED,
        theta_validation=lambda _density: AD_FULL,
        log_validation=lambda _density: AD_FULL,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )
    surfaces.append(
        _make_surface(
            mean="ARMA",
            volatility="GJR-GARCH",
            density="GED",
            kind="linked joint",
            theta_validation=AD_FULL,
            log_validation=AD_FULL,
            default_engine=NATIVE_ENGINE,
            default_log_mode=True,
        )
    )
    for density in GH_DENSITIES:
        surfaces.append(
            _make_surface(
                mean="ARMA",
                volatility="GJR-GARCH",
                density=density,
                kind="linked joint",
                theta_validation=AD_FULL,
                log_validation=AD_FULL,
                default_engine=NATIVE_ENGINE,
                default_log_mode=True,
            )
        )
    _extend(
        surfaces,
        mean="ARMA",
        volatility="EGARCH",
        kind="linked joint",
        densities=DENSITIES,
        log_nll=C_FUSED,
        log_grad=C_FUSED,
        theta_validation=_arma_egarch_validation,
        log_validation=_arma_egarch_validation,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )
    _extend(
        surfaces,
        mean="ARMA",
        volatility="APARCH",
        kind="linked joint",
        densities=DENSITIES,
        log_nll=PY_C,
        log_grad=PY_C,
        theta_validation=lambda _density: AD_FULL,
        log_validation=lambda _density: AD_FULL,
        default_engine=NATIVE_ENGINE,
        default_log_mode=True,
    )

    for mean in ("ARX", "HARX"):
        for density in DENSITIES:
            surfaces.append(
                _make_surface(
                    mean=mean,
                    volatility=None,
                    density=density,
                    kind="standalone mean",
                    constrained=CPP_JET,
                    log_nll=CPP_JET,
                    log_grad=CPP_JET,
                    log_hess=CPP_JET,
                    theta_validation=AD_FULL,
                    log_validation=AD_FULL,
                    default_engine=NATIVE_ENGINE,
                    default_log_mode=density in (*FUSED_DENSITIES,"VarianceGamma"),
                )
            )
        for volatility in ("GARCH", "GJR-GARCH"):
            for density in DENSITIES:
                surfaces.append(
                    _make_surface(
                        mean=mean,
                        volatility=volatility,
                        density=density,
                        kind="linked joint",
                        constrained=CPP_JET,
                        log_nll=CPP_JET,
                        log_grad=CPP_JET,
                        log_hess=CPP_JET,
                        theta_validation=AD_FULL,
                        log_validation=AD_FULL,
                        default_engine=NATIVE_ENGINE,
                        default_log_mode=False,
                    )
                )
        _extend(
            surfaces,
            mean=mean,
            volatility="EGARCH",
            kind="linked joint",
            densities=DENSITIES,
            log_nll=PY_C,
            log_grad=PY_C,
            theta_validation=_meanx_egarch_theta_validation,
            log_validation=lambda _density: AD_FULL,
            default_log_mode=False,
        )

    return tuple(surfaces)


SURFACES = _surfaces()
SURFACE_BY_ID = {surface.surface_id: surface for surface in SURFACES}


def _native_jets() -> tuple[NativeJetContract, ...]:
    contracts = []
    for surface in SURFACES:
        if surface.mean in ("ARX", "HARX") and surface.volatility in ("GARCH", "GJR-GARCH"):
            contracts.append(
                NativeJetContract(
                    surface_id=surface.surface_id,
                    status="complete",
                    order_axes=(
                        "lags" if surface.mean == "ARX" else "horizons",
                        "n_exog", "p_vol", "q_vol",
                    ),
                    general_order=True,
                    generated_fixed_orders=True,
                    precompiled_orders=("columns=4,11", "columns=7,11"),
                    additive_objective=True,
                    sparse_requests=True,
                    transformed_hessian=True,
                    model_domain=True,
                    theta_ad=True,
                    transformed_ad=True,
                    sparse_projection=True,
                    fixed_runtime_parity=True,
                    ad_evidence="tests/test_native_joint_mean.py",
                    dispatch_evidence="tests/test_native_joint_mean.py",
                    performance_evidence="benchmark_native_joint.py",
                    replacement_default=True,
                )
            )
            continue
        is_aparch = surface.volatility == "APARCH"
        is_volatility_only = (
            surface.mean is None
            and surface.volatility in ("GARCH", "GJR-GARCH", "EGARCH", "APARCH")
        )
        is_joint = (
            surface.mean == "ARMA"
            and surface.volatility in ("GARCH", "GJR-GARCH", "EGARCH", "APARCH")
        )
        is_mean = surface.volatility is None and surface.mean in ("ARMA","ARX","HARX")
        if not (is_volatility_only or is_joint or is_mean):
            continue
        if is_mean:
            contracts.append(NativeJetContract(
                surface_id=surface.surface_id,status="complete",
                order_axes=("p_ar","q_ma") if surface.mean=="ARMA" else (
                    "lags" if surface.mean=="ARX" else "horizons","n_exog"),
                general_order=True,generated_fixed_orders=True,
                precompiled_orders=("11",) if surface.mean=="ARMA" else ("columns=4",),
                additive_objective=True,sparse_requests=True,transformed_hessian=True,
                model_domain=True,theta_ad=True,transformed_ad=True,sparse_projection=True,
                fixed_runtime_parity=True,ad_evidence="tests/test_native_mean.py",
                dispatch_evidence="tests/test_native_mean.py",
                performance_evidence="benchmark_native_mean.py",replacement_default=True,
            ))
            continue
        contracts.append(NativeJetContract(
            surface_id=surface.surface_id,
            status="complete",
            order_axes=("p_vol", "q_vol") if is_volatility_only else (
                "p_ar", "q_ma", "p_vol", "q_vol"
            ),
            general_order=True,
            generated_fixed_orders=True,
            precompiled_orders=(
                ("11", "12", "21", "22")
                if is_volatility_only else ("1111",)
            ),
            additive_objective=True,
            sparse_requests=True,
            transformed_hessian=True,
            model_domain=True,
            theta_ad=True,
            transformed_ad=True,
            sparse_projection=True,
            fixed_runtime_parity=True,
            ad_evidence=(
                "tests/test_native_aparch.py"
                if is_aparch else "tests/test_native_fit.py"
            ),
            dispatch_evidence=(
                "tests/test_native_aparch.py"
                if is_aparch else "tests/test_native_fit.py"
            ),
            performance_evidence=(
                "benchmark_native_aparch.py"
                if is_aparch else "benchmark_native_tensor.py"
            ),
            replacement_default=surface.default_engine == NATIVE_ENGINE,
        ))
    return tuple(contracts)


NATIVE_JETS = _native_jets()
NATIVE_JET_BY_ID = {contract.surface_id: contract for contract in NATIVE_JETS}


def _ids(predicate: Callable[[SurfaceContract], bool]) -> tuple[str, ...]:
    return tuple(surface.surface_id for surface in SURFACES if predicate(surface))


VERIFICATION_EXCEPTIONS: tuple[VerificationException, ...] = ()


def shipped_surface_ids() -> tuple[str, ...]:
    return tuple(surface.surface_id for surface in SURFACES)


def canonical_tensor_markdown() -> str:
    rows = [
        (None, "GARCH", "vol only"),
        (None, "GJR-GARCH", "vol only"),
        (None, "EGARCH", "vol only"),
        (None, "APARCH", "vol only"),
        ("ARMA", None, "standalone mean"),
        ("ARMA", "GARCH", "linked joint"),
        ("ARMA", "GJR-GARCH", "linked joint"),
        ("ARMA", "EGARCH", "linked joint"),
        ("ARMA", "APARCH", "linked joint"),
        ("ARX", None, "standalone mean"),
        ("ARX", "GARCH", "linked joint"),
        ("ARX", "GJR-GARCH", "linked joint"),
        ("ARX", "EGARCH", "linked joint"),
        ("HARX", None, "standalone mean"),
        ("HARX", "GARCH", "linked joint"),
        ("HARX", "GJR-GARCH", "linked joint"),
        ("HARX", "EGARCH", "linked joint"),
    ]
    lines = [
        "| Mean | Volatility | Kind | Normal | StudentT | SkewT | GED | GeneralizedHyperbolic | Hyperbolic | NormalInverseGaussian | VarianceGamma |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mean, volatility, kind in rows:
        values = []
        for density in DENSITIES:
            surface_id = _surface_id(mean, volatility, density)
            values.append("`F`" if surface_id in SURFACE_BY_ID else "`—`")
        lines.append(
            f"| `{mean or '—'}` | `{volatility or '—'}` | {kind} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines)


def verification_exceptions_markdown() -> str:
    if not VERIFICATION_EXCEPTIONS:
        return "No known verification exceptions."

    lines = [
        "| Surface group | Missing evidence | Current protection | Closure target |",
        "|---|---|---|---|",
    ]
    for item in VERIFICATION_EXCEPTIONS:
        lines.append(f"| {item.group} | {item.missing} | {item.current} | {item.target} |")
    return "\n".join(lines)


def native_jet_markdown() -> str:
    lines = [
        "| Jet family | Densities | Status | General order | Generated fixed path | Evidence | Default |",
        "|---|---|---|---|---|---|---|",
    ]
    groups = (
        ("GARCH", None, "GARCH", "`(p_vol,q_vol)`", "11, 12, 21, 22"),
        ("GJR-GARCH", None, "GJR-GARCH", "`(p_vol,q_vol)`", "11, 12, 21, 22"),
        ("EGARCH", None, "EGARCH", "`(p_vol,q_vol)`", "11, 12, 21, 22"),
        ("APARCH", None, "APARCH", "`(p_vol,q_vol)`", "11, 12, 21, 22"),
        ("ARMA + GARCH", "ARMA", "GARCH", "`(p_ar,q_ma,p_vol,q_vol)`", "1111"),
        ("ARMA + GJR-GARCH", "ARMA", "GJR-GARCH", "`(p_ar,q_ma,p_vol,q_vol)`", "1111"),
        ("ARMA + EGARCH", "ARMA", "EGARCH", "`(p_ar,q_ma,p_vol,q_vol)`", "1111"),
        ("ARMA + APARCH", "ARMA", "APARCH", "`(p_ar,q_ma,p_vol,q_vol)`", "1111"),
        ("ARMA", "ARMA", None, "`(p_ar,q_ma)`", "11"),
        ("ARX", "ARX", None, "`(lags,n_exog)`", "4 design columns"),
        ("HARX", "HARX", None, "`(horizons,n_exog)`", "4 design columns"),
        ("ARX + GARCH", "ARX", "GARCH", "`(lags,n_exog,p_vol,q_vol)`", "4/7 design columns, 11"),
        ("ARX + GJR-GARCH", "ARX", "GJR-GARCH", "`(lags,n_exog,p_vol,q_vol)`", "4/7 design columns, 11"),
        ("HARX + GARCH", "HARX", "GARCH", "`(horizons,n_exog,p_vol,q_vol)`", "4/7 design columns, 11"),
        ("HARX + GJR-GARCH", "HARX", "GJR-GARCH", "`(horizons,n_exog,p_vol,q_vol)`", "4/7 design columns, 11"),
    )
    for label, mean, volatility, order, fixed in groups:
        contracts = [
            contract for contract in NATIVE_JETS
            if SURFACE_BY_ID[contract.surface_id].mean == mean
            and SURFACE_BY_ID[contract.surface_id].volatility == volatility
        ]
        densities = ", ".join(
            SURFACE_BY_ID[contract.surface_id].density for contract in contracts
        )
        default = ", ".join(
            SURFACE_BY_ID[contract.surface_id].density for contract in contracts
            if contract.replacement_default
        ) or "Explicit only"
        benchmark = (
            "benchmark_native_mean.py" if volatility is None else "benchmark_native_joint.py"
            if mean in ("ARX", "HARX") else "benchmark_native_aparch.py"
            if volatility == "APARCH" else "benchmark_native_tensor.py"
        )
        lines.append(
            f"| {label} | {densities} | complete | {order} runtime | "
            f"template-generated; precompiled {fixed} | AD full; dispatch and "
            f"`{benchmark}` | {default} |"
        )
    return "\n".join(lines)
