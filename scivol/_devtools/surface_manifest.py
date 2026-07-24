from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional


C = "C"
C_FUSED = "C-fused"
PY_C = "Py+C"
AD_FULL = "AD full"
AD_PARTIAL = "AD partial"
RUNTIME = "Runtime"
NOT_APPLICABLE = "N/A"

DENSITIES = ("Normal", "StudentT", "SkewT", "GED")
ABSENT_SURFACE_IDS = (
    "mean:arma+vol:none+density:studentt",
    "mean:arma+vol:none+density:skewt",
)


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
    default_log_mode: Optional[bool]
    log_nll_wrapper: Optional[str] = None
    log_grad_wrapper: Optional[str] = None

    @property
    def label(self) -> str:
        parts = [part for part in (self.mean, self.volatility, self.density) if part is not None]
        return " + ".join(parts)

    def build(self):
        from ..components import ARMA, ARX, EGARCH, GARCH, GED, GJRGARCH, HARX, Normal, SkewT, StudentT

        mean_factories = {
            "ARMA": lambda: ARMA(1, 1),
            "ARX": lambda: _with_exog(ARX(1)),
            "HARX": lambda: _with_exog(HARX((1, 5))),
        }
        vol_factories = {
            "GARCH": lambda: GARCH(1, 1),
            "GJR-GARCH": lambda: GJRGARCH(1, 1),
            "EGARCH": lambda: EGARCH(1, 1),
        }
        density_factories = {
            "Normal": Normal,
            "StudentT": StudentT,
            "SkewT": SkewT,
            "GED": GED,
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
    default_log_mode: Optional[bool] | Callable[[str], Optional[bool]] = False,
) -> None:
    for density in densities:
        default_for_density = (
            default_log_mode(density) if callable(default_log_mode) else default_log_mode
        )
        surfaces.append(
            _make_surface(
                mean=mean,
                volatility=volatility,
                density=density,
                kind=kind,
                log_nll=log_nll if density in DENSITIES else PY_C,
                log_grad=log_grad if density in DENSITIES else PY_C,
                theta_validation=theta_validation(density),
                log_validation=log_validation(density),
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
        default_log_mode=lambda density: False if density == "GED" else True,
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
        default_log_mode=lambda density: True if density == "Normal" else False,
    )
    surfaces.append(
        _make_surface(
            mean=None,
            volatility="GJR-GARCH",
            density="GED",
            kind="vol only",
            theta_validation=AD_FULL,
            log_validation=AD_FULL,
            default_log_mode=False,
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
        default_log_mode=False,
    )

    for density in ("Normal", "GED"):
        surfaces.append(
            _make_surface(
                mean="ARMA",
                volatility=None,
                density=density,
                kind="standalone mean",
                log_nll=C_FUSED,
                log_grad=C_FUSED,
                theta_validation=AD_FULL,
                log_validation=AD_FULL,
                default_log_mode=False,
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
        default_log_mode=False,
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
        default_log_mode=False,
    )
    surfaces.append(
        _make_surface(
            mean="ARMA",
            volatility="GJR-GARCH",
            density="GED",
            kind="linked joint",
            theta_validation=AD_FULL,
            log_validation=AD_FULL,
            default_log_mode=False,
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
        default_log_mode=False,
    )

    for mean in ("ARX", "HARX"):
        for density in DENSITIES:
            surfaces.append(
                _make_surface(
                    mean=mean,
                    volatility=None,
                    density=density,
                    kind="standalone mean",
                    theta_validation=AD_FULL,
                    log_validation=AD_FULL,
                    default_log_mode=False,
                )
            )
        _extend(
            surfaces,
            mean=mean,
            volatility="GARCH",
            kind="linked joint",
            densities=DENSITIES,
            log_nll=PY_C,
            log_grad=PY_C,
            theta_validation=lambda _density: AD_FULL,
            log_validation=lambda _density: AD_FULL,
            default_log_mode=False,
        )
        _extend(
            surfaces,
            mean=mean,
            volatility="GJR-GARCH",
            kind="linked joint",
            densities=DENSITIES,
            log_nll=PY_C,
            log_grad=PY_C,
            theta_validation=_meanx_gjr_theta_validation,
            log_validation=lambda _density: AD_FULL,
            default_log_mode=False,
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
        ("ARMA", None, "standalone mean"),
        ("ARMA", "GARCH", "linked joint"),
        ("ARMA", "GJR-GARCH", "linked joint"),
        ("ARMA", "EGARCH", "linked joint"),
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
        "| Mean | Volatility | Kind | Normal | StudentT | SkewT | GED |",
        "|---|---|---|---|---|---|---|",
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
