"""Exogenous regressors: where they go and what they are called.

`x` is one block of regressors, or a mapping from layer ("mean", "volatility")
to a block. A bare block goes to the only layer that takes regressors. A block
is a DataFrame (its columns name the regressors), a named Series, or an array
(columns named x[1], x[2], ...). The regressors' names are the names of their
coefficients. A specification fitted to data takes every column it is given;
a model whose regressors are already named (a fitted model, or a candidate
from a search) takes its columns by name, or an array of exactly its width in
order.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .roles import Role

LAYERS = ("mean", "volatility")


@dataclass(frozen=True, slots=True)
class Regressors:
    """Regressor rows (rows x columns, C-contiguous float64) and their names."""

    values: NDArray[np.float64]
    names: tuple[str, ...]
    named: bool  # names came with the data (a DataFrame or a named Series)

    def rows(self, start: int, stop: Optional[int] = None) -> "Regressors":
        return Regressors(np.ascontiguousarray(self.values[start:stop]), self.names, self.named)

    def select(self, names: Sequence[str]) -> Optional[NDArray[np.float64]]:
        """The columns `names`, in that order; an unnamed block of exactly that
        width is taken in order."""
        names = tuple(names)
        if not names:
            return None
        missing = [name for name in names if name not in self.names]
        if not missing:
            index = [self.names.index(name) for name in names]
            return np.ascontiguousarray(self.values[:, index])
        if not self.named and len(self.names) == len(names):
            return self.values
        raise ValueError(f"x has no column {', '.join(missing)}; the model's regressors are {', '.join(names)}, "
                         f"and x has {', '.join(self.names)}.")


def default_names(columns: int) -> tuple[str, ...]:
    return tuple(f"x[{i}]" for i in range(1, columns + 1))


def block(x: Any, rows: Optional[int] = None) -> Optional[Regressors]:
    """One block of regressors with its names, checked to be finite with `rows` rows."""
    if x is None or isinstance(x, Regressors):
        out = x
    else:
        columns = getattr(x, "columns", None)
        if columns is not None:  # a DataFrame
            names, named = tuple(str(c) for c in columns), True
            values = np.asarray(x.to_numpy(), dtype=np.float64)
        else:
            values = np.asarray(x, dtype=np.float64)
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2:
                raise ValueError("Regressors must be a 1-D or 2-D array, a Series or a DataFrame.")
            series_name = getattr(x, "name", None) if hasattr(x, "index") else None
            named = series_name is not None
            names = (str(series_name),) if named else default_names(values.shape[1])
        if len(set(names)) != len(names):
            raise ValueError(f"Regressor names must be distinct; x has columns {', '.join(names)}.")
        if not np.isfinite(values).all():
            raise ValueError("Regressors contain NaN or infinite values")
        out = Regressors(np.ascontiguousarray(values), names, named)
    if out is not None and rows is not None and out.values.shape[0] != rows:
        raise ValueError(f"Regressors must have the same number of rows as the data. Got {out.values.shape[0]} "
                         f"and {rows}.")
    return out


def layers(x: Any) -> dict[str, Any]:
    """`x` by layer: a mapping's entries, checked, or {None: x} for a bare block."""
    if x is None:
        return {}
    if isinstance(x, Mapping):
        unknown = [str(k) for k in x if k not in LAYERS]
        if unknown:
            raise ValueError(f"x names unknown layers {', '.join(unknown)}; the layers are 'mean' and 'volatility'.")
        return {k: v for k, v in x.items() if v is not None}
    return {None: x}


def mean_regressors(spec: Any, x: Any, rows: Optional[int] = None) -> Optional[Regressors]:
    """The mean's regressors in `x` for `spec`: a bare block goes to the only
    layer that takes regressors, which is the mean. A model whose regressors are
    already named ignores a bare block it has no regressors for."""
    entries = layers(x)
    if not entries:
        return None
    if "volatility" in entries:
        volatility = spec.get_component(Role.VOLATILITY)
        raise ValueError(f"{volatility.signature if volatility is not None else 'the variance'} takes no regressors.")
    definition = getattr(spec, "_native_definition", None)
    if spec.get_component(Role.MEAN) is None and not (definition is not None and None in entries):
        raise ValueError(f"x was passed, but {spec} has no component that takes regressors; add a mean such as "
                         "ARMA(0, 0).")
    return block(entries.get("mean", entries.get(None)), rows)


def rows(x: Any, start: int, stop: int) -> Any:
    """Rows start..stop of `x` in whatever form it was given."""
    if x is None:
        return None
    if isinstance(x, Mapping):
        return {k: rows(v, start, stop) for k, v in x.items()}
    if isinstance(x, Regressors):
        return x.rows(start, stop)
    if hasattr(x, "iloc"):
        return x.iloc[start:stop]
    return np.asarray(x)[start:stop]


def length(x: Any) -> Optional[int]:
    """The number of rows of `x` (of each layer, which must agree)."""
    if x is None:
        return None
    if isinstance(x, Mapping):
        counts = {length(v) for v in x.values() if v is not None}
        if len(counts) > 1:
            raise ValueError("Each layer's regressors must have the same number of rows.")
        return counts.pop() if counts else None
    if isinstance(x, Regressors):
        return x.values.shape[0]
    return len(x)
