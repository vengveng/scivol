# scivol/components/density.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..components.base import Component
from ..roles import Role

class ARMA(Component):
    role = Role.MEAN
    
    def __init__(self, p: int, q: int):
        self.p, self.q = p, q
        self.fitted_params = None
        self.fitted_values = None
        
    @property
    def signature(self): return f"ARMA({self.p},{self.q})"
    
    @property
    def n_params(self): return 1 + self.p + self.q  # constant + AR + MA
    
    def pack(self, params_dict):
        return np.array([params_dict['const']] + 
                         params_dict['ar'] + 
                         params_dict['ma'])
    
    def unpack(self, flat_params):
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}
            
        self.fitted_params = {
            'const': flat_params[0],
            'ar': flat_params[1:1+self.p].tolist(),
            'ma': flat_params[1+self.p:].tolist()
        }
        return self.fitted_params


class ARX(Component):
    role = Role.MEAN

    def __init__(self, lags: int = 0, *, constant: bool = True):
        if lags < 0:
            raise ValueError("lags must be non-negative")
        self.lags = int(lags)
        self.constant = bool(constant)
        self.n_exog: int = 0
        self.fitted_params = None
        self.fitted_values = None

    @property
    def signature(self):
        return f"ARX({self.lags})"

    @property
    def n_params(self):
        return int(self.constant) + self.lags + self.n_exog

    def set_n_exog(self, n_exog: int) -> None:
        if n_exog < 0:
            raise ValueError("n_exog must be non-negative")
        self.n_exog = int(n_exog)

    @property
    def required_hold_back(self) -> int:
        return self.lags

    def pack(self, params_dict):
        const = [params_dict["const"]] if self.constant else []
        ar = list(params_dict.get("ar", []))
        exog = list(params_dict.get("exog", []))
        return np.array(const + ar + exog, dtype=np.float64)

    def unpack(self, flat_params):
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}

        offset = 0
        const = 0.0
        if self.constant:
            const = float(flat_params[offset])
            offset += 1

        ar = flat_params[offset:offset + self.lags].tolist()
        offset += self.lags
        if self.n_exog == 0 and len(flat_params) > offset:
            self.n_exog = int(len(flat_params) - offset)
        exog = flat_params[offset:offset + self.n_exog].tolist()
        self.fitted_params = {"const": const, "ar": ar, "exog": exog}
        return self.fitted_params


class HARX(Component):
    role = Role.MEAN

    def __init__(self, horizons: Sequence[int], *, constant: bool = True):
        if not horizons:
            raise ValueError("horizons must contain at least one positive integer")
        cleaned = tuple(int(h) for h in horizons)
        if any(h <= 0 for h in cleaned):
            raise ValueError("horizons must be strictly positive")
        if list(cleaned) != sorted(cleaned):
            raise ValueError("horizons must be sorted in ascending order")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("horizons must be unique")

        self.horizons = cleaned
        self.constant = bool(constant)
        self.n_exog: int = 0
        self.fitted_params = None
        self.fitted_values = None

    @property
    def signature(self):
        horizons = ",".join(str(h) for h in self.horizons)
        return f"HARX({horizons})"

    @property
    def n_params(self):
        return int(self.constant) + len(self.horizons) + self.n_exog

    def set_n_exog(self, n_exog: int) -> None:
        if n_exog < 0:
            raise ValueError("n_exog must be non-negative")
        self.n_exog = int(n_exog)

    @property
    def required_hold_back(self) -> int:
        return max(self.horizons)

    def pack(self, params_dict):
        const = [params_dict["const"]] if self.constant else []
        har = list(params_dict.get("har", []))
        exog = list(params_dict.get("exog", []))
        return np.array(const + har + exog, dtype=np.float64)

    def unpack(self, flat_params):
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}

        offset = 0
        const = 0.0
        if self.constant:
            const = float(flat_params[offset])
            offset += 1

        n_har = len(self.horizons)
        har = flat_params[offset:offset + n_har].tolist()
        offset += n_har
        if self.n_exog == 0 and len(flat_params) > offset:
            self.n_exog = int(len(flat_params) - offset)
        exog = flat_params[offset:offset + self.n_exog].tolist()
        self.fitted_params = {"const": const, "har": har, "exog": exog}
        return self.fitted_params


class AutoMean(Component):
    """Placeholder for bounded automatic conditional-mean selection.

    The default grid screens a zero mean and every ARMA(p, q) combination
    through order two. When composed with automatic volatility or density
    components, only the best screened mean specifications enter the full
    cross-product search.
    """

    role = Role.MEAN
    CANDIDATES = ("Zero", "ARMA")

    def __init__(
        self,
        candidates: Optional[Sequence[str]] = None,
        *,
        max_p: int = 2,
        max_q: int = 2,
        candidate_budget: int = 120,
        finalists: int = 2,
    ) -> None:
        selected = tuple(candidates) if candidates is not None else self.CANDIDATES
        invalid = [name for name in selected if name not in self.CANDIDATES]
        if invalid:
            raise ValueError(
                f"Unknown AutoMean candidates {invalid}; valid choices are "
                f"{list(self.CANDIDATES)}."
            )
        if max_p < 0 or max_q < 0:
            raise ValueError("AutoMean max_p and max_q must be non-negative.")
        if candidate_budget <= 0:
            raise ValueError("AutoMean candidate_budget must be positive.")
        if finalists <= 0:
            raise ValueError("AutoMean finalists must be positive.")
        self.candidates = selected
        self.max_p = int(max_p)
        self.max_q = int(max_q)
        self.candidate_budget = int(candidate_budget)
        self.finalists = int(finalists)
        self.fitted_params: Dict[str, Any] = {}
        self._is_auto = True

    @property
    def signature(self) -> str:
        return "AutoMean"

    @property
    def n_params(self) -> int:
        return 0

    def pack(self, params_dict: Dict[str, Any]) -> np.ndarray:
        return np.array([], dtype=np.float64)

    def unpack(self, flat_params: np.ndarray) -> Dict[str, Any]:
        return {}

    def get_candidates(self) -> List[Tuple[str, Tuple[Any, ...]]]:
        candidates: List[Tuple[str, Tuple[Any, ...]]] = []
        if "Zero" in self.candidates:
            candidates.append(("Zero", ()))
        if "ARMA" in self.candidates:
            orders = [
                (p, q)
                for p in range(self.max_p + 1)
                for q in range(self.max_q + 1)
            ]
            orders.sort(key=lambda order: (sum(order), order[0], order[1]))
            candidates.extend(("ARMA", order) for order in orders)
        return candidates
