# scivol/components/density.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

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
    
    def default_start(self, data):
        c  = np.mean(data)
        ar = [0.1] * self.p
        ma = [0.1] * self.q
        return np.array([c] + ar + ma)
    
    def bounds(self):
        c_bound   = [(-10.0, 10.0)]
        ar_bounds = [(-0.99, 0.99)] * self.p
        ma_bounds = [(-0.99, 0.99)] * self.q
        return c_bound + ar_bounds + ma_bounds
    
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

    def default_start(self, data):
        const = [float(np.mean(data))] if self.constant else []
        ar = [0.0] * self.lags
        exog = [0.0] * self.n_exog
        return np.array(const + ar + exog, dtype=np.float64)

    def bounds(self):
        const_bounds = [(-10.0, 10.0)] if self.constant else []
        ar_bounds = [(-0.99, 0.99)] * self.lags
        exog_bounds = [(-10.0, 10.0)] * self.n_exog
        return const_bounds + ar_bounds + exog_bounds

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

    def default_start(self, data):
        const = [float(np.mean(data))] if self.constant else []
        har = [0.0] * len(self.horizons)
        exog = [0.0] * self.n_exog
        return np.array(const + har + exog, dtype=np.float64)

    def bounds(self):
        const_bounds = [(-10.0, 10.0)] if self.constant else []
        har_bounds = [(-0.99, 0.99)] * len(self.horizons)
        exog_bounds = [(-10.0, 10.0)] * self.n_exog
        return const_bounds + har_bounds + exog_bounds

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