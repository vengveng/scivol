# scivol/components/vol.py
from __future__ import annotations
from ..roles import Role
import numpy as np
from ..components.base import Component


class APARCH(Component):
    """Asymmetric Power ARCH volatility component.

    The conditional scale follows

    ``sigma_t**delta = omega + sum(alpha_i * (|eps|-gamma_i*eps)**delta)
    + sum(beta_j * sigma_{t-j}**delta)``.
    """

    role = Role.VOLATILITY

    def __init__(self, p: int = 1, q: int = 1):
        if any(type(order) is not int or order < 0 for order in (p, q)):
            raise ValueError("APARCH orders must be non-negative integers.")
        self.p = p
        self.q = q
        self.fitted_params = None
        self.fitted_values = None
        self._data = None

    @classmethod
    def up_to(cls, p: int, q: int):
        """Declare a maximal generator; native restrictions supply descendants."""
        return cls(p, q)

    @property
    def signature(self) -> str:
        return f"APARCH({self.p},{self.q})"

    @property
    def n_params(self) -> int:
        return 2 + 2 * self.p + self.q

    def pack(self, params_dict: dict) -> np.ndarray:
        return np.asarray(
            [params_dict["omega"]]
            + list(params_dict["alpha"])
            + list(params_dict["gamma"])
            + list(params_dict["beta"])
            + [params_dict["delta"]],
            dtype=np.float64,
        )

    def unpack(self, flat_params: np.ndarray) -> dict:
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}
        beta_start = 1 + 2 * self.p
        self.fitted_params = {
            "omega": float(flat_params[0]),
            "alpha": [float(x) for x in flat_params[1 : 1 + self.p]],
            "gamma": [float(x) for x in flat_params[1 + self.p : beta_start]],
            "beta": [float(x) for x in flat_params[beta_start : beta_start + self.q]],
            "delta": float(flat_params[beta_start + self.q]),
        }
        return self.fitted_params

class GJRGARCH(Component):
    """GJRGARCH(p, q) conditional volatility."""

    role = Role.VOLATILITY

    def __init__(self, p: int = 1, q: int = 1):
        if any(type(order) is not int or order < 0 for order in (p, q)):
            raise ValueError("orders must be non-negative integers")
        self.p, self.q = p, q
        self.fitted_params = None
        self.fitted_values = None
        self._data = None

    @classmethod
    def up_to(cls, p: int, q: int):
        """Declare a maximal generator; native restrictions supply descendants."""
        return cls(p, q)

    @property
    def signature(self): return f"GJR-GARCH({self.p},{self.q})"

    @property
    def n_params(self): return 1 + 2 * self.p + self.q  # omega + p alphas + p gammas + q betas

    def pack(self, params_dict: dict) -> np.ndarray:
        return np.array([params_dict['omega']] +
                         params_dict['alpha']  +
                         params_dict['gamma']  +
                         params_dict['beta'])

    def unpack(self, flat_params: np.ndarray) -> dict:
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}

        self.fitted_params = {
            'omega': flat_params[0],
            'alpha': flat_params[1:1+self.p].tolist(),
            'gamma': flat_params[1+self.p:1+2*self.p].tolist(),
            'beta': flat_params[1+2*self.p:1+2*self.p+self.q].tolist()
        }
        return self.fitted_params

class EGARCH(Component):
    """
    EGARCH(p, q) volatility component.

    The shipped recursion is:

        log h_t = omega
                + sum_i alpha_i * (|z_{t-i}| - E|z|)
                + sum_i gamma_i * z_{t-i}
                + sum_j beta_j  * log h_{t-j}

    where z_t = epsilon_t / sqrt(h_t).
    """
    role = Role.VOLATILITY

    def __init__(self, p: int = 1, q: int = 1):
        if any(type(order) is not int or order < 0 for order in (p, q)):
            raise ValueError("EGARCH orders must be non-negative integers.")
        self.p = int(p)
        self.q = int(q)
        self.fitted_params = None
        self.fitted_values = None
        self._data = None

    @classmethod
    def up_to(cls, p: int, q: int):
        """Declare a maximal generator; native restrictions supply descendants."""
        return cls(p, q)

    @property
    def signature(self) -> str:
        return f"EGARCH({self.p},{self.q})"

    @property
    def n_params(self) -> int:
        return 1 + 2 * self.p + self.q

    def pack(self, params_dict: dict) -> np.ndarray:
        return np.array(
            [params_dict["omega"]] + list(params_dict["alpha"]) + list(params_dict["gamma"]) + list(params_dict["beta"]),
            dtype=np.float64,
        )

    def unpack(self, flat_params: np.ndarray) -> dict:
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}

        alpha = flat_params[1 : 1 + self.p]
        gamma = flat_params[1 + self.p : 1 + 2 * self.p]
        beta = flat_params[1 + 2 * self.p : 1 + 2 * self.p + self.q]
        self.fitted_params = {
            "omega": float(flat_params[0]),
            "alpha": [float(x) for x in alpha],
            "gamma": [float(x) for x in gamma],
            "beta": [float(x) for x in beta],
        }
        return self.fitted_params

class GARCH(Component):
    """GARCH(p, q) conditional volatility."""

    role = Role.VOLATILITY

    def __init__(self, p: int = 1, q: int = 1):
        if any(type(order) is not int or order < 0 for order in (p, q)):
            raise ValueError("orders must be non-negative integers")
        self.p, self.q = p, q
        self.fitted_params = None
        self.fitted_values = None
        self._data = None

    @classmethod
    def up_to(cls, p: int, q: int):
        """Declare a maximal generator; native restrictions supply descendants."""
        return cls(p, q)

    @property
    def signature(self): return f"GARCH({self.p},{self.q})"

    @property
    def n_params(self): return 1 + self.p + self.q

    def pack(self, params_dict: dict) -> np.ndarray:
        """Convert dict to flat array"""
        return np.array([params_dict['omega']] +
                         params_dict['alpha']  +
                         params_dict['beta'])

    def unpack(self, flat_params: np.ndarray) -> dict:
        """Convert flat array to dict - MUTATES SELF"""
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}

        self.fitted_params = {
            'omega': flat_params[0],
            'alpha': flat_params[1:1+self.p].tolist(),
            'beta': flat_params[1+self.p:1+self.p+self.q].tolist()
        }
        return self.fitted_params
