# scivol/components/density.py
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..components.base import Component
from ..roles import Role

class Normal(Component):
    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = {}

    @property
    def signature(self): return 'Normal'

    @property
    def n_params(self): return 0

    def pack(self, params_dict): return np.array([])
    def unpack(self, flat_params):
        self.fitted_params = {}
        return {}

class StudentT(Component):
    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self): return "StudentT"

    @property
    def n_params(self): return 1  # reciprocal tail

    def pack(self, params_dict): return np.array([params_dict['eta']])
    def unpack(self, flat_params):
        self.fitted_params = {'eta': flat_params[0]} if len(flat_params) > 0 else {}
        return self.fitted_params


class SkewT(Component):
    """
    Skewed Student-t distribution (Hansen, 1994 parameterization).

    Parameters
    ----------
    eta : float
        Reciprocal degrees of freedom in [0, 1/2), with the Normal limit at zero
    lam : float
        Skewness parameter (-1, 1), where lam=0 gives symmetric Student-t
    """
    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self): return "SkewT"

    @property
    def n_params(self): return 2  # reciprocal tail + skewness

    def pack(self, params_dict):
        return np.array([params_dict['eta'], params_dict['lam']])

    def unpack(self, flat_params):
        if len(flat_params) >= 2:
            self.fitted_params = {'eta': flat_params[0], 'lam': flat_params[1]}
        else:
            self.fitted_params = {}
        return self.fitted_params


class GED(Component):
    """
    Generalized Error Distribution (exponential power), standardized to unit variance.

    The shape parameter ``nu`` controls tail thickness and peakedness. ``nu=2``
    recovers the Normal distribution, while smaller values imply heavier tails.
    """

    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self):
        return "GED"

    @property
    def n_params(self):
        return 1

    def pack(self, params_dict):
        return np.array([params_dict["nu"]], dtype=np.float64)

    def unpack(self, flat_params):
        self.fitted_params = {"nu": flat_params[0]} if len(flat_params) > 0 else {}
        return self.fitted_params


class GeneralizedHyperbolic(Component):
    """Standardized Generalized Hyperbolic conditional density."""

    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self):
        return "GeneralizedHyperbolic"

    @property
    def n_params(self):
        return 3

    def pack(self, params_dict):
        return np.array(
            [params_dict["rho"], params_dict["zeta"], params_dict["lambda"]],
            dtype=np.float64,
        )

    def unpack(self, flat_params):
        if len(flat_params) >= 3:
            self.fitted_params = {
                "rho": flat_params[0],
                "zeta": flat_params[1],
                "lambda": flat_params[2],
            }
        else:
            self.fitted_params = {}
        return self.fitted_params


class NormalInverseGaussian(Component):
    """Standardized Normal Inverse Gaussian conditional density."""

    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self):
        return "NormalInverseGaussian"

    @property
    def n_params(self):
        return 2

    def pack(self, params_dict):
        return np.array(
            [params_dict["rho"], params_dict["zeta"]], dtype=np.float64
        )

    def unpack(self, flat_params):
        if len(flat_params) >= 2:
            self.fitted_params = {
                "rho": flat_params[0],
                "zeta": flat_params[1],
            }
        else:
            self.fitted_params = {}
        return self.fitted_params


class Hyperbolic(Component):
    """Standardized Hyperbolic conditional density."""

    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self):
        return "Hyperbolic"

    @property
    def n_params(self):
        return 2

    def pack(self, params_dict):
        return np.array(
            [params_dict["rho"], params_dict["zeta"]], dtype=np.float64
        )

    def unpack(self, flat_params):
        if len(flat_params) >= 2:
            self.fitted_params = {
                "rho": flat_params[0],
                "zeta": flat_params[1],
            }
        else:
            self.fitted_params = {}
        return self.fitted_params


class VarianceGamma(Component):
    """Standardized Variance-Gamma conditional density.

    The GH boundary index ``lambda`` is restricted above 1.5 so the density
    has a finite observed Hessian at its center.
    """

    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self):
        return "VarianceGamma"

    @property
    def n_params(self):
        return 2

    def pack(self, params_dict):
        return np.array(
            [params_dict["rho"], params_dict["lambda"]], dtype=np.float64
        )

    def unpack(self, flat_params):
        if len(flat_params) >= 2:
            self.fitted_params = {
                "rho": flat_params[0],
                "lambda": flat_params[1],
            }
        else:
            self.fitted_params = {}
        return self.fitted_params
