from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
from ctypes import CDLL, c_double, POINTER, c_size_t
from utils.stats import Test
from numpy.typing import NDArray

if TYPE_CHECKING:
    import _core
else:
    import importlib as _implib
    _core = _implib.import_module(__package__ + "._core")

from utils.tools import ModelResult, CombinedModelResult, DataHandler


