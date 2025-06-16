"""
volkit
------

High-performance volatility-model helpers.
The compiled extension lives in `volkit._core`.
"""

from importlib.metadata import version as _v

__version__ = _v(__name__)

from . import _core as _core

__all__: list[str] = []