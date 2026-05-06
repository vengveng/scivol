"""
Global settings for scivol display and formatting.

Usage::

    import scivol

    # Rename parameters for display
    scivol.settings.names.gamma = "zeta"
    scivol.settings.names.alpha = "a"
    scivol.settings.names.nu = "df"

    # Reset all overrides
    scivol.settings.names.reset()

Display names only affect user-facing output (summary tables, to_dict keys,
printed standard-error reports).  Internal dictionary keys, dataclass
attributes, and C function signatures are **not** changed.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# Regex to split indexed names like "alpha[1]" → ("alpha", "[1]")
_INDEXED_RE = re.compile(r"^([a-zA-Z_]+)(\[.+\])$")


class ParamNames:
    """Registry that maps canonical parameter names to user-chosen display names.

    Supports bare names (``"omega"``) and indexed names (``"alpha[1]"``).
    For indexed names the base is resolved and the index suffix reattached.
    """

    _defaults: Dict[str, str] = {
        "omega": "omega",
        "alpha": "alpha",
        "gamma": "gamma",
        "beta": "beta",
        "nu": "nu",
        "lambda": "lambda",
        "lam": "lam",
        "const": "const",
        "ar": "ar",
        "ma": "ma",
    }

    def __init__(self) -> None:
        self._overrides: Dict[str, str] = {}

    # -- attribute access (get / set) ------------------------------------

    def __getattr__(self, name: str) -> str:
        # Avoid infinite recursion for private / dunder attributes
        if name.startswith("_"):
            raise AttributeError(name)
        return self._overrides.get(name, self._defaults.get(name, name))

    def __setattr__(self, name: str, value: str) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if not isinstance(value, str):
                raise TypeError(
                    f"Display name must be a string, got {type(value).__name__}"
                )
            self._overrides[name] = value

    # -- programmatic API -----------------------------------------------

    def resolve(self, name: str) -> str:
        """Return the display name for *name*.

        Handles indexed names like ``"alpha[1]"`` by resolving the base
        (``"alpha"``) and reattaching the index suffix.
        """
        m = _INDEXED_RE.match(name)
        if m:
            base, suffix = m.group(1), m.group(2)
            display_base = self._overrides.get(base, self._defaults.get(base, base))
            return display_base + suffix
        return self._overrides.get(name, self._defaults.get(name, name))

    def reset(self) -> None:
        """Clear all user overrides, reverting to default names."""
        self._overrides.clear()

    def __repr__(self) -> str:
        items = {**self._defaults, **self._overrides}
        changed = {k: v for k, v in items.items() if v != self._defaults.get(k, k)}
        if changed:
            parts = ", ".join(f"{k}={v!r}" for k, v in sorted(changed.items()))
            return f"ParamNames(overrides: {parts})"
        return "ParamNames(defaults)"


class Settings:
    """Top-level container for scivol global settings."""

    def __init__(self) -> None:
        self.names: ParamNames = ParamNames()
        self.show_progress: bool = False

    def __repr__(self) -> str:
        return f"Settings(names={self.names!r}, show_progress={self.show_progress!r})"


# Module-level singleton
settings = Settings()
