"""
volkit._devtools - Development and debugging tools
===================================================

This module provides internal development tools for validating and debugging
the volkit library, particularly the C extension implementations.

These helpers are not part of the public user-facing API.

Tools:
- validate_derivatives: Validate analytical derivatives against an AD oracle
- quick_check: Quick pass/fail check for derivatives
- check_routine: Detailed derivative validation (from derivcheck.py)
"""

from .diagnostic import validate_derivatives, quick_check, DerivativeValidationReport

__all__ = [
    "validate_derivatives",
    "quick_check",
    "DerivativeValidationReport",
]
