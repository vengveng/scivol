"""The text report returned by every result's summary()."""
from __future__ import annotations


class Summary(str):
    """A result report: a string that displays as its text at the prompt and in notebooks."""

    def __repr__(self) -> str:
        return str(self)
