"""Progress display for ordered thread-pool results."""
from __future__ import annotations

from typing import Any
from tqdm.auto import tqdm


def get_progress_bar(iterable: Any, *, total=None, desc=None, disable=False):
    return iterable if disable else tqdm(iterable, total=total, desc=desc)
