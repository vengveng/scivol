# scivol/_progress.py
"""
Progress bar utilities for scivol.

Provides a thin wrapper around tqdm that gracefully falls back to a no-op
iterator when tqdm is not installed.  Also provides a context manager for
integrating tqdm with joblib parallel execution.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Any, Iterator, Optional

try:
    from tqdm.auto import tqdm as _tqdm

    HAS_TQDM = True
except ImportError:  # pragma: no cover
    HAS_TQDM = False
    _tqdm = None  # type: ignore[assignment]

_WARNED_ONCE = False


def _warn_no_tqdm() -> None:
    """Emit a one-time warning when tqdm is requested but not installed."""
    global _WARNED_ONCE
    if not _WARNED_ONCE:
        warnings.warn(
            "Install tqdm for progress bars: pip install tqdm",
            stacklevel=3,
        )
        _WARNED_ONCE = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_progress_bar(
    iterable: Any,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    disable: bool = False,
) -> Any:
    """
    Wrap *iterable* in a tqdm progress bar if available.

    Parameters
    ----------
    iterable : iterable
        The iterable to wrap.
    total : int, optional
        Expected number of iterations.
    desc : str, optional
        Short description shown next to the bar.
    disable : bool
        If True, return the bare iterable (no progress bar).

    Returns
    -------
    iterable
        Either a ``tqdm`` wrapper or the original iterable.
    """
    if disable:
        return iterable

    if HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc)

    _warn_no_tqdm()
    return iterable


@contextlib.contextmanager
def tqdm_joblib(
    *,
    total: int,
    desc: Optional[str] = None,
    disable: bool = False,
) -> Iterator[None]:
    """
    Context manager that patches joblib to report progress via tqdm.

    Usage::

        with tqdm_joblib(total=100, desc="Fitting"):
            results = Parallel(n_jobs=4)(delayed(fn)(x) for x in tasks)

    When *disable* is True or tqdm is not installed the context manager
    is a no-op.
    """
    if disable or not HAS_TQDM:
        if not disable and not HAS_TQDM:
            _warn_no_tqdm()
        yield
        return

    import joblib.parallel

    pbar = _tqdm(total=total, desc=desc)

    # Monkey-patch BatchCompletionCallBack to update the bar
    _OriginalCallback = joblib.parallel.BatchCompletionCallBack

    class _TqdmCallback(_OriginalCallback):  # type: ignore[misc]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _TqdmCallback  # type: ignore[misc]
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = _OriginalCallback  # type: ignore[misc]
        pbar.close()
