# scivol/_progress.py
"""
Progress bar utilities for scivol.

Provides a thin wrapper around tqdm plus a context manager for integrating
tqdm with joblib parallel execution.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional

from tqdm.auto import tqdm as _tqdm


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
    Wrap *iterable* in a tqdm progress bar.

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
        Either a ``tqdm`` wrapper or the original iterable when disabled.
    """
    if disable:
        return iterable

    return _tqdm(iterable, total=total, desc=desc)


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

    When *disable* is True the context manager is a no-op.
    """
    if disable:
        yield
        return

    import joblib.parallel  # pyright: ignore[reportMissingImports]

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
