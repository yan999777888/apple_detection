"""Bounded replacement retry for transient Windows sharing/access conflicts.

The caller must finish writing and close the temporary file before calling.
No target is deleted, copied in place, or silently discarded on failure.
"""
from __future__ import annotations

import os
import time


_RETRYABLE_WINDOWS_ERRORS = frozenset({5, 32, 33})
_DELAYS_SECONDS = (0.05, 0.10, 0.20, 0.40, 0.50, 0.50, 0.50)


def replace_with_retry(source, destination, *, max_attempts=8):
    """Atomically replace, retrying only Windows errors 5/32/33.

    Eight attempts schedule at most 2.25 seconds of sleeping. Resource
    supervisors can use six attempts for a 1.25-second sleep budget. OS call
    and scheduling time are outside those bounds. A persistent or unrelated
    error is raised; the existing target and completed temporary file remain
    available. This helper never rewrites/serializes the temporary file.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 8:
        raise ValueError("max_attempts must be an integer between 1 and 8")
    for attempt in range(max_attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if (os.name != "nt" or getattr(exc, "winerror", None) not in _RETRYABLE_WINDOWS_ERRORS
                    or attempt + 1 == max_attempts):
                raise
            time.sleep(_DELAYS_SECONDS[attempt])
