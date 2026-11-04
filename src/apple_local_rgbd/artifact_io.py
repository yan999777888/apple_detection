"""Atomic JSON artifacts resilient to short Windows reader sharing locks.

Callers own each output path exclusively. Write the same-directory temporary
file once, then retry only replacement, never unlinking the existing artifact.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any


WINDOWS_REPLACE_RETRY_ERRORS = frozenset({5, 32, 33})


def write_json(path: str | Path, value: Any, *, replace_attempts: int = 16,
               initial_backoff_seconds: float = .01,
               max_backoff_seconds: float = .25) -> None:
    """Publish complete JSON or raise, preserving the old complete JSON on failure.

    Default replacement retries wait at most 2.81 seconds in total. Only a
    PermissionError with Windows error 5/32/33 qualifies for retry. A persistent
    lock retains the pending .tmp for diagnosis and raises the original error;
    disk, serialization and other I/O errors are never hidden or retried.
    """
    if (not isinstance(replace_attempts, int) or not 1 <= replace_attempts <= 64
            or not math.isfinite(initial_backoff_seconds) or initial_backoff_seconds <= 0
            or not math.isfinite(max_backoff_seconds) or max_backoff_seconds <= 0):
        raise ValueError("Atomic JSON replacement requires finite bounded retry parameters")
    path = Path(path)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    delay = min(initial_backoff_seconds, max_backoff_seconds)
    for attempt in range(replace_attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in WINDOWS_REPLACE_RETRY_ERRORS:
                raise
            if attempt + 1 == replace_attempts:
                error.add_note(f"Atomic JSON replace failed after {replace_attempts} attempts; pending file: {temporary}")
                raise
            time.sleep(delay)
            delay = min(delay * 2, max_backoff_seconds)

