"""Retry transient Windows sharing conflicts without changing training math."""
import hashlib
import os
from pathlib import Path
import time

_ORIGINAL_REPLACE = os.replace
_RETRY_SECONDS = 10.0
_POLL_SECONDS = 0.05


def replace_with_retry(source, destination, *args, **kwargs):
    deadline = time.monotonic() + _RETRY_SECONDS
    while True:
        try:
            return _ORIGINAL_REPLACE(source, destination, *args, **kwargs)
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(_POLL_SECONDS)


def install_windows_replace_retry():
    enabled = os.name == 'nt'
    if enabled and not getattr(os.replace, '_apple_windows_retry', False):
        replace_with_retry._apple_windows_retry = True
        os.replace = replace_with_retry
    with Path(__file__).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'enabled': enabled, 'retry_seconds': _RETRY_SECONDS,
            'scope': 'transient Windows os.replace sharing errors only',
            'source_sha256': digest, 'training_math_changed': False}