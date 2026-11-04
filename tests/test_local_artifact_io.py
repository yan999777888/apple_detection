"""Atomic JSON publication, including real Windows no-delete-share readers."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from apple_local_rgbd.artifact_io import write_json


class ArtifactIOTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[1] / "tmp/local_artifact_io_tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.root = Path(self.temporary.name)
        self.path = self.root / "pipeline_status.json"
        self.path.write_text(json.dumps({"phase": "old", "complete": True}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_publish_complete_json_atomically_and_remove_pending_source(self):
        write_json(self.path, {"phase": "new", "中文": "有效"})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"phase": "new", "中文": "有效"})
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

    def test_nonfinite_serialization_preserves_original_and_never_replaces(self):
        before = self.path.read_bytes()
        with patch("apple_local_rgbd.artifact_io.os.replace") as replace:
            with self.assertRaises(ValueError):
                write_json(self.path, {"bad": float("nan")})
        replace.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_unrelated_io_errors_raise_immediately(self):
        before = self.path.read_bytes()
        for error in (OSError("disk failure"), PermissionError("non-Windows permission error")):
            with self.subTest(error=type(error).__name__):
                with patch("apple_local_rgbd.artifact_io.os.replace", side_effect=error) as replace:
                    with patch("apple_local_rgbd.artifact_io.time.sleep") as sleep:
                        with self.assertRaises(type(error)):
                            write_json(self.path, {"phase": "new"})
                self.assertEqual(replace.call_count, 1)
                sleep.assert_not_called()
                self.assertEqual(self.path.read_bytes(), before)

    def windows_reader(self):
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # GENERIC_READ; SHARE_READ | SHARE_WRITE deliberately excludes SHARE_DELETE.
        handle = kernel.CreateFileW(str(self.path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return kernel, handle

    @unittest.skipUnless(os.name == "nt", "Requires real Windows sharing semantics")
    def test_real_windows_reader_release_allows_retry_without_partial_target(self):
        kernel, handle = self.windows_reader()
        original = self.path.read_bytes()
        observed, released = [], threading.Event()
        def release():
            time.sleep(.08)
            observed.append(self.path.read_bytes())
            kernel.CloseHandle(handle)
            released.set()
        thread = threading.Thread(target=release)
        thread.start()
        try:
            started = time.monotonic()
            write_json(self.path, {"phase": "new", "complete": True},
                       replace_attempts=20, initial_backoff_seconds=.01, max_backoff_seconds=.02)
            self.assertGreaterEqual(time.monotonic() - started, .07)
            self.assertEqual(observed, [original])
            self.assertEqual(json.loads(self.path.read_text()), {"phase": "new", "complete": True})
        finally:
            thread.join(timeout=2)
            if not released.is_set():
                kernel.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Requires real Windows sharing semantics")
    def test_real_persistent_reader_lock_fails_bounded_and_preserves_valid_original(self):
        kernel, handle = self.windows_reader()
        original = self.path.read_bytes()
        started = time.monotonic()
        try:
            with self.assertRaises(PermissionError) as failure:
                write_json(self.path, {"phase": "pending"}, replace_attempts=4,
                           initial_backoff_seconds=.01, max_backoff_seconds=.02)
            self.assertIn(failure.exception.winerror, {5, 32, 33})
            self.assertGreaterEqual(time.monotonic() - started, .04)
            self.assertLess(time.monotonic() - started, 1.5)
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(json.loads(self.path.read_text()), {"phase": "old", "complete": True})
            pending = self.path.with_name(self.path.name + ".tmp")
            self.assertEqual(json.loads(pending.read_text()), {"phase": "pending"})
        finally:
            kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
