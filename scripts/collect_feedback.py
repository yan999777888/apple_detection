"""Collect a bounded, explicitly selected handoff feedback ZIP.

Only diagnostics under configs/benchmarks/reports/runs are selected. Checkpoints,
raw assets, environments, conversations, credentials and reparse points are
excluded. No environment variable values or sys.path dump are collected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import zipfile


ALLOWED_DIRECTORIES = ("configs", "benchmarks", "reports", "runs")
ALLOWED_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".csv", ".png"}
CODE_DIRECTORIES = ("src", "scripts", "tests", "reference")
CODE_SUFFIXES = {".py", ".ps1", ".sh", ".toml", ".yaml", ".yml"}
PACKAGES = ("torch", "torchvision", "transformers", "numpy", "scipy", "Pillow", "psutil", "safetensors", "tokenizers", "huggingface-hub")
DENIED_COMPONENTS = {".git", ".aws", ".codex", ".agents", ".ssh", ".venv", "venv", "env", "envs",
                     "datasets", "dataset", "assets", "weights", "checkpoints", "conversations", "conversation",
                     "credentials", "secrets", "chat_history", "session_history"}
DENIED_NAME = re.compile(r"(?:^|[_.-])(?:conversations?|credentials?|secrets?|passwords?|tokens?|auth|authorization|cookies?|api[_-]?keys?|chat[_-]?history|session[_-]?history)(?:[_.-]|$)", re.I)


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reparse(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _absolute(path):
    return Path(os.path.abspath(Path(path).expanduser()))


def _reject_reparse_ancestors(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if _reparse(current):
            raise ValueError("Feedback root/destination cannot follow a symlink/reparse point")


def _denied(relative):
    return any(part.lower() in DENIED_COMPONENTS for part in relative.parts) or bool(DENIED_NAME.search(relative.name)) or relative.name.lower().startswith(".env")


def _scan(root, directories, suffixes, skipped):
    selected = []
    for name in directories:
        base = root / name
        if not base.exists():
            continue
        if _reparse(base) or not _inside(_absolute(base), root):
            skipped.append({"path": name, "reason": "symlink/reparse point or outside root"})
            continue
        for current, dirs, files in os.walk(base, followlinks=False):
            current = Path(current)
            for child in list(dirs):
                path = current / child
                relative = path.relative_to(root)
                if _denied(relative) or _reparse(path) or not _inside(_absolute(path), root):
                    dirs.remove(child)
                    skipped.append({"path": relative.as_posix(), "reason": "excluded directory or symlink/reparse point"})
            for child in files:
                path = current / child
                relative = path.relative_to(root)
                if _denied(relative):
                    skipped.append({"path": relative.as_posix(), "reason": "conversation/credential/environment/raw-asset exclusion"})
                elif _reparse(path) or not _inside(_absolute(path), root):
                    skipped.append({"path": relative.as_posix(), "reason": "symlink/reparse point or outside root"})
                elif path.suffix.lower() not in suffixes:
                    skipped.append({"path": relative.as_posix(), "reason": "suffix outside diagnostic/source allowlist"})
                elif path.is_file():
                    selected.append((relative, _absolute(path)))
    return selected


def _environment():
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "packages": packages, "CUDA_queried": False, "model_or_checkpoint_loaded": False,
            "environment_variable_values_collected": False, "sys_path_collected": False}


def collect_feedback(root, *, max_file_bytes=200 * 1024**2, max_png_bytes=10 * 1024**2, max_total_bytes=2 * 1024**3):
    root = _absolute(root)
    if not root.is_dir():
        raise FileNotFoundError("Feedback repository root is missing")
    _reject_reparse_ancestors(root)
    if any(type(value) is not int or value < 1 for value in (max_file_bytes, max_png_bytes, max_total_bytes)):
        raise ValueError("All size limits must be positive integer byte counts")
    skipped = []
    candidates = _scan(root, ALLOWED_DIRECTORIES, ALLOWED_SUFFIXES, skipped)
    # Preserve compact status/failure logs before large raw prediction files.
    candidates.sort(key=lambda item: (item[1].stat().st_size, item[0].as_posix()))
    source_skips = []
    code = _scan(root, CODE_DIRECTORIES, CODE_SUFFIXES, source_skips)
    code_hashes = []
    for relative, path in sorted(code, key=lambda item: item[0].as_posix()):
        if path.stat().st_size > max_file_bytes:
            source_skips.append({"path": relative.as_posix(), "reason": "code hash size limit"})
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        code_hashes.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    feedback = root / "feedback"
    feedback.mkdir(parents=True, exist_ok=True)
    if _reparse(feedback) or not _inside(_absolute(feedback), root):
        raise ValueError("Feedback destination must be an ordinary directory inside root")
    output = feedback / f"feedback_{stamp}_{os.getpid()}.zip"
    included, total = [], 0
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in candidates:
            before = path.stat()
            limit = min(max_file_bytes, max_png_bytes) if path.suffix.lower() == ".png" else max_file_bytes
            if before.st_size > limit:
                skipped.append({"path": relative.as_posix(), "bytes": before.st_size, "reason": "per-file/PNG size limit", "limit_bytes": limit})
                continue
            if total + before.st_size > max_total_bytes:
                skipped.append({"path": relative.as_posix(), "bytes": before.st_size, "reason": "total uncompressed diagnostic size limit"})
                continue
            arcname = "files/" + relative.as_posix()
            digest, captured, remaining = hashlib.sha256(), 0, before.st_size
            # Capture a bounded snapshot prefix when a live log grows. It may
            # be incomplete JSONL/LOG; the manifest records that explicitly.
            with path.open("rb") as source, archive.open(arcname, "w", force_zip64=True) as target:
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
                    captured += len(chunk)
                    remaining -= len(chunk)
            after = path.stat()
            total += captured
            included.append({"path": relative.as_posix(), "archive_path": arcname, "captured_bytes": captured,
                             "sha256": digest.hexdigest(), "source_size_at_selection": before.st_size,
                             "changed_while_collecting": before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or captured != before.st_size})
        manifest = {"schema": "apple_handoff_feedback_selection_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "allowed_directories": list(ALLOWED_DIRECTORIES), "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
                    "limits_bytes": {"single_file": max_file_bytes, "PNG": max_png_bytes, "total_uncompressed_diagnostics": max_total_bytes},
                    "included": included, "captured_total_bytes": total, "skipped": skipped,
                    "source_hash_skips": source_skips, "source_files_hashed": len(code_hashes),
                    "excluded_scopes": ["weights/checkpoints/PT", "raw datasets/assets", "environments", "conversations", "credentials", "all symlinks/reparse points"],
                    "environment_variable_values_collected": False, "sys_path_collected": False}
        for name, value in (("selection_manifest.json", manifest), ("environment.json", _environment()), ("code_sha256.json", code_hashes)):
            archive.writestr(name, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-file-mib", type=int, default=200)
    parser.add_argument("--max-png-mib", type=int, default=10)
    parser.add_argument("--max-total-mib", type=int, default=2048)
    args = parser.parse_args()
    output, manifest = collect_feedback(args.root, max_file_bytes=args.max_file_mib * 1024**2,
                                        max_png_bytes=args.max_png_mib * 1024**2, max_total_bytes=args.max_total_mib * 1024**2)
    print(json.dumps({"archive": str(output), "files": len(manifest["included"]), "captured_bytes": manifest["captured_total_bytes"],
                      "skipped": len(manifest["skipped"])}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
