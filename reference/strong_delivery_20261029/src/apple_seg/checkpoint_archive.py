"""Retain validated local checkpoints while last.pt continues to rotate."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import torch

from .inference import _audit_trained_payload


class CheckpointChanging(RuntimeError):
    """The writer advanced between reading the sidecar and taking a snapshot."""


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retain_checkpoint(output, training_commit, *, name="last.pt"):
    """Hardlink an exact, hash-verified formal checkpoint with its own validation.

    A temporary hardlink pins the old file even if training replaces last.pt.
    Files are never edited in place. Unvalidated updates are left unarchived.
    """
    if name not in ("last.pt", "best.pt"):
        raise ValueError("Only the trainer's last.pt or best.pt may be retained")
    if re.fullmatch("[0-9a-f]{40}", training_commit) is None:
        raise ValueError("A full training code commit is required")
    output = Path(output).absolute()
    source = output / name
    expected = source.with_suffix(".pt.sha256").read_text(encoding="ascii").strip()
    if re.fullmatch("[0-9a-f]{64}", expected) is None:
        raise CheckpointChanging("Checkpoint sidecar is incomplete")
    archive = output / "validated_checkpoints"
    archive.mkdir(exist_ok=True)
    capture = archive / (".capture_" + uuid.uuid4().hex + ".pt")
    try:
        os.link(source, capture)
        if file_sha(capture) != expected:
            raise CheckpointChanging("Checkpoint and sidecar advanced during capture")
        # An exception traceback can keep mapped tensor storage alive on Windows
        # after del payload, preventing capture cleanup. Use an ordinary CPU load.
        payload = torch.load(capture, map_location="cpu", weights_only=False, mmap=False)
        try:
            metadata = _audit_trained_payload(payload, require_calibration=False)
            if metadata["calibration"] is None:
                return None
            record = {
                "schema": "apple_validated_checkpoint_archive_v1",
                "training_code_commit": training_commit,
                "checkpoint_sha256": expected,
                "successful_updates": metadata["successful_updates"],
                "calibration": metadata["calibration"],
                "identity": payload["identity"],
                "weights_uploaded_to_git": False,
            }
        finally:
            del payload
        target = archive / f"update_{metadata['successful_updates']:06d}_{expected[:16]}.pt"
        sidecar = target.with_suffix(".pt.json")
        if target.exists():
            if file_sha(target) != expected:
                raise ValueError("Existing archive has different checkpoint bytes")
        else:
            os.link(capture, target)
        hash_sidecar = target.with_suffix(".pt.sha256")
        if not hash_sidecar.exists():
            with hash_sidecar.open("x", encoding="ascii") as stream:
                stream.write(expected)
        elif hash_sidecar.read_text(encoding="ascii").strip() != expected:
            raise ValueError("Existing archive SHA256 sidecar differs")
        if not sidecar.exists():
            record["retained_utc"] = datetime.now(timezone.utc).isoformat()
            with sidecar.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            for key, value in record.items():
                if saved.get(key) != value:
                    raise ValueError("Existing archive metadata differs: " + key)
        return target
    finally:
        capture.unlink(missing_ok=True)
