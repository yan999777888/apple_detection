"""CPU/file fixtures for portable preflight and bounded feedback collection."""
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("_test_handoff_" + name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


preflight = load_script("preflight")
feedback = load_script("collect_feedback")


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    (root / "configs").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "assets").mkdir()
    (root / "assets" / "papple").mkdir()
    (root / "assets" / "minneapple").mkdir()
    parent = root / "weights" / "parent.pt"
    parent.write_bytes(b"fixture checkpoint bytes; never deserialized")
    (root / "requirements.txt").write_text("pytest==" + metadata.version("pytest") + "\n", encoding="utf-8")
    config = {"path_base": "..", "paths": {"papple_root": "assets/papple", "minne_root": "assets/minneapple",
        "minne_manifest": "assets/minne_manifest.json", "parent_checkpoint": "weights/parent.pt",
        "papple_provenance": "assets/provenance.json", "output": "runs/pilot"},
        "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(), "training": {"input_size": 768}}
    path = root / "configs" / "pilot.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return root, path, config


def test_cpu_preflight_writes_report_and_never_calls_cuda_or_loads_checkpoint(package, monkeypatch):
    root, path, _ = package
    monkeypatch.setattr(preflight, "check_data", lambda *args: {"passed": True, "fixture_sample_check": True})
    class ForbiddenCUDA:
        def __getattr__(self, name):
            raise AssertionError("CPU preflight touched CUDA: " + name)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=ForbiddenCUDA()))
    report, output = preflight.run_preflight(path)
    assert report["passed"] and report["exit_code"] == 0 and output.parent == root / "reports"
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["checkpoint_deserialized"] is False and saved["model_constructed"] is False
    cuda = next(check for check in saved["checks"] if check["name"] == "cuda")
    assert cuda["details"]["status"] == "disabled" and cuda["details"]["CUDA_queried"] is False
    parent = next(check for check in saved["checks"] if check["name"] == "parent_checkpoint_bytes")
    assert parent["details"]["sha256"] == hashlib.sha256((root / "weights" / "parent.pt").read_bytes()).hexdigest()


def test_preflight_failure_is_saved_with_nonzero_exit(package, monkeypatch):
    root, path, _ = package
    (root / "weights" / "parent.pt").write_bytes(b"changed checkpoint")
    monkeypatch.setattr(preflight, "check_data", lambda *args: (_ for _ in ()).throw(FileNotFoundError("missing sample data")))
    report, output = preflight.run_preflight(path, verify_all_assets=True)
    assert report["passed"] is False and report["exit_code"] == 1 and output.is_file()
    failures = {row["name"] for row in report["checks"] if not row["passed"]}
    assert {"parent_checkpoint_bytes", "data_and_sample_contracts", "all_asset_and_package_SHA256"} <= failures


def test_preflight_cli_returns_nonzero_and_preserves_failure_report(package):
    root, path, _ = package
    process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "preflight.py"), "--config", str(path)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 1
    message = json.loads(process.stdout)
    saved = json.loads(Path(message["report"]).read_text(encoding="utf-8"))
    assert saved["passed"] is False and saved["exit_code"] == 1
    assert Path(message["report"]).parent == root / "reports"


def test_manifest_full_sha_catches_same_length_tampering_and_missing_manifest(package):
    root, _, _ = package
    data = root / "assets" / "file.bin"
    data.write_bytes(b"123456")
    manifest = {"files": [{"path": "assets/file.bin", "bytes": 6, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}], "total_bytes": 6}
    (root / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert preflight.verify_manifests(root)["passed"]
    data.write_bytes(b"654321")
    result = preflight.verify_manifests(root)
    assert result["passed"] is False and result["manifests"][0]["mismatches"][0]["path"] == "assets/file.bin"


def test_repository_paths_cannot_escape_via_config_or_manifest(package, tmp_path):
    root, path, config = package
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    config["paths"]["parent_checkpoint"] = "../outside.txt"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        preflight.resolve_config(path)
    (root / "asset_manifest.json").write_text(json.dumps({"files": [{"path": "../outside.txt", "sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        preflight.verify_manifests(root)


def test_actual_minne_manifest_format_keeps_official_train570_val100_only(tmp_path):
    path = tmp_path / "manifest.json"
    rows = [{"official_split": "train", "split": "train", "source_filename": f"train_{i}.png"} for i in range(570)]
    rows += [{"official_split": "train", "split": "val", "source_filename": f"val_{i}.png"} for i in range(100)]
    rows += [{"official_split": "test", "split": "test", "source_filename": f"test_{i}.png"} for i in range(331)]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert preflight._manifest_counts(path) == {"train": 570, "val": 100}
    rows[0]["official_split"] = "test"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="train570/val100"):
        preflight._manifest_counts(path)


def test_feedback_zip_includes_failure_status_and_raw_predictions_but_excludes_private_and_weights(tmp_path, monkeypatch):
    root = tmp_path / "package"
    for directory in ("configs", "benchmarks", "reports", "runs", "assets", "weights", "src"):
        (root / directory).mkdir(parents=True)
    (root / "runs" / "status.json").write_text('{"status":"failed","error":"out of memory"}', encoding="utf-8")
    (root / "runs" / "failure.log").write_text("RuntimeError: failed update\n", encoding="utf-8")
    (root / "runs" / "raw_predictions.json").write_text('[{"score":0.9}]', encoding="utf-8")
    (root / "runs" / "model.pt").write_bytes(b"weights-must-not-be-collected")
    (root / "runs" / "conversation.json").write_text("private conversation", encoding="utf-8")
    (root / "runs" / "credentials.json").write_text("secret credentials", encoding="utf-8")
    (root / "runs" / "access_token.txt").write_text("secret access token", encoding="utf-8")
    (root / "assets" / "raw_data.json").write_text("raw dataset", encoding="utf-8")
    (root / "src" / "example.py").write_text("print('source hashed, not embedded')", encoding="utf-8")
    monkeypatch.setenv("PRIVATE_FEEDBACK_TEST_TOKEN", "must-not-leak-to-feedback")
    output, selection = feedback.collect_feedback(root)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "files/runs/status.json" in names and "files/runs/failure.log" in names
        assert "files/runs/raw_predictions.json" in names
        assert not any(name.endswith(".pt") or "conversation.json" in name or "credentials.json" in name
                       or "access_token.txt" in name or "raw_data.json" in name for name in names)
        assert b"failed" in archive.read("files/runs/status.json")
        assert not any(b"must-not-leak-to-feedback" in archive.read(name) for name in names)
        environment = json.loads(archive.read("environment.json"))
        assert environment["environment_variable_values_collected"] is False and environment["sys_path_collected"] is False
        code = json.loads(archive.read("code_sha256.json"))
        assert code[0]["path"] == "src/example.py"
        for row in selection["included"]:
            assert hashlib.sha256(archive.read(row["archive_path"])).hexdigest() == row["sha256"]


def test_feedback_size_caps_are_explicit_and_skips_are_recorded(tmp_path):
    root = tmp_path / "package"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "status.json").write_text('{"status":"failed"}', encoding="utf-8")
    (root / "runs" / "rawpred.json").write_bytes(b"r" * 100)
    (root / "runs" / "too_large.json").write_bytes(b"x" * 201)
    (root / "runs" / "large.png").write_bytes(b"p" * 31)
    output, manifest = feedback.collect_feedback(root, max_file_bytes=200, max_png_bytes=30, max_total_bytes=150)
    assert manifest["limits_bytes"] == {"single_file": 200, "PNG": 30, "total_uncompressed_diagnostics": 150}
    assert {row["path"] for row in manifest["included"]} == {"runs/status.json", "runs/rawpred.json"}
    assert {row["path"] for row in manifest["skipped"]} >= {"runs/too_large.json", "runs/large.png"}
    assert output.is_file()


def test_feedback_cli_creates_a_readable_zip_with_failure_log(tmp_path):
    root = tmp_path / "package"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "status.log").write_text("status=failed\n", encoding="utf-8")
    process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "collect_feedback.py"), "--root", str(root)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0
    message = json.loads(process.stdout)
    with zipfile.ZipFile(message["archive"]) as archive:
        assert archive.read("files/runs/status.log") == (root / "runs" / "status.log").read_bytes()


def test_feedback_rejects_real_external_symlink_or_windows_junction(tmp_path):
    root, outside = tmp_path / "package", tmp_path / "external"
    (root / "runs").mkdir(parents=True)
    outside.mkdir()
    (outside / "external.log").write_text("outside-root-private-data", encoding="utf-8")
    link = root / "runs" / "external_link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        if created.returncode:
            pytest.skip("This Windows process cannot create symlinks/junctions; rerun with normal filesystem privileges")
    assert link.exists()
    output, manifest = feedback.collect_feedback(root)
    with zipfile.ZipFile(output) as archive:
        assert not any("external.log" in name for name in archive.namelist())
        assert not any(b"outside-root-private-data" in archive.read(name) for name in archive.namelist())
    assert any(row["path"] == "runs/external_link" and "symlink" in row["reason"] for row in manifest["skipped"])
