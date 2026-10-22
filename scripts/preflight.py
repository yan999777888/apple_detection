"""CPU-first handoff checks; CUDA is queried only with --require-cuda.

This script hashes checkpoints but never deserializes them or constructs a
model. A report is written even when configuration or individual checks fail.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys


SCRIPT_ROOT = Path(os.path.abspath(__file__)).parents[1]


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def absolute(path):
    # Windows handle-based realpath may be denied even for files that can be
    # opened normally. Normalize lexically, then reject every existing reparse
    # component rather than following links outside the selected repository.
    return Path(os.path.abspath(Path(path).expanduser()))


def reject_reparse(path):
    path = absolute(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError("Symlink/reparse paths are not portable repository assets: " + str(current))


def resolve_config(config_path):
    config_path = absolute(config_path)
    reject_reparse(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    root = absolute(config_path.parent / config.get("path_base", ".."))
    reject_reparse(root)
    if not root.is_dir():
        raise FileNotFoundError("Repository root is missing")
    paths = {}
    for name, value in config["paths"].items():
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"Config paths must be repository-relative: {name}")
        resolved = absolute(root / relative)
        if not inside(resolved, root):
            raise ValueError(f"Config path escapes the repository: {name}")
        reject_reparse(resolved)
        paths[name] = resolved
    required = {"papple_root", "minne_root", "minne_manifest", "parent_checkpoint", "papple_provenance", "output"}
    if not required <= paths.keys():
        raise ValueError("Required config paths are missing: " + ", ".join(sorted(required - paths.keys())))
    return root, config, paths


def check_dependencies(root):
    path = root / "requirements.txt"
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise ValueError("Preflight requires explicit package==version pins: " + line)
        name, expected = match.groups()
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        # CUDA wheels carry a local version suffix such as +cu128.
        passed = installed is not None and installed.split("+", 1)[0] == expected
        records.append({"package": name, "expected": expected, "installed": installed, "passed": passed})
    if not records:
        raise ValueError("No pinned dependencies found")
    return {"passed": all(row["passed"] for row in records), "packages": records}


def probe_cuda(require_cuda):
    if not require_cuda:
        return {"passed": True, "status": "disabled", "CUDA_queried": False,
                "reason": "CPU preflight; pass --require-cuda on the training computer to probe its GPU"}
    import torch
    result = {"passed": False, "status": "requested", "CUDA_queried": True,
              "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
              "available": bool(torch.cuda.is_available()), "model_loaded": False}
    if not result["available"]:
        return result
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": properties.name, "VRAM_bytes": properties.total_memory,
                        "compute_capability": [properties.major, properties.minor],
                        "is_RTX_3080": "3080" in properties.name})
    result["devices"] = devices
    try:
        command = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20, check=True)
        result["nvidia_smi"] = command.stdout.strip().splitlines()
        result["driver_query_succeeded"] = True
    except (OSError, subprocess.SubprocessError) as error:
        result["driver_query_succeeded"] = False
        result["driver_query_error"] = str(error)
    result["passed"] = bool(devices) and result["driver_query_succeeded"]
    return result


def verify_manifests(root):
    manifests = [path for path in (root / "asset_manifest.json", root / "package_manifest.json") if path.is_file()]
    if not manifests:
        raise FileNotFoundError("--verify-all-assets requires asset_manifest.json or package_manifest.json")
    results = []
    for path in manifests:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        entries = document.get("files")
        if isinstance(entries, dict):
            entries = [{"path": name, **(value if isinstance(value, dict) else {"sha256": value})}
                       for name, value in entries.items()]
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest files must be a nonempty list or path mapping")
        mismatches, seen, total = [], set(), 0
        for entry in entries:
            relative = Path(entry["path"])
            if relative.is_absolute():
                raise ValueError("Manifest contains an absolute path")
            target = absolute(root / relative)
            if not inside(target, root) or str(relative) in seen:
                raise ValueError("Manifest path escapes repository or is duplicated: " + str(relative))
            reject_reparse(target)
            seen.add(str(relative))
            size = target.stat().st_size
            digest = file_sha(target)
            total += size
            if digest != entry.get("sha256") or ("bytes" in entry and size != entry["bytes"]):
                mismatches.append({"path": relative.as_posix(), "expected_sha256": entry.get("sha256"),
                                   "actual_sha256": digest, "expected_bytes": entry.get("bytes"), "actual_bytes": size})
        declared = document.get("total_bytes")
        if declared is not None and total != declared:
            mismatches.append({"total_bytes_expected": declared, "total_bytes_actual": total})
        results.append({"path": path.name, "manifest_sha256": file_sha(path), "files_checked": len(entries),
                        "bytes_hashed": total, "mismatches": mismatches, "passed": not mismatches})
    return {"passed": all(row["passed"] for row in results), "manifests": results, "GPU_used": False}


def _manifest_counts(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, list):
        # The portable source manifest also inventories the separate official
        # test split. Never load those images or masks during TRAIN/VAL checks.
        counts = {split: sum(row.get("official_split") == "train" and row.get("split") == split for row in value)
                  for split in ("train", "val")}
        if counts != {"train": 570, "val": 100}:
            raise ValueError(f"Minne official-train split must retain train570/val100: {counts}")
        return counts
    splits = value.get("splits", value)
    counts = {}
    for split in ("train", "val"):
        entries = splits.get(split)
        if isinstance(entries, dict):
            entries = entries.get("images", entries.get("files", entries.get("image_ids")))
        if not isinstance(entries, list):
            raise ValueError("Minne manifest must enumerate train/val images")
        counts[split] = len(entries)
    if counts != {"train": 570, "val": 100}:
        raise ValueError(f"Minne split must retain train570/val100: {counts}")
    return counts


def _quick_target(item, dataset_name, split):
    import torch
    target = item["target"]
    source, validity = target["source"], target["annotation_validity"]
    known_a = dataset_name == "PApple"
    if source["dataset"] != dataset_name or source["split"] != split:
        raise ValueError("Loader source dataset/split differs")
    if validity != {"visible": True, "amodal": known_a}:
        raise ValueError("Loader annotation validity differs")
    labels, ids, hw = target["labels"], target["instance_ids"], target["original_hw"]
    if labels.device.type != "cpu" or labels.dtype != torch.int64 or labels.ndim != 1 or not bool((labels == 0).all()):
        raise ValueError("Loader must return original CPU int64 apple=0 labels")
    if (split == "train" and len(labels) > 100) or len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("GT capacity/instance inventory differs")
    for branch in ("visible_masks", "amodal_masks"):
        masks = target[branch]
        if branch == "amodal_masks" and not known_a:
            if masks is not None:
                raise ValueError("Unknown Minne A must be None, never fabricated")
            continue
        if (not isinstance(masks, torch.Tensor) or masks.device.type != "cpu"
                or masks.dtype not in (torch.bool, torch.uint8) or tuple(masks.shape) != (len(labels), *hw)):
            raise ValueError("Loader must retain original CPU binary masks")
        for mask in masks:
            if int(mask.max()) > 1 or not bool(mask.any()):
                raise ValueError("Binary/nonempty original instance mask required")
    pixels, mask = item["pixel_values"], item["pixel_mask"]
    if pixels.device.type != "cpu" or pixels.ndim != 3 or pixels.shape[0] != 3 or not torch.isfinite(pixels).all():
        raise ValueError("Loader must return finite CPU RGB[3,H,W]")
    if mask.device.type != "cpu" or tuple(mask.shape) != tuple(pixels.shape[-2:]) or not bool(mask.all()):
        raise ValueError("Loader whole-frame pixel_mask must be CPU/all-one")
    return {"dataset": dataset_name, "split": split, "image_id": source["image_id"], "instances": len(labels),
            "original_hw": list(hw), "input_hw": list(pixels.shape[-2:]), "amodal_known": known_a}


def check_data(root, config, paths):
    for name in ("papple_root", "minne_root"):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"Missing data directory: {name}")
    counts = _manifest_counts(paths["minne_manifest"])
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    data = import_module("apple_seg.data")
    evaluation = import_module("apple_seg.evaluate")
    contract = evaluation.load_papple_contract(paths["papple_provenance"])
    size = config.get("training", {}).get("input_size", 768)
    papple = {split: data.PAppleDataset(paths["papple_root"], split=split, input_size=size, augment=False)
              for split in ("train", "val")}
    minne = {split: data.MinneAppleDataset(paths["minne_root"], paths["minne_manifest"], split=split,
                                         input_size=size, tiled=split == "train", augment=False) for split in ("train", "val")}
    tile_count = len(data.MINNE_TILES)
    if len(papple["val"]) != 814 or len(minne["train"]) != 570 * tile_count or len(minne["val"]) != 100:
        raise ValueError("Actual loader split counts differ from full inventories")
    if papple["val"].annotation_sha256 != contract["provenance"]["binding"]["current_annotation_sha256"]:
        raise ValueError("PApple loader annotation SHA identity differs from provenance")
    samples = []
    for name, datasets in (("PApple", papple), ("MinneApple", minne)):
        for split, dataset in datasets.items():
            if not len(dataset):
                raise ValueError("An expected nonempty dataset split is empty")
            item = dataset[0]
            samples.append(_quick_target(item, name, split))
            if name == "PApple" and split == "val":
                frozen = contract["images"][0]
                evaluation._audit_target(item["target"], frozen, contract["truth"][frozen["image_id"]])
                evaluation._audit_rgb(evaluation._rgb_path(item["target"]), frozen)
            del item
    return {"passed": True, "Minne_manifest_counts": counts,
            "loader_counts": {"PApple_train": len(papple["train"]), "PApple_val": len(papple["val"]),
                              "Minne_train_views": len(minne["train"]), "Minne_train_original_images": 570,
                              "Minne_val": len(minne["val"])},
            "Minne_fixed_training_tiles": [list(tile) for tile in data.MINNE_TILES], "official_test_images_opened": False,
            "sample_contracts": samples, "PApple_provenance_sha256": contract["provenance_sha256"],
            "scope": "Four CPU loader samples, including exact first PApple VAL GT/RGB identity. Full VAL evaluates all 814 identities later.",
            "model_loaded": False, "checkpoint_deserialized": False, "GPU_used": False}


def run_preflight(config_path, *, require_cuda=False, verify_all_assets=False):
    root, report = SCRIPT_ROOT, {"schema": "apple_handoff_preflight_v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                               "checks": [], "checkpoint_deserialized": False, "model_constructed": False,
                               "require_cuda": require_cuda, "verify_all_assets": verify_all_assets}
    def record(name, operation):
        try:
            details = operation()
            passed = details.get("passed", True)
            report["checks"].append({"name": name, "passed": bool(passed), "details": details})
            return details if passed else None
        except Exception as error:
            report["checks"].append({"name": name, "passed": False, "error_type": type(error).__name__, "error": str(error)})
            return None
    try:
        root, config, paths = resolve_config(config_path)
        report.update(repository_root=str(root), config_path=str(Path(config_path).absolute()), config_sha256=file_sha(config_path))
        record("python", lambda: {"passed": sys.version_info[:2] == (3, 11), "version": platform.python_version(),
                                   "implementation": platform.python_implementation(), "required": "3.11"})
        record("dependencies", lambda: check_dependencies(root))
        record("cuda", lambda: probe_cuda(require_cuda))
        def parent():
            expected = config.get("parent_sha256", "")
            if re.fullmatch(r"[a-fA-F0-9]{64}", expected) is None:
                raise ValueError("Explicit complete parent_sha256 is required")
            actual = file_sha(paths["parent_checkpoint"])
            return {"passed": actual == expected.lower(), "sha256": actual, "expected_sha256": expected.lower(),
                    "bytes": paths["parent_checkpoint"].stat().st_size, "checkpoint_deserialized": False}
        record("parent_checkpoint_bytes", parent)
        record("data_and_sample_contracts", lambda: check_data(root, config, paths))
        if verify_all_assets:
            record("all_asset_and_package_SHA256", lambda: verify_manifests(root))
        else:
            report["asset_verification"] = "not requested; launch preflight with --verify-all-assets to hash all manifest entries"
    except Exception as error:
        report["checks"].append({"name": "configuration", "passed": False, "error_type": type(error).__name__, "error": str(error)})
    report["passed"] = bool(report["checks"]) and all(check["passed"] for check in report["checks"])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["exit_code"] = 0 if report["passed"] else 1
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = reports / f"preflight_{stamp}_{os.getpid()}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--verify-all-assets", action="store_true")
    args = parser.parse_args()
    report, output = run_preflight(args.config, require_cuda=args.require_cuda, verify_all_assets=args.verify_all_assets)
    print(json.dumps({"passed": report["passed"], "exit_code": report["exit_code"], "report": str(output)}, ensure_ascii=True))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
