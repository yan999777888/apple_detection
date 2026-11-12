"""Bounded dataset-attribute ROI linear-head pilot using an existing COCO backbone.

Labels retain the exact semantics declared in the real source manifest. They
do not establish physiological maturity, internal quality, or market grades. No test
records are accepted. Without an explicitly documented independent group split,
only engineering training loss and label mappings are reported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", YOLO_AUTOINSTALL="False")

WEIGHT_SHA = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
DEFAULT_WEIGHTS = r"F:\研究生毕业设计\模块一定位补偿以及预测\apple_vision\local_amodal\cache\torch\hub\checkpoints\maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
SEMANTIC_LIMITS = "Dataset-specific ROI attribute labels do not establish physiological maturity, internal fruit quality, or commodity grade; region labels do not establish whole-fruit quality"


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    p.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS))
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--check-only", action="store_true", help="Validate records and input hashes; no torch/GPU, no output files")
    return p


def resolve_input(value, parent):
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def validate_manifest(document, manifest_parent):
    require(isinstance(document.get("label_semantics"), str) and bool(document["label_semantics"].strip()), "Manifest requires nonempty label_semantics from its actual dataset")
    records = document.get("records")
    require(isinstance(records, list) and records, "Manifest needs a nonempty records list")
    # Reject any test declaration before resolving or opening image/annotation files.
    require(all(isinstance(r, dict) for r in records), "Every record must be an object")
    require(all(r.get("split") in ("train", "val") for r in records), "Only train/val records allowed; test and unknown splits are refused")
    declared = document.get("class_names")
    if declared is not None:
        require(isinstance(declared, list) and len(declared) >= 2 and all(isinstance(x, str) and x.strip() for x in declared), "class_names must contain at least two nonempty strings")
        require(len(set(declared)) == len(declared), "Duplicate class names")
    seen, source_splits, group_splits, path_splits = set(), defaultdict(set), defaultdict(set), defaultdict(set)
    normalized = []
    for i, record in enumerate(records):
        for field in ("image_path", "bbox_xyxy", "label", "source_image_id", "group_id", "split"):
            require(field in record and record[field] is not None, f"Record {i}: missing {field}")
        require(isinstance(record["source_image_id"], str) and record["source_image_id"].strip(), f"Record {i}: empty source_image_id")
        require(isinstance(record["group_id"], str) and record["group_id"].strip(), f"Record {i}: empty group_id")
        label = record["label"]
        if isinstance(label, int) and not isinstance(label, bool):
            require(declared is not None and 0 <= label < len(declared), f"Record {i}: numeric labels require valid class_names")
            label = declared[label]
        require(isinstance(label, str) and label.strip(), f"Record {i}: label must be a nonempty category")
        require(declared is None or label in declared, f"Record {i}: undeclared label")
        bbox = record["bbox_xyxy"]
        require(isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox), f"Record {i}: invalid bbox")
        x1, y1, x2, y2 = map(float, bbox)
        require(0 <= x1 < x2 and 0 <= y1 < y2, f"Record {i}: invalid/empty bbox")
        image = resolve_input(record["image_path"], manifest_parent)
        identity = (str(image), tuple(map(float, bbox)))
        require(identity not in seen, f"Record {i}: duplicated ROI (possibly conflicting labels)")
        seen.add(identity)
        source_splits[record["source_image_id"]].add(record["split"])
        group_splits[record["group_id"]].add(record["split"])
        path_splits[str(image)].add(record["split"])
        row = dict(record, image_path=str(image), label=label, bbox_xyxy=list(map(float, bbox)), record_index=i)
        if record.get("annotation_path"):
            row["annotation_path"] = str(resolve_input(record["annotation_path"], manifest_parent))
        normalized.append(row)
    for name, values in (("source image", source_splits), ("group", group_splits), ("image path", path_splits)):
        require(all(len(v) == 1 for v in values.values()), f"{name} crosses train/val split")
    train = [r for r in normalized if r["split"] == "train"]
    require(bool(train), "No training records")
    train_labels = set(r["label"] for r in train)
    names = declared or sorted(train_labels)
    require(len(names) >= 2, "At least two genuinely labeled training classes required")
    require(set(names) == train_labels, "Every declared class must have real training records")
    require(all(r["label"] in train_labels for r in normalized), "Validation contains a class absent from training")
    policy = document.get("split_policy", {})
    require(isinstance(policy, dict), "split_policy must be an object")
    independent = (policy.get("independent_group_holdout") is True
                   and isinstance(policy.get("evidence"), str) and bool(policy["evidence"].strip())
                   and isinstance(policy.get("group_definition"), str) and bool(policy["group_definition"].strip()))
    val = [r for r in normalized if r["split"] == "val"] if independent else []
    mapping = {name: index for index, name in enumerate(names)}
    for row in normalized:
        row["label_id"] = mapping[row["label"]]
    return normalized, train, val, names, independent


def preflight(args):
    require(1 <= args.max_steps <= 256, "max_steps must be explicit and bounded in [1,256]; default is 32")
    require(1 <= args.batch_size <= 32, "batch_size must be in [1,32]")
    require(math.isfinite(args.learning_rate) and 0 < args.learning_rate <= .1, "Invalid learning rate")
    manifest, output, weights = args.manifest.resolve(), args.output.resolve(), args.weights.resolve()
    require(not output.exists(), "Output exists; refusing to overwrite")
    document = json.loads(manifest.read_text(encoding="utf-8-sig"))
    records, train, val, names, independent = validate_manifest(document, manifest.parent)
    receipts = {str(manifest): sha(manifest), str(weights): sha(weights)}
    require(receipts[str(weights)] == WEIGHT_SHA, "Weights must be the exact verified official COCO Mask R-CNN ResNet50 FPN checkpoint")
    # Unreliable val records are not opened at all.
    for row in train + val:
        for path_field, hash_field in (("image_path", "image_sha256"), ("annotation_path", "annotation_sha256")):
            if row.get(path_field):
                path = row[path_field]
                if path not in receipts:
                    receipts[path] = sha(path)
                require(not row.get(hash_field) or row[hash_field].lower() == receipts[path], f"Input hash mismatch: {path}")
    from PIL import Image
    dimensions, content_splits = {}, defaultdict(set)
    for row in train + val:
        path = row["image_path"]
        if path not in dimensions:
            with Image.open(path) as image:
                dimensions[path] = image.size
                image.verify()
        width, height = dimensions[path]
        require(row["bbox_xyxy"][2] <= width and row["bbox_xyxy"][3] <= height, f"ROI outside actual image: record {row['record_index']}")
        content_splits[receipts[path]].add(row["split"])
    require(all(len(splits) == 1 for splits in content_splits.values()), "Exact duplicate image content crosses train/val")
    counts = {split: dict(Counter(r["label"] for r in records if r["split"] == split)) for split in ("train", "val")}
    return document, records, train, val, names, receipts, counts, independent


def fingerprint(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        data = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(data.dtype).encode()); h.update(str(data.shape).encode())
        h.update(memoryview(data).cast("B"))
    return h.hexdigest()


def latency_summary(values):
    values = sorted(float(v) for v in values)
    require(bool(values) and all(math.isfinite(v) and v > 0 for v in values), "Invalid benchmark timings")
    q = .95 * (len(values) - 1)
    low, high = math.floor(q), math.ceil(q)
    mean = statistics.mean(values)
    return {"n_ROIs": len(values), "mean_ms": mean, "p95_ms": values[low] + (values[high] - values[low]) * (q - low), "ROIs_per_second": 1000 / mean}


def run(args):
    document, records, train, val, names, receipts, counts, independent = preflight(args)
    if args.check_only:
        print(json.dumps({"phase": "check_only_passed", "classes": names, "counts": counts,
            "independent_validation_eligible": independent and bool(val), "validation_records_to_read": len(val),
            "max_steps": args.max_steps, "input_hash_count": len(receipts), "GPU_initialized": False,
            "label_semantics": document["label_semantics"], "semantic_limits": SEMANTIC_LIMITS}, ensure_ascii=False, indent=2))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    for variable in ("TEMP", "TMP", "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "NUMBA_CACHE_DIR"):
        directory = runtime / variable.lower(); directory.mkdir(); os.environ[variable] = str(directory)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    script_sha = sha(__file__)
    status = {"schema": "dataset_attribute_frozen_resnet50_linear_pilot_v1", "phase": "initializing",
        "pid": os.getpid(), "created_utc": now(), "seed": args.seed, "manifest_sha256": receipts[str(args.manifest.resolve())],
        "script_sha256": script_sha, "label_semantics": document["label_semantics"], "semantic_limits": SEMANTIC_LIMITS, "class_names": names,
        "label_to_id": {name: i for i, name in enumerate(names)}, "label_counts": counts,
        "split_policy": document.get("split_policy", {}), "independent_validation_eligible": independent and bool(val),
        "validation_skipped_reason": None if val else "No documented independent group holdout; only engineering training evidence",
        "test_used": False, "source": document.get("source"), "source_metadata": document.get("source_metadata"),
        "weights": str(args.weights.resolve()), "weights_sha256": WEIGHT_SHA,
        "weights_role": "Official COCO detection backbone initialization; contains no dataset-attribute head training",
        "recipe": {"max_steps": args.max_steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                   "optimizer": "AdamW", "weight_decay": .0001, "dtype": "FP32", "feature_dim": 2048,
                   "backbone": "ResNet50 body through layer4, FrozenBatchNorm2d, frozen/eval",
                   "head": f"Linear(2048,{len(names)})", "crop": "floor x1/y1, ceil x2/y2; all bounds validated; no clipping",
                   "resize": "224x224 PIL bilinear", "normalization_mean": [.485,.456,.406], "normalization_std": [.229,.224,.225]},
        "input_sha256_before": receipts, "actual_optimizer_updates": 0, "steps": []}
    write(output / "status.json", status)
    try:
        import numpy as np
        import torch
        import torchvision
        import PIL
        from PIL import Image
        from torchvision.models import resnet50
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d
        from torchvision.transforms import functional as TF
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        if args.device.startswith("cuda"):
            require(torch.cuda.is_available(), "CUDA unavailable; no silent CPU fallback")
            torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        status["runtime"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "Pillow": PIL.__version__, "device": str(device), "python": sys.version,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        body_state = {key[len("backbone.body."):]: value for key, value in state.items() if key.startswith("backbone.body.")}
        backbone = IntermediateLayerGetter(resnet50(weights=None, norm_layer=FrozenBatchNorm2d), return_layers={"layer4":"features"})
        restored = backbone.load_state_dict(body_state, strict=True)
        require(len(body_state) == 265, "Unexpected official backbone key count")
        del state, body_state
        backbone.requires_grad_(False).eval().to(device)
        require(all(not p.requires_grad for p in backbone.parameters()), "Backbone not frozen")
        backbone_before = fingerprint(backbone)
        head = torch.nn.Linear(2048, len(names)).to(device)
        head_before = fingerprint(head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=.0001)
        criterion = torch.nn.CrossEntropyLoss()
        status.update(backbone_strict_restore=str(restored), backbone_tensor_sha256_before=backbone_before,
                      head_tensor_sha256_before=head_before)
        def tensor_from_rgb(rgb, row):
            x1,y1,x2,y2 = row["bbox_xyxy"]
            crop = rgb.crop((math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2)))
            resized = crop.resize((224,224), resample=Image.Resampling.BILINEAR)
            return TF.normalize(TF.to_tensor(resized), [.485,.456,.406], [.229,.224,.225])

        def read_batch(rows):
            tensors = []
            for row in rows:
                with Image.open(row["image_path"]) as image:
                    tensors.append(tensor_from_rgb(image.convert("RGB"), row))
            return torch.stack(tensors).to(device)

        def features(x):
            require(not backbone.training and all(not module.training for module in backbone.modules()), "Backbone/BN was put in training mode")
            with torch.no_grad():
                return torch.nn.functional.adaptive_avg_pool2d(backbone(x)["features"], 1).flatten(1)

        # Fixed actual TRAIN ROIs only: a fitting diagnostic, never model selection.
        probe_rows = []
        for name in names:
            probe_rows.extend([row for row in train if row["label"] == name][:8])
        def probe_loss():
            total, per_class, count_per_class = 0., Counter(), Counter()
            head.eval()
            with torch.no_grad():
                for offset in range(0, len(probe_rows), args.batch_size):
                    batch = probe_rows[offset:offset+args.batch_size]
                    y = torch.tensor([row["label_id"] for row in batch], dtype=torch.long, device=device)
                    loss_values = torch.nn.functional.cross_entropy(head(features(read_batch(batch))), y, reduction="none")
                    require(torch.isfinite(loss_values).all().item(), "Nonfinite fixed training probe loss")
                    for row, value in zip(batch, loss_values.cpu().tolist()):
                        total += value; per_class[row["label"]] += value; count_per_class[row["label"]] += 1
            return {"mean_CE": total / len(probe_rows),
                    "per_class_mean_CE": {name: per_class[name] / count_per_class[name] for name in names}}
        status.update(phase="training_probe_before", updated_utc=now())
        write(output / "status.json", status)
        status["training_fit_check"] = {
            "scope": "Fixed actual training ROIs, first up to eight per class in manifest order",
            "record_indices": [row["record_index"] for row in probe_rows], "n_ROIs": len(probe_rows),
            "label_counts": dict(Counter(row["label"] for row in probe_rows)),
            "diagnostic_only": True, "accuracy_reported": False,
            "used_for_model_selection": False, "used_to_extend_training_or_tune_hyperparameters": False,
            "before": probe_loss(),
        }

        order, cursor, epoch = [], 0, 0
        rng = random.Random(args.seed)
        head.train()
        event_file = (output / "steps.jsonl").open("w", encoding="utf-8")
        try:
            for step in range(args.max_steps):
                selected = []
                for _ in range(args.batch_size):
                    if cursor == len(order):
                        order = list(range(len(train))); rng.shuffle(order); cursor = 0; epoch += 1
                    selected.append(train[order[cursor]]); cursor += 1
                started = time.perf_counter()
                x = read_batch(selected)
                y = torch.tensor([r["label_id"] for r in selected], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(features(x)); loss = criterion(logits, y)
                require(torch.isfinite(loss).item(), "Nonfinite loss; no skipped update")
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in head.parameters()), "Missing/nonfinite head gradient; no skipped update")
                require(all(p.grad is None for p in backbone.parameters()), "Backbone received gradients")
                grad_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in head.parameters())).item())
                before_update = fingerprint(head)
                optimizer.step()
                require(all(torch.isfinite(p).all().item() for p in head.parameters()), "Nonfinite head after update")
                head_changed = fingerprint(head) != before_update
                row = {"step": step + 1, "loss": float(loss.item()), "gradient_finite": True,
                       "gradient_norm": grad_norm, "optimizer_step_completed": True,
                       "head_values_changed": head_changed, "sample_pass": epoch,
                       "record_indices": [r["record_index"] for r in selected],
                       "batch_label_counts": dict(Counter(r["label"] for r in selected)),
                       "step_wall_ms_including_IO_and_audit": (time.perf_counter() - started) * 1000}
                event_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); event_file.flush()
                status["steps"].append(row)
                status.update(phase="training", actual_optimizer_updates=step + 1, updated_utc=now())
                write(output / "status.json", status)
                print(f"ATTRIBUTE_PILOT step={step+1}/{args.max_steps} loss={row['loss']:.6f}", flush=True)
        finally:
            event_file.close()
        head.eval()
        status.update(phase="training_probe_after", updated_utc=now())
        write(output / "status.json", status)
        status["training_fit_check"]["after"] = probe_loss()
        status["training_fit_check"]["mean_CE_change_after_minus_before"] = (
            status["training_fit_check"]["after"]["mean_CE"] - status["training_fit_check"]["before"]["mean_CE"])
        if val:
            status.update(phase="validation", updated_utc=now())
            write(output / "status.json", status)
            confusion = np.zeros((len(names), len(names)), dtype=np.int64)
            validation_loss, validation_count, predictions = 0., 0, []
            with torch.no_grad():
                for offset in range(0, len(val), args.batch_size):
                    batch = val[offset:offset+args.batch_size]
                    y = torch.tensor([r["label_id"] for r in batch], device=device)
                    logits = head(features(read_batch(batch)))
                    validation_loss += float(criterion(logits, y).item()) * len(batch)
                    predicted = logits.argmax(1).cpu().tolist()
                    for row, predicted_id in zip(batch, predicted):
                        confusion[row["label_id"], predicted_id] += 1
                        predictions.append({"record_index": row["record_index"], "label_id": row["label_id"], "predicted_id": predicted_id})
                    validation_count += len(batch)
                    status.update(validation_records_completed=validation_count, updated_utc=now())
                    write(output / "status.json", status)
            status["validation"] = {"scope": "Manifest-declared independent group development holdout; not an independent final test",
                "n_ROIs": validation_count, "loss": validation_loss / validation_count,
                "accuracy": float(np.trace(confusion) / validation_count), "confusion_matrix": confusion.tolist(),
                "class_order": names, "group_count": len({r["group_id"] for r in val})}
            write(output / "validation_predictions.json", predictions)
        else:
            status["validation"] = None

        # Single-ROI engineering throughput. Decode happens before the timer.
        status.update(phase="benchmark", updated_utc=now())
        write(output / "status.json", status)
        bench_rows = train[:min(8, len(train))]
        bench_rgb = []
        for row in bench_rows:
            with Image.open(row["image_path"]) as image:
                bench_rgb.append(image.convert("RGB"))
        def sync():
            if device.type == "cuda": torch.cuda.synchronize(device)
        def timed_roi(rgb, row):
            sync(); started = time.perf_counter()
            x = tensor_from_rgb(rgb, row).unsqueeze(0).to(device)
            with torch.no_grad():
                probability = head(features(x)).softmax(1)[0].cpu().tolist()
            predicted = max(range(len(probability)), key=probability.__getitem__)
            sync(); elapsed = (time.perf_counter() - started) * 1000
            return {"record_index": row["record_index"], "ms": elapsed, "predicted_id": predicted}
        for i in range(3): timed_roi(bench_rgb[i % len(bench_rgb)], bench_rows[i % len(bench_rows)])
        timings = [dict(timed_roi(rgb, row), repeat=repeat+1) for repeat in range(3) for rgb,row in zip(bench_rgb,bench_rows)]
        status["benchmark"] = {"scope": "One annotated ROI: predecoded RGB -> crop -> resize224 -> tensor/normalize -> H2D -> frozen ResNet50 -> linear head -> softmax and CPU output",
            "unit": "ROIs/s, not detector frames/s or end-to-end robot FPS", "excludes": ["disk decode", "detector", "camera", "depth", "tracking", "IO export"],
            "sample_split": "train", "warmup": 3, "rounds": 3, "frames": timings,
            "summary": latency_summary([r["ms"] for r in timings])}
        backbone_after, head_after = fingerprint(backbone), fingerprint(head)
        require(backbone_after == backbone_before, "Frozen backbone values changed")
        require(all(p.grad is None for p in backbone.parameters()), "Backbone gradient invariant failed")
        require(sha(__file__) == script_sha, "Pilot source changed during run")
        input_after = {path: sha(path) for path in receipts}
        require(input_after == receipts, "Source data, manifest or initialization weights changed")
        checkpoint = {"schema": status["schema"], "label_semantics": document["label_semantics"],
            "semantic_limits": SEMANTIC_LIMITS, "class_names": names,
            "head_state_dict": {k:v.detach().cpu() for k,v in head.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "backbone_source_weights_sha256": WEIGHT_SHA,
            "backbone_tensor_sha256": backbone_after, "recipe": status["recipe"], "seed": args.seed,
            "manifest_sha256": status["manifest_sha256"], "actual_optimizer_updates": args.max_steps,
            "training_fit_check": status["training_fit_check"],
            "checkpoint_role": "Engineering pilot, not production-ready or physiologically validated"}
        torch.save(checkpoint, output / "linear_head_pilot.pt")
        status.update(phase="complete", completed_utc=now(), backbone_tensor_sha256_after=backbone_after,
            frozen_backbone_unchanged=True, backbone_received_no_gradients=True, head_tensor_sha256_after=head_after,
            head_changed_from_initialization=head_after != head_before, input_sha256_after=input_after,
            checkpoint_sha256=sha(output / "linear_head_pilot.pt"), production_model_claim=False)
        write(output / "pilot_report.json", status); write(output / "status.json", status)
        return 0
    except BaseException as error:
        status.update(phase="failed", failed_utc=now(), error=str(error), traceback=traceback.format_exc())
        write(output / "status.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
