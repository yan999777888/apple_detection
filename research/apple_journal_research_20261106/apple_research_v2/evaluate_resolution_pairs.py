"""Frozen-weight paired 640/960 full-view development evaluation.

Primary metric is class-aware BOX IoU>=0.5, conf>=0.25, score-first one-to-one
matching. Stage GT boxes enclose author polygons; no claim of visible-mask GT.
No training, learned routing, test access, camera use or robot FPS. FP32 only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import socket
import struct
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SIZES = (640, 960)
CONF = .25
MATCH_IOU = .5


def float32_predecessor(value):
    """Exact positive finite FP32 predecessor for strict backend thresholds."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected positive finite FP32 threshold")
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    return struct.unpack(">f", struct.pack(">I", bits-1))[0]


BACKEND_CONF = float32_predecessor(CONF)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def names_dict(value):
    if isinstance(value, list):
        value = dict(enumerate(value))
    require(isinstance(value, dict) and bool(value), "Class names must be a nonempty list/dict")
    result = {int(k): v for k, v in value.items()}
    require(len(result) == len(value) and sorted(result) == list(range(len(result))), "Class IDs must be contiguous")
    require(all(isinstance(v, str) and v.strip() for v in result.values()), "Class names must be explicit")
    return result


def valid_box(box):
    return (len(box) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box)
            and box[2] > box[0] and box[3] > box[1])


def box_iou(a, b):
    require(valid_box(a) and valid_box(b), "Invalid or zero-area xyxy box")
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union


def match_boxes(predictions, ground_truth, class_names, threshold=CONF, iou_threshold=MATCH_IOU):
    """Stable score-first greedy class-aware matching; every class is reported."""
    require(0 <= threshold <= 1 and 0 < iou_threshold <= 1, "Invalid evaluation threshold")
    names = names_dict(class_names)
    require(len({g["GT_id"] for g in ground_truth}) == len(ground_truth), "GT IDs must be unique within an image")
    per_class = {k: {"class_name": v, "support": 0, "TP": 0, "FP": 0, "FN": 0} for k, v in names.items()}
    for gt in ground_truth:
        require(gt["class_id"] in names and valid_box(gt["box_xyxy"]), "Invalid GT class/box")
        per_class[gt["class_id"]]["support"] += 1
    for pred in predictions:
        require(pred["class_id"] in names and valid_box(pred["box_xyxy"]), "Invalid prediction class/box")
        require(math.isfinite(pred["score"]) and 0 <= pred["score"] <= 1, "Invalid confidence")
    kept = [i for i, pred in enumerate(predictions) if pred["score"] >= threshold]
    used = set()
    matches, false_positives = [], []
    for pi in sorted(kept, key=lambda i: (-predictions[i]["score"], i)):
        pred = predictions[pi]
        candidates = [(box_iou(pred["box_xyxy"], gt["box_xyxy"]), gi)
                      for gi, gt in enumerate(ground_truth)
                      if gi not in used and gt["class_id"] == pred["class_id"]]
        candidates = [(iou, gi) for iou, gi in candidates if iou >= iou_threshold]
        if candidates:
            iou, gi = min(candidates, key=lambda item: (-item[0], item[1]))
            used.add(gi)
            per_class[pred["class_id"]]["TP"] += 1
            matches.append({"prediction_index": pi, "GT_index": gi,
                            "GT_id": ground_truth[gi]["GT_id"], "IoU": iou})
        else:
            per_class[pred["class_id"]]["FP"] += 1
            false_positives.append(pi)
    missed = [i for i in range(len(ground_truth)) if i not in used]
    for gi in missed:
        per_class[ground_truth[gi]["class_id"]]["FN"] += 1
    totals = {key: sum(row[key] for row in per_class.values()) for key in ("support", "TP", "FP", "FN")}
    totals["FP_plus_FN"] = totals["FP"]+totals["FN"]
    return {**totals, "per_class": per_class, "matches": matches,
            "false_positive_prediction_indices": false_positives,
            "missed_GT_indices": missed, "retained_predictions": len(kept)}


def pair_delta(low, high):
    low_ids = {row["GT_id"] for row in low["matches"]}
    high_ids = {row["GT_id"] for row in high["matches"]}
    delta = low["FP_plus_FN"]-high["FP_plus_FN"]
    return {"risk_640": low["FP_plus_FN"], "risk_960": high["FP_plus_FN"],
            "risk_reduction_640_minus_960": delta,
            "sign": "positive" if delta > 0 else "negative" if delta < 0 else "zero",
            "recovered_GT_ids": sorted(high_ids-low_ids), "lost_GT_ids": sorted(low_ids-high_ids),
            "delta_FP_960_minus_640": high["FP"]-low["FP"],
            "delta_FN_960_minus_640": high["FN"]-low["FN"],
            "GT_oracle_min_risk": min(low["FP_plus_FN"], high["FP_plus_FN"]),
            "GT_oracle_uses_labels_not_deployable": True}


def within(path, root):
    result = Path(path).resolve()
    require(result.is_relative_to(root), f"Path escapes research root: {result}")
    return result


def reject_test_path(path):
    tokens = [token for part in Path(path).parts for token in re.split(r"[^a-z0-9]+", part.lower())]
    require(not {"test", "tests", "testing"} & set(tokens), f"Test-labelled input is forbidden: {path}")


def yolo_val_paths(spec, root, workspace):
    """Resolve only explicit YAML val paths, never a test split or recursive glob."""
    result = []
    for entry in spec if isinstance(spec, list) else [spec]:
        require(isinstance(entry, str) and not any(c in entry for c in '*?[]'), "Explicit val path required")
        path = Path(entry)
        path = path if path.is_absolute() else root/path
        path = within(path, workspace)
        reject_test_path(path)
        if path.is_dir():
            result += sorted(p.resolve() for p in path.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
        elif path.is_file() and path.suffix.lower() == '.txt':
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    image = Path(line.strip())
                    image = within(image if image.is_absolute() else root/image, workspace)
                    reject_test_path(image)
                    result.append(image)
        else:
            raise ValueError("Expected explicit val directory or image list")
    for image in result:
        within(image, workspace)
        reject_test_path(image)
    require(len(result) == len(set(result)), "Duplicate YAML val images")
    return result


def read_dataset(data_yaml, manifest_path, kind, workspace):
    import yaml
    from PIL import Image
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    names = names_dict(config["names"])
    root = Path(config.get("path", data_yaml.parent))
    root = (root if root.is_absolute() else data_yaml.parent/root).resolve()
    within(root, workspace)
    declared_paths = yolo_val_paths(config["val"], root, workspace)
    source = read_json(manifest_path)
    input_hashes = {str(data_yaml): sha(data_yaml), str(manifest_path): sha(manifest_path)}
    if kind == "stage":
        require(source["schema"] == "mendeley_stage_segmentation_v1", "Wrong stage manifest schema")
        require(names == names_dict(source["class_names"]), "Stage YAML/source class mapping mismatch")
        entries = [r for r in source["images"] if r["split"] == "val"]
        semantics = {k: source[k] for k in ("label_semantics", "mask_semantics", "evaluation_role")}
        expected_count = source["counts"]["val"]
    else:
        require(source["schema"] == "ssda_date_manifest_v2", "Wrong SSDA manifest schema")
        audit_path = within(manifest_path.parent/source["report"], workspace)
        audit = read_json(audit_path)
        input_hashes[str(audit_path)] = sha(audit_path)
        require(names == names_dict(audit["numeric_classes"]), "SSDA YAML/source numeric names mismatch")
        entries = [r for r in source["samples"] if r["partition"] == "dev"]
        semantics = {"label_semantics": "Author numeric defect-region boxes only; no normal fruit, stage or grade labels",
                     "mask_semantics": "No mask GT", "evaluation_role": "Date-held-out development; physical fruit independence unknown"}
        expected_count = {"images": audit["partitions"]["dev"], "instances": sum(audit["class_instance_counts"]["dev"].values())}
    rows = []
    for entry in entries:
        image = within(entry["image_path"] if kind == "stage" else entry["rgb_path"], workspace)
        label = within(entry["label_path"], workspace)
        reject_test_path(image)
        reject_test_path(label)
        if kind == "stage":
            annotation = within(entry["annotation_path"], workspace)
            reject_test_path(annotation)
            if str(annotation) not in input_hashes:
                require(sha(annotation) == entry["annotation_sha256"], "Original annotation hash changed")
                input_hashes[str(annotation)] = entry["annotation_sha256"]
        expected_image_sha = entry["image_sha256"] if kind == "stage" else entry["rgb_sha256"]
        require(sha(image) == expected_image_sha and sha(label) == entry["label_sha256"], "Image/label SHA changed")
        input_hashes[str(image)], input_hashes[str(label)] = expected_image_sha, entry["label_sha256"]
        with Image.open(image) as im:
            width, height = im.size
            im.verify()
        require([height, width] == entry["native_hw"], "Native image dimensions mismatch")
        lines = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(len(lines) == len(entry["instances"]), "Manifest/file instance count mismatch")
        gt = []
        for index, (tokens, instance) in enumerate(zip(lines, entry["instances"])):
            numeric = [float(v) for v in tokens]
            require(all(math.isfinite(v) for v in numeric), "Nonfinite label")
            label_id = int(numeric[0])
            require(label_id == numeric[0] and label_id in names, "Invalid label class")
            if kind == "stage":
                points = instance["source_vertices_xy"]
                require(label_id == instance["label_id"] and names[label_id] == instance["label_original"], "Stage class changed")
                expected = [coordinate for point in points for coordinate in (point[0]/width, point[1]/height)]
                require(len(numeric) == len(expected)+1 and all(abs(a-b) <= 1e-14 for a,b in zip(numeric[1:], expected)), "Source polygon differs from YOLO label")
                box = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
                extra = {"source_region_index": instance["source_region_index"], "source_polygon_xy": points,
                         "box_semantics": "Axis-aligned envelope of unchanged author polygon; not amodal/visible GT relabeling"}
            else:
                require(label_id == instance["class_id"] and len(numeric) == 5, "SSDA label/class changed")
                norm = instance["xywh_normalized"]
                require(all(abs(a-b) <= 1e-14 for a,b in zip(numeric[1:],norm)), "SSDA label coordinates changed")
                xc,yc,w,h = norm
                box = [(xc-w/2)*width, (yc-h/2)*height, (xc+w/2)*width, (yc+h/2)*height]
                extra = {"source_line": instance["source_line"], "box_semantics": "Original normalized defect-region box mapped to native pixels"}
            require(valid_box(box) and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height, "Invalid/out-of-bounds original GT")
            gt.append({"GT_id": f"{image.name}#{index}", "class_id": label_id, "class_name": names[label_id],
                       "box_xyxy": box, **extra})
        rows.append({"image_id": image.name, "image_path": str(image), "label_path": str(label),
                     "image_sha256": expected_image_sha, "label_sha256": entry["label_sha256"],
                     "native_hw": [height,width], "known_group_id": entry["group_id"] if kind == "stage" else entry["same_second_group"],
                     "fruit_identity_known": False, "ground_truth": gt})
    require(rows and len({r["image_id"] for r in rows}) == len(rows), "Empty/duplicate image identities")
    require(set(Path(r["image_path"]) for r in rows) == set(declared_paths), "YAML val images differ from manifest val/dev")
    require(len(rows) == expected_count["images"] and sum(len(r["ground_truth"]) for r in rows) == expected_count["instances"], "Development counts mismatch")
    return names, rows, semantics, input_hashes


def configure_runtime(output):
    runtime = output/"runtime"
    mapping = {"YOLO_CONFIG_DIR":"yolo", "MPLCONFIGDIR":"matplotlib", "TORCH_HOME":"torch", "HF_HOME":"hf",
               "XDG_CACHE_HOME":"cache", "CUDA_CACHE_PATH":"cuda", "TORCH_EXTENSIONS_DIR":"extensions", "PIP_CACHE_DIR":"pip", "TEMP":"tmp", "TMP":"tmp"}
    for key, subdir in mapping.items():
        directory = runtime/subdir
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", YOLO_AUTOINSTALL="0", YOLO_OFFLINE="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", COMET_MODE="DISABLED",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    def refuse_network(*args, **kwargs):
        raise RuntimeError("Network is disabled in this local paired evaluation")
    socket.create_connection = refuse_network
    socket.socket.connect = refuse_network
    socket.socket.connect_ex = refuse_network


def counts_summary(results, class_names):
    per_class = {str(k): {"class_name": v, **{key: 0 for key in ("support","TP","FP","FN")}}
                 for k,v in class_names.items()}
    for result in results:
        for key, row in result["per_class"].items():
            for metric in ("support","TP","FP","FN"):
                per_class[str(key)][metric] += row[metric]
    total = {key: sum(row[key] for row in per_class.values()) for key in ("support","TP","FP","FN")}
    for row in [total, *per_class.values()]:
        row["precision"] = row["TP"]/(row["TP"]+row["FP"]) if row["TP"]+row["FP"] else None
        row["recall"] = row["TP"]/row["support"] if row["support"] else None
        row["FP_plus_FN"] = row["FP"]+row["FN"]
    return {**total, "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stage","ssda"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-official-val", action="store_true", help="Separate standard YOLO val at each size after paired predictions; root schedules GPU")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    require(workspace.drive.upper() == "F:", "All runtime/source/output paths must be on F")
    marker = workspace/"relocation_complete.json"
    require(marker.is_file(), "Relocation marker missing")
    output = within(args.output, workspace)
    require(not output.exists(), "New output directory required; no overwrite/resume")
    data, manifest = within(args.data, workspace), within(args.manifest, workspace)
    reject_test_path(data)
    reject_test_path(manifest)
    weights = args.weights.resolve()
    require(weights.drive.upper() == "F:" and weights.is_file() and weights.suffix == '.pt', "Existing local F-drive .pt weights required")
    require(sha(weights) == args.expected_weights_sha256.lower(), "Weight SHA mismatch")
    require(1 <= args.warmup <= 20 and 1 <= args.max_det <= 1000, "Invalid bounded warmup/max_det")
    require(args.device.startswith('cuda'), "Executed evaluation is CUDA-only; check-only does not initialize GPU")
    require(not (args.check_only and args.run_official_val), "check-only and official-val are mutually exclusive")
    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(output)
    if args.check_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    names, rows, semantics, input_hashes = read_dataset(data, manifest, args.kind, workspace)
    input_hashes[str(weights)] = args.expected_weights_sha256.lower()
    input_hashes[str(marker)] = sha(marker)
    preflight = {"schema":"resolution_pairs_preflight_v1", "kind":args.kind, "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha256":sha(__file__), "source_paths_sha256":input_hashes, "weights":str(weights),
        "data":str(data), "manifest":str(manifest), "class_names":names, "images":len(rows),
        "GT_instances":sum(len(r["ground_truth"]) for r in rows),
        "class_support":dict(Counter(gt["class_name"] for r in rows for gt in r["ground_truth"])),
        "semantics":semantics, "development_only":True, "test_read":False,
        "check_only":args.check_only, "GPU_used":False, "class_mapping_in_checkpoint_checked":False}
    save(output/"preflight.json", preflight)
    save(output/"input_ground_truth.json", rows)
    if args.check_only:
        print(json.dumps({k:v for k,v in preflight.items() if k != "source_paths_sha256"},ensure_ascii=False))
        return
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    require(ultralytics.__version__ == "8.4.115", "Protocol requires installed Ultralytics 8.4.115")
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    model = YOLO(str(weights), task="segment" if args.kind == "stage" else "detect")
    require(names_dict(model.names) == names, "Trained weight class names differ from audited data")
    require(model.task == ("segment" if args.kind == "stage" else "detect"), "Checkpoint task differs from protocol")

    model.model.float().eval()
    checkpoint_metadata = {"epoch_index_zero_based": model.ckpt.get("epoch"),
        "epoch_completed": model.ckpt.get("audit", {}).get("epoch_completed"),
        "selection": "Explicit supplied checkpoint; no model selection by this script"}

    def decode(row):
        bgr = cv2.imdecode(np.fromfile(row["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        require(bgr is not None and list(bgr.shape[:2]) == row["native_hw"], "Decode/native size mismatch")
        return bgr

    def predict(bgr, size):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.predict(source=bgr, imgsz=size, conf=BACKEND_CONF, iou=.7, max_det=args.max_det,
            device=args.device, quantize=32, batch=1, rect=False, augment=False, retina_masks=args.kind == "stage",
            verbose=False, save=False, save_txt=False, save_conf=False, show=False,
            project=str(output), name="predict", exist_ok=True)[0]
        boxes = result.boxes.data.detach().float().cpu().numpy().copy()
        masks = (result.masks.data.detach().cpu().numpy().copy() if result.masks is not None else None)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        require(not model.predictor.model.fp16, "Backend unexpectedly FP16")
        require(boxes.ndim == 2 and boxes.shape[1] == 6 and np.isfinite(boxes).all(), "Unexpected prediction box schema")
        # Ultralytics 8.4.115 uses score > conf. A float32 predecessor retains
        # exactly .25; explicitly enforce the published inclusive threshold.
        keep = boxes[:, 4] >= CONF
        boxes = boxes[keep]
        if masks is not None:
            require(len(masks) == len(keep), "Raw box/mask count mismatch")
            masks = masks[keep]
        predictions = []
        for index, values in enumerate(boxes):
            x1,y1,x2,y2,score,label = [float(v) for v in values]
            require(label == int(label) and int(label) in names, "Prediction class outside audited mapping")
            require(score >= CONF, "Backend returned score below fixed .25 threshold")
            predictions.append({"prediction_index":index, "class_id":int(label), "class_name":names[int(label)],
                                "score":score, "box_xyxy":[x1,y1,x2,y2]})
        if args.kind == "stage":
            if masks is None:
                require(not len(boxes), "Boxes without instance masks")
                masks = np.zeros((0,*bgr.shape[:2]),dtype=np.uint8)
            require(masks.shape == (len(boxes),*bgr.shape[:2]) and np.isin(masks,[0,1]).all(), "Expected native binary mask rasters")
            masks = masks.astype(np.uint8,copy=False)
        return predictions, masks, elapsed

    first = decode(rows[0])
    for size in SIZES:
        for _ in range(args.warmup):
            predict(first,size)
    paired, measurements = [], {size:[] for size in SIZES}
    (output/"predictions").mkdir()
    for index,row in enumerate(rows):
        bgr = decode(row)
        order = SIZES if index % 2 == 0 else tuple(reversed(SIZES))
        decoded_sha = hashlib.sha256(bgr.tobytes()).hexdigest()
        case = {"decoded_BGR_sha256":decoded_sha, "image_id":row["image_id"], "known_group_id":row["known_group_id"], "order":list(order), "resolutions":{}}
        for size in order:
            predictions,masks,elapsed = predict(bgr,size)
            require(hashlib.sha256(bgr.tobytes()).hexdigest() == decoded_sha, "Predictor modified input between paired passes")
            result = match_boxes(predictions,row["ground_truth"],names)
            detail = {"image_id":row["image_id"], "imgsz":size, "native_hw":row["native_hw"],
                "weights_sha256":args.expected_weights_sha256.lower(), "ground_truth":row["ground_truth"],
                "predictions":predictions, "box_matching":result, "native_prediction_call_ms":elapsed,
                "full_view":True, "GT_mask_metrics_computed":False}
            if masks is not None:
                mask_file = output/"predictions"/f"{index:05d}_{size}_masks.npz"
                np.savez_compressed(mask_file,masks=masks)
                detail["native_mask_export"] = {"path":str(mask_file.relative_to(output)),"sha256":sha(mask_file),"shape":list(masks.shape),"dtype":str(masks.dtype)}
            detail_file = output/"predictions"/f"{index:05d}_{size}.json"
            save(detail_file,detail)
            case["resolutions"][str(size)] = {"box_matching":result, "native_prediction_call_ms":elapsed,
                "prediction_file":str(detail_file.relative_to(output)),"prediction_file_sha256":sha(detail_file)}
            measurements[size].append(elapsed)
        case["delta"] = pair_delta(case["resolutions"]["640"]["box_matching"],case["resolutions"]["960"]["box_matching"])
        paired.append(case)
    save(output/"paired_images.json",paired)
    totals = {str(size):counts_summary([r["resolutions"][str(size)]["box_matching"] for r in paired], names) for size in SIZES}
    latency = {str(size):{"samples":len(values),"mean_ms":float(np.mean(values)),"p50_ms":float(np.percentile(values,50)),
                "p95_ms":float(np.percentile(values,95)),"p99_ms":float(np.percentile(values,99))} for size,values in measurements.items()}
    summary = {"schema":"resolution_pairs_v1", "phase":"complete", "created_utc":datetime.now(timezone.utc).isoformat(),
        "preflight_sha256":sha(output/"preflight.json"), "kind":args.kind, "weights":str(weights),
        "weights_sha256":args.expected_weights_sha256.lower(), "checkpoint":checkpoint_metadata, "class_names":names,"semantics":semantics,
        "images":len(rows),"counts":totals,"image_delta_sign_counts":{sign:sum(r["delta"]["sign"] == sign for r in paired) for sign in ("positive","zero","negative")},
        "per_class_risk_reduction_640_minus_960":{str(k):totals["640"]["per_class"][str(k)]["FP_plus_FN"]-totals["960"]["per_class"][str(k)]["FP_plus_FN"] for k in names},
        "risk_reduction_640_minus_960":totals["640"]["FP_plus_FN"]-totals["960"]["FP_plus_FN"],
        "GT_oracle": {"min_total_FP_plus_FN":sum(r["delta"]["GT_oracle_min_risk"] for r in paired),
            "uses_GT":True,"deployable":False,"meaning":"Per-image min of 640/960 risk; no learned policy and no budget-feasible guarantee"},
        "protocol":{"confidence_inclusive":CONF,"box_IoU_inclusive":MATCH_IOU,"class_aware":True,
            "matching":"descending confidence, stable prediction-index ties, best unused same-class GT; not official COCO AP",
            "requested_predict_iou":.7,
            "end2end_head":bool(getattr(model.model,"end2end",False)),
            "postprocess":("end2end confidence filter and max_det; requested iou is inactive, no NMS"
                           if bool(getattr(model.model,"end2end",False)) else "class-aware NMS with requested iou=.7"),
            "backend_confidence_strict_floor":BACKEND_CONF,
            "inclusive_confidence_enforcement":"score > FP32 predecessor(.25), then explicit score >= .25 with aligned mask filtering",
            "max_det":args.max_det,"batch":1,"precision":"FP32","rect":False,
            "TF32_enabled":False,"sizes":list(SIZES),"order":"640->960 for even manifest index; reverse for odd; no GT-based scheduling",
            "warmup_per_resolution":args.warmup,"native_stage_masks_exported":args.kind == "stage"},
        "prediction_latency":latency,
        "timing_scope":"Predecoded BGR through GPU prediction and CPU boxes/native masks; includes CUDA synchronization; one measured call per image/size",
        "timing_excludes":["decode","GT matching","hashing/export","camera","depth","3D","tracking","routing","robot communication"],
        "camera_to_output_latency":None,"robot_FPS":None,"official_AP_from_fixed_threshold":False,
        "training_performed":False,"learned_policy_evaluated":False,"test_read":False,"independent_fruit_test":False,
        "environment":{"python":sys.version,"torch":torch.__version__,"ultralytics":ultralytics.__version__,"GPU":torch.cuda.get_device_name(),"torch_threads":torch.get_num_threads()},
        "limitations":["Stage risk uses boxes derived from author polygons, not mask IoU or verified visible-mask truth",
            "Stage class labels represent author color stages, not physiological maturity",
            "SSDA has numeric defect types, no normal fruit or whole-fruit quality classes",
            "No verified physical fruit identities; results are development evidence only",
            "Higher resolution may harm predictions; negative deltas are retained",
            "Inference at 960 uses the same 640-trained weight, not a separately trained 960 model"]}
    summary["source_inputs_unchanged"] = all(sha(path) == digest for path,digest in input_hashes.items())
    require(summary["source_inputs_unchanged"], "Source/weights changed during paired evaluation")
    save(output/"summary.json",summary)
    if args.run_official_val:
        official = {"scope":"Separate official development AP; not fixed-.25 matching and not timing comparison", "resolutions":{}}
        for size in SIZES:
            metrics = model.val(data=str(data), split="val", imgsz=size, batch=1, device=args.device, quantize=32,
                conf=.001,iou=.7,max_det=args.max_det,plots=False,save_json=False,save_txt=False,cache=False,
                project=str(output/"official_val"),name=str(size),exist_ok=False,workers=0,verbose=False)
            official["resolutions"][str(size)] = {"metrics":{str(k):float(v) for k,v in metrics.results_dict.items()},
                "config":{"imgsz":size,"conf":.001,"requested_iou":.7,"max_det":args.max_det,"quantize":32,
                    "effective_rect":bool(model.validator.args.rect),
                    "end2end_head":bool(getattr(model.model,"end2end",False))},
                "weight_sha256":args.expected_weights_sha256.lower()}
        require(all(sha(path) == digest for path,digest in input_hashes.items()), "Input changed during official validation")
        save(output/"official_validation.json",official)
    print(json.dumps({"summary":str(output/"summary.json"),"counts":totals,"image_delta_sign_counts":summary["image_delta_sign_counts"],
                      "risk_reduction":summary["risk_reduction_640_minus_960"],"prediction_latency":latency},ensure_ascii=False))


if __name__ == "__main__":
    main()
