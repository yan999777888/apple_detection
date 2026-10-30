"""Same-resolution flip TTA and deterministic box fusion for the OD detector.

Prediction uses only RGB and model outputs; GT is copied from the published raw
records solely for subsequent evaluation. Fusion never reads GT. Final metrics
use the repository's exact greedy matcher and complete distinct-score curve.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tmp"))

from apple_seg.data import PAppleDataset  # noqa: E402
from apple_seg.od_data import ODInputDataset  # noqa: E402
from apple_seg.od_model import build_od_amodal_detector  # noqa: E402

METRICS_PATH = ROOT / "reference/local_amodal/scripts/metrics.py"
SPEC = importlib.util.spec_from_file_location("apple_exact_metrics", METRICS_PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_records(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Prediction file must contain a list")
    return records


def flip_boxes(boxes, image_hw, axis):
    """Map exclusive XYXY boxes through a horizontal or vertical image flip."""
    array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    height, width = (int(value) for value in image_hw)
    result = array.copy()
    if axis in ("h", "hv"):
        result[:, [0, 2]] = width - array[:, [2, 0]]
    if axis in ("v", "hv"):
        result[:, [1, 3]] = height - array[:, [3, 1]]
    if axis not in ("h", "v", "hv"):
        raise ValueError("axis must be h, v or hv")
    return result


def _rescale(score, factor):
    """Multiply the odds by a factor so the result stays a valid probability."""
    if factor == 1.0:
        return score
    clipped = min(max(float(score), 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-(math.log(clipped / (1 - clipped)) + math.log(factor))))


def fuse_entries(entries, iou_threshold=0.55, score_mode="max", box_mode="score_weighted",
                 coverage_floor=1.0, view_weights=None, consensus_groups=None,
                 consensus_bonus=1.0, isolation_penalty=1.0, cluster_metadata=None):
    """Greedy cross-view clustering with at most one member from each view.

    ``view_weights`` optionally trusts one view more than another when the two
    disagree. It only reweights members inside an existing cluster, so the
    candidate set and the one-box-per-view rule are unchanged.
    """
    if not entries:
        raise ValueError("At least one prediction view is required")
    if score_mode not in ("max", "mean") or box_mode not in ("score_weighted", "representative"):
        raise ValueError("Unsupported fusion mode")
    if not 0 <= coverage_floor <= 1:
        raise ValueError("coverage_floor must be in [0,1]")
    if view_weights is None:
        view_weights = np.ones(len(entries))
    view_weights = np.asarray(view_weights, dtype=np.float64)
    if (view_weights.shape != (len(entries),) or not np.isfinite(view_weights).all()
            or (view_weights <= 0).any()):
        raise ValueError("view_weights must be one positive finite weight per view")
    # Cross-architecture agreement: a cluster confirmed by two different
    # backbones is far less likely to be a single network's hallucination, while
    # an isolated single-view box is the most likely background false positive.
    if consensus_groups is None:
        groups = np.arange(len(entries))
    else:
        groups = np.asarray(consensus_groups, dtype=np.int64)
        if groups.shape != (len(entries),):
            raise ValueError("consensus_groups must give one group id per view")
    for value in (consensus_bonus, isolation_penalty):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 < value <= 4):
            raise ValueError("consensus bonus and isolation penalty must be in (0,4]")
    boxes = np.concatenate([np.asarray(entry["pred"]["boxes"], dtype=np.float64).reshape(-1, 4)
                            for entry in entries])
    scores = np.concatenate([np.asarray(entry["pred"]["scores"], dtype=np.float64)
                             for entry in entries])
    views = np.concatenate([np.full(len(entry["pred"]["scores"]), index, dtype=np.int64)
                            for index, entry in enumerate(entries)])
    if not len(boxes):
        return boxes, scores
    order = np.argsort(-scores, kind="stable")
    boxes, scores, views = boxes[order], scores[order], views[order]
    taken = np.zeros(len(boxes), dtype=bool)
    clusters = []
    for index in range(len(boxes)):
        if taken[index]:
            continue
        members = [index]
        used_views = {int(views[index])}
        taken[index] = True
        overlaps = metrics.box_iou(boxes[index:index + 1], boxes[index + 1:])[0]
        for offset, overlap in enumerate(overlaps):
            candidate = index + 1 + offset
            view = int(views[candidate])
            if not taken[candidate] and view not in used_views and overlap >= iou_threshold:
                members.append(candidate)
                used_views.add(view)
                taken[candidate] = True
        clusters.append(members)
    fused_boxes, fused_scores = [], []
    for members in clusters:
        if cluster_metadata is not None:
            cluster_metadata.append({"member_views": [int(views[index]) for index in members],
                                     "member_scores": [float(scores[index]) for index in members],
                                     "member_boxes": boxes[members].tolist()})
        member_boxes, member_scores = boxes[members], scores[members]
        member_views = views[members]
        trust = view_weights[member_views]
        if box_mode == "score_weighted":
            weights = member_scores * trust
            weights = weights / weights.sum()
            fused_boxes.append((member_boxes * weights[:, None]).sum(axis=0))
        else:
            fused_boxes.append(member_boxes[0])
        if score_mode == "max":
            score = member_scores.max()
        else:
            score = float((member_scores * trust).sum() / trust.sum())
        if len(entries) > 1:
            distinct = len(set(int(groups[view]) for view in member_views))
            if distinct > 1:
                score = _rescale(score, consensus_bonus)
            elif len(members) == 1:
                score = _rescale(score, isolation_penalty)
        if len(entries) > 1 and coverage_floor < 1:
            agreement = (coverage_floor
                         + (1 - coverage_floor) * (len(members) - 1) / (len(entries) - 1))
            score *= agreement
        fused_scores.append(score)
    return np.asarray(fused_boxes).reshape(-1, 4), np.asarray(fused_scores)


def fuse_records(layers, iou_threshold, score_mode, box_mode, coverage_floor=1.0,
                 view_weights=None, consensus_groups=None, consensus_bonus=1.0,
                 isolation_penalty=1.0):
    names = [record["image_id"] for record in layers[0]]
    if any([record["image_id"] for record in layer] != names for layer in layers[1:]):
        raise ValueError("Prediction layers differ in image identity or order")
    fused = []
    for entries in zip(*layers):
        if any(entry["gt"] != entries[0]["gt"] for entry in entries[1:]):
            raise ValueError("Prediction layers differ in GT identity or values")
        boxes, scores = fuse_entries(entries, iou_threshold, score_mode, box_mode,
                                     coverage_floor, view_weights, consensus_groups,
                                     consensus_bonus, isolation_penalty)
        fused.append({"image_id": entries[0]["image_id"], "gt": entries[0]["gt"],
                      "pred": {"boxes": boxes.tolist(), "scores": scores.tolist()}})
    return fused


def evaluate(records):
    calibrated = metrics.calibrate_threshold(records, 0.5, return_curve=False)
    summary = calibrated["summary"]
    return {"threshold": calibrated["confidence"], "precision": summary["precision"],
            "recall": summary["recall"], "min_precision_recall": summary["min_precision_recall"],
            "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"],
            "candidate_thresholds": calibrated["candidate_thresholds"]}


def verify_unflipped(model, dataset, base_records, device, count):
    """The custom inference path must reproduce the published boxes and scores."""
    differences = []
    with torch.inference_mode():
        for index in range(min(count, len(dataset))):
            item = dataset[index]
            output = model(item["pixel_values"].unsqueeze(0).to(device))[0]
            original_h, original_w = item["target"]["original_hw"]
            input_h, input_w = output["image_hw"]
            scale = np.asarray([original_w / input_w, original_h / input_h] * 2)
            boxes = output["boxes"].cpu().numpy() * scale
            scores = output["scores"].cpu().numpy()
            expected = base_records[index]["pred"]
            expected_boxes = np.asarray(expected["boxes"], dtype=np.float64).reshape(-1, 4)
            expected_scores = np.asarray(expected["scores"], dtype=np.float64)
            if boxes.shape != expected_boxes.shape or scores.shape != expected_scores.shape:
                raise RuntimeError(f"Unflipped output shape differs at image {index}")
            difference = max(float(np.max(np.abs(boxes - expected_boxes), initial=0.0)),
                             float(np.max(np.abs(scores - expected_scores), initial=0.0)))
            if difference > 1e-4:
                raise RuntimeError(f"Unflipped reproduction differs at image {index}: {difference}")
            differences.append(difference)
    return differences


def predict(args):
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    dataset = ODInputDataset(PAppleDataset(ROOT / "assets/papple", "val", 32),
                             short_sides=(args.short_side,), max_size=args.max_size)
    base_records = load_records(args.base_predictions)
    if len(base_records) != len(dataset):
        raise ValueError("Base predictions and dataset differ in length")
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    reproduction = verify_unflipped(model, dataset, base_records, device, args.verify_base)
    print(f"unflipped reproduction passed for {len(reproduction)} images; max diff "
          f"{max(reproduction, default=0.0):.3g}", flush=True)
    records = []
    started = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.json")
    with torch.inference_mode():
        for index in range(count):
            item = dataset[index]
            base = base_records[index]
            source_name = Path(item["target"]["source"]["rgb_path"]).name
            if source_name != base["image_id"]:
                raise RuntimeError(f"Image identity differs: {source_name} != {base['image_id']}")
            pixels = item["pixel_values"]
            if args.axis == "h":
                pixels = torch.flip(pixels, dims=(-1,))
            elif args.axis == "v":
                pixels = torch.flip(pixels, dims=(-2,))
            else:
                pixels = torch.flip(pixels, dims=(-2, -1))
            output = model(pixels.unsqueeze(0).to(device))[0]
            original_h, original_w = item["target"]["original_hw"]
            input_h, input_w = output["image_hw"]
            boxes = flip_boxes(output["boxes"].cpu().numpy(), (input_h, input_w), args.axis)
            boxes *= np.asarray([original_w / input_w, original_h / input_h] * 2)
            scores = output["scores"].cpu().numpy()
            records.append({"image_id": base["image_id"], "gt": base["gt"],
                            "pred": {"boxes": boxes.tolist(), "scores": scores.tolist()}})
            if (index + 1) % args.progress_every == 0 or index + 1 == count:
                partial.write_text(json.dumps(records), encoding="utf-8")
                print(f"FLIP {index + 1}/{count}; predictions={sum(len(r['pred']['scores']) for r in records)}; "
                      f"elapsed={(time.time() - started) / 60:.1f} min", flush=True)
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    metadata = {"schema": "apple_flip_tta_predictions_v1", "complete_full_val": count == len(dataset) == 814,
                "images": count, "axis": args.axis, "short_side": args.short_side, "max_size": args.max_size,
                "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
                "base_predictions": str(Path(args.base_predictions).resolve()),
                "base_predictions_sha256": sha256(args.base_predictions),
                "unflipped_reproduction_images": len(reproduction),
                "unflipped_reproduction_max_abs_diff": max(reproduction, default=0.0),
                "gt_used_for_inference": False}
    out.with_name("prediction_metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print("wrote", out, flush=True)


def fuse_command(args):
    layers = [load_records(path) for path in args.predictions]
    report = {"schema": "apple_flip_tta_fusion_v1", "images": len(layers[0]),
              "gt": sum(len(record["gt"]["boxes"]) for record in layers[0]),
              "prediction_files": [str(Path(path).resolve()) for path in args.predictions],
              "individual": [evaluate(layer) for layer in layers], "arms": []}
    best = None
    weight_sets = args.view_weight_sets or [None]
    for view_weights in weight_sets:
        for coverage_floor in args.coverage_floors:
            for score_mode in args.score_modes:
                for box_mode in args.box_modes:
                    for iou in args.ious:
                        fused = fuse_records(layers, iou, score_mode, box_mode, coverage_floor,
                                             view_weights, args.consensus_groups,
                                             args.consensus_bonus, args.isolation_penalty)
                        row = {"iou": iou, "score_mode": score_mode, "box_mode": box_mode,
                               "coverage_floor": coverage_floor,
                               "view_weights": view_weights,
                               "consensus_bonus": args.consensus_bonus,
                               "isolation_penalty": args.isolation_penalty,
                               "candidates": sum(len(record["pred"]["scores"]) for record in fused),
                               **evaluate(fused)}
                        report["arms"].append(row)
                        print(f"w={view_weights} cov={coverage_floor:.2f} {score_mode:4s} "
                              f"{box_mode:14s} IoU={iou:.2f} "
                              f"thr={row['threshold']:.4f} P={row['precision'] * 100:.2f} "
                              f"R={row['recall'] * 100:.2f} TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}",
                              flush=True)
                        key = (row["min_precision_recall"], row["precision"] + row["recall"],
                               row["threshold"])
                        if best is None or key > best[0]:
                            best = (key, row, fused)
    report["best"] = best[1]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.write_best_predictions:
        best_path = out.with_name(out.stem + "_raw_predictions.json")
        best_path.write_text(json.dumps(best[2], indent=1), encoding="utf-8")
        print("wrote", best_path)
    print("BEST", json.dumps(best[1]), "\nwrote", out, flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pred = subparsers.add_parser("predict")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--base-predictions", required=True)
    pred.add_argument("--short-side", type=int, default=1024)
    pred.add_argument("--max-size", type=int, default=1280)
    pred.add_argument("--axis", choices=("h", "v", "hv"), default="h")
    pred.add_argument("--device", default="cuda")
    pred.add_argument("--limit", type=int)
    pred.add_argument("--verify-base", type=int, default=2)
    pred.add_argument("--progress-every", type=int, default=25)
    pred.add_argument("--out", required=True)
    pred.set_defaults(function=predict)
    fuse = subparsers.add_parser("fuse")
    fuse.add_argument("--predictions", nargs="+", required=True)
    fuse.add_argument("--ious", type=float, nargs="+", default=[0.45, 0.55, 0.65])
    fuse.add_argument("--score-modes", nargs="+", default=["max", "mean"])
    fuse.add_argument("--box-modes", nargs="+", default=["score_weighted", "representative"])
    fuse.add_argument("--coverage-floors", type=float, nargs="+", default=[1.0])
    fuse.add_argument("--view-weight-sets", type=float, nargs="+", action="append",
                      help="Each occurrence adds one view-weight vector, e.g. --view-weight-sets 0.6 0.4")
    fuse.add_argument("--consensus-groups", type=int, nargs="+", default=None,
                      help="One architecture id per view; two ids mean cross-backbone fusion")
    fuse.add_argument("--consensus-bonus", type=float, default=1.0)
    fuse.add_argument("--isolation-penalty", type=float, default=1.0)
    fuse.add_argument("--write-best-predictions", action="store_true")
    fuse.add_argument("--out", required=True)
    fuse.set_defaults(function=fuse_command)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
