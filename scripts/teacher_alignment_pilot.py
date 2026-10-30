"""Bounded TRAIN-only teacher/GT alignment diagnostic; never trains or pseudo-labels.

Run base, small and query as separate sequential processes. The four diagnostic
images are deliberately selected for empty/dense/ordinary coverage, not a test
set and not evidence of overall model accuracy. ROI mask IoU is conditional on
one-to-one A-box matching; unmatched GT and candidates are reported separately.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

REPO = Path(__file__).absolute().parents[1]
WORKSPACE = REPO.parent
OLD = WORKSPACE / "Apple_DualBranch_3080_20261022"
PACKAGE = WORKSPACE / "含权重/苹果检测成果包_20261029"
RUNTIME = PACKAGE / "08_本机接力/runtime"
INDICES = (63, 475, 0, 2060)
SHA = {
    "base": "0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b",
    "small": "e0c1f659c2509d7a01d617e12cb83c673768b440ea5b4b96cb4f763e4207e7e3",
    "query": "aac6f257bf43af3b945dd3469ba66214de99d7ae878e13e85ec7bef0e20b3227",
}
METRICS_SHA = "1b0a7e385940fd0c6160cc08b75ddaf2e1f43f3ea00f5bd37d547fea6b08010d"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def inverse_hflip_boxes(boxes, width):
    result = boxes.clone()
    result[:, 0] = width - boxes[:, 2]
    result[:, 2] = width - boxes[:, 0]
    return result


def gt_boxes_from_masks(masks, torch):
    result = []
    for mask in masks:
        ys, xs = torch.where(mask.bool())
        if not len(xs):
            raise ValueError("Original GT must not contain an empty A mask")
        result.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return torch.tensor(result, dtype=torch.float32).reshape(-1, 4)


def crop_mask_iou(prediction, window, truth):
    left, top, right, bottom = window
    truth = truth.bool()
    intersection = int((prediction & truth[top:bottom, left:right]).sum())
    union = int(prediction.sum()) + int(truth.sum()) - intersection
    return intersection / union if union else 0.0


def full_mask_iou(prediction, truth):
    truth = truth.bool()
    intersection = int((prediction & truth).sum())
    union = int(prediction.sum()) + int(truth.sum()) - intersection
    return intersection / union if union else 0.0


def report_matches(boxes, scores, target, metrics, confidence, mask_iou):
    import torch
    gt_boxes = gt_boxes_from_masks(target["amodal_masks"], torch)
    matched = metrics.match_image(gt_boxes.numpy(), boxes.numpy(), scores, .5, confidence)
    pairs, unmatched_predictions = [], []
    for pred_index, gt_index, overlap in zip(matched["pred_indices"], matched["gt_indices"], matched["matched_ious"], strict=True):
        pred_index, gt_index = int(pred_index), int(gt_index)
        if gt_index < 0:
            best = float(metrics.box_iou(boxes[pred_index:pred_index + 1].numpy(), gt_boxes.numpy()).max()) if len(gt_boxes) else None
            unmatched_predictions.append({"prediction_index": pred_index, "score": float(scores[pred_index]),
                                          "max_GT_box_iou": best, "annotation_action": "none; audit candidate only"})
            continue
        pair = {"prediction_index": pred_index, "gt_index": gt_index,
                "original_instance_id": target["instance_ids"][gt_index], "score": float(scores[pred_index]),
                "box_iou": float(overlap), "box_original_xyxy": boxes[pred_index].tolist()}
        pair.update(mask_iou(pred_index, gt_index))
        pairs.append(pair)
    return {"GT_count": len(gt_boxes), "raw_prediction_count": len(boxes),
            "selected_prediction_count": len(matched["pred_indices"]), "matched_GT_count": len(pairs),
            "unmatched_GT_instance_ids": [target["instance_ids"][int(g)] for g in matched["unmatched_gt_indices"]],
            "matched_pairs": pairs, "unmatched_selected_predictions": unmatched_predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "small", "query"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        raise ValueError("confidence must be a probability")
    output = args.output or REPO / "reports/teacher_alignment_pilot_20261030" / (args.model + ".json")
    output = output.absolute()
    if output.exists():
        raise FileExistsError("Use a fresh output path to retain earlier evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "apple_TRAIN_alignment_pilot_v1", "model": args.model,
              "started_utc": datetime.now(timezone.utc).isoformat(), "phase": "initializing",
              "scope": "Four deliberately chosen TRAIN images; conditional mask quality and matching diagnosis only",
              "overall_accuracy_claimed": False, "training_performed": False, "validation_or_test_used": False,
              "teacher_views_fused": False, "distillation_implemented": False, "pseudo_GT_created": False,
              "confidence_threshold": args.confidence, "box_match_iou_threshold": .5, "mask_threshold": .5,
              "sample_indices": list(INDICES), "device": args.device, "images": [],
              "formal_checkpoint_source_calibration_audit_passed": False}
    started = time.monotonic()
    try:
        import torch
        from PIL import Image
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261030)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        source = REPO if args.model == "query" else RUNTIME
        sys.path.insert(0, str(source / "src"))
        from apple_seg.data import PAppleDataset, _pixels
        dataset = PAppleDataset(OLD / "assets/papple", split="train", input_size=32, augment=False)
        assert len(dataset) == 2304 and dataset.split == "train"
        metrics_path = OLD / "reference/local_amodal/scripts/metrics.py"
        assert sha(metrics_path) == METRICS_SHA
        metrics = module(metrics_path, "_alignment_pinned_metrics")
        if args.model == "query":
            from apple_seg.model import DualMaskModel
            checkpoint = OLD / "weights/query_epoch006.pt"
            model, initialization = DualMaskModel.from_checkpoint(checkpoint, SHA[args.model])
            restore = initialization["strict_model_load"]
            post = module(OLD / "reference/local_amodal/temp/query_dual_mask_candidate_20261021/query_dual_postprocess.py", "_alignment_query_post")
            assert sha(OLD / "reference/local_amodal/temp/query_dual_mask_candidate_20261021/query_dual_postprocess.py") == "d19cedb68be01e68957acff0026b3c50a6c6e4d7fe2bf176b317e37057cfbf9d"
        else:
            from apple_seg.od_training import load_checkpoint
            from apple_seg.od_model import build_od_amodal_detector
            from apple_seg.od_inference import _strict_restore
            from apple_seg.od_data import legacy_rgb_tensor
            from apple_seg.od_evaluate import _validate_prediction
            from apple_seg.od_roi import paste_roi_mask
            folder = "convnext_base_c2_2304" if args.model == "base" else "convnext_small_joint2_2304"
            checkpoint = PACKAGE / "05_模型权重" / folder / "validated_002304.pt"
            assert sha(checkpoint) == SHA[args.model]
            payload = load_checkpoint(checkpoint)
            model = build_od_amodal_detector(weights=None, **payload["constructor_config"])
            restore = _strict_restore(model, payload["model"])
            assert model.constructor_config == payload["constructor_config"]
            del payload
            gc.collect()
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=SHA[args.model], strict_model_restore=restore,
                      source_model_sha256=sha(source / "src/apple_seg" / ("model.py" if args.model == "query" else "od_model.py")),
                      metrics_sha256=METRICS_SHA, train_annotation_sha256=dataset.annotation_sha256)
        model.requires_grad_(False).eval().to(device)
        report.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                      parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                      model_precision="float32; autocast disabled",
                      model_receives_GT=False, native_mask_rendering="one matched instance at a time")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        report["phase"] = "inference"
        for index in INDICES:
            target = dataset[index]["target"]
            assert target["source"]["split"] == "train" and target["source"]["dataset"] == "PApple"
            assert target["annotation_validity"]["amodal"] and target["amodal_masks"] is not None
            with Image.open(target["source"]["rgb_path"]) as opened:
                rgb = opened.convert("RGB")
            original_hw = (rgb.height, rgb.width)
            row = {"index": index, "image_id": target["source"]["image_id"],
                   "RGB_sha256": sha(Path(target["source"]["rgb_path"])), "original_hw": list(original_hw),
                   "GT_count": len(target["instance_ids"]), "views": []}
            views = ("original",) if args.model == "query" else ("original", "hflip")
            for view in views:
                pixels = _pixels(rgb, 768) if args.model == "query" else legacy_rgb_tensor(rgb, 1024, 1280)
                if view == "hflip":
                    pixels = pixels.flip(-1).contiguous()
                input_hw = tuple(pixels.shape[-2:])
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                    outputs = model(pixels[None].to(device), pixel_mask=torch.ones((1, *input_hw), dtype=torch.bool, device=device))
                if args.model == "query":
                    prediction = post.postprocess_single_frame(outputs, original_hw, return_masks=False)
                    boxes, scores = prediction["boxes"], prediction["scores"].tolist()
                    def get_quality(p, g):
                        query = int(prediction["query_indices"][p])
                        result = {"query_index": query}
                        for branch in ("visible", "amodal"):
                            binary = post._binary_chunk(outputs[branch + "_masks_queries_logits"], [query], original_hw,
                                                       torch, torch.nn.functional.interpolate)[0]
                            result[branch + "_mask_iou"] = full_mask_iou(binary, target[branch + "_masks"][g])
                            del binary
                        return result
                    inverse_check = None
                else:
                    prediction = _validate_prediction(outputs, input_hw, original_hw, mask_chunk_size=8)
                    boxes, scores = prediction["boxes"], prediction["scores"]
                    inverse_check = None
                    if view == "hflip":
                        native_flipped_boxes = boxes.clone()
                        boxes = inverse_hflip_boxes(boxes, rgb.width)
                        recovered = inverse_hflip_boxes(boxes, rgb.width)
                        error = float((recovered - native_flipped_boxes).abs().max()) if len(boxes) else 0.
                        assert error < .001
                        for branch in ("visible", "amodal"):
                            original_masks = prediction[branch]
                            prediction[branch] = original_masks.flip(-1).contiguous()
                            assert torch.equal(prediction[branch].flip(-1), original_masks)
                        inverse_check = {"boxes_double_hflip_max_abs_error_pixels": error,
                                         "ROI_masks_double_hflip_exact": True,
                                         "boxes_and_masks_restored_to_original_orientation": True}
                    def get_quality(p, g):
                        result = {}
                        for branch in ("visible", "amodal"):
                            crop, window = paste_roi_mask(prediction[branch][p], boxes[p], original_hw)
                            result[branch + "_mask_iou"] = crop_mask_iou(crop, window, target[branch + "_masks"][g])
                            del crop
                        return result
                evidence = report_matches(boxes, scores, target, metrics, args.confidence, get_quality)
                evidence.update(view=view, input_hw=list(input_hw), inverse_hflip_check=inverse_check,
                                mask_quality_scope="only one-to-one GT A-box matches; coverage and unmatched predictions retained")
                row["views"].append(evidence)
                print(json.dumps({"model": args.model, "image": row["image_id"], "view": view,
                                  "GT": evidence["GT_count"], "matched": evidence["matched_GT_count"],
                                  "selected_predictions": evidence["selected_prediction_count"]}), flush=True)
                del outputs, prediction, pixels, get_quality
            report["images"].append(row)
            del target, rgb
        aggregates = {}
        for row in report["images"]:
            for view in row["views"]:
                item = aggregates.setdefault(view["view"], {"GT": 0, "matched_GT": 0, "unmatched_selected_predictions": 0,
                                                           "visible_mask_ious": [], "amodal_mask_ious": []})
                item["GT"] += view["GT_count"]
                item["matched_GT"] += view["matched_GT_count"]
                item["unmatched_selected_predictions"] += len(view["unmatched_selected_predictions"])
                for pair in view["matched_pairs"]:
                    for branch in ("visible", "amodal"):
                        item[branch + "_mask_ious"].append(pair[branch + "_mask_iou"])
        for item in aggregates.values():
            for branch in ("visible", "amodal"):
                values = item.pop(branch + "_mask_ious")
                item[branch + "_mean_iou_on_box_matches"] = sum(values) / len(values) if values else None
                item[branch + "_mask_iou_ge_0_5_on_box_matches"] = sum(v >= .5 for v in values)
        report.update(phase="complete", per_view_conditional_diagnostics=aggregates,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
    except Exception as error:
        report.update(phase="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"phase": report["phase"], "report": str(output)}, ensure_ascii=False), flush=True)
    return 0 if report["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
