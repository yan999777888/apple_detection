"""Final-layer direct amodal-box decoding for the residual geometry trial.

Standard-library-only import. Mask binarization delegates to the exact frozen
dual-query helper. Valid decoded boxes survive even when their A mask is empty;
independent mask metrics will still count those unmatched masks as FP.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_POST = ROOT / "temp/query_dual_mask_candidate_20261021/query_dual_postprocess.py"
OLD_POST_SHA256 = "d19cedb68be01e68957acff0026b3c50a6c6e4d7fe2bf176b317e37057cfbf9d"


def _old_post():
    source = OLD_POST.read_bytes()
    if hashlib.sha256(source).hexdigest() != OLD_POST_SHA256:
        raise ValueError("Frozen binary-mask postprocessor changed")
    module = ModuleType("_residual_frozen_binary_mask_postprocessor")
    module.__file__ = str(OLD_POST)
    exec(compile(source, str(OLD_POST), "exec"), module.__dict__)
    return module


_OLD = _old_post()
_image_size = _OLD._image_size
_binary_chunk = _OLD._binary_chunk
RAW_SCORE_THRESHOLD = .001
MAX_CANDIDATES = 100
MASK_CHUNK_SIZE = 8
POSTPROCESS_POLICY = {
    "schema": "residual_amodal_box_direct_postprocess_v1",
    "class_mapping": {"apple_logit": 0, "null_logit": 1, "output_apple_label": 1},
    "score": "float32 softmax final class logits apple probability only; no score calibration",
    "raw_score_filter": {"operator": ">", "threshold": RAW_SCORE_THRESHOLD, "comparison_dtype": "float32"},
    "order": "descending score; exact ties retain original query index order",
    "max_candidates_before_invalid_box_drop": MAX_CANDIDATES,
    "box_source": "final amodal_boxes normalized cxcywh; independent of thresholded mask support",
    "box_format": "continuous original-image XYXY; x=cx*W,y=cy*H,w=w*W,h=h*H; clip edges to[0,W]x[0,H]",
    "invalid_geometry": "drop only nonpositive clipped width/height; nonfinite logits or box values fail",
    "empty_amodal": "retain valid box; its empty mask participates as unmatched FP in independent mask metrics",
    "empty_visible": "retain valid box and empty visible mask",
    "mask_transform": _OLD.POSTPROCESS_POLICY["mask_transform"],
    "mask_helper_source_sha256": OLD_POST_SHA256,
    "align_corners": False, "mask_chunk_size": MASK_CHUNK_SIZE,
    "spatial_contract": "unpadded full image; no crop, flip or letterbox inversion",
    "cross_branch_projection": "none; visible is not intersected with amodal",
    "instance_fusion": "none; no NMS, argmax panoptic map, or exclusive pixel assignment",
    "auxiliary_outputs": "penultimate training auxiliary is ignored at inference; final outputs only",
    "ground_truth_used": False, "accuracy_or_ap_claimed": False,
    "output_device": "cpu", "output_mask_dtype": "bool",
}


def postprocess_single_frame(outputs, original_size, *, return_masks=False):
    original_size = _image_size(original_size)
    if not isinstance(outputs, Mapping) or not isinstance(return_masks, bool):
        raise TypeError("Require output mapping and boolean return_masks")
    import torch
    from torch.nn.functional import interpolate
    classes = outputs.get("class_queries_logits")
    visible = outputs.get("visible_masks_queries_logits")
    amodal = outputs.get("amodal_masks_queries_logits")
    boxes_normalized = outputs.get("amodal_boxes")
    for name, value in (("classes", classes), ("visible", visible), ("amodal", amodal), ("boxes", boxes_normalized)):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("Floating final output tensor required: " + name)
    if tuple(classes.shape) != (1, 100, 2) or tuple(boxes_normalized.shape) != (1, 100, 4):
        raise ValueError("Final class[1,100,2] and normalized cxcywh box[1,100,4] contract required")
    for value in (visible, amodal):
        if value.ndim != 4 or tuple(value.shape[:2]) != (1, 100) or min(value.shape[-2:]) < 1:
            raise ValueError("Each final mask branch must be[1,100,h,w]")
    with torch.inference_mode():
        logits = classes.detach().float()
        geometry = boxes_normalized.detach().to(device="cpu", dtype=torch.float32)[0]
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(geometry).all()):
            raise ValueError("Final class logits and amodal boxes must be finite")
        scores = logits.softmax(-1)[0, :, 0].cpu()
        raw = torch.nonzero(scores > RAW_SCORE_THRESHOLD).flatten().tolist()
        ranked = sorted(raw, key=lambda q: -float(scores[q]))[:MAX_CANDIDATES]
        h, w = original_size
        center, extent = geometry[:, :2], geometry[:, 2:]
        xyxy = torch.cat((center - extent / 2, center + extent / 2), dim=1)
        xyxy *= torch.tensor([w, h, w, h], dtype=torch.float32)
        xyxy[:, 0::2].clamp_(0, w); xyxy[:, 1::2].clamp_(0, h)
        kept, invalid = [], []
        for q in ranked:
            if bool((extent[q] > 0).all() & (xyxy[q, 2:] > xyxy[q, :2]).all()):
                kept.append(q)
            else:
                invalid.append({"query_index": q, "score": float(scores[q]), "reason": "nonpositive_clipped_box_extent"})
        empty_a, empty_v = [], []
        a_buffer = v_buffer = None
        if return_masks:
            a_buffer = torch.empty((len(kept), *original_size), dtype=torch.bool, device="cpu")
            v_buffer = torch.empty_like(a_buffer)
        for start in range(0, len(kept), MASK_CHUNK_SIZE):
            chunk = kept[start:start + MASK_CHUNK_SIZE]
            a = _binary_chunk(amodal, chunk, original_size, torch, interpolate)
            v = _binary_chunk(visible, chunk, original_size, torch, interpolate)
            empty_a.extend(q for q, valid in zip(chunk, a.flatten(1).any(1).tolist()) if not valid)
            empty_v.extend(q for q, valid in zip(chunk, v.flatten(1).any(1).tolist()) if not valid)
            if return_masks:
                a_buffer[start:start + len(chunk)].copy_(a)
                v_buffer[start:start + len(chunk)].copy_(v)
            del a, v
        result = {"boxes": xyxy[kept].clone().reshape(-1, 4), "scores": scores[kept].clone(),
            "labels": torch.ones(len(kept), dtype=torch.int64, device="cpu"),
            "query_indices": torch.tensor(kept, dtype=torch.int64, device="cpu"),
            "metadata": {"policy": deepcopy(POSTPROCESS_POLICY), "original_size_hw": list(original_size),
                "branch_logit_size_hw": {"visible": list(visible.shape[-2:]), "amodal": list(amodal.shape[-2:])},
                "raw_score_pass_count": len(raw), "capped_candidate_count": len(ranked), "kept_count": len(kept),
                "invalid_candidates": invalid, "empty_amodal_query_indices": empty_a,
                "empty_visible_query_indices": empty_v, "returned_masks": return_masks}}
        if return_masks:
            result.update(visible_masks=v_buffer, amodal_masks=a_buffer, masks=a_buffer)
        return result
