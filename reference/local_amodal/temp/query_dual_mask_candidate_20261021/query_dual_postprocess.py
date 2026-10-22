"""Bounded, single-frame postprocessing for the source-only query candidate.

Import is standard-library-only. Torch is imported only when the function is
called. This module runs no model, reads no annotations and computes no metric.
The caller must supply logits from an unpadded whole image: a crop, letterbox or
nonzero padding cannot be inferred or corrected from these tensors alone.
"""
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral


RAW_SCORE_THRESHOLD = 0.001
MAX_CANDIDATES = 100
MASK_CHUNK_SIZE = 8

POSTPROCESS_POLICY = {
    "schema": "query_dual_postprocess_v1",
    "source_wrapper_sha256": "cef59347445af673eceff65cd5d0ed7caa5769683c5e78eb7f4610deb2e48878",
    "class_mapping": {"apple_logit": 0, "null_logit": 1, "output_apple_label": 1},
    "score": "float32 softmax(class_queries_logits, dim=-1)[0, query, 0] only",
    "raw_score_filter": {"operator": ">", "threshold": RAW_SCORE_THRESHOLD,
                         "comparison_dtype": "float32"},
    "order": "descending score; exact ties retain original query index order",
    "max_candidates_before_empty_amodal_drop": MAX_CANDIDATES,
    "mask_transform": "each branch independently: float32 logits -> bilinear resize to original H,W -> sigmoid >= 0.5",
    "align_corners": False,
    "spatial_contract": "unpadded full image; no crop, flip or letterbox inversion",
    "padding_verification": "caller contract; logits cannot prove absence of padding",
    "box_format": "XYXY continuous pixel extents: xmin,ymin,xmax+1,ymax+1 of binary amodal mask",
    "box_convention_source": "scripts/dataset.py:_numpy_target; scripts/metrics.py",
    "empty_amodal": "record query index and score; drop candidate because no box exists",
    "empty_visible": "retain candidate with an empty visible mask",
    "cross_branch_projection": "none; visible is not intersected with amodal",
    "instance_fusion": "none; no NMS, argmax panoptic map, or exclusive pixel assignment",
    "overlapping_amodal_instances": "retained independently",
    "mask_chunk_size": MASK_CHUNK_SIZE,
    "output_device": "cpu",
    "output_mask_dtype": "bool",
    "ground_truth_used": False,
    "accuracy_or_ap_claimed": False,
}


def _image_size(original_size):
    if not isinstance(original_size, (tuple, list)) or len(original_size) != 2:
        raise ValueError("original_size must be (height, width)")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
           for v in original_size):
        raise ValueError("original_size must contain two positive integers")
    return tuple(int(v) for v in original_size)


def _binary_chunk(logits, query_indices, original_size, torch, interpolate):
    """Only this <=8-query branch holds a full-resolution float tensor."""
    indices = torch.tensor(query_indices, dtype=torch.int64, device=logits.device)
    low_resolution = logits.detach().index_select(1, indices).to(dtype=torch.float32)
    if not bool(torch.isfinite(low_resolution).all()):
        raise ValueError("Selected mask logits must remain finite in float32")
    resized = interpolate(low_resolution, size=original_size,
                          mode="bilinear", align_corners=False)
    del low_resolution
    # The float buffer is local and discarded; the returned buffer is bool CPU.
    return resized.sigmoid_().ge(0.5)[0].to(device="cpu")


def postprocess_single_frame(outputs, original_size, *, return_masks=False):
    """Return CPU tensors and JSON-ready policy/diagnostics for one frame.

    Required keys match query_dual_mask.py: class_queries_logits [1,100,2],
    visible_masks_queries_logits and amodal_masks_queries_logits [1,100,h,w].
    Each branch can have its own positive h,w. original_size is (H,W) before
    the unpadded whole-image resize. Logits must be floating tensors; the
    selected masks and all class logits must be finite after float32 conversion.

    boxes [N,4] and scores [N] are float32; labels and query_indices [N] are
    int64. Model apple class 0 is explicitly mapped to PAppleDataset label 1.
    With return_masks=True, visible_masks and amodal_masks are CPU bool[N,H,W].
    The masks alias names the same amodal tensor for the existing metrics schema.
    Without returned masks, only <=8 original-size bool masks per branch are
    retained at a time. Optional returned mask buffers use at most 100 bool
    images per branch, allocated on CPU; no 100-query float resize is created.
    """
    original_size = _image_size(original_size)
    if not isinstance(return_masks, bool):
        raise TypeError("return_masks must be bool")
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be the candidate output mapping")

    import torch
    from torch.nn.functional import interpolate

    keys = ("class_queries_logits", "visible_masks_queries_logits",
            "amodal_masks_queries_logits")
    for key in keys:
        value = outputs.get(key)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{key} must be a floating Torch tensor")
        if key == keys[0]:
            if tuple(value.shape) != (1, 100, 2):
                raise ValueError("class_queries_logits must have shape [1,100,2]")
        elif (value.ndim != 4 or tuple(value.shape[:2]) != (1, 100)
              or min(value.shape[-2:]) <= 0):
            raise ValueError(f"{key} must have shape [1,100,h,w] with positive h,w")

    classes, visible, amodal = (outputs[key] for key in keys)
    with torch.inference_mode():
        classes_float = classes.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(classes_float).all()):
            raise ValueError("Class logits must remain finite in float32")
        scores_cpu = classes_float.softmax(dim=-1)[0, :, 0].to(device="cpu")
        score_values = scores_cpu.tolist()
        raw_indices = torch.nonzero(scores_cpu > RAW_SCORE_THRESHOLD).flatten().tolist()
        # nonzero gives ascending query indices; Python sorting is stable.
        ranked_indices = sorted(raw_indices, key=lambda q: -score_values[q])
        candidate_indices = ranked_indices[:MAX_CANDIDATES]
        boxes, kept_indices, invalid_candidates, empty_visible_indices = [], [], [], []
        visible_buffer = amodal_buffer = None

        for start in range(0, len(candidate_indices), MASK_CHUNK_SIZE):
            chunk = candidate_indices[start:start + MASK_CHUNK_SIZE]
            amodal_chunk = _binary_chunk(amodal, chunk, original_size, torch, interpolate)
            valid_flags = amodal_chunk.flatten(1).any(dim=1).tolist()
            valid_local = [i for i, valid in enumerate(valid_flags) if valid]
            for i, valid in enumerate(valid_flags):
                if not valid:
                    invalid_candidates.append({"query_index": chunk[i],
                                               "score": score_values[chunk[i]],
                                               "reason": "empty_amodal_mask"})
            if not valid_local:
                del amodal_chunk
                continue
            valid_queries = [chunk[i] for i in valid_local]
            visible_chunk = _binary_chunk(visible, valid_queries, original_size,
                                          torch, interpolate)
            if return_masks and visible_buffer is None:
                capacity = (len(candidate_indices), *original_size)
                visible_buffer = torch.empty(capacity, dtype=torch.bool, device="cpu")
                amodal_buffer = torch.empty_like(visible_buffer)
            for v_index, a_index in enumerate(valid_local):
                query_index = chunk[a_index]
                a_mask, v_mask = amodal_chunk[a_index], visible_chunk[v_index]
                xs = torch.nonzero(a_mask.any(dim=0)).flatten()
                ys = torch.nonzero(a_mask.any(dim=1)).flatten()
                boxes.append([int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1])
                if not bool(v_mask.any()):
                    empty_visible_indices.append(query_index)
                if return_masks:
                    visible_buffer[len(kept_indices)].copy_(v_mask)
                    amodal_buffer[len(kept_indices)].copy_(a_mask)
                kept_indices.append(query_index)
            del a_mask, v_mask, xs, ys, amodal_chunk, visible_chunk

        result = {
            "boxes": torch.tensor(boxes, dtype=torch.float32, device="cpu").reshape(-1, 4),
            "scores": scores_cpu[kept_indices].clone(),
            "labels": torch.ones(len(kept_indices), dtype=torch.int64, device="cpu"),
            "query_indices": torch.tensor(kept_indices, dtype=torch.int64, device="cpu"),
            "metadata": {"policy": deepcopy(POSTPROCESS_POLICY),
                         "original_size_hw": list(original_size),
                         "branch_logit_size_hw": {"visible": list(visible.shape[-2:]),
                                                  "amodal": list(amodal.shape[-2:])},
                         "raw_score_pass_count": len(raw_indices),
                         "capped_candidate_count": len(candidate_indices),
                         "kept_count": len(kept_indices),
                         "invalid_candidates": invalid_candidates,
                         "empty_visible_query_indices": empty_visible_indices,
                         "returned_masks": return_masks},
        }
        if return_masks:
            if visible_buffer is None:
                visible_buffer = torch.empty((0, *original_size), dtype=torch.bool, device="cpu")
                amodal_buffer = torch.empty_like(visible_buffer)
            result["visible_masks"] = visible_buffer[:len(kept_indices)]
            result["amodal_masks"] = amodal_buffer[:len(kept_indices)]
            result["masks"] = result["amodal_masks"]
        return result
