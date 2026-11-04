"""Measure what a ConvNeXt-Small backbone swap can actually inherit.

The proposal asks for a hot swap that keeps RPN and every ROI head, on the
argument that only the backbone body changes. This probe builds the replacement
detector without touching the audited builder, then reports:

1. how many of the parent's tensors transfer by name and shape, and
2. the real per-step training cost at the proven 1024 evaluation scale.

No training happens here and no checkpoint is written.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torchvision.models.detection import MaskRCNN  # noqa: E402
from torchvision.models.detection.backbone_utils import BackboneWithFPN  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: E402
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool  # noqa: E402
from torchvision.ops.misc import FrozenBatchNorm2d  # noqa: E402
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: E402
from torchvision.models import ConvNeXt_Small_Weights, convnext_small  # noqa: E402

from apple_seg.od_model import ODAmodalDetector  # noqa: E402


def build_convnext_fpn():
    body = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).features
    return BackboneWithFPN(
        body,
        return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    constructor = dict(payload["constructor_config"])
    options = {key: constructor[key] for key in (
        "mask_roi_size", "mask_hidden_channels", "roi_chunk_size", "mask_projection_chunk_size",
        "score_thresh", "detections_per_img", "pre_mask_nms_topk", "checkpoint_roi",
        "occlusion_weight", "focal_gamma", "use_visible_gate", "use_amodal_refinement",
        "final_nms", "hidden_tower_mode", "zero_initialize_hidden_predictor",
        "unknown_amodal_classifier_policy", "occlusion_verifier")}

    source = MaskRCNN(build_convnext_fpn(), num_classes=2,
                      rpn_pre_nms_top_n_train=constructor["rpn_pre_nms_top_n_train"],
                      rpn_pre_nms_top_n_test=constructor["rpn_pre_nms_top_n_test"],
                      rpn_post_nms_top_n_train=constructor["rpn_post_nms_top_n_train"],
                      rpn_post_nms_top_n_test=constructor["rpn_post_nms_top_n_test"],
                      box_batch_size_per_image=constructor["box_batch_size_per_image"],
                      box_positive_fraction=constructor["box_positive_fraction"])
    source.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((int(size),) for size in constructor["anchor_sizes"]),
        aspect_ratios=((.5, 1., 2.),) * 5)
    model = ODAmodalDetector(source, **options)

    old = payload["model"]
    new = model.state_dict()
    shared = {key for key in old if key in new and old[key].shape == new[key].shape}
    backbone_shared = {key for key in shared if key.startswith("backbone.")}
    report = {
        "parent_tensors": len(old),
        "new_tensors": len(new),
        "shared_name_and_shape": len(shared),
        "shared_outside_backbone": len(shared - backbone_shared),
        "shared_backbone": len(backbone_shared),
        "unloaded_parent_tensors": sorted(set(old) - shared)[:20],
        "uninitialized_new_tensors": sorted(set(new) - shared)[:20],
        "new_tensor_count": len(set(new) - shared),
    }
    outside_parent = set(old) - {key for key in old if key.startswith("backbone.")}
    outside_new = set(new) - {key for key in new if key.startswith("backbone.")}
    report["non_backbone_parent"] = len(outside_parent)
    report["non_backbone_transferred"] = len(shared & outside_parent)
    report["non_backbone_missing"] = sorted(outside_parent - shared)

    if torch.cuda.is_available() and args.steps > 0:
        model = model.cuda().train()
        images = torch.rand(1, 3, 1024, 1280, device="cuda")
        import numpy as np
        mask = np.zeros((1, 40, 80), dtype=np.uint8)
        mask[0, 15:23, 31:42] = 1
        targets = {"boxes": torch.tensor([[100., 120., 340., 380.]]),
                   "labels": torch.zeros(1, dtype=torch.int64),
                   "instance_ids": [0], "original_hw": [40, 80],
                   "visible_masks": torch.from_numpy(mask), "amodal_masks": torch.from_numpy(mask),
                   "source": {"dataset": "PApple"},
                   "annotation_validity": {"visible": True, "amodal": True}}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
        timings = []
        for step in range(args.steps):
            if step == 1:
                started = time.time()
            output = model(images, targets=[targets])
            losses = output[1] if isinstance(output, tuple) else output
            loss = sum(value for value in losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step:
                timings.append(time.time() - started)
                started = time.time()
        report["steps"] = args.steps
        report["seconds_per_step"] = [round(value, 3) for value in timings]
        report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        report["parameters_millions"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("parent_tensors", "shared_name_and_shape", "non_backbone_parent",
                       "non_backbone_transferred", "new_tensor_count")}, ensure_ascii=False))
    print("missing non-backbone:", report["non_backbone_missing"][:6])
    if "seconds_per_step" in report:
        print("seconds/step", report["seconds_per_step"], "peak", report["peak_gpu_gb"], "GB",
              "params", report["parameters_millions"], "M")


if __name__ == "__main__":
    main()
