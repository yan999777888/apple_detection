"""Measure the frozen apple maturity detector at its deployment threshold on labeled fruit."""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import box_iou

from class0_patch_refinement import CP, ROOT, atomic_json, configure, load_image, sha
from deployment_adapter_v1 import FrozenDeploymentAdapter
from joint_model_v1 import build_model


OUT = ROOT / "experiments/grade_stage_validation_v1"
DATA = ROOT / "apple_research_v2/datasets/mendeley_stage_seg_v1"
NAMES = ["immature", "semi-mature", "mature"]
MODEL_SHA = "f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"


def references(label_file, shape):
    h, w = shape[:2]
    boxes, stages = [], []
    for row in label_file.read_text(encoding="utf8").splitlines():
        if not row.strip():
            continue
        cls, *coords = row.split()
        vertices = np.asarray(coords, dtype=np.float32).reshape(-1, 2) * [w, h]
        if len(vertices) < 3 or int(cls) not in range(3):
            raise ValueError(f"Invalid reference polygon: {label_file}")
        boxes.append([*vertices.min(axis=0), *vertices.max(axis=0)])
        stages.append(int(cls))
    return torch.as_tensor(np.asarray(boxes, np.float32).reshape(-1, 4)), stages


@torch.inference_mode()
def measure():
    if sha(CP) != MODEL_SHA:
        raise ValueError("Unexpected maturity checkpoint")
    model = build_model("separate", device="cpu")
    model.load_state_dict(torch.load(CP, map_location="cpu", weights_only=False)["model"])
    worker = FrozenDeploymentAdapter(model.eval().requires_grad_(False), imgsz=640, conf=.25)
    images = sorted(p for p in (DATA / "images/val").iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    matrix = np.zeros((3, 4), dtype=int)
    unmatched = np.zeros(3, dtype=int)
    for path in images:
        image = load_image(path)
        truth, labels = references(DATA / "labels/val" / (path.stem + ".txt"), image.shape)
        pred = worker.predict(image, tasks=("stage",))["stage"][0].boxes
        overlaps = box_iou(truth, pred.xyxy).numpy()
        pairings = {}
        if overlaps.size:
            # Lexicographic objective: maximize count at IoU .50, then summed IoU.
            valid = overlaps >= .50
            weight = valid * (min(overlaps.shape) + 1 + overlaps)
            rows, cols = linear_sum_assignment(weight, maximize=True)
            pairings = {int(i): int(j) for i, j in zip(rows, cols) if valid[i, j]}
        classes = pred.cls.int().tolist()
        for i, label in enumerate(labels):
            matrix[label, classes[pairings[i]] if i in pairings else 3] += 1
        assigned = set(pairings.values())
        for j, cls in enumerate(classes):
            if j not in assigned:
                unmatched[cls] += 1
    if len(images) != 72 or matrix.sum() != 114:
        raise ValueError("Unexpected source development split")
    summary = dict(source="Mendeley apple stage original development", backend="CPU FP32",
                   checkpoint_sha256=MODEL_SHA, threshold=.25, box_iou=.50, images=len(images),
                   gt_instances=int(matrix.sum()), truth_rows=NAMES, prediction_columns=NAMES+["undetected"],
                   confusion=matrix.tolist(), unmatched_predictions_by_stage=unmatched.tolist(),
                   matched_correct=int(np.trace(matrix[:, :3])), matched_any_stage=int(matrix[:, :3].sum()),
                   meaning="Appearance-stage ground truth. Matching uses bounding boxes from native polygons. "
                           "This is not a physiological maturity or AFruitDB joint-grade validation.")
    atomic_json(OUT / "results.json", summary)
    print(summary)


if __name__ == "__main__":
    configure(OUT, "cpu")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    measure()
