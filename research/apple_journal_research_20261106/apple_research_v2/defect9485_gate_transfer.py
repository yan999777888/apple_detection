"""Transfer the two training-fold pit veto policies to fixed new detector outputs."""

from defect9485_pit_gate import (
    OUT,
    ROOT,
    manifest,
    configure,
    read,
    prepare,
    atomic_json,
    metrics,
    sha,
)
import numpy as np
from class0_patch_refinement import load_image
import joblib
import cv2

SOURCES = {
    "small_flip": "defect9485_small_flip_v1/fixed_predictions.json",
    "small_cross": "defect9485_cross/yolo26s/one_fixed_plus_other_support_and_frozen_set_predictions.json",
    "small_flip_cross": "defect9485_cross/yolo26s_flip/one_fixed_plus_other_support_and_frozen_set_predictions.json",
}


def apply(im, boxes, model, limit):
    b = np.asarray(boxes, np.float32).reshape(-1, 6)
    ids, x = prepare(im, b)
    keep = np.ones(len(b), bool)
    if len(ids):
        keep[ids] = model.predict_proba(x)[:, 1] >= limit
    return b[keep]


if __name__ == "__main__":
    configure(OUT, "cpu")
    cv2.setNumThreads(1)
    cfg = read(OUT / "frozen.json")
    path = OUT / "pit.joblib"
    assert sha(path) == cfg["sha256"]
    model = joblib.load(path)
    lookup = manifest()
    result = {}
    for name, src in SOURCES.items():
        rows = read(ROOT / "experiments" / src)
        arms = {k: [] for k in cfg["limits"]}
        for r in rows:
            sid = r["image_id"]
            assert lookup[sid]["partition"] == "dev"
            b = np.asarray(r["pred"], np.float32).reshape(-1, 6)
            ids, x = prepare(load_image(lookup[sid]["rgb_path"]), b)
            q = np.ones(len(b))
            if len(ids):
                q[ids] = model.predict_proba(x)[:, 1]
            for k, v in cfg["limits"].items():
                arms[k].append(dict(image_id=sid, GT=r["GT"], pred=b[q >= v].tolist()))
        for k, rr in arms.items():
            key = name + "_" + k.replace(".", "")
            atomic_json(OUT / f"{key}_predictions.json", rr)
            result[key] = metrics(rr)
    atomic_json(OUT / "transfer_metrics.json", result)
    print({k: v["total"] for k, v in result.items()})
