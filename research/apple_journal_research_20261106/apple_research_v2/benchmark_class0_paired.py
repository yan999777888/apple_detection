"""Measure incumbent and fused FP32 crop verifier on identical alternating frames."""

import time
from class0_crop_deploy import (
    OUT,
    CP,
    Inference,
    ROOT,
    configure,
    atomic_json,
    load_image,
    read,
)


def main():
    configure(OUT, "cuda:0")
    import cv2
    import numpy as np
    import torch
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from defect_four_orientations_20261118 import make_worker
    from defect_coordinate_vote_20261118 import predict_vote
    from audit_defect_causes_20261112 import match, ious
    from scipy.optimize import linear_sum_assignment

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    old = make_worker(CP, False)
    fused = Inference(False)
    models = {
        "coordinate_vote": lambda im: predict_vote(old, im),
        "crop_expert_fp32": fused,
    }
    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims.extend(
            load_image(paths[round(i * (len(paths) - 1) / 19)]) for i in range(20)
        )
    times = {name: [] for name in models}
    for im in ims[:5]:
        for model in models.values():
            model(im)
    names = list(models)
    for rep in range(15):
        for i, im in enumerate(ims):
            for name in names if (rep + i) % 2 == 0 else names[::-1]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                models[name](im)
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - start) * 1000)
    # Confirm that the numerical changes preserve the physical locations of FPs.
    new = read(OUT / "fp32/dev_predictions.json")
    before = read(ROOT / "experiments/class0_crop_expert_v1/dev_predictions.json")
    assert len(new) == len(before) == 113
    min_overlap = 1.0
    slots = 0
    for r, s in zip(new, before):
        assert (r["image_id"], r["GT"]) == (s["image_id"], s["GT"])
        x, y = (
            np.asarray(r["pred"]).reshape(-1, 6),
            np.asarray(s["pred"]).reshape(-1, 6),
        )
        a, _ = match(x, r["GT"])
        b, _ = match(y, r["GT"])
        assert set(a.values()) == set(b.values())
        for cls in (0, 1):
            m = [p[:4] for i, p in enumerate(x) if p[5] == cls and i not in a]
            n = [p[:4] for i, p in enumerate(y) if p[5] == cls and i not in b]
            assert len(m) == len(n)
            if n:
                overlap = ious(m, n)
                ii, jj = linear_sum_assignment(-overlap)
                min_overlap = min(min_overlap, float(overlap[ii, jj].min()))
                slots += len(m)
    assert slots == 45 and min_overlap >= 0.5
    result = {name: summary(t) for name, t in times.items()}
    atomic_json(
        OUT / "paired_benchmark.json",
        dict(
            benchmark=result,
            fp32_covered_same_GT=True,
            matched_FP_count=slots,
            minimum_FP_pair_IoU=min_overlap,
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
