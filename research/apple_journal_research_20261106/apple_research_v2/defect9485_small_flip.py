"""Frozen horizontal two-view coordinate voting on the final 768-small checkpoint."""

import numpy as np
import torch
from defect9485_small import ROOT, SIZE, PRIMARY, deploy_model, square, configure
from defect9485_p2 import evaluate
from defect_coordinate_vote_20261118 import vote

OUT = ROOT / "experiments/defect9485_small_flip_v1"


@torch.inference_mode()
def predict(model, im):
    arr, (dx, dy, sx, sy) = square(im)
    t = torch.from_numpy(arr).cuda().permute(2, 0, 1)[None].float().div_(255)
    raw = model(torch.cat([t, t.flip((-1,))], 0))
    pred = raw[0] if isinstance(raw, tuple) else raw
    assert pred.ndim == 3 and pred.shape[0] == 2 and pred.shape[2] == 6
    h, w = im.shape[:2]
    src = {}
    for i, key in enumerate(("o2o", "flip")):
        p = pred[i][pred[i, :, 4] >= 0.001].cpu().numpy()
        if i:
            p[:, [0, 2]] = SIZE - p[:, [2, 0]]
        p[:, [0, 2]] = ((p[:, [0, 2]] - dx) / sx).clip(0, w)
        p[:, [1, 3]] = ((p[:, [1, 3]] - dy) / sy).clip(0, h)
        src[key] = p
    return vote(src, 0.5, [0.001, 0.001])


if __name__ == "__main__":
    import cv2

    configure(OUT, "cuda:0")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    evaluate(deploy_model(), OUT, predict)
