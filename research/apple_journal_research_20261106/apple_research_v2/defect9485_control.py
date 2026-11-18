"""Matched unmodified YOLO26-nano control at 800 px, single view."""

import torch
from defect9485_p2 import NATIVE, SHA, ROOT, configure, sha, square_bgr, nms, evaluate

OUT = ROOT / "experiments/defect9485_control800_v1"


@torch.inference_mode()
def predict(model, im):
    arr, (dx, dy, sx, sy) = square_bgr(im)
    t = torch.from_numpy(arr).cuda().permute(2, 0, 1)[None].float().div_(255)
    output = model(t)
    pred = output[0][0] if isinstance(output, tuple) else output[0]
    assert pred.ndim == 2 and pred.shape[1] == 6
    p = pred[pred[:, 4] >= 0.001].cpu().numpy()
    h, w = im.shape[:2]
    p[:, [0, 2]] = ((p[:, [0, 2]] - dx) / sx).clip(0, w)
    p[:, [1, 3]] = ((p[:, [1, 3]] - dy) / sy).clip(0, h)
    return nms(p)


if __name__ == "__main__":
    configure(OUT, "cuda:0")
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    assert sha(NATIVE) == SHA
    ck = torch.load(NATIVE, map_location="cpu", weights_only=False)
    model = (
        (ck.get("ema") or ck["model"])
        .float()
        .cuda()
        .eval()
        .fuse(verbose=False)
        .requires_grad_(False)
    )
    evaluate(model, OUT, predict)
