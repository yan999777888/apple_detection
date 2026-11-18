"""Train a 768 px YOLO26-small defect detector from general COCO weights."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from class0_patch_refinement import ROOT, read, sha, atomic_json, configure, load_image
from defect9485_p2 import samples, diagnostic, BATCH
from defect_algorithm_modes_20261112 import nms
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_small_v1"
BASE = ROOT.parent / "apple_vision/models/yolo26s.pt"
BASE_SHA = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
SIZE = 768
EPOCHS = 36
ACCUM = 3
PRIMARY = [0.40, 0.34]
SEED = 49485


def architecture():
    from ultralytics.nn.tasks import DetectionModel

    assert sha(BASE) == BASE_SHA
    state = torch.load(BASE, map_location="cpu", weights_only=False)
    ref = (state.get("ema") or state["model"]).float()
    assert (
        ref.model[-1].nc == 80
        and ref.names[0] == "person"
        and state["train_args"]["data"].endswith("coco.yaml")
    )
    model = DetectionModel(deepcopy(ref.yaml), ch=3, nc=2, verbose=False)
    model.load(ref, verbose=False)
    # COCO channels are unrelated to the two SSDA classes. Both terminal heads start fresh.
    assert model.model[-1].nc == 2 and model.end2end
    return model


def square(im):
    import cv2

    h, w = im.shape[:2]
    ratio = SIZE / max(h, w)
    nh, nw = round(h * ratio), round(w * ratio)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x, y = (SIZE - nw) // 2, (SIZE - nh) // 2
    if h != w:
        resized = cv2.copyMakeBorder(
            resized,
            y,
            SIZE - nh - y,
            x,
            SIZE - nw - x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    return np.ascontiguousarray(resized[:, :, ::-1]), (x, y, nw / w, nh / h)


def prepare():
    rows = samples("train")
    assert len(rows) == 596
    images = []
    labels = []
    for r in rows:
        im = load_image(r["rgb_path"])
        assert im.shape[:2] == (1024, 1024)
        images.append(square(im)[0])
        labels.append(
            np.array(
                [[v["class_id"], *v["xywh_normalized"]] for v in r["instances"]],
                np.float32,
            ).reshape(-1, 5)
        )
    assert sum(map(len, labels)) == 1276
    return images, labels


def augment(images, labels, ids, rng):
    import cv2

    out = []
    gt = []
    inds = []
    for k, i in enumerate(ids):
        im = images[i]
        v = labels[i].copy()
        if rng.random() < 0.5:
            im = im[:, ::-1]
            v[:, 1] = 1 - v[:, 1]
        mat = cv2.getRotationMatrix2D(
            (SIZE / 2, SIZE / 2), rng.uniform(-12, 12), rng.uniform(0.85, 1.14)
        )
        mat[:, 2] += rng.uniform(-0.035, 0.035, 2) * SIZE
        im = cv2.warpAffine(im, mat, (SIZE, SIZE), borderValue=(114, 114, 114))
        if len(v):
            a = v[:, 1:3] - v[:, 3:5] / 2
            b = v[:, 1:3] + v[:, 3:5] / 2
            corners = np.stack(
                [a, b, np.c_[a[:, 0], b[:, 1]], np.c_[b[:, 0], a[:, 1]]], axis=1
            )
            xy = corners @ mat[:, :2].T + mat[None, None, :, 2] / SIZE
            rawlo = xy.min(1)
            rawhi = xy.max(1)
            lo = rawlo.clip(0, 1)
            hi = rawhi.clip(0, 1)
            extent = hi - lo
            keep = (extent.min(1) * SIZE >= 1) & (
                extent.prod(1) >= 0.3 * (rawhi - rawlo).prod(1)
            )
            v[:, 1:3] = (lo + hi) / 2
            v[:, 3:5] = extent
            v = v[keep]
        out.append(im)
        gt.append(v)
        inds.extend([k] * len(v))
    x = torch.from_numpy(np.stack(out)).to("cuda").permute(0, 3, 1, 2).float() / 255
    gain = rng.uniform(0.82, 1.18, (len(ids), 1, 1, 1)) * rng.uniform(
        0.92, 1.08, (len(ids), 3, 1, 1)
    )
    x = (x * torch.as_tensor(gain, device="cuda", dtype=x.dtype)).clamp(0, 1)
    v = torch.from_numpy(np.concatenate(gt)).cuda()
    return x, dict(
        cls=v[:, :1],
        bboxes=v[:, 1:],
        batch_idx=torch.tensor(inds, device="cuda", dtype=torch.float32),
    )


def fit():
    from ultralytics.utils.loss import E2ELoss
    from ultralytics.utils.torch_utils import ModelEMA
    from joint_model_v1 import LossView

    if (OUT / "weights.pt").exists():
        raise ValueError("Final checkpoint already exists")
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = architecture().cuda().train()
    ema = ModelEMA(model, decay=0.997, tau=200)
    crit = E2ELoss(LossView(model.model[-1], EPOCHS))
    groups = [
        dict(
            params=[p for p in model.parameters() if p.requires_grad and p.ndim > 1],
            weight_decay=0.0005,
        ),
        dict(
            params=[p for p in model.parameters() if p.requires_grad and p.ndim <= 1],
            weight_decay=0.0,
        ),
    ]
    opt = torch.optim.AdamW(groups, lr=0.0008)
    atomic_json(
        OUT / "protocol.json",
        dict(
            base_sha=BASE_SHA,
            base_metadata="COCO 80-class YOLO26-small",
            epochs=EPOCHS,
            primary="final epoch EMA",
            imgsz=SIZE,
            batch=BATCH,
            effective_batch=BATCH * ACCUM,
            seed=SEED,
            primary_thresholds=PRIMARY,
            lr=0.0008,
            lr_schedule="2-epoch linear warmup, cosine to .05",
            augmentation="±12 degrees, .85–1.14 size, ±.035 translation, .5 horizontal flip, .82–1.18 exposure, per-channel mild color jitter",
            train_images=596,
            gt_boxes=1276,
            train_precision="bfloat16 CUDA forward; FP32 assignment/loss/optimizer",
            inference="single-view, FP32 fused, explicit class NMS .7",
            dev_used_before_final=False,
        ),
    )
    ims, gt = prepare()
    epochs = []
    updates = 0
    start = time.perf_counter()
    for e in range(EPOCHS):
        perm = rng.permutation(len(ims))
        losses = []
        batches = np.array_split(perm, len(perm) // BATCH)
        for start_idx in range(0, len(batches), ACCUM):
            group = batches[start_idx : start_idx + ACCUM]
            ep = e + start_idx / len(batches)
            gain = (
                0.05
                + 0.95 * (1 + math.cos(math.pi * max(0, ep - 2) / (EPOCHS - 2))) / 2
            ) * min(1, (ep + 0.05) / 2)
            for g in opt.param_groups:
                g["lr"] = 0.0008 * gain
            opt.zero_grad(set_to_none=True)
            for ids in group:
                x, labels = augment(ims, gt, ids, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model(x)
                raw = {
                    name: {
                        k: (
                            [t.float() for t in val]
                            if isinstance(val, list)
                            else val.float()
                        )
                        for k, val in branch.items()
                    }
                    for name, branch in raw.items()
                }
                loss = crit(raw, labels)[0].sum() / len(ids)
                (loss / len(group)).backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0, error_if_nonfinite=True
            )
            opt.step()
            ema.update(model)
            updates += 1
        crit.update()
        info = dict(
            epoch=e + 1,
            updates=updates,
            mean_loss=float(torch.stack(losses).mean()),
            seconds=time.perf_counter() - start,
        )
        epochs.append(info)
        atomic_json(OUT / "training.json", epochs)
        print(info, flush=True)
    torch.save(ema.ema.float().cpu().state_dict(), OUT / "weights.pt")
    atomic_json(
        OUT / "receipt.json",
        dict(weight_sha=sha(OUT / "weights.pt"), epochs=len(epochs), updates=updates),
    )


def deploy_model():
    assert sha(OUT / "weights.pt") == read(OUT / "receipt.json")["weight_sha"]
    model = architecture().eval()
    model.load_state_dict(
        torch.load(OUT / "weights.pt", map_location="cpu", weights_only=True)
    )
    # Upstream fuse may delete unused training-time one-to-many head, as intended for deployment.
    return model.cuda().fuse(verbose=False).requires_grad_(False)


@torch.inference_mode()
def boxes(model, im):
    a, (dx, dy, sx, sy) = square(im)
    x = torch.from_numpy(a).cuda().permute(2, 0, 1)[None].float() / 255
    output = model(x)
    p = output[0][0] if isinstance(output, tuple) else output[0]
    assert p.ndim == 2 and p.shape[1] == 6
    p = p[p[:, 4] >= 0.001].cpu().numpy()
    h, w = im.shape[:2]
    p[:, [0, 2]] = ((p[:, [0, 2]] - dx) / sx).clip(0, w)
    p[:, [1, 3]] = ((p[:, [1, 3]] - dy) / sy).clip(0, h)
    return nms(p)


def evaluate():
    from audit_defect_causes_20261112 import match
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from class0_crop_deploy import Inference

    m = deploy_model()
    rows = []
    for r in samples("dev"):
        im = load_image(r["rgb_path"])
        p = boxes(m, im)
        gt = []
        for v in r["instances"]:
            x, y, w, h = v["xywh_normalized"]
            gt.append(
                dict(
                    box=[
                        (x - w / 2) * 1024,
                        (y - h / 2) * 1024,
                        (x + w / 2) * 1024,
                        (y + h / 2) * 1024,
                    ],
                    **{"class": v["class_id"]},
                )
            )
        rows.append(dict(image_id=r["image_id"], GT=gt, pred=p.tolist()))
    assert len(rows) == 113 and sum(len(r["GT"]) for r in rows) == 230
    atomic_json(OUT / "low_predictions.json", rows)
    fixed = [
        dict(r, pred=[p for p in r["pred"] if p[4] >= PRIMARY[int(p[5])]]) for r in rows
    ]
    atomic_json(OUT / "fixed_predictions.json", fixed)

    def hits(rr):
        return {
            (r["image_id"], i)
            for r in rr
            for i in match(r["pred"], r["GT"])[0].values()
        }

    old = hits(
        read(ROOT / "experiments/class0_crop_deploy_v1/fp32/dev_predictions.json")
    )
    p2 = hits(read(ROOT / "experiments/defect9485_p2_v1/fixed_predictions.json"))
    nh = hits(fixed)
    report = dict(
        fixed=metrics(fixed),
        dev_diagnostic=diagnostic(rows),
        oracle_development=dict(
            fixed_union_with_P2=len(p2 | nh),
            fixed_union_with_old=len(old | nh),
            fixed_union_three=len(old | p2 | nh),
            novel_vs_old=len(nh - old),
        ),
    )
    atomic_json(OUT / "metrics.json", report)
    worker = Inference(False).worker

    @torch.inference_mode()
    def both(im):
        p = boxes(m, im)
        predictor = worker.predictors["stage"]
        t = predictor.preprocess([im])
        predictor.batch = (["frame"], [im], [""])
        stage = [
            v.cpu()
            for v in predictor.postprocess(
                worker.model.predict_source(t, "stage"), t, [im]
            )
        ]
        return stage, p[p[:, 4] >= np.array(PRIMARY)[p[:, 5].astype(int)]]

    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims.extend(
            load_image(paths[round(i * (len(paths) - 1) / 19)]) for i in range(20)
        )
    for im in ims[:5]:
        both(im)
    times = []
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize()
            start = time.perf_counter()
            both(im)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
    atomic_json(OUT / "speed.json", summary(times))
    print(report["fixed"]["total"], summary(times), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["fit", "evaluate"])
    args = p.parse_args()
    configure(OUT, "cuda:0")
    import cv2

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.action == "fit":
        fit()
    else:
        evaluate()
