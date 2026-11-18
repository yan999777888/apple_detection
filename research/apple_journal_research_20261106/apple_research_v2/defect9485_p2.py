"""Fixed-epoch fine-tuning with a stride-4 detail path into the stride-8 detector."""

from pathlib import Path
import argparse
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from class0_patch_refinement import (
    ROOT,
    CP,
    read,
    sha,
    atomic_json,
    configure,
    load_image,
)
from defect_algorithm_modes_20261112 import nms
from run_defect_zero_train_20261112 import metrics
from audit_defect_causes_20261112 import match

OUT = ROOT / "experiments/defect9485_p2_v1"
NATIVE = ROOT / "runs/ssda_yolo26n_detect_640_e40_s20261106/weights/last.pt"
SHA = "1d10a4a02a852c85bdd4ea5a34eed11f72c70367445ec6176cd411b9b8b674bc"
SIZE = 800
EPOCHS = 6
PRIMARY = [0.40, 0.34]
BATCH = 4


class P2Detector(nn.Module):
    def __init__(self):
        super().__init__()
        assert sha(NATIVE) == SHA
        d = torch.load(NATIVE, map_location="cpu", weights_only=False)
        base = (d.get("ema") or d["model"]).float()
        self.trunk = nn.ModuleList(list(base.model[:-1])).eval().requires_grad_(False)
        self.save = set(base.save) | {2}
        self.head = base.model[-1].requires_grad_(True)
        self.head_indices = self.head.f
        channels_in = self.trunk[2].cv2.conv.out_channels * 4
        channels_out = self.head.cv2[0][0].conv.in_channels
        self.detail = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(channels_in, 24, 1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, channels_out, 1, bias=False),
        )
        nn.init.zeros_(self.detail[-1].weight)

    def features(self, x):
        outputs = []
        with torch.no_grad():
            for layer in self.trunk:
                if layer.f != -1:
                    x = (
                        outputs[layer.f]
                        if isinstance(layer.f, int)
                        else [x if j == -1 else outputs[j] for j in layer.f]
                    )
                x = layer(x)
                outputs.append(x if layer.i in self.save else None)
        features = [outputs[i] for i in self.head_indices]
        features[0] = features[0] + self.detail(outputs[2])
        return features

    def forward(self, x):
        return self.head(self.features(x))

    def fuse(self):
        from ultralytics.nn.modules.conv import Conv
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        self.eval()
        for mod in self.modules():
            if isinstance(mod, Conv) and isinstance(
                getattr(mod, "bn", None), nn.BatchNorm2d
            ):
                mod.conv = fuse_conv_bn_eval(mod.conv, mod.bn)
                del mod.bn
                mod.forward = mod.forward_fuse
        return self


def samples(partition):
    return [
        r
        for r in read(ROOT / "apple_research_v2/datasets/ssda_date_v2/manifest.json")[
            "samples"
        ]
        if r["partition"] == partition
    ]


def square_bgr(im):
    import cv2

    h, w = im.shape[:2]
    scale = SIZE / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    small = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x, y = (SIZE - nw) // 2, (SIZE - nh) // 2
    if nw != SIZE or nh != SIZE:
        small = cv2.copyMakeBorder(
            small,
            y,
            SIZE - nh - y,
            x,
            SIZE - nw - x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    return np.ascontiguousarray(small[:, :, ::-1]), (x, y, nw / w, nh / h)


def prepare():
    rows = samples("train")
    assert len(rows) == 596
    images = []
    targets = []
    for r in rows:
        im = load_image(r["rgb_path"])
        assert im.shape[:2] == (1024, 1024)
        images.append(square_bgr(im)[0])
        targets.append(
            np.asarray(
                [[v["class_id"], *v["xywh_normalized"]] for v in r["instances"]],
                np.float32,
            ).reshape(-1, 5)
        )
    return images, targets


def batch_data(images, targets, ids, rng):
    batch = []
    labels = []
    indices = []
    for k, i in enumerate(ids):
        im = images[i]
        v = targets[i].copy()
        if rng.random() < 0.5:
            im = im[:, ::-1]
            v[:, 1] = 1 - v[:, 1]
        batch.append(im)
        labels.append(v)
        indices.extend([k] * len(v))
    x = torch.from_numpy(np.stack(batch)).to("cuda").permute(0, 3, 1, 2).float() / 255
    gains = torch.tensor(
        rng.uniform(0.88, 1.12, (len(batch), 3, 1, 1)), device="cuda", dtype=x.dtype
    )
    x = (x * gains).clamp(0, 1)
    v = torch.from_numpy(np.concatenate(labels)).cuda()
    return x, dict(
        batch_idx=torch.tensor(indices, device="cuda", dtype=torch.float32),
        cls=v[:, :1],
        bboxes=v[:, 1:],
    )


def train():
    from ultralytics.utils.loss import E2ELoss
    from joint_model_v1 import LossView

    if (OUT / "weights.pt").exists():
        raise ValueError("Existing final weights")
    rng = np.random.default_rng(9485)
    torch.manual_seed(9485)
    random.seed(9485)
    model = P2Detector().cuda()
    model.trunk.eval()
    model.head.train()
    for m in model.head.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    criterion = E2ELoss(LossView(model.head, EPOCHS))
    # Preserve image-space weights initially; the new residual starts exactly at zero.
    opt = torch.optim.AdamW(
        [
            dict(params=model.head.parameters(), lr=1.2e-4),
            dict(params=model.detail.parameters(), lr=7e-4),
        ],
        weight_decay=0.001,
    )
    atomic_json(
        OUT / "protocol.json",
        dict(
            native=SHA,
            imgsz=SIZE,
            epochs=EPOCHS,
            batch=BATCH,
            train_images=596,
            split="original date split",
            detection_thresholds=PRIMARY,
            low_floor=0.001,
            incremental_selection="use final epoch",
            proposal="detector one-to-one head; classwise NMS .7",
            features="stride4 pixel-unshuffle to stride8; 24-channel learned projection + residual",
            loss="native YOLO E2E o2m + o2o",
            optimizer="AdamW, frozen batch statistics",
            lr_head=0.00012,
            lr_detail=0.0007,
        ),
    )
    images, targets = prepare()
    start = time.perf_counter()
    epochs = []
    for epoch in range(EPOCHS):
        perm = rng.permutation(len(images))
        losses = []
        for j in range(0, len(perm), BATCH):
            x, gt = batch_data(images, targets, perm[j : j + BATCH], rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
            # Explicit float32 decoding and assignment preserve IoU/box-loss arithmetic.
            pred = {
                key: {
                    k: (
                        [v.float() for v in val]
                        if isinstance(val, list)
                        else val.float()
                    )
                    for k, val in branch.items()
                }
                for key, branch in pred.items()
            }
            loss, _ = criterion(pred, gt)
            loss = loss.sum() / len(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.head.parameters(), *model.detail.parameters()],
                10.0,
                error_if_nonfinite=True,
            )
            opt.step()
            losses.append(float(loss.detach()))
        criterion.update()
        info = dict(
            epoch=epoch + 1,
            loss=float(np.mean(losses)),
            updates=len(losses),
            seconds=time.perf_counter() - start,
        )
        epochs.append(info)
        atomic_json(OUT / "training.json", epochs)
        print(info, flush=True)
    torch.save(
        dict(
            detail=model.detail.cpu().state_dict(), head=model.head.cpu().state_dict()
        ),
        OUT / "weights.pt",
    )
    atomic_json(
        OUT / "receipt.json",
        dict(
            sha256=sha(OUT / "weights.pt"),
            epochs=EPOCHS,
            updates=sum(e["updates"] for e in epochs),
        ),
    )


def deploy_model():
    assert sha(OUT / "weights.pt") == read(OUT / "receipt.json")["sha256"]
    state = torch.load(OUT / "weights.pt", map_location="cpu", weights_only=True)
    net = P2Detector()
    net.head.load_state_dict(state["head"])
    net.detail.load_state_dict(state["detail"])
    return net.fuse().requires_grad_(False).cuda()


@torch.inference_mode()
def boxes(net, im):
    arr, (dx, dy, sx, sy) = square_bgr(im)
    t = torch.from_numpy(arr).cuda().permute(2, 0, 1)[None].float().div_(255)
    feat = net.features(t)
    raw = net.head.forward_head(feat, **net.head.one2one)
    pred = net.head.postprocess(net.head._inference(raw).permute(0, 2, 1))[0]
    p = pred[pred[:, 4] >= 0.001].cpu().numpy()
    h, w = im.shape[:2]
    p[:, [0, 2]] = ((p[:, [0, 2]] - dx) / sx).clip(0, w)
    p[:, [1, 3]] = ((p[:, [1, 3]] - dy) / sy).clip(0, h)
    return nms(p)


def diagnostic(rows):
    from itertools import groupby

    flags = []
    for r in rows:
        h, _ = match(r["pred"], r["GT"])
        flags.extend((p[4], int(i in h)) for i, p in enumerate(r["pred"]))
    tp = fp = 0
    points = []
    for s, recs in groupby(sorted(flags, reverse=True), key=lambda a: a[0]):
        hits = [v for _, v in recs]
        tp += sum(hits)
        fp += len(hits) - sum(hits)
        points.append(dict(threshold=s, TP=tp, FP=fp, P=tp / (tp + fp), R=tp / 230))
    return dict(
        best_R_at_P94=max(
            (p for p in points if p["P"] >= 0.94), key=lambda p: p["TP"], default=None
        ),
        best_P_at_R85=max(
            (p for p in points if p["R"] >= 0.85), key=lambda p: p["P"], default=None
        ),
    )


def evaluate(model=None, destination=OUT, predict=boxes):
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from class0_crop_deploy import Inference

    if model is None:
        model = deploy_model()
    rows = []
    for r in samples("dev"):
        im = load_image(r["rgb_path"])
        p = predict(model, im)
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
    atomic_json(destination / "low_predictions.json", rows)
    thresholded = [
        dict(r, pred=[p for p in r["pred"] if p[4] >= PRIMARY[int(p[5])]]) for r in rows
    ]
    atomic_json(destination / "fixed_predictions.json", thresholded)
    report = dict(fixed=metrics(thresholded), dev_diagnostic=diagnostic(rows))
    atomic_json(destination / "metrics.json", report)
    print(report["fixed"]["total"], flush=True)
    worker = Inference(False).worker

    @torch.inference_mode()
    def both(im):
        p = predict(model, im)
        pred = worker.predictors["stage"]
        t = pred.preprocess([im])
        pred.batch = (["frame"], [im], [""])
        stage = [
            v.cpu()
            for v in pred.postprocess(worker.model.predict_source(t, "stage"), t, [im])
        ]
        return stage, p[p[:, 4] >= np.array(PRIMARY)[p[:, 5].astype(int)]]

    ims = []
    for task in ("stage", "defect"):
        pp = source_spec(task)["images"]
        ims.extend(load_image(pp[round(i * (len(pp) - 1) / 19)]) for i in range(20))
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
    atomic_json(destination / "speed.json", summary(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate"])
    args = parser.parse_args()
    configure(OUT, "cuda:0")
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if args.action == "train":
        train()
    else:
        evaluate()
