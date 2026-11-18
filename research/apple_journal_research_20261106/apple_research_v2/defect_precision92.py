"""Contextual RGB verification with held-fold calibration of the P/R tradeoff."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from class0_patch_refinement import (
    CP,
    ROOT,
    OLD,
    sha,
    read,
    atomic_json,
    configure,
    load_image,
    manifest,
)
from audit_defect_causes_20261112 import ious, match
from defect_coordinate_vote_20261118 import vote
from defect_algorithm_modes_20261112 import nms
from joint_model_v1 import OFFICIAL, OFFICIAL_SHA
from class0_crop_deploy import Inference
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect_precision92_v1"
DEVICE = "cuda"


class RGBFeatures(nn.Module):
    """Frozen general-purpose image filters preserve detail at native crop scale."""

    def __init__(self):
        super().__init__()
        assert sha(OFFICIAL) == OFFICIAL_SHA
        obj = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
        trunk = (obj.get("ema") or obj["model"]).float().model
        self.trunk = (
            nn.Sequential(*list(trunk.children())[:7]).requires_grad_(False).eval()
        )

    @torch.inference_mode()
    def forward(self, batch):
        x = batch.to(DEVICE, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        pooled = [F.adaptive_avg_pool2d(x, 3).flatten(1)]
        for i, layer in enumerate(self.trunk):
            x = layer(x)
            if i in (2, 4, 6):
                pooled.append(F.adaptive_avg_pool2d(x, 3).flatten(1))
        return torch.cat(pooled, 1).cpu().numpy()


def proposals(src):
    boxes = vote(src, 0.5, [0.055, 0.055])
    dense = src["o2m"]
    dense = dense[dense[:, 4] >= 0.10]
    extra = []
    for c in (0, 1):
        b = boxes[boxes[:, 5] == c]
        d = dense[dense[:, 5] == c]
        if len(d) and len(b):
            d = d[ious(d[:, :4], b[:, :4]).max(1) < 0.5]
        extra.append(d)
    return np.concatenate([boxes, *extra])


def extract(src, im, encoder, gt=None, rng=None):
    import cv2

    src = {
        k: np.asarray(src[k], np.float32).reshape(-1, 6) for k in ("o2o", "flip", "o2m")
    }
    b = proposals(src)
    count = len(b)
    h, w = im.shape[:2]
    if gt:
        # Train with well-centred and modestly shifted examples of both defect types.
        extra = []
        for g in gt:
            box = np.array(g["box"])
            size = box[2:] - box[:2]
            for _ in range(2):
                delta = rng.uniform(-0.14, 0.14, 4) * np.tile(size, 2)
                extra.append(
                    [*(box + delta).clip([0, 0, 0, 0], [w, h, w, h]), 0, g["class"]]
                )
        b = np.concatenate((b, np.asarray(extra, np.float32)))
    cols = []
    for pool in src.values():
        for c in (0, 1):
            p = pool[pool[:, 5] == c]
            s = np.zeros(len(b))
            if len(b) and len(p):
                s = (ious(b[:, :4], p[:, :4]) ** 2 * p[None, :, 4]).max(1)
            cols.append(s)
    wh = np.maximum(1, b[:, 2:4] - b[:, :2])
    attrs = np.column_stack([*cols, np.log(wh / [w, h])]).astype(np.float32)
    crops = []
    for box, sz in zip(b, wh):
        lo = np.maximum(np.floor(box[:2] - 0.42 * sz).astype(int), 0)
        hi = np.minimum(np.ceil(box[2:4] + 0.42 * sz).astype(int), [w, h])
        region = im[lo[1] : hi[1], lo[0] : hi[0]]
        if not region.size:
            region = np.zeros((1, 1, 3), np.uint8)
        crops.append(
            cv2.resize(region, (80, 80), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        )
    if crops:
        z = encoder(torch.from_numpy(np.stack(crops)))
        x = np.column_stack([attrs, z]).astype(np.float32)
    else:
        x = np.empty((0, 0), np.float32)
    return b, x, count


class Verifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        self.hidden = nn.Sequential(
            nn.Linear(dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 48),
            nn.SiLU(),
        )
        self.cls = nn.Linear(48, 3)
        self.box = nn.Linear(48, 4)

    def forward(self, x):
        z = self.hidden(((x - self.mean) / self.scale).clamp(-6, 6))
        return self.cls(z), self.box(z)


def learn(rows):
    torch.manual_seed(92)
    feats = []
    labels = []
    deltas = []
    for r in rows:
        b, x, _ = r["data"]
        if not len(b):
            continue
        gt = r["GT"]
        y = np.full(len(b), 2, np.int64)
        d = np.zeros((len(b), 4), np.float32)
        if gt:
            g = np.array([v["box"] for v in gt])
            ov = ious(b[:, :4], g)
            j = ov.argmax(1)
            pos = ov.max(1) >= 0.46
            y[pos] = np.array([v["class"] for v in gt])[j[pos]]
            wh = np.maximum(b[:, 2:4] - b[:, :2], 1)
            d = ((g[j] - b[:, :4]) / np.tile(wh, (1, 2))).clip(-0.65, 0.65)
        feats.append(x)
        labels.append(y)
        deltas.append(d.astype(np.float32))
    x, y, d = [
        torch.from_numpy(np.concatenate(v)).to(DEVICE) for v in (feats, labels, deltas)
    ]
    m = Verifier(x.shape[1]).to(DEVICE)
    m.mean.copy_(x.mean(0))
    m.scale.copy_(x.std(0).clamp_min(0.03))
    opt = torch.optim.AdamW(m.parameters(), lr=0.0012, weight_decay=0.012)
    for epoch in range(24):
        for batch in torch.randperm(len(y), device=DEVICE).split(128):
            cls, box = m(x[batch])
            pos = y[batch] != 2
            loss = F.cross_entropy(cls, y[batch], label_smoothing=0.04)
            if pos.any():
                loss += 1.5 * F.smooth_l1_loss(box[pos], d[batch][pos], beta=0.12)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return m.eval().cpu()


@torch.inference_mode()
def rank(data, model, hw):
    b, x, n = data
    if not n:
        return np.empty((0, 6), np.float32)
    b = b[:n].copy()
    logits, delta = model(torch.from_numpy(x[:n]))
    q = logits.softmax(1).numpy()
    # Match the proposed class to visual evidence; combine with independent detector support.
    ids = b[:, 5].astype(int)
    s = q[np.arange(n), ids].clip(0.001, 0.999)
    support = x[:n, :6].reshape(n, 3, 2)[np.arange(n), :, ids].max(1).clip(0.001, 0.999)
    b[:, 4] = 1 / (
        1 + np.exp(-(0.7 * np.log(s / (1 - s)) + 0.3 * np.log(support / (1 - support))))
    )
    b[:, :4] += (
        0.50
        * delta.numpy().clip(-0.45, 0.45)
        * np.tile(np.maximum(b[:, 2:4] - b[:, :2], 1), (1, 2))
    )
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, hw[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, hw[0])
    return nms(b)


def cut(boxes, thresholds):
    return boxes[boxes[:, 4] >= np.asarray(thresholds)[boxes[:, 5].astype(int)]]


def calibrate(rows):
    grid = np.linspace(0.01, 0.99, 99)
    counts = np.zeros((2, len(grid), 2), int)
    for r in rows:
        p = r["oof"]
        hits, _ = match(p, r["GT"])
        for j, b in enumerate(p):
            counts[int(b[5]), grid <= b[4], 0 if j in hits else 1] += 1
    tp = counts[0, :, 0, None] + counts[1, None, :, 0]
    fp = counts[0, :, 1, None] + counts[1, None, :, 1]
    prec = tp / np.maximum(1, tp + fp)
    totals = sum(len(r["GT"]) for r in rows)
    modes = {}
    for name, target in [
        ("balanced", None),
        ("precision92", 0.92),
        ("precision95_margin", 0.95),
    ]:
        if target is None:
            # Preserve at least the prior verifier's class-0 OOF recall and class-1 baseline recall.
            feasible = (counts[0, :, 0, None] >= 311) & (
                counts[1, None, :, 0] >= sum(r["ref1"] for r in rows)
            )
            objective = prec
        else:
            feasible = (
                (prec >= target)
                & (counts[0, :, 0, None] >= 50)
                & (counts[1, None, :, 0] >= 50)
            )
            objective = tp.astype(float)
        if not feasible.any():
            continue
        i, j = max(
            zip(*np.where(feasible)), key=lambda ij: (objective[ij], prec[ij], tp[ij])
        )
        modes[name] = dict(
            thresholds=[float(grid[i]), float(grid[j])],
            TP=int(tp[i, j]),
            FP=int(fp[i, j]),
            P=float(prec[i, j]),
            R=float(tp[i, j] / totals),
        )
    return modes


def fit():
    rows = []
    lookup = manifest()
    rng = np.random.default_rng(92)
    encoder = RGBFeatures().to(DEVICE).eval()
    for fold in range(3):
        for r in read(OLD / f"all_mode_OOF_calibration/fold{fold}_sources.json")[
            "rows"
        ]:
            sid = r["image_id"]
            assert lookup[sid]["partition"] == "train"
            im = load_image(lookup[sid]["rgb_path"])
            base = vote(r["sources"], 0.5, [0.4, 0.34])
            hits, _ = match(base, r["GT"])
            rows.append(
                dict(
                    image_id=sid,
                    GT=r["GT"],
                    fold=fold,
                    hw=im.shape[:2],
                    ref1=sum(r["GT"][i]["class"] == 1 for i in hits.values()),
                    data=extract(r["sources"], im, encoder, r["GT"], rng),
                )
            )
    assert len({r["image_id"] for r in rows}) == 596
    for fold in range(3):
        model = learn([r for r in rows if r["fold"] != fold])
        for r in rows:
            if r["fold"] == fold:
                r["oof"] = rank(r["data"], model, r["hw"])
    modes = calibrate(rows)
    model = learn(rows)
    torch.save(model.state_dict(), OUT / "verifier.pt")
    atomic_json(
        OUT / "frozen.json",
        dict(
            modes=modes,
            checkpoint_sha=sha(CP),
            verifier_sha=sha(OUT / "verifier.pt"),
            train_images=len(rows),
        ),
    )
    print(modes, flush=True)


class Deployment:
    def __init__(self):
        self.incumbent = Inference(False)
        self.encoder = RGBFeatures().to(DEVICE).eval()
        cfg = read(OUT / "frozen.json")
        assert cfg["verifier_sha"] == sha(OUT / "verifier.pt") and cfg[
            "checkpoint_sha"
        ] == sha(CP)
        state = torch.load(OUT / "verifier.pt", map_location="cpu", weights_only=True)
        self.verifier = Verifier(len(state["mean"])).eval()
        self.verifier.load_state_dict(state)
        self.modes = cfg["modes"]

    @torch.inference_mode()
    def predict(self, im, mode=None, both=False):
        worker = self.incumbent.worker
        pred = worker.predictors["stage"]
        tensor = pred.preprocess([im])
        src = worker.sources(tensor, im.shape[:2], ("o2o", "o2m"), True)
        p = rank(extract(src, im, self.encoder), self.verifier, im.shape[:2])
        if mode:
            p = cut(p, self.modes[mode]["thresholds"])
        if not both:
            return p
        pred.batch = (["frame"], [im], [""])
        maturity = [
            r.cpu()
            for r in pred.postprocess(
                worker.model.predict_source(tensor, "stage"), tensor, [im]
            )
        ]
        return maturity, p


def evaluate():
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary

    model = Deployment()
    lookup = manifest()
    rows = []
    prior = read(ROOT / "experiments/class0_crop_deploy_v1/fp32/dev_predictions.json")
    for r in prior:
        sid = r["image_id"]
        assert lookup[sid]["partition"] == "dev"
        p = model.predict(load_image(lookup[sid]["rgb_path"]))
        rows.append(dict(image_id=sid, GT=r["GT"], all=p))
    assert len(rows) == 113
    result = {}
    for mode, setting in model.modes.items():
        rr = [
            dict(
                image_id=r["image_id"],
                GT=r["GT"],
                pred=cut(r["all"], setting["thresholds"]).tolist(),
            )
            for r in rows
        ]
        atomic_json(OUT / f"{mode}_predictions.json", rr)
        result[mode] = metrics(rr)
    atomic_json(OUT / "dev_metrics.json", result)
    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims.extend(
            load_image(paths[round(i * (len(paths) - 1) / 19)]) for i in range(20)
        )
    timings = {s: [] for s in model.modes}
    for mode in timings:
        for im in ims[:5]:
            model.predict(im, mode, True)
    for k in range(15):
        for i, im in enumerate(ims):
            names = list(timings)
            for mode in names[(i + k) % len(names) :] + names[: (i + k) % len(names)]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                model.predict(im, mode, True)
                torch.cuda.synchronize()
                timings[mode].append((time.perf_counter() - start) * 1000)
    atomic_json(OUT / "speed.json", {k: summary(v) for k, v in timings.items()})
    print({k: r["total"] for k, r in result.items()}, flush=True)


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
