"""Class-agnostic DINOv2 crop verifier for defect types and imperfect box extent."""

import argparse
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from class0_patch_refinement import (
    ROOT,
    OLD,
    configure,
    sha,
    read,
    atomic_json,
    load_image,
    manifest,
    appearance,
)
from audit_defect_causes_20261112 import ious, match
from run_defect_zero_train_20261112 import metrics
from defect9485_repr_encoder import Encoder
from defect9485_set import operating_points, cut

OUT = ROOT / "experiments/defect9485_repr"
WEIGHTS = OUT / "dinov2_vits14_pretrain.pth"
META_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def encoder():
    assert sha(WEIGHTS) == META_SHA
    m = Encoder()
    m.load_state_dict(
        torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True
    )
    return m.eval().requires_grad_(False)


def prep(sources, im, gt=()):
    arr = [
        np.asarray(sources[k], np.float32).reshape(-1, 6)
        for k in ("o2o", "flip", "o2m")
    ]
    merged = np.concatenate(arr)
    merged = merged[merged[:, 4] >= 0.008]
    ids = nms(
        torch.from_numpy(merged[:, :4]), torch.from_numpy(merged[:, 4]), 0.62
    ).numpy()[:36]
    b = merged[ids, :4].copy()
    n = len(b)
    h, w = im.shape[:2]
    if gt:
        b = np.concatenate([b, np.asarray([g["box"] for g in gt], np.float32)])
    wh = np.maximum(b[:, 2:4] - b[:, :2], 1)
    cols = []
    for a in arr:
        for cls in (0, 1):
            pool = a[a[:, 5] == cls]
            s = np.zeros(len(b))
            ov = s.copy()
            if len(b) and len(pool):
                m = ious(b, pool[:, :4])
                st = m**2 * pool[None, :, 4]
                idx = st.argmax(1)
                s = st[np.arange(len(b)), idx]
                ov = m[np.arange(len(b)), idx]
            cols.extend([s, ov])
    crops = []
    side = []
    for box, size in zip(b, wh):
        lo = np.maximum(0, (box[:2] - 0.42 * size).astype(int))
        hi = np.minimum([w, h], np.ceil(box[2:4] + 0.42 * size).astype(int))
        roi = im[lo[1] : hi[1], lo[0] : hi[0]]
        if not roi.size:
            roi = np.zeros((1, 1, 3), np.uint8)
        crops.append(
            cv2.resize(roi, (84, 84), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        )
        side.append(appearance(im, box))
    a = (
        np.column_stack([*cols, np.stack(side)]).astype(np.float32)
        if crops
        else np.empty((0, 0), np.float32)
    )
    return b, n, a, np.array(crops, dtype=np.uint8)


def embed(m, crops):
    return (
        torch.cat([m(batch) for batch in torch.from_numpy(crops).split(32)]).numpy()
        if len(crops)
        else np.empty((0, 2304), np.float32)
    )


def collect(raw, lookup, m, part, fold=None):
    out = []
    for r in raw:
        s = lookup[r["image_id"]]
        assert s["partition"] == part
        im = load_image(s["rgb_path"])
        b, n, side, crops = prep(r["sources"], im, r["GT"] if part == "train" else ())
        e = embed(m, crops)
        out.append(
            dict(
                image_id=r["image_id"],
                GT=r["GT"],
                hw=im.shape[:2],
                fold=fold,
                b=b,
                n=n,
                side=side,
                e=e.astype(np.float16),
            )
        )
    return out


class Head(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        self.body = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(n, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.28),
            nn.Linear(96, 48),
            nn.SiLU(),
        )
        self.cls = nn.Linear(48, 3)
        self.box = nn.Linear(48, 4)

    def forward(self, x):
        h = self.body(((x - self.mean) / self.scale).clamp(-6, 6))
        return self.cls(h), self.box(h)


def inputs(r, dino):
    return (
        np.column_stack([r["side"], r["e"].astype(np.float32)]) if dino else r["side"]
    )


def learn(rows, dino):
    feat = []
    labels = []
    delta = []
    for r in rows:
        b, gt = r["b"], r["GT"]
        if not len(b):
            continue
        feat.append(inputs(r, dino))
        y = np.full(len(b), 2)
        off = np.zeros((len(b), 4))
        if gt:
            g = np.array([v["box"] for v in gt])
            ov = ious(b, g)
            j = ov.argmax(1)
            pos = ov.max(1) >= 0.43
            y[pos] = np.array([v["class"] for v in gt])[j[pos]]
            off = (
                (g[j] - b) / np.tile(np.maximum(b[:, 2:4] - b[:, :2], 1), (1, 2))
            ).clip(-0.6, 0.6)
        labels.extend(y)
        delta.extend(off)
    x = torch.from_numpy(np.concatenate(feat))
    y = torch.tensor(labels)
    d = torch.from_numpy(np.stack(delta)).float()
    torch.manual_seed(9485)
    m = Head(x.shape[1])
    m.mean.copy_(x.mean(0))
    m.scale.copy_(x.std(0).clamp_min(0.04))
    opt = torch.optim.AdamW(m.parameters(), lr=0.0009, weight_decay=0.04)
    for epoch in range(19):
        for ids in torch.randperm(len(x)).split(192):
            logits, z = m(x[ids])
            pos = y[ids] != 2
            loss = F.cross_entropy(logits, y[ids], label_smoothing=0.035)
            if pos.any():
                loss += 1.25 * F.smooth_l1_loss(z[pos], d[ids][pos], beta=0.15)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return m.eval()


@torch.inference_mode()
def rank(r, m, dino):
    n = r["n"]
    if n == 0:
        return np.empty((0, 6), np.float32)
    x = inputs(r, dino)[:n]
    logits, z = m(torch.from_numpy(x))
    q = logits.softmax(1)[:, :2].numpy().clip(0.001, 0.999)
    support = x[:, :12:2].reshape(n, 3, 2).max(1).clip(0.001, 0.999)
    odds = 0.8 * np.log(q / (1 - q)) + 0.2 * np.log(support / (1 - support))
    cls = odds.argmax(1)
    scores = 1 / (1 + np.exp(-odds[np.arange(n), cls]))
    b = r["b"][:n].copy()
    b += (
        0.5
        * z.numpy().clip(-0.45, 0.45)
        * np.tile(np.maximum(b[:, 2:4] - b[:, :2], 1), (1, 2))
    )
    h, w = r["hw"]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    idx = nms(torch.from_numpy(b), torch.from_numpy(scores), 0.55).numpy()
    return np.column_stack([b, scores, cls])[idx]


def train():
    lookup = manifest()
    m = encoder()
    rows = []
    start = time.perf_counter()
    for f in range(3):
        raw = read(OLD / f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"]
        rows += collect(raw, lookup, m, "train", f)
        print(
            "train feature fold",
            f,
            "elapsed",
            round(time.perf_counter() - start),
            flush=True,
        )
    assert len({r["image_id"] for r in rows}) == 596
    for name, dino in [("texture", False), ("dino", True)]:
        for f in range(3):
            head = learn([r for r in rows if r["fold"] != f], dino)
            for r in rows:
                if r["fold"] == f:
                    r["raw"] = rank(r, head, dino)
        profiles = operating_points(rows)
        p = OUT / f"{name}_head.pt"
        torch.save(learn(rows, dino).state_dict(), p)
        atomic_json(
            OUT / f"{name}_frozen.json",
            dict(profiles=profiles, sha256=sha(p), encoder_sha256=META_SHA),
        )
        print(name, profiles, flush=True)


def evaluate():
    raw = read(OLD / "phase1/source_predictions.json")
    dev = collect(raw, manifest(), encoder(), "dev")
    assert len(dev) == 113
    for name, dino in [("texture", False), ("dino", True)]:
        cfg = read(OUT / f"{name}_frozen.json")
        p = OUT / f"{name}_head.pt"
        assert sha(p) == cfg["sha256"]
        state = torch.load(p, weights_only=True, map_location="cpu")
        head = Head(len(state["mean"]))
        head.load_state_dict(state)
        head.eval()
        for r in dev:
            r["raw"] = rank(r, head, dino)
        low = [
            dict(image_id=r["image_id"], GT=r["GT"], pred=r["raw"].tolist())
            for r in dev
        ]
        atomic_json(OUT / f"{name}_low_predictions.json", low)
        res = {}
        for mode, point in cfg["profiles"].items():
            rr = [
                dict(
                    image_id=r["image_id"],
                    GT=r["GT"],
                    pred=cut(r["raw"], point["thresholds"]).tolist(),
                )
                for r in dev
            ]
            met = metrics(rr)
            res[mode] = {k: met[k] for k in ["total", "class"]}
            atomic_json(OUT / f"{name}_{mode}_predictions.json", rr)
        atomic_json(
            OUT / f"{name}_metrics.json",
            dict(frozen=res, dev_grid_diagnostic=operating_points(dev)),
        )
        print(name, {k: v["total"] for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("action", choices=["train", "evaluate"])
    args = a.parse_args()
    configure(OUT, "cpu")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.action == "train":
        train()
    else:
        evaluate()
