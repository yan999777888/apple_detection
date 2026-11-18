"""Shared image features for local defect verification and image-level presence supervision."""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from defect_precision92 import RGBFeatures, proposals, cut
from class0_patch_refinement import (
    ROOT,
    OLD,
    manifest,
    configure,
    atomic_json,
    load_image,
    read,
    sha,
    CP,
)
from audit_defect_causes_20261112 import match, ious
from defect_algorithm_modes_20261112 import nms
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_scene"


class SceneFeatures(RGBFeatures):
    @torch.inference_mode()
    def forward(self, im, boxes):
        import cv2

        canvas = cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA)
        x = (
            torch.from_numpy(canvas[:, :, ::-1].copy())
            .permute(2, 0, 1)[None]
            .to(next(self.parameters()).device)
            .float()
            / 255
        )
        b = torch.as_tensor(boxes[:, :4], device=x.device).clone()
        b[:, [0, 2]] *= 384 / im.shape[1]
        b[:, [1, 3]] *= 384 / im.shape[0]
        size = (b[:, 2:] - b[:, :2]).clamp_min(2)
        wide = torch.cat([b[:, :2] - 0.45 * size, b[:, 2:] + 0.45 * size], 1).clamp(
            0, 384
        )
        whole = [F.adaptive_avg_pool2d(x, 3).flatten()]
        local = []
        for i, m in enumerate(self.trunk):
            x = m(x)
            if i not in (2, 4, 6):
                continue
            whole.append(F.adaptive_avg_pool2d(x, 2).flatten())
            for region in (b, wide):
                pooled = roi_align(
                    x,
                    [region],
                    (3, 3),
                    spatial_scale=x.shape[-1] / 384,
                    aligned=True,
                    sampling_ratio=2,
                )
                local.append(pooled.flatten(1))
        return torch.cat(local, 1).cpu(), torch.cat(whole).cpu()


def prepare(src, im, enc, gt=None, rng=None):
    src = {
        k: np.asarray(src[k], np.float32).reshape(-1, 6) for k in ("o2o", "o2m", "flip")
    }
    b = proposals(src)
    n = len(b)
    if gt:
        extras = []
        for g in gt:
            box = np.array(g["box"])
            size = box[2:] - box[:2]
            d = np.tile(size, 2) * rng.uniform(-0.16, 0.16, 4)
            extras.append(
                [*(box + d).clip(0, [im.shape[1], im.shape[0]] * 2), 0, g["class"]]
            )
        b = np.concatenate((b, np.asarray(extras, np.float32)))
    attrs = []
    for pool in src.values():
        for c in (0, 1):
            p = pool[pool[:, 5] == c]
            col = np.zeros(len(b))
            if len(b) and len(p):
                col = (ious(b[:, :4], p[:, :4]) ** 2 * p[None, :, 4]).max(1)
            attrs.append(col)
    attrs = np.stack(attrs, axis=1)
    local, scene = enc(im, b)
    wh = np.maximum(1, b[:, 2:4] - b[:, :2])
    x = np.column_stack((attrs, np.log(wh / [im.shape[1], im.shape[0]]), local.numpy()))
    return b, torch.as_tensor(x, dtype=torch.float32), scene, n


class SceneVerifier(nn.Module):
    def __init__(self, xd, sd):
        super().__init__()
        self.register_buffer("xm", torch.zeros(xd))
        self.register_buffer("xs", torch.ones(xd))
        self.register_buffer("sm", torch.zeros(sd))
        self.register_buffer("ss", torch.ones(sd))
        self.local = nn.Sequential(
            nn.Linear(xd, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.3)
        )
        self.global_branch = nn.Sequential(
            nn.Linear(sd, 48), nn.LayerNorm(48), nn.SiLU(), nn.Dropout(0.35)
        )
        self.present = nn.Linear(48, 2)
        self.joint = nn.Sequential(nn.Linear(144, 64), nn.LayerNorm(64), nn.SiLU())
        self.classifier = nn.Linear(64, 3)
        self.reg = nn.Linear(64, 4)

    def forward(self, x, s, idx):
        frame = self.global_branch(((s - self.sm) / self.ss).clamp(-5, 5))
        feat = self.local(((x - self.xm) / self.xs).clamp(-5, 5))
        z = self.joint(torch.cat((feat, frame[idx]), 1))
        return self.classifier(z), self.reg(z), self.present(frame)


def learn(rows):
    torch.manual_seed(9485)
    xt = []
    sc = []
    ys = []
    ds = []
    pres = []
    idx = []
    for i, r in enumerate(rows):
        b, x, s, n = r["data"]
        g = r["GT"]
        xt.append(x)
        sc.append(s)
        ov = ious(b[:, :4], [t["box"] for t in g])
        y = np.full(len(b), 2)
        d = np.zeros((len(b), 4))
        if len(b) and g:
            j = ov.argmax(1)
            pos = ov.max(1) >= 0.45
            y[pos] = np.array([t["class"] for t in g])[j[pos]]
            size = np.tile(np.maximum(b[:, 2:4] - b[:, :2], 1), (1, 2))
            d = ((np.array([t["box"] for t in g])[j] - b[:, :4]) / size).clip(-0.5, 0.5)
        ys.extend(y)
        ds.extend(d)
        idx.extend([i] * len(b))
        pres.append([any(t["class"] == c for t in g) for c in (0, 1)])
    x = torch.cat(xt)
    s = torch.stack(sc)
    idx = torch.tensor(idx)
    y = torch.tensor(ys)
    d = torch.as_tensor(np.asarray(ds), dtype=torch.float32)
    pres = torch.tensor(pres, dtype=torch.float32)
    model = SceneVerifier(x.shape[1], s.shape[1])
    model.xm.copy_(x.mean(0))
    model.xs.copy_(x.std(0).clamp_min(0.03))
    model.sm.copy_(s.mean(0))
    model.ss.copy_(s.std(0).clamp_min(0.03))
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.015)
    for epoch in range(20):
        for ii in torch.randperm(len(y)).split(128):
            scenes, inv = idx[ii].unique(return_inverse=True)
            cls, offset, p = model(x[ii], s[scenes], inv)
            loss = F.cross_entropy(
                cls, y[ii], label_smoothing=0.035
            ) + 0.32 * F.binary_cross_entropy_with_logits(p, pres[scenes])
            pos = y[ii] != 2
            if pos.any():
                loss += F.smooth_l1_loss(offset[pos], d[ii][pos], beta=0.12)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    return model.eval()


@torch.inference_mode()
def rank(data, model, hw):
    b, x, s, n = data
    if not n:
        return np.empty((0, 6), np.float32)
    b = b[:n].copy()
    x = x[:n]
    score, delta, present = model(x, s[None], torch.zeros(n, dtype=torch.long))
    q = score.softmax(1).numpy()[:, :2].clip(0.001, 0.999)
    support = x[:, :6].numpy().reshape(n, 3, 2).max(1).clip(0.001, 0.999)
    logits = 0.72 * np.log(q / (1 - q)) + 0.28 * np.log(support / (1 - support))
    c = logits.argmax(1)
    b[:, 5] = c
    b[:, 4] = 1 / (1 + np.exp(-logits[np.arange(n), c]))
    b[:, :4] += (
        0.5
        * delta.numpy().clip(-0.5, 0.5)
        * np.tile(np.maximum(1, b[:, 2:4] - b[:, :2]), (1, 2))
    )
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, hw[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, hw[0])
    return nms(b)


def policy(rows):
    grid = np.linspace(0.001, 0.999, 160)
    counter = np.zeros((2, len(grid), 2), int)
    for r in rows:
        p = r["oof"]
        hits, _ = match(p, r["GT"])
        for i, b in enumerate(p):
            counter[int(b[5]), grid <= b[4], int(i not in hits)] += 1
    tp = counter[0, :, None, 0] + counter[1, None, :, 0]
    fp = counter[0, :, None, 1] + counter[1, None, :, 1]
    pr = tp / np.maximum(1, tp + fp)
    re = tp / sum(len(r["GT"]) for r in rows)
    res = {}
    for name, valid, obj in (
        ("joint", tp >= 0, np.minimum(pr / 0.94, re / 0.85)),
        ("recall85", re >= 0.85, pr),
        ("precision94", pr >= 0.94, re),
    ):
        valid = valid & (counter[0, :, None, 0] > 0) & (counter[1, None, :, 0] > 0)
        if not valid.any():
            continue
        i, j = max(zip(*np.where(valid)), key=lambda z: (obj[z], tp[z], pr[z]))
        res[name] = dict(
            thresholds=[float(grid[i]), float(grid[j])],
            P=float(pr[i, j]),
            R=float(re[i, j]),
            TP=int(tp[i, j]),
            FP=int(fp[i, j]),
        )
    return res


def run_cpu():
    lookup = manifest()
    enc = SceneFeatures().eval()
    rows = []
    rng = np.random.default_rng(9485)
    for f in range(3):
        for r in read(OLD / f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"]:
            im = load_image(lookup[r["image_id"]]["rgb_path"])
            assert lookup[r["image_id"]]["partition"] == "train"
            rows.append(
                dict(
                    fold=f,
                    image_id=r["image_id"],
                    GT=r["GT"],
                    data=prepare(r["sources"], im, enc, r["GT"], rng),
                    hw=im.shape[:2],
                )
            )
    assert len({r["image_id"] for r in rows}) == 596
    for f in range(3):
        model = learn([r for r in rows if r["fold"] != f])
        for r in rows:
            if r["fold"] == f:
                r["oof"] = rank(r["data"], model, r["hw"])
    policies = policy(rows)
    model = learn(rows)
    torch.save(model.state_dict(), OUT / "model.pt")
    atomic_json(
        OUT / "frozen.json",
        dict(policies=policies, sha=sha(OUT / "model.pt"), base=sha(CP)),
    )
    dev = []
    for r in read(OLD / "phase1/source_predictions.json"):
        im = load_image(lookup[r["image_id"]]["rgb_path"])
        assert lookup[r["image_id"]]["partition"] == "dev"
        b = rank(prepare(r["sources"], im, enc), model, im.shape[:2])
        dev.append(dict(image_id=r["image_id"], GT=r["GT"], pred=b.tolist()))
    atomic_json(OUT / "low_predictions.json", dev)
    results = {}
    for mode, p in policies.items():
        rr = [
            dict(
                image_id=r["image_id"],
                GT=r["GT"],
                pred=cut(np.array(r["pred"]).reshape(-1, 6), p["thresholds"]).tolist(),
            )
            for r in dev
        ]
        atomic_json(OUT / f"{mode}_predictions.json", rr)
        results[mode] = metrics(rr)
    atomic_json(OUT / "metrics.json", results)
    print({k: v["total"] for k, v in results.items()})


if __name__ == "__main__":
    configure(OUT, "cpu")
    import cv2

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run_cpu()
