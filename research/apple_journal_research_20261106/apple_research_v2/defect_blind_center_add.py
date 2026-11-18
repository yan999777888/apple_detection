"""Train a new defect center predictor with reviewed-point MIL on frozen parent features."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import batched_nms

from audit_defect_causes_20261112 import ious, match
from class0_patch_refinement import ROOT, atomic_json, configure, load_image, read, sha
from defect_blind_review_pack import MANIFEST, OUT as PACK, native_gt
from defect_blind_review_update import SOURCE_SHA, base_network, replay
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect_blind_center_add_v1"
REVIEW = PACK / "核对结果/cases_audit.json"
SEED = 68186
EPOCHS = 20
STRIDE = 8
SIDE = 768
GRID = SIDE // STRIDE
CUTS = [0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]


def split_training():
    audit = read(REVIEW)
    src = [r for r in read(MANIFEST)["samples"] if r["partition"] == "train"]
    reviewed = {r["image_id"] for r in audit["rows"]}
    minute = {r["image_id"]: r["same_second_group"][:-2] for r in src}
    exclude = {minute[i] for i in reviewed}
    available = sorted(
        set(minute.values()) - exclude,
        key=lambda k: hashlib.sha256(f"{SEED}:{k}".encode()).digest(),
    )
    selected, count = set(), 0
    freq = Counter(minute.values())
    for group in available:
        selected.add(group)
        count += freq[group]
        if count >= 110:
            break
    folds = {
        r["image_id"]: ("screen" if minute[r["image_id"]] in selected else "fit")
        for r in src
    }
    if not (count >= 110 and all(folds[i] == "fit" for i in reviewed)):
        raise ValueError("Invalid training-date screen split")
    return src, folds, audit


def freeze():
    src, folds, audit = split_training()
    exports = list((PACK / "盲审材料/导出结果").glob("*.json"))
    prior = read(ROOT / "experiments/defect_blind_review_update_v1/frozen_review.json")
    if audit["export_sha256"] != prior["export_sha256"] or {
        sha(p) for p in exports
    } != {prior["export_sha256"]}:
        raise ValueError(
            "The previous paired study's locked export differs, or another export has arrived"
        )
    payload = dict(
        seed=SEED,
        epochs=EPOCHS,
        parent_scratch_sha=SOURCE_SHA,
        export_sha256=audit["export_sha256"],
        manifest_sha256=sha(MANIFEST),
        split=folds,
        split_counts=dict(Counter(folds.values())),
        class_order=["scratch", "insect"],
        scope="Recall-only ablation. The branch cannot remove the parent detector's false positives; even 230 TP with all 21 original FP retained would give only 91.63% precision.",
        heldout="minute-group training-date screen, unseen by the new center model. Frozen parent saw all 596 images.",
        point_training="Human class and marked central point only. Use a 3x3 spatial bag. No candidate box or learned proposal enters center or size regression.",
        native_training="Native positive centers, their nearest four geometrical neighbors; geometry from native boxes only. Sparse negative sampling from smooth bright surfaces or dark plain background outside original boxes, held-out OOF detections and reviewed locations.",
        ablation="Same initialization, source, sampling, native targets and 20 epochs. Reviewed region MIL active only in review arm.",
        fusion="Keep all frozen original workpoint detections. Suppress center candidates overlapping any original output IoU>=.30 or whose center falls within an original output. Within the center stream class-aware NMS .45.",
        thresholds=CUTS,
        conditional_validation="Try review-arm candidates from high to low on training-date screen. At least two net new matched native GT on two images, FP increase <= TP increase, no lost native hits, P loss <= 0.01. Rank by new TP then fewer FP then higher cutoff. Use the same review-selected cutoff for the control. Run development only if a cutoff passes.",
        benchmark="600 frame serial stage+defect benchmark only for accepted review and control, identical original protocol.",
        training_sources=sorted({r["capture_filename_tokens"]["date"] for r in src}),
    )
    path = OUT / "protocol.json"
    if path.exists() and read(path) != payload:
        raise ValueError("Existing frozen protocol differs")
    atomic_json(path, payload)
    return src, folds, audit


def maps(det, image):
    rgb, (dx, dy, sx, sy) = replay.original.square(image)
    x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255
    fine, levels = det.locked.features(x)
    pest, scratch = det.locked.detections(levels), det.scratch(fine, levels)
    first = levels[0]
    pooled = torch.cat(
        [
            F.avg_pool2d(fine[:, ::4], 2),
            first,
            F.avg_pool2d(first[:, ::4], 5, stride=1, padding=2),
            F.interpolate(levels[1][:, ::4], size=first.shape[-2:], mode="nearest"),
        ],
        1,
    )
    arr = []
    h, w = image.shape[:2]
    for raw, cls in ((pest, 1), (scratch, 0)):
        p = raw[0]
        p = p[p[:, 4] >= 0.001].cpu().numpy()
        p[:, [0, 2]] = ((p[:, [0, 2]] - dx) / sx).clip(0, w)
        p[:, [1, 3]] = ((p[:, [1, 3]] - dy) / sy).clip(0, h)
        p = replay.original.nms(p)
        arr.append(p[(p[:, 5] == cls) & (p[:, 4] >= replay.original.PRIMARY[cls])])
    parent = np.concatenate(arr)
    return (
        pooled[0].permute(1, 2, 0).contiguous().half(),
        parent[np.argsort(-parent[:, 4], kind="stable")],
        (dx, dy, sx, sy),
    )


def descriptors(targets, srcrow, oofrow, weak, im, grid, rng):
    """Select native center training targets and independent background support."""
    assert im.shape[:2] == (1024, 1024)
    features, classes, geometry, geometry_classes = [], [], [], []
    grid = grid.cpu()
    gt = native_gt(srcrow)
    allowed = np.ones((GRID, GRID), dtype=bool)

    def exclude(box, margin):
        l, t, r, b = np.asarray(box) * GRID / 1024
        allowed[
            max(0, int(t) - margin) : min(GRID, int(b) + margin + 1),
            max(0, int(l) - margin) : min(GRID, int(r) + margin + 1),
        ] = False

    for g in gt:
        exclude(g["box"], 2)
    for p in oofrow:
        if p[4] >= 0.05:
            exclude(p[:4], 1)
    if weak:
        cx, cy = np.asarray(weak["center_xy"]) * GRID / 1024
        exclude([weak["center_xy"][0], weak["center_xy"][1]] * 2, 4)
        offsets = [
            (np.clip(int(cy) + dy, 0, GRID - 1), np.clip(int(cx) + dx, 0, GRID - 1))
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        targets["bags"].append(torch.stack([grid[y, x] for y, x in offsets]))
        targets["answers"].append(
            {"other": -1, "scratch": 0, "insect": 1}[weak["manual"]]
        )
    for g in gt:
        l, t, r, b = np.asarray(g["box"]) * GRID / 1024
        cx, cy = (l + r) * 0.5, (t + b) * 0.5
        c = g["class"]
        for j, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]):
            x, y = int(cx) + dx, int(cy) + dy
            if not (0 <= x < GRID and 0 <= y < GRID):
                continue
            # Neighboring cells can regress to the same physical defect. Only
            # the central cell is interpreted as a class-positive center.
            features.append(grid[y, x])
            classes.append(c if j == 0 else -2)
            geometry.append(
                [cx - x, cy - y, np.log(max(0.25, r - l)), np.log(max(0.25, b - t))]
            )
            geometry_classes.append(c)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    avg = cv2.resize(gray, (GRID, GRID), interpolation=cv2.INTER_AREA)
    sq = cv2.resize(gray * gray, (GRID, GRID), interpolation=cv2.INTER_AREA)
    smooth = np.sqrt(np.maximum(0, sq - avg**2)) < 4
    hsv = cv2.resize(
        cv2.cvtColor(im, cv2.COLOR_BGR2HSV), (GRID, GRID), interpolation=cv2.INTER_AREA
    )
    easy = smooth & ((avg < 24) | ((avg > 80) & (hsv[:, :, 1] > 22)))
    y, x = np.where(allowed & easy)
    for k in rng.choice(len(x), min(28, len(x)), replace=False):
        features.append(grid[y[k], x[k]])
        classes.append(-1)
        geometry.append([0.0] * 4)
        geometry_classes.append(-1)
    if features:
        targets["features"].append(torch.stack(features))
        targets["labels"].extend(classes)
        targets["box"].extend(geometry)
        targets["geom_classes"].extend(geometry_classes)


@torch.no_grad()
def cache():
    src, folds, audit = freeze()
    if (OUT / "train.pt").exists():
        raise ValueError("Feature cache exists")
    weak = {
        r["image_id"]: r
        for r in audit["rows"]
        if r["kind"] == "unmatched" and r["manual"] in ("insect", "scratch", "other")
    }
    oof = {}
    for i in range(3):
        folder = (
            ROOT
            / "experiments/defect_algorithm_steps_20261112_v2/oof_v2"
            / f"fold{i}"
            / "predictions.json"
        )
        oof.update({r["image_id"]: r["pred"] for r in read(folder)["rows"]})
    det = base_network()
    rng = np.random.default_rng(SEED)
    targets = defaultdict(list)
    screen = []
    for r in src:
        name = r["image_id"]
        im = load_image(r["rgb_path"])
        grid, parent, trans = maps(det, im)
        assert tuple(grid.shape[:2]) == (GRID, GRID) and trans == (0, 0, 0.75, 0.75)
        if folds[name] == "fit":
            descriptors(targets, r, oof[name], weak.get(name), im, grid, rng)
        else:
            screen.append(
                dict(
                    image_id=name,
                    parent=parent.tolist(),
                    gt=native_gt(r),
                    grid=grid.cpu(),
                )
            )
    x = dict(
        features=torch.cat(targets["features"]),
        labels=torch.tensor(targets["labels"]),
        box=torch.tensor(targets["box"], dtype=torch.float32),
        geom_classes=torch.tensor(targets["geom_classes"]),
        bags=torch.stack(targets["bags"]),
        answers=torch.tensor(targets["answers"]),
    )
    replay.atomic_latest(OUT / "train.pt", x)
    replay.atomic_latest(OUT / "screen.pt", screen)
    atomic_json(
        OUT / "cache.json",
        dict(
            features=list(x["features"].shape),
            native_centers=int((x["labels"] >= 0).sum()),
            geometry_rows=int((x["geom_classes"] >= 0).sum()),
            weak_counts=dict(Counter(targets["answers"])),
            screen_images=len(screen),
            screen_gt=sum(len(v["gt"]) for v in screen),
            train_sha=sha(OUT / "train.pt"),
            screen_sha=sha(OUT / "screen.pt"),
        ),
    )


class CenterHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
        )
        self.logits = nn.Linear(64, 2)
        self.geometry = nn.Linear(64, 8)
        with torch.no_grad():
            self.geometry.bias.copy_(torch.tensor([0.5, 0.5, 1.9, 1.9] * 2))

    def forward(self, features):
        z = self.trunk(features.float())
        return self.logits(z), self.geometry(z).reshape(*z.shape[:-1], 2, 4)


def fit():
    freeze()
    cache_info = read(OUT / "cache.json")
    assert sha(OUT / "train.pt") == cache_info["train_sha"]
    data = torch.load(OUT / "train.pt", map_location="cuda", weights_only=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    init = CenterHead(data["features"].shape[1]).cuda()
    models = {tag: deepcopy(init) for tag in ("control", "review")}
    optim = {
        tag: torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
        for tag, net in models.items()
    }
    train = data["features"]
    labels = data["labels"]
    gclass = data["geom_classes"]
    native = torch.where(labels >= 0)[0].cpu().numpy()
    negatives = torch.where(labels == -1)[0].cpu().numpy()
    geom = torch.where(gclass >= 0)[0].cpu().numpy()
    hist = defaultdict(list)
    steps = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(native)
        batches = np.array_split(order, max(1, len(order) // 48))
        loss_rows = defaultdict(list)
        for pos in batches:
            neg = rng.choice(negatives, len(pos) * 4, replace=False)
            selected = np.r_[pos, neg]
            gi = rng.choice(geom, len(pos) * 4, replace=False)
            t = (
                F.one_hot(labels[selected].clamp_min(0), 2).float()
                * (labels[selected] >= 0)[:, None]
            )
            for name, net in models.items():
                score, reg = net(train[np.r_[selected, gi]])
                logits = score[: len(selected)]
                p = logits.sigmoid()
                bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
                focal = torch.where(t.bool(), 1 - p, p).square()
                loss = (bce * focal).mean() + 0.4 * F.smooth_l1_loss(
                    reg[len(selected) :][
                        torch.arange(len(gi), device="cuda"), gclass[gi]
                    ],
                    data["box"][gi],
                    beta=0.25,
                )
                if name == "review":
                    manual = data["answers"]
                    weak_score = net(data["bags"])[0].amax(1)
                    # Max-over-neighborhood supervision asks only whether a
                    # lesion is present there. No weak geometry gradients.
                    weak_target = (
                        F.one_hot(manual.clamp_min(0), 2).float()
                        * (manual >= 0)[:, None]
                    )
                    grouped = F.binary_cross_entropy_with_logits(
                        weak_score, weak_target, reduction="none"
                    ).mean(1)
                    loss += (
                        0.15
                        * torch.stack(
                            [grouped[manual == c].mean() for c in (-1, 0, 1)]
                        ).mean()
                    )
                optim[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), 5.0, error_if_nonfinite=True
                )
                optim[name].step()
                loss_rows[name].append(float(loss.detach()))
            steps += 1
        for name in models:
            hist[name].append(float(np.mean(loss_rows[name])))
    for name, net in models.items():
        where = OUT / name
        where.mkdir(exist_ok=True)
        replay.atomic_latest(where / "weights.pt", net.cpu().state_dict())
        atomic_json(
            where / "receipt.json",
            dict(
                epochs=EPOCHS,
                updates=steps,
                weight_sha=sha(where / "weights.pt"),
                feature_dim=train.shape[1],
                loss=hist[name],
                cache_sha=cache_info["train_sha"],
            ),
        )


@torch.no_grad()
def decode(net, grid, trans, hw):
    logits, geom = net(grid)
    scores = logits.sigmoid().permute(2, 0, 1).unsqueeze(0)
    peaks = scores * (scores == F.max_pool2d(scores, 5, 1, 2))
    scores, idx = peaks.flatten().topk(64)
    keep = scores >= min(CUTS)
    scores, idx = scores[keep], idx[keep]
    c = idx // (GRID * GRID)
    y = (idx // GRID) % GRID
    x = idx % GRID
    reg = geom[y, x, c]
    centers = (torch.stack((x, y), 1) + reg[:, :2].clamp(-1.5, 2.5)) * STRIDE
    wh = reg[:, 2:].clamp(-1, 4.5).exp() * STRIDE
    dx, dy, sx, sy = trans
    h, w = hw
    b = torch.cat([centers - wh / 2, centers + wh / 2], 1)
    b = (b - b.new_tensor([dx, dy, dx, dy])) / b.new_tensor([sx, sy, sx, sy])
    b[:, ::2].clamp_(0, w)
    b[:, 1::2].clamp_(0, h)
    valid = (b[:, 2:] - b[:, :2]).amin(1) > 0
    b, c, scores = b[valid], c[valid], scores[valid]
    ids = batched_nms(b, scores, c, 0.45)
    return torch.cat([b[ids], scores[ids, None], c[ids, None]], 1).cpu().numpy()


def attach(parent, new, cut):
    original = np.asarray(parent, np.float32).reshape(-1, 6)
    additions = new[new[:, 4] >= cut]
    if len(original) and len(additions):
        centers = (additions[:, :2] + additions[:, 2:4]) / 2
        inside = (
            (
                (centers[:, None] >= original[None, :, :2])
                & (centers[:, None] <= original[None, :, 2:4])
            )
            .all(2)
            .any(1)
        )
        keep = (ious(additions[:, :4], original[:, :4]).max(1) < 0.30) & ~inside
        additions = additions[keep]
    return np.concatenate((original, additions))


def model(name):
    where = OUT / name
    receipt = read(where / "receipt.json")
    assert sha(where / "weights.pt") == receipt["weight_sha"]
    m = CenterHead(receipt["feature_dim"]).cuda()
    m.load_state_dict(
        torch.load(where / "weights.pt", map_location="cpu", weights_only=True)
    )
    return m.eval()


@torch.no_grad()
def screen():
    assert sha(OUT / "screen.pt") == read(OUT / "cache.json")["screen_sha"]
    items = torch.load(OUT / "screen.pt", map_location="cpu", weights_only=True)
    base_rows = [
        dict(image_id=v["image_id"], pred=v["parent"], GT=v["gt"]) for v in items
    ]

    def hits(rows):
        return {
            (r["image_id"], j)
            for r in rows
            for j in match(r["pred"], r["GT"])[0].values()
        }

    base = metrics(base_rows)
    base_hits = hits(base_rows)
    tested = defaultdict(list)
    for name in ("control", "review"):
        net = model(name)
        new = [
            decode(net, r["grid"].cuda(), (0, 0, 0.75, 0.75), (1024, 1024))
            for r in items
        ]
        for cut in CUTS:
            rows = [
                dict(r, pred=attach(r["pred"], pred, cut).tolist())
                for r, pred in zip(base_rows, new, strict=True)
            ]
            result = metrics(rows)
            got = hits(rows)
            extra = result["total"]["TP"] - base["total"]["TP"]
            added_fp = result["total"]["FP"] - base["total"]["FP"]
            gained = sorted(got - base_hits)
            lost = sorted(base_hits - got)
            acceptable = (
                extra >= 2
                and len({k for k, _ in gained}) >= 2
                and not lost
                and added_fp <= extra
                and result["total"]["P"] >= base["total"]["P"] - 0.01
            )
            tested[name].append(
                dict(
                    cut=cut,
                    metrics=dict(total=result["total"], by_class=result["class"]),
                    gained=gained,
                    lost=lost,
                    added_fp=added_fp,
                    extra_tp=extra,
                    pass_gate=acceptable,
                )
            )
    passing = [r for r in tested["review"] if r["pass_gate"]]
    best = (
        max(passing, key=lambda r: (r["extra_tp"], -r["added_fp"], r["cut"]))
        if passing
        else None
    )
    out = dict(
        parent={"total": base["total"], "class": base["class"]},
        screens=tested,
        selected=best["cut"] if best else None,
    )
    atomic_json(OUT / "gate.json", out)
    print(
        "Original",
        base["total"],
        "review",
        [(r["cut"], r["extra_tp"], r["added_fp"]) for r in tested["review"]],
        "selected",
        out["selected"],
        flush=True,
    )


def evaluate(name):
    threshold = read(OUT / "gate.json")["selected"]
    if threshold is None:
        print("No training-date candidate passed the frozen screen")
        return
    m = model(name)

    @torch.inference_mode()
    def predict(det, im):
        features, base, trans = maps(det, im)
        return attach(base, decode(m, features, trans, im.shape[:2]), threshold)

    original = replay.original
    old = original.deploy_model, original.boxes
    try:
        original.deploy_model, original.boxes = base_network, predict
        with replay.evaluation_output(OUT / name):
            original.evaluate()
    finally:
        original.deploy_model, original.boxes = old


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "action", choices=["freeze", "cache", "fit", "screen", "control", "review"]
    )
    action = ap.parse_args().action
    configure(OUT, "cpu" if action == "freeze" else "cuda")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    with replay.exclusive_run(OUT):
        if action in ("control", "review"):
            evaluate(action)
        else:
            {"freeze": freeze, "cache": cache, "fit": fit, "screen": screen}[action]()
