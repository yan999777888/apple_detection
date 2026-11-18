"""Fine-tune the last DINOv2 blocks on native-detail defect crops with same-frame negatives."""

import copy
import hashlib

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from class0_patch_refinement import ROOT, OLD, CP, configure, atomic_json, read, sha, load_image, manifest
from defect9485_repr import encoder, META_SHA
from defect9485_scene import policy
from audit_defect_causes_20261112 import ious
from defect_algorithm_modes_20261112 import nms as final_nms
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect_continue_metric"
SIDE = 140
GRID = SIDE // 14
FLOOR = 0.004
EPOCHS = 8
SEED = 4621
BLENDS = (0.35, 0.65, 0.9)


def status(phase, **info):
    atomic_json(OUT / "status.json", dict(phase=phase, **info))
    print(phase, info, flush=True)


def native_crops(im, boxes):
    crops = []
    sizes = []
    for box in boxes:
        wh = np.maximum(2, box[2:] - box[:2])
        center = (box[:2] + box[2:]) / 2
        extent = max(70, 1.7 * wh.max())
        ratio = SIDE / extent
        transform = np.float32([[ratio, 0, SIDE / 2 - ratio * center[0]],
                                [0, ratio, SIDE / 2 - ratio * center[1]]])
        crop = cv2.warpAffine(im, transform, (SIDE, SIDE), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
        crops.append(crop[:, :, ::-1])
        sizes.append(wh / extent)
    return np.asarray(crops, np.uint8), np.asarray(sizes, np.float32).reshape(-1, 2)


def sources_to_boxes(src, ground_truth, shape, rng):
    pools = [np.asarray(src[key], np.float32).reshape(-1, 6) for key in ("o2o", "flip", "o2m")]
    seed = np.concatenate(pools)
    seed = seed[(seed[:, 4] >= FLOOR) & ((seed[:, 2:4] - seed[:, :2]).min(1) > 1)]
    ids = nms(torch.from_numpy(seed[:, :4]), torch.from_numpy(seed[:, 4]), 0.65)[:48].numpy()
    boxes = seed[ids, :4]
    n = len(boxes)
    if ground_truth:
        gt = np.asarray([g["box"] for g in ground_truth], np.float32)
        size = np.tile(np.maximum(1, gt[:, 2:] - gt[:, :2]), (1, 2))
        jitter = gt + size * rng.uniform(-.16, .16, gt.shape)
        boxes = np.concatenate([boxes, gt, jitter.clip(0, [shape[1], shape[0]] * 2)]).astype(np.float32)
    columns = []
    for pool in pools:
        for c in (0, 1):
            candidates = pool[pool[:, 5] == c]
            conf = np.zeros(len(boxes), np.float32)
            if len(candidates) and len(boxes):
                conf = (ious(boxes, candidates[:, :4]) ** 2 * candidates[None, :, 4]).max(1)
            columns.append(conf)
    scores = np.stack(columns, axis=1).astype(np.float32)
    return boxes, scores, n


@torch.no_grad()
def prefix(enc, crops):
    device = enc.cls_token.device
    x = torch.as_tensor(crops, device=device).permute(0, 3, 1, 2).float().div(255)
    x = enc.patch_embed["proj"]((x - enc.rgb_mean) / enc.rgb_std)
    spatial = enc.pos_embed[:, 1:].reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
    pos = F.interpolate(spatial, scale_factor=((GRID + .1) / 37,) * 2, mode="bicubic", align_corners=False)
    x = torch.cat([(enc.cls_token + enc.pos_embed[:, :1]).expand(len(x), -1, -1),
                   (x + pos).flatten(2).transpose(1, 2)], 1)
    for block in enc.blocks[:10]:
        x = block(x)
    return x.to(torch.float16)


class Tail(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.blocks = copy.deepcopy(enc.blocks[10:]).requires_grad_(True)
        self.norm = copy.deepcopy(enc.norm).requires_grad_(True)
        self.visual = nn.Sequential(nn.LayerNorm(384 * 3), nn.Linear(384 * 3, 192),
                                    nn.LayerNorm(192), nn.GELU(), nn.Dropout(.21))
        self.detector = nn.Sequential(nn.Linear(10, 48), nn.LayerNorm(48), nn.GELU())
        self.cls = nn.Linear(240, 3)
        self.box = nn.Linear(240, 4)
        nn.init.zeros_(self.box.weight)
        nn.init.zeros_(self.box.bias)
        self.register_buffer("grid", (torch.arange(GRID) + .5) / GRID - .5)

    def forward(self, tokens, attr):
        x = tokens.float()
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        feat = x[:, 1:].reshape(len(x), GRID, GRID, -1)
        radius = (.5 * attr[:, 6:8]).clamp_min(.07)
        dx = self.grid[None, None, :] / radius[:, 0, None, None]
        dy = self.grid[None, :, None] / radius[:, 1, None, None]
        center = torch.exp(-1.5 * (dx.square() + dy.square()))
        center = center / center.sum((1, 2), keepdim=True).clamp_min(.001)
        core = (feat * center[..., None]).sum((1, 2))
        style = self.visual(torch.cat([x[:, 0], core, core - feat.mean((1, 2))], 1))
        attr = torch.cat([torch.logit(attr[:, :6].clamp(.001, .999)) / 4, attr[:, 6:]], 1)
        fused = torch.cat([style, self.detector(attr)], 1)
        return self.cls(fused).float(), .55 * self.box(fused).tanh().float()


def collect(raw, lookup, enc, train):
    rows, buffers, xy = [], [], []
    rng = np.random.default_rng(SEED)
    def flush():
        if buffers:
            batch = np.concatenate(buffers)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for lo in range(0, len(batch), 40):
                    xy.append(prefix(enc, batch[lo:lo + 40]).cpu())
            buffers.clear()
    offset = 0
    for idx, (fold, row) in enumerate(raw):
        item = lookup[row["image_id"]]
        assert item["partition"] == ("train" if train else "dev")
        im = load_image(item["rgb_path"])
        boxes, scores, n = sources_to_boxes(row["sources"], row["GT"] if train else (), im.shape, rng)
        crops, frac = native_crops(im, boxes)
        wh = np.maximum(1, boxes[:, 2:] - boxes[:, :2])
        attrs = np.column_stack([scores, frac, np.log(wh / 64)]).astype(np.float32)
        valid = np.ones(len(boxes), bool)
        cls = np.full(len(boxes), 2, np.int64)
        delta = np.zeros((len(boxes), 4), np.float32)
        if train and row["GT"] and len(boxes):
            gt = np.asarray([g["box"] for g in row["GT"]], np.float32)
            ov = ious(boxes, gt)
            close = ov.argmax(1)
            best = ov.max(1)
            pos = best >= .42
            valid = pos | (best < .20)
            cls[pos] = np.array([g["class"] for g in row["GT"]])[close[pos]]
            delta = ((gt[close] - boxes) / np.tile(wh, (1, 2))).clip(-.55, .55).astype(np.float32)
        if len(crops):
            buffers.append(crops)
        rows.append(dict(image_id=row["image_id"], GT=row["GT"], boxes=boxes,
                         attrs=attrs, n=n, fold=fold, hw=im.shape[:2], offset=offset,
                         valid=valid, cls=cls, delta=delta))
        offset += len(boxes)
        if sum(map(len, buffers)) >= 96 or idx + 1 == len(raw):
            flush()
        if (idx + 1) % 100 == 0:
            status("prefix_train" if train else "prefix_dev", images=idx+1, crops=offset)
    x = torch.cat(xy)
    assert len(x) == offset
    return rows, x


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def learn(rows, tokens, enc, excluded):
    selected = [r for r in rows if r["fold"] != excluded]
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = Tail(enc).cuda().train()
    initial_sha = tensor_sha(model.blocks[1].attn["qkv"].weight)
    optimizer = torch.optim.AdamW([
        {"params": list(model.blocks.parameters()) + list(model.norm.parameters()), "lr": 3e-5},
        {"params": list(model.visual.parameters()) + list(model.detector.parameters()) +
                   list(model.cls.parameters()) + list(model.box.parameters()), "lr": .0007}],
        weight_decay=.03)
    a = torch.from_numpy(np.concatenate([r["attrs"] for r in rows])).cuda()
    y = torch.from_numpy(np.concatenate([r["cls"] for r in rows])).cuda()
    d = torch.from_numpy(np.concatenate([r["delta"] for r in rows])).cuda()
    pos, neg, hard_pairs = [], [], []
    for r in selected:
        ids = r["offset"] + np.arange(len(r["cls"]))
        p = ids[r["valid"] & (r["cls"] != 2)]
        z = ids[r["valid"] & (r["cls"] == 2)]
        pos.extend(p)
        neg.extend(z)
        if len(p) and len(z):
            hard_pairs.append((p, z[:min(6, len(z))]))
    steps = max(24, int(np.ceil(len(selected) / 7)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps * EPOCHS, eta_min=2e-6)
    for ep in range(EPOCHS):
        mean_loss = 0.
        for step in range(steps):
            selected_pairs = [hard_pairs[i] for i in rng.integers(len(hard_pairs), size=10)]
            pairs = np.array([(rng.choice(p), rng.choice(z)) for p, z in selected_pairs])
            ids = np.r_[pairs[:, 0], pairs[:, 1], rng.choice(pos, 12), rng.choice(neg, 12)]
            batch = tokens[torch.as_tensor(ids)].cuda()
            target = y[ids]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, delta = model(batch, a[ids])
            positive = target != 2
            label = target[:10]
            idx = torch.arange(10, device=target.device)
            pair_scores = logits[idx, label] - logits[:10, 2] - logits[idx+10, label] + logits[10:20, 2]
            loss = (F.cross_entropy(logits, target, label_smoothing=.035) +
                    .80 * F.smooth_l1_loss(delta[positive], d[ids][positive], beta=.10) +
                    .22 * F.softplus(1 - pair_scores).mean())
            if not loss.isfinite():
                raise RuntimeError("nonfinite objective")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            scheduler.step()
            mean_loss += loss.item()
        status("tail_training", excluded=excluded, epoch=ep+1, steps=steps, loss=round(mean_loss/steps, 5))
    final_sha = tensor_sha(model.blocks[1].attn["qkv"].weight)
    assert initial_sha != final_sha
    return model.eval(), dict(initial_qkv_sha=initial_sha, final_qkv_sha=final_sha,
                              optimizer_steps=steps*EPOCHS, train_images=len(selected))


@torch.inference_mode()
def infer(model, rows, tokens):
    out = []
    for r in rows:
        n = r["n"]
        results = {b: np.empty((0, 6), np.float32) for b in BLENDS}
        if n:
            attrs = torch.from_numpy(r["attrs"][:n]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, delta = model(tokens[r["offset"]:r["offset"]+n].cuda(), attrs)
            visual = logits.softmax(-1)[:, :2].cpu().numpy().clip(.00001, 1)
            priors = r["attrs"][:n, :6].reshape(n, 3, 2).max(1).clip(.0001, 1)
            boxes = r["boxes"][:n]
            size = np.tile(np.maximum(2, boxes[:, 2:] - boxes[:, :2]), (1, 2))
            boxes = (boxes + .75 * delta.cpu().numpy() * size).clip(0, [r["hw"][1], r["hw"][0]] * 2)
            for b in BLENDS:
                scores = visual**b * priors**(1-b)
                pred = np.concatenate([np.column_stack([boxes, scores[:, c], np.full(n, c)]) for c in (0, 1)])
                results[b] = final_nms(pred[pred[:, 4] >= .001].astype(np.float32))
        out.append(dict(image_id=r["image_id"], GT=r["GT"], raw=results))
    return out


def main():
    configure(OUT, "cuda:0")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    enc = encoder().cuda()
    lookup = manifest()
    raw = [(i, r) for i in range(3) for r in read(OLD / f"all_mode_OOF_calibration/fold{i}_sources.json")["rows"]]
    assert len({r["image_id"] for _, r in raw}) == 596
    cache = OUT / "prefix_train.pt"
    if cache.exists():
        rows, tokens = torch.load(cache, weights_only=False, map_location="cpu")
    else:
        rows, tokens = collect(raw, lookup, enc, True)
        torch.save((rows, tokens), cache)
    oof = []
    history = {}
    for f in range(3):
        model, history[f] = learn(rows, tokens, enc, f)
        pred = infer(model, [r for r in rows if r["fold"] == f], tokens)
        oof.extend(pred)
        del model
    candidates = {}
    for blend in BLENDS:
        candidates[blend] = policy([dict(GT=r["GT"], oof=r["raw"][blend]) for r in oof])
    frozen = {}
    for name in ("joint", "recall85", "precision94"):
        eligible = [(b, p[name]) for b, p in candidates.items() if name in p]
        if not eligible:
            continue
        def value(item):
            p = item[1]
            if name == "joint":
                return min(p["P"]/.94, p["R"]/.85), p["P"] * p["R"]
            return (p["P"], p["R"]) if name == "recall85" else (p["R"], p["P"])
        blend, opts = max(eligible, key=value)
        frozen[name] = dict(blend=blend, **opts)
    model, history["final"] = learn(rows, tokens, enc, None)
    torch.save(model.state_dict(), OUT / "tail.pt")
    atomic_json(OUT / "frozen.json", dict(points=frozen, scans=candidates, training=history,
                detector_sha=sha(CP), dino_sha=META_SHA, tail_sha=sha(OUT/"tail.pt"), benchmark_points="joint",
                feature="RGB 140px DINOv2-S/14 prefix 10 BF16, last two transformer blocks fine-tuned",
                source="cached 3 fold held-out detector proposals; only 596 train images for adaptation and calibration"))
    status("frozen", points=frozen)
    del tokens, rows
    raw_dev = [(None, r) for r in read(OLD / "phase1/source_predictions.json")]
    assert len(raw_dev) == 113
    dev_rows, dev_tok = collect(raw_dev, lookup, enc, False)
    dev = infer(model, dev_rows, dev_tok)
    for blend in BLENDS:
        raw_pred = [dict(image_id=r["image_id"], GT=r["GT"], pred=r["raw"][blend].tolist()) for r in dev]
        atomic_json(OUT / f"low_blend_{blend:.2f}.json", raw_pred)
    results = {}
    for name, wp in frozen.items():
        t = np.asarray(wp["thresholds"])
        preds = []
        for row in dev:
            p = row["raw"][wp["blend"]]
            p = p[p[:, 4] >= t[p[:, 5].astype(int)]]
            preds.append(dict(image_id=row["image_id"], GT=row["GT"], pred=p.tolist()))
        atomic_json(OUT / f"{name}_predictions.json", preds)
        results[name] = metrics(preds)
    atomic_json(OUT/"metrics.json", results)
    status("validation_complete", metrics={k:v["total"] for k,v in results.items()})


if __name__ == "__main__":
    main()
