"""Train a COCO-initialized Faster R-CNN on apple defects and freeze its operating point."""

import argparse
import math
import time

import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from class0_patch_refinement import ROOT, atomic_json, configure, read, sha, load_image
from defect_continue_transplant import OUT, SIDE, sample, split_train
from defect9485_p2 import samples
from audit_defect_causes_20261112 import match
from run_defect_zero_train_20261112 import metrics
from joint_state_v1 import atomic_checkpoint, load_checkpoint

COCO = ROOT.parent / "apple_vision/local_amodal/cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
COCO_SHA = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
SEED = 94185
EPOCHS = 18
CHECKS = (6, 12, 18)


def detector():
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None,
                                   norm_layer=FrozenBatchNorm2d, trainable_layers=3)
    anchors = AnchorGenerator(tuple((a,) for a in (16, 32, 64, 128, 256)), ((0.5, 1, 2),) * 5)
    m = FasterRCNN(backbone, num_classes=91, min_size=SIDE, max_size=SIDE,
                   rpn_anchor_generator=anchors, rpn_pre_nms_top_n_train=1500,
                   rpn_post_nms_top_n_train=600, rpn_pre_nms_top_n_test=600,
                   rpn_post_nms_top_n_test=160, box_batch_size_per_image=256,
                   box_score_thresh=0.005, box_nms_thresh=0.55, box_detections_per_img=80)
    return m


def initialized():
    assert sha(COCO) == COCO_SHA
    m = detector()
    source = torch.load(COCO, weights_only=True, map_location="cpu")
    mask_only = ("roi_heads.mask_",)
    m.load_state_dict({k: v for k, v in source.items() if not k.startswith(mask_only)}, strict=True)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 3)
    return m


def tensor(im):
    rgb = np.ascontiguousarray(im[:, :, ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1).to("cuda", dtype=torch.float32) / 255


def train_sample(fit, i, rng, transfer_probability):
    im, lab, _ = sample(fit, i, rng, transfer_probability)
    lab = lab.copy()
    if rng.random() < 0.5:
        im = im[:, ::-1]
        lab[:, [0, 2]] = SIDE - lab[:, [2, 0]]
    # The camera captures isolated apples in different orientations.
    for _ in range(int(rng.integers(4))):
        im = np.rot90(im)
        lab[:, :4] = np.column_stack((lab[:, 1], SIDE-lab[:, 2], lab[:, 3], SIDE-lab[:, 0]))
    gain = rng.uniform(0.80, 1.22) * rng.uniform(0.89, 1.12, 3)
    im = np.clip(im * gain.astype(np.float32), 0, 255).astype(np.uint8)
    return tensor(im), dict(boxes=torch.from_numpy(lab[:, :4].copy()).cuda(),
                            labels=torch.from_numpy(lab[:, 4].astype(np.int64) + 1).cuda())


def save_epoch(folder, ep, model, optim, rng, history, transfer_probability):
    return atomic_checkpoint(folder / f"epoch{ep:02}.pt", dict(
        schema="joint_exact_training_checkpoint_v1", resumable=True, epoch_completed=ep,
        model=model.state_dict(), optimizer=optim.state_dict(),
        numpy_rng=rng.bit_generator.state, cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        history=list(history), side=SIDE, seed=SEED, initial_coco=COCO_SHA,
        transfer_probability=transfer_probability,
    ))


def restore_epoch(state, model, optim, rng, transfer_probability):
    if (state["side"], state["seed"], state["initial_coco"], state["transfer_probability"]) != (
            SIDE, SEED, COCO_SHA, transfer_probability):
        raise ValueError("Checkpoint belongs to a different training recipe")
    model.load_state_dict(state["model"], strict=True)
    optim.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["epoch_completed"], state["history"]


def train(folder, transfer_probability, resume=False):
    fit, calib = split_train()
    if not resume and (folder / "training.json").exists():
        raise ValueError("Pass --resume for an existing run")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = initialized().cuda().train()
    slow, fast = [], []
    for k, p in model.named_parameters():
        if p.requires_grad:
            (slow if k.startswith("backbone.body") else fast).append(p)
    optim = torch.optim.SGD([dict(params=slow), dict(params=fast)], lr=0.004,
                            momentum=0.9, weight_decay=0.0001)
    history = []
    batches = math.ceil(len(fit) / 2)
    groups = math.ceil(batches / 2)
    total_steps = groups * EPOCHS
    first = 1
    if resume:
        receipts = sorted(folder.glob("epoch??.sha256.json"))
        if not receipts:
            raise ValueError("No complete checkpoint with a verified receipt")
        saved = load_checkpoint(receipts[-1].with_suffix("").with_suffix(".pt"))
        completed, history = restore_epoch(saved, model, optim, rng, transfer_probability)
        del saved
        first = completed + 1
    for ep in range(first, EPOCHS+1):
        start = time.perf_counter()
        order = rng.permutation(len(fit))
        losses = []
        for group in range(groups):
            step = groups * (ep-1) + group
            scale = min(1, (step+1)/120) * (0.08 + 0.92*(1 + math.cos(math.pi*step/total_steps))/2)
            for g, lr in zip(optim.param_groups, [0.001, 0.004]):
                g["lr"] = lr * scale
            ids = order[group*4:(group+1)*4]
            optim.zero_grad(set_to_none=True)
            for offset in range(0, len(ids), 2):
                pairs = [train_sample(fit, i, rng, transfer_probability) for i in ids[offset:offset+2]]
                xs, ys = zip(*pairs)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = sum(model(list(xs), list(ys)).values())
                (loss * len(xs) / len(ids)).backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(slow + fast, 10, error_if_nonfinite=True)
            optim.step()
        entry = dict(epoch=ep, updates=groups * ep, mean_loss=float(torch.stack(losses).mean()),
                     seconds=round(time.perf_counter()-start, 2))
        history.append(entry)
        if ep % 3 == 0:
            save_epoch(folder, ep, model, optim, rng, history, transfer_probability)
        atomic_json(folder / "training.json", history)
        print(entry, flush=True)
    atomic_json(folder / "fit.json", dict(coco_sha=COCO_SHA, fit_ids=[r["image_id"] for r in fit],
                calibration_ids=[r["image_id"] for r in calib], epochs=EPOCHS, seed=SEED,
                transfer_probability=transfer_probability, updates=total_steps))


@torch.inference_mode()
def predict(model, im):
    height, width = im.shape[:2]
    if im.shape[:2] != (SIDE, SIDE):
        im = cv2.resize(im, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model([tensor(im)])[0]
    scales = out["boxes"].new_tensor([width, height, width, height], dtype=torch.float32) / SIDE
    return torch.cat((out["boxes"].float() * scales, out["scores"][:, None].float(),
                      out["labels"][:, None]-1), dim=1).cpu().numpy()


def predictions(model, rows):
    recs = []
    for r in rows:
        gt = []
        for v in r["instances"]:
            x, y, w, h = v["xywh_normalized"]
            gt.append({"box":[(x-w/2)*1024,(y-h/2)*1024,(x+w/2)*1024,(y+h/2)*1024],
                       "class":v["class_id"]})
        recs.append(dict(image_id=r["image_id"], GT=gt, pred=predict(model,load_image(r["rgb_path"])).tolist()))
    return recs


def clipped(rows, cut):
    return [dict(r, pred=[p for p in r["pred"] if p[4] >= cut[int(p[5])]]) for r in rows]


def calibrate(rows):
    evidence = [[], []]
    n = sum(len(r["GT"]) for r in rows)
    for r in rows:
        pairs, _ = match(r["pred"], r["GT"])
        for i, p in enumerate(r["pred"]):
            evidence[int(p[5])].append((p[4], int(i in pairs)))
    thresholds = np.arange(.025, 1.0, .025)
    prefix = []
    for data in evidence:
        e = np.asarray(data, dtype=float).reshape(-1,2)
        prefix.append([(t, int(np.sum(e[e[:,0]>=t,1])), int(np.sum(e[:,0]>=t)))
                       for t in thresholds])
    best = None
    for t0,a,x in prefix[0]:
        for t1,b,y in prefix[1]:
            p = (a+b)/max(x+y,1)
            r = (a+b)/n
            # Prefer the joint requirement; otherwise balance the two relative shortfalls.
            key = (min(p/.94, r/.85), p+r)
            if best is None or key > best[0]:
                best = key, [float(t0),float(t1)]
    fixed = metrics(clipped(rows, best[1]))["total"]
    return dict(key=list(best[0]), cut=best[1], fixed=fixed)


def select_model(folder):
    _, calib = split_train()
    model = initialized().cuda().eval()
    records = []
    for ep in CHECKS:
        w = folder / f"epoch{ep:02}.pt"
        model.load_state_dict(load_checkpoint(w)["model"])
        result = calibrate(predictions(model, calib))
        records.append(dict(epoch=ep, sha256=sha(w), **result))
    best = max(records, key=lambda z: z["key"])
    atomic_json(folder / "frozen.json", dict(selected=best, calibration=records,
                 rule="maximize min(P/0.94,R/0.85) on reserved training minutes"))
    print("calibration", best, flush=True)


def evaluate(folder):
    point = read(folder / "frozen.json")["selected"]
    w = folder / f"epoch{point['epoch']:02}.pt"
    assert sha(w) == point["sha256"]
    model = initialized().cuda().eval()
    model.load_state_dict(load_checkpoint(w)["model"])
    rows = predictions(model, samples("dev"))
    assert len(rows) == 113 and sum(len(r["GT"]) for r in rows) == 230
    fixed = clipped(rows, point["cut"])
    atomic_json(folder / "low_predictions.json", rows)
    atomic_json(folder / "fixed_predictions.json", fixed)
    m = metrics(fixed)
    atomic_json(folder / "metrics.json", m)
    print("development", m["total"], m["class"], flush=True)
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from class0_crop_deploy import Inference
    worker = Inference(False).worker
    stage = worker.predictors["stage"]
    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims += [load_image(paths[round(i*(len(paths)-1)/19)]) for i in range(20)]

    @torch.inference_mode()
    def both(im):
        t = stage.preprocess([im])
        stage.batch = (["frame"], [im], [""])
        maturity = [v.cpu() for v in stage.postprocess(worker.model.predict_source(t,"stage"), t, [im])]
        p = predict(model, im)
        return maturity, p[p[:,4] >= np.array(point["cut"])[p[:,5].astype(int)]]

    for im in ims[:5]:
        both(im)
    latency = []
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize()
            start = time.perf_counter()
            both(im)
            torch.cuda.synchronize()
            latency.append(1000 * (time.perf_counter()-start))
    timing = summary(latency)
    atomic_json(folder / "speed.json", timing)
    print("dual task latency", timing, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "train", "evaluate", "all"])
    p.add_argument("--native-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    folder = OUT / ("native_v2" if args.native_only else "transfer_v2")
    configure(folder, "cpu" if args.action == "check" else "cuda:0")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.action == "check":
        m = initialized().eval()
        with torch.inference_mode():
            out = m([torch.zeros(3,SIDE,SIDE)])[0]
        assert out["boxes"].shape[-1] == 4
        print("Strict COCO box-branch load and CPU forward OK", flush=True)
    if args.action in ("train","all"):
        train(folder, 0 if args.native_only else 0.5, args.resume)
        select_model(folder)
    if args.action in ("evaluate","all"):
        evaluate(folder)
