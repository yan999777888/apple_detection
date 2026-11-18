"""Verify class-0 proposals using their native RGB patch and detector evidence."""

import argparse
import time

from class0_patch_refinement import (
    ROOT,
    OLD,
    CP,
    read,
    load_image,
    manifest,
    configure,
    atomic_json,
    sha,
)

OUT = ROOT / "experiments/class0_crop_expert_v1"

import numpy as np
import torch
from torch import nn


def extract(src, im):
    import cv2
    from defect_coordinate_vote_20261118 import vote
    from defect_algorithm_modes_20261112 import nms
    from audit_defect_causes_20261112 import ious

    src = {
        k: np.asarray(src[k], np.float32).reshape(-1, 6) for k in ("o2o", "flip", "o2m")
    }
    ref = vote(src, 0.5, [0.035, 0.34])
    old, other = ref[ref[:, 5] == 0], ref[ref[:, 5] == 1]
    dense = src["o2m"]
    dense = nms(dense[(dense[:, 5] == 0) & (dense[:, 4] >= 0.06)])
    if len(old) and len(dense):
        dense = dense[ious(dense[:, :4], old[:, :4]).max(1) < 0.55]
    boxes = np.concatenate((old, dense))
    inherited = np.r_[old[:, 4], np.zeros(len(dense))].astype(np.float32)
    attrs = [inherited]
    for arr in src.values():
        for cls in (0, 1):
            p = arr[arr[:, 5] == cls]
            col = np.zeros(len(boxes))
            if len(boxes) and len(p):
                col = (ious(boxes[:, :4], p[:, :4]) ** 2 * p[None, :, 4]).max(1)
            attrs.append(col)
    h, w = im.shape[:2]
    wh = np.maximum(boxes[:, 2:4] - boxes[:, :2], 1)
    attrs.extend(np.log(wh / [w, h]).T)
    crops = []
    for (x, y, X, Y), (bw, bh) in zip(boxes[:, :4], wh):
        roi = im[
            max(0, int(y - 0.35 * bh)) : min(h, int(Y + 0.35 * bh) + 1),
            max(0, int(x - 0.35 * bw)) : min(w, int(X + 0.35 * bw) + 1),
        ]
        crops.append(
            cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
            if roi.size
            else np.zeros((32, 32, 3), np.uint8)
        )
    return (
        boxes,
        torch.from_numpy(np.asarray(crops, np.uint8).reshape(-1, 32, 32, 3)).permute(
            0, 3, 1, 2
        ),
        torch.tensor(np.stack(attrs, 1), dtype=torch.float32),
        other,
    )


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for inputs, outputs in [(3, 12), (12, 20), (20, 24)]:
            layers.extend(
                [
                    nn.Conv2d(inputs, outputs, 3, stride=2, padding=1),
                    nn.GroupNorm(4, outputs),
                    nn.SiLU(),
                ]
            )
        self.encoder = nn.Sequential(
            *layers, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        self.head = nn.Sequential(nn.Linear(96 + 9, 40), nn.SiLU(), nn.Linear(40, 3))

    def forward(self, crops, attrs):
        x = crops.float() / 255
        x = (x - x.mean((2, 3), keepdim=True)) / (0.05 + x.std((2, 3), keepdim=True))
        return self.head(torch.cat((self.encoder(x), attrs), 1))


def train(rows, device):
    from audit_defect_causes_20261112 import ious
    from torch.nn import functional as F

    torch.manual_seed(61)
    labels = []
    for row in rows:
        b = row["data"][0]
        overlaps = ious(b[:, :4], [g["box"] for g in row["GT"]])
        y = np.full(len(b), 2)
        if len(b) and len(row["GT"]):
            top = overlaps.argmax(1)
            for i in np.where(overlaps.max(1) >= 0.5)[0]:
                y[i] = row["GT"][top[i]]["class"]
        labels.extend(y)
    patches = torch.cat([r["data"][1] for r in rows]).to(device)
    attrs = torch.cat([r["data"][2] for r in rows]).to(device)
    labels = torch.tensor(labels, device=device)
    model = Expert().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    for epoch in range(14):
        for idx in torch.randperm(len(labels), device=device).split(128):
            x = patches[idx]
            if epoch % 2:
                x = x.flip([-1])
            loss = F.cross_entropy(
                model(x, attrs[idx]),
                labels[idx],
                weight=x.new_tensor([1.7, 1.3, 1.0], dtype=torch.float32),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval().cpu()


def scores(data, model):
    with torch.inference_mode():
        return (
            model(data[1], data[2]).softmax(1)[:, 0].numpy()
            if len(data[0])
            else np.empty(0)
        )


def assemble(data, q, admit, veto):
    from defect_algorithm_modes_20261112 import nms

    b, _, a, other = data
    old = a[:, 0].numpy()
    existing = old >= 0.40
    mask = (existing & (q >= veto)) | (~existing & (q >= admit))
    p = b[mask].copy()
    p[:, 4] = np.where(existing[mask], 0.5 + 0.5 * old[mask], 0.5 * q[mask])
    return np.concatenate((nms(p), other))


def fit():
    from run_defect_zero_train_20261112 import metrics

    rows = []
    lookup = manifest()
    for f in range(3):
        for r in read(OLD / f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"]:
            s = lookup[r["image_id"]]
            assert s["partition"] == "train"
            rows.append(
                dict(
                    image_id=r["image_id"],
                    GT=r["GT"],
                    fold=f,
                    data=extract(r["sources"], load_image(s["rgb_path"])),
                )
            )
    assert len({r["image_id"] for r in rows}) == 596
    for f in range(3):
        model = train([r for r in rows if r["fold"] != f], "cuda")
        for r in rows:
            if r["fold"] == f:
                r["q"] = scores(r["data"], model)
    choices = []
    for admit in (0.55, 0.65, 0.75, 0.85, 0.95, 1.01):
        for veto in (0, 0.10, 0.20, 0.30, 0.40):
            m = metrics(
                [
                    dict(
                        image_id=r["image_id"],
                        GT=r["GT"],
                        pred=assemble(r["data"], r["q"], admit, veto).tolist(),
                    )
                    for r in rows
                ]
            )["class"]["0"]
            choices.append(dict(admit=admit, veto=veto, **m))
    budget = read(ROOT / "experiments/class0_patch_refinement_v1/frozen.json")[
        "reference"
    ]["FP"]
    chosen = max(
        (m for m in choices if m["FP"] <= budget), key=lambda m: (m["TP"], -m["FP"])
    )
    path = OUT / "expert.pt"
    torch.save(train(rows, "cuda").state_dict(), path)
    atomic_json(
        OUT / "frozen.json",
        dict(selection=chosen, model_sha256=sha(path), training_images=len(rows)),
    )


def predict(worker, model, cfg, im, both=True):
    from ultralytics.engine.results import Results

    with torch.inference_mode():
        pred = worker.predictors["stage"]
        t = pred.preprocess([im])
        src = worker.sources(t, im.shape[:2], ("o2o", "o2m"), True)
        data = extract(src, im)
        p = assemble(data, scores(data, model), cfg["admit"], cfg["veto"])
        if not both:
            return p
        pred.batch = (["frame"], [im], [""])
        return {
            "defect": [
                Results(
                    im, "frame", worker.model.names["defect"], boxes=torch.from_numpy(p)
                )
            ],
            "stage": [
                v.cpu()
                for v in pred.postprocess(
                    worker.model.predict_source(t, "stage"), t, [im]
                )
            ],
        }


def evaluate():
    from defect_four_orientations_20261118 import make_worker
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from run_defect_zero_train_20261112 import metrics

    cfg = read(OUT / "frozen.json")
    assert (
        sha(OUT / "expert.pt") == cfg["model_sha256"]
        and sha(CP)
        == "f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
    )
    model = Expert().eval()
    model.load_state_dict(
        torch.load(OUT / "expert.pt", map_location="cpu", weights_only=True)
    )
    worker = make_worker(CP, False)
    lookup = manifest()
    rows = []
    for r in read(OLD / "phase1/source_predictions.json"):
        p = predict(
            worker,
            model,
            cfg["selection"],
            load_image(lookup[r["image_id"]]["rgb_path"]),
            False,
        )
        rows.append(dict(image_id=r["image_id"], GT=r["GT"], pred=p.tolist()))
    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims.extend(
            load_image(paths[round(i * (len(paths) - 1) / 19)]) for i in range(20)
        )
    for im in ims[:5]:
        predict(worker, model, cfg["selection"], im)
    times = []
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict(worker, model, cfg["selection"], im)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    atomic_json(OUT / "dev_predictions.json", rows)
    atomic_json(OUT / "results.json", dict(metrics=metrics(rows), speed=summary(times)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["fit", "evaluate"])
    args = p.parse_args()
    configure(OUT, "cuda:0")
    torch.set_num_threads(2)
    if args.action == "fit":
        fit()
    else:
        evaluate()
