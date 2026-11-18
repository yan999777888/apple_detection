"""Use OOF-frozen image risk to conditionally evaluate the second detector."""

import argparse
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results
from class0_patch_refinement import (
    ROOT,
    configure,
    atomic_json,
    read,
    load_image,
    manifest,
    sha,
)
from class0_crop_expert import extract, scores, assemble
from defect9485_router import OUT as FIT, features
from defect9485_cross_deploy import CrossDeploy, boxes, apply_gate
from defect9485_set_supplement import audit_pair
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_router_deploy"


class Router(CrossDeploy):
    def __init__(self):
        super().__init__()
        c = read(FIT / "frozen.json")
        p = FIT / "risk.joblib"
        if sha(p) != c["model_sha256"]:
            raise ValueError("Image router checksum mismatch")
        self.risk = joblib.load(p)
        self.threshold = c["settings"]["top60"]["threshold"]

    @torch.inference_mode()
    def __call__(self, image, both=True):
        pred = self.worker.predictors["stage"]
        t = pred.preprocess([image])
        src = self.worker.sources(t, image.shape[:2], ("o2o", "o2m"), True)
        high = bool(self.risk.predict(features(src)[None])[0] >= self.threshold)
        if high:
            small = boxes(self.small, image)
        # Complete the maturity forward on CUDA while this frame's CPU verifiers run.
        raw = self.worker.model.predict_source(t, "stage") if both else None
        if high:
            p = apply_gate(
                image, self.confirm(src, image, small), self.pit, self.limits["0.99"]
            )
        else:
            data = extract(src, image)
            b, x, a, other = data
            idx = a[:, 0] < 0.4
            q = np.ones(len(b), np.float32)
            if idx.any():
                keep = idx.numpy()
                q[keep] = scores((b[keep], x[idx], a[idx], other), self.confirm.expert)
            p = assemble(data, q, self.confirm.crop_policy["admit"], 0)
        if not both:
            return p, high
        pred.batch = (["frame"], [image], [""])
        stage = [v.cpu() for v in pred.postprocess(raw, t, [image])]
        return {
            "stage": stage,
            "defect": [
                Results(
                    image,
                    "frame",
                    self.worker.model.names["defect"],
                    boxes=torch.from_numpy(p),
                )
            ],
        }, high


def evaluate():
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary

    model = Router()
    lookup = manifest()
    rr = []
    routed = []
    ref = read(FIT / "top60_predictions.json")
    for r in ref:
        sid = r["image_id"]
        assert lookup[sid]["partition"] == "dev"
        p, high = model(load_image(lookup[sid]["rgb_path"]), False)
        if high:
            routed.append(sid)
        rr.append(dict(image_id=sid, GT=r["GT"], pred=p.tolist()))
    met = metrics(rr)
    atomic_json(OUT / "predictions.json", rr)
    atomic_json(
        OUT / "metrics.json",
        dict(
            metrics={k: met[k] for k in ("total", "class")},
            upgraded=routed,
            n=len(rr),
            vs_cache=audit_pair(ref, rr),
        ),
    )
    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims.extend(
            (task, load_image(paths[round(i * (len(paths) - 1) / 19)]))
            for i in range(20)
        )
    for _, im in ims[:5]:
        model(im)
    times = []
    flags = {}
    for rep in range(15):
        for i, (task, im) in enumerate(ims):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _, high = model(im)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
            if rep == 0:
                flags[i] = high
            else:
                assert flags[i] == high
    speed = dict(
        **summary(times),
        by_domain={
            task: summary(
                [t for j, t in enumerate(times) if ims[j % len(ims)][0] == task]
            )
            for task in ("stage", "defect")
        },
        distinct_stage_upgrades=sum(flags[i] for i in range(20)),
        distinct_defect_upgrades=sum(flags[i] for i in range(20, 40)),
    )
    atomic_json(OUT / "speed.json", speed)
    print(met["total"], len(routed), speed, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["evaluate", "predict"])
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    configure(OUT, "cuda:0")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if a.action == "evaluate":
        evaluate()
    else:
        if not a.image or not a.output or a.output.exists():
            p.error("Provide an image and a fresh project output folder")
        configure(a.output, "cuda:0")
        r, high = Router()(load_image(a.image))
        atomic_json(
            a.output / "predictions.json",
            dict(upgraded=high, **{k: v[0].boxes.data.tolist() for k, v in r.items()}),
        )
        for k, v in r.items():
            Image.fromarray(v[0].plot()[:, :, ::-1]).save(a.output / f"{k}.jpg")
