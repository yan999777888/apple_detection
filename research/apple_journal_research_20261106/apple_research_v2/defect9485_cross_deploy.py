"""Deploy the frozen two-defect-detector agreement and set rescue policy on one frame."""

import argparse
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results

from class0_patch_refinement import (
    ROOT,
    CP,
    OLD,
    load_image,
    manifest,
    configure,
    read,
    sha,
    atomic_json,
)
from class0_crop_deploy import Inference
from class0_crop_expert import OUT as CROP, Expert, assemble, extract, scores
from defect9485_set import OUT as SET, rank, cut, prepare
from defect9485_set_supplement import audit_pair
from defect9485_small import OUT as SMALL, PRIMARY, boxes, deploy_model, square
from defect9485_tree_runtime import pack, PackedTrees
from defect9485_gate_transfer import apply as apply_gate
from defect9485_pit_gate import OUT as PIT
from defect9485_small_flip import predict as flip_boxes
from defect9485_cross import OUT as CROSS, extend, paired, with_supplement
from defect_coordinate_vote_20261118 import vote
from defect_algorithm_modes_20261112 import nms
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_cross_deploy"
REFERENCE = (
    CROSS / "yolo26s/one_fixed_plus_other_support_and_frozen_set_predictions.json"
)


class Confirm:
    def __init__(self, expert, crop_policy):
        self.expert = expert
        self.crop_policy = crop_policy
        if crop_policy["veto"] != 0:
            raise ValueError("Requires the frozen original crop policy")
        cfg = read(SET / "frozen.json")
        weights = SET / "models.joblib"
        assert sha(weights) == cfg["models_sha256"]
        self.trees = pack(joblib.load(weights))
        self.set_thresholds = cfg["OOF_profiles"]["P94"]["thresholds"]

    @torch.inference_mode()
    def __call__(self, src, im, small_low):
        data = extract(src, im)
        b, x, attrs, other = data
        idx = attrs[:, 0] < 0.4
        q = np.ones(len(b), np.float32)
        if idx.any():
            ii = idx.numpy()
            q[ii] = scores((b[ii], x[idx], attrs[idx], other), self.expert)
        primary = assemble(data, q, self.crop_policy["admit"], 0)
        low0 = vote(src, 0.5, [0.055, 0.055])
        b0 = extend(primary, low0)
        strong = cut(small_low, PRIMARY)
        b1 = extend(strong, small_low[small_low[:, 4] >= 0.055])
        both, levels = paired(b0, b1)
        agreement = nms(both[levels >= 1])
        supplement = cut(
            rank(prepare(src, im), self.trees, im.shape[:2]), self.set_thresholds
        )
        return with_supplement(agreement, supplement)


@torch.inference_mode()
def half_boxes(net, im):
    a, (dx, dy, sx, sy) = square(im)
    t = torch.from_numpy(a).cuda().permute(2, 0, 1)[None].float().div_(255).half()
    pred = net(t)
    p = (pred[0][0] if isinstance(pred, tuple) else pred[0]).float()
    p = p[p[:, 4] >= 0.001].cpu().numpy()
    h, w = im.shape[:2]
    p[:, [0, 2]] = ((p[:, [0, 2]] - dx) / sx).clip(0, w)
    p[:, [1, 3]] = ((p[:, [1, 3]] - dy) / sy).clip(0, h)
    return nms(p)


class CrossDeploy:
    def __init__(self, small_half=False):
        incumbent = Inference(False)
        self.worker = incumbent.worker
        self.confirm = Confirm(incumbent.expert, incumbent.policy)
        self.small = deploy_model()
        if small_half:
            self.small.half()
        limits = read(PIT / "frozen.json")
        assert sha(PIT / "pit.joblib") == limits["sha256"]
        self.pit = PackedTrees(joblib.load(PIT / "pit.joblib"))
        self.limits = limits["limits"]

    @torch.inference_mode()
    def __call__(self, image, both=True, mode="gate99"):
        worker = self.worker
        pred = worker.predictors["stage"]
        tensor = pred.preprocess([image])
        src = worker.sources(tensor, image.shape[:2], ("o2o", "o2m"), True)
        if mode == "gate99_small_half":
            low = half_boxes(self.small, image)
        elif mode == "flip_gate95":
            low = flip_boxes(self.small, image)
        else:
            low = boxes(self.small, image)
        # The CPU verifiers can run while CUDA evaluates maturity on the shared input.
        early = both and mode == "gate99_small_half"
        raw = worker.model.predict_source(tensor, "stage") if early else None
        p = self.confirm(src, image, low)
        if mode != "raw":
            p = apply_gate(
                image,
                p,
                self.pit,
                self.limits["0.95" if mode == "flip_gate95" else "0.99"],
            )
        if not both:
            return p
        pred.batch = (["frame"], [image], [""])
        if raw is None:
            raw = worker.model.predict_source(tensor, "stage")
        stage = [r.cpu() for r in pred.postprocess(raw, tensor, [image])]
        return {
            "stage": stage,
            "defect": [
                Results(
                    image,
                    "frame",
                    worker.model.names["defect"],
                    boxes=torch.from_numpy(p),
                )
            ],
        }


def cached_check():
    cfg = read(CROP / "frozen.json")
    assert sha(CROP / "expert.pt") == cfg["model_sha256"]
    expert = Expert().eval()
    expert.load_state_dict(
        torch.load(CROP / "expert.pt", map_location="cpu", weights_only=True)
    )
    confirm = Confirm(expert, cfg["selection"])
    lookup = manifest()
    small = {r["image_id"]: r for r in read(SMALL / "low_predictions.json")}
    srcrows = {r["image_id"]: r for r in read(OLD / "phase1/source_predictions.json")}
    refrows = read(REFERENCE)
    # Sample sparse and crowded inputs, including zero predictions if present.
    ordered = sorted(refrows, key=lambda r: len(r["pred"]))
    chosen = [
        ordered[i]
        for i in sorted(
            set([0, len(ordered) // 3, len(ordered) // 2, len(ordered) - 1])
        )
    ]
    result = []
    for r in chosen:
        sid = r["image_id"]
        sources = {
            k: np.asarray(v, np.float32).reshape(-1, 6)
            for k, v in srcrows[sid]["sources"].items()
        }
        raw = np.asarray(small[sid]["pred"], np.float32).reshape(-1, 6)
        p = confirm(sources, load_image(lookup[sid]["rgb_path"]), raw)
        trial = dict(image_id=sid, GT=r["GT"], pred=p.tolist())
        quality = audit_pair([r], [trial])
        assert (
            not quality["gained"] and not quality["lost"] and len(p) == len(r["pred"])
        )
        result.append(dict(image_id=sid, n_pred=len(p), **quality))
    atomic_json(OUT / "cached_check.json", result)


def evaluate(mode):
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary

    model = CrossDeploy(small_half=mode == "gate99_small_half")
    ref_path = {
        "raw": REFERENCE,
        "gate99": PIT / "small_cross_099_predictions.json",
        "gate99_small_half": PIT / "small_cross_099_predictions.json",
        "flip_gate95": PIT / "small_flip_cross_095_predictions.json",
    }[mode]
    dest = OUT / mode
    ref = read(ref_path)
    lookup = manifest()
    rows = []
    for r in ref:
        sid = r["image_id"]
        assert lookup[sid]["partition"] == "dev"
        p = model(load_image(lookup[sid]["rgb_path"]), False, mode)
        rows.append(dict(image_id=sid, GT=r["GT"], pred=p.tolist()))
    assert len(rows) == 113 and sum(len(r["GT"]) for r in rows) == 230
    out = metrics(rows)
    atomic_json(dest / "fixed_predictions.json", rows)
    atomic_json(
        dest / "metrics.json",
        dict(
            fixed={k: out[k] for k in ("total", "class")},
            comparison=audit_pair(ref, rows),
            reference_sha256=sha(ref_path),
            pit_sha256=sha(PIT / "pit.joblib"),
            source_sha256=sha(CP),
            small_sha256=sha(SMALL / "weights.pt"),
            set_sha256=sha(SET / "models.joblib"),
            crop_sha256=sha(CROP / "expert.pt"),
            policy_sha256=sha(CROSS / "protocol.json"),
        ),
    )
    print("Full validation", out["total"], flush=True)
    ims = []
    for task in ("stage", "defect"):
        pool = source_spec(task)["images"]
        ims.extend(load_image(pool[round(i * (len(pool) - 1) / 19)]) for i in range(20))
    for im in ims[:5]:
        model(im, mode=mode)
    timings = []
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            model(im, mode=mode)
            torch.cuda.synchronize()
            timings.append(1000 * (time.perf_counter() - begin))
    speed = summary(timings)
    atomic_json(dest / "speed.json", speed)
    print("Speed", speed, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("action", choices=["cached-check", "evaluate", "predict"])
    cli.add_argument(
        "--mode",
        choices=["raw", "gate99", "gate99_small_half", "flip_gate95"],
        default="gate99",
    )
    cli.add_argument("--image", type=Path)
    cli.add_argument("--output", type=Path)
    a = cli.parse_args()
    configure(OUT, "cpu" if a.action == "cached-check" else "cuda:0")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if a.action == "cached-check":
        cached_check()
    elif a.action == "evaluate":
        evaluate(a.mode)
    else:
        if not a.image or not a.output or a.output.exists():
            cli.error("Supply image and fresh output directory")
        configure(a.output, "cuda:0")
        im = load_image(a.image)
        rr = CrossDeploy(small_half=a.mode == "gate99_small_half")(im, mode=a.mode)
        atomic_json(
            a.output / "predictions.json",
            {k: v[0].boxes.data.tolist() for k, v in rr.items()},
        )
        for k, v in rr.items():
            Image.fromarray(v[0].plot()[:, :, ::-1]).save(a.output / f"{k}.jpg")
