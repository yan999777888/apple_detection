"""Run balanced crop + set verification on one original/flipped detector forward."""

import argparse
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image
import torch
from ultralytics.engine.results import Results
from class0_crop_expert import OUT as CROP, Expert, extract, scores, assemble
from class0_crop_deploy import Inference
from class0_patch_refinement import (
    ROOT,
    OLD,
    read,
    configure,
    atomic_json,
    load_image,
    manifest,
    sha,
)
from defect9485_set import OUT as SET, prepare, rank, cut
from defect9485_set_supplement import audit_pair
from defect9485_tree_runtime import pack
from defect9485_surface import (
    prepare as surface_prepare,
    ranked as surface_rank,
    OUT as SURFACE,
)
from defect9485_local_fusion import add, OUT as LOCAL_FUSION
from audit_defect_causes_20261112 import ious
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_set_deploy"


def union(old, new):
    existing = np.asarray(old, np.float32).reshape(-1, 6).copy()
    extra = np.asarray(new, np.float32).reshape(-1, 6).copy()
    if len(existing) and len(extra):
        ov = ious(extra[:, :4], existing[:, :4])
        extra = extra[
            ~((ov >= 0.50) & (extra[:, 5, None] == existing[None, :, 5])).any(1)
        ]
    existing[:, 4] = 0.5 + 0.5 * existing[:, 4]
    extra[:, 4] *= 0.5
    return np.concatenate([existing, extra])


class CandidateHeads:
    def __init__(self, expert, policy):
        self.expert = expert
        self.policy = policy
        assert self.policy["veto"] == 0
        cfg = read(SET / "frozen.json")
        weights = SET / "models.joblib"
        if sha(weights) != cfg["models_sha256"]:
            raise ValueError("Set model checksum mismatch")
        self.models = pack(joblib.load(weights))
        self.thresholds = cfg["OOF_profiles"]["P94"]["thresholds"]

        surface_cfg = read(SURFACE / "frozen.json")
        path = SURFACE / "model.joblib"
        if sha(path) != surface_cfg["sha"]:
            raise ValueError("Surface model checksum mismatch")
        self.surface = pack(joblib.load(path))
        self.surface_thresholds = surface_cfg["policies"]["joint"]["thresholds"]

    @torch.inference_mode()
    def __call__(self, src, image, mode):
        data = extract(src, image)
        boxes, crops, attrs, other = data
        unresolved = attrs[:, 0] < 0.4
        q = np.ones(len(boxes), np.float32)
        if unresolved.any():
            ids = unresolved.numpy()
            q[ids] = scores(
                (boxes[ids], crops[unresolved], attrs[unresolved], other), self.expert
            )
        balanced = assemble(data, q, self.policy["admit"], 0)
        if mode != "surface_only":
            high = cut(
                rank(prepare(src, image), self.models, image.shape[:2]), self.thresholds
            )
        results = {}
        if mode in ("set", "all"):
            results["set"] = union(balanced, high)
        if mode in ("surface_only", "union", "all"):
            surface = cut(
                surface_rank(
                    surface_prepare(src, image), self.surface, image.shape[:2]
                ),
                self.surface_thresholds,
            )
            base_surface = add(balanced, surface)
            if mode in ("surface_only", "all"):
                results["surface_only"] = base_surface
            if mode in ("union", "all"):
                results["union"] = add(base_surface, high.astype(np.float32))
        return results if mode == "all" else results[mode]


class Deploy:
    def __init__(self):
        baseline = Inference(False)
        self.worker = baseline.worker
        self.heads = CandidateHeads(baseline.expert, baseline.policy)

    @torch.inference_mode()
    def __call__(self, image, mode="set", both=True):
        pred = self.worker.predictors["stage"]
        tensor = pred.preprocess([image])
        src = self.worker.sources(tensor, image.shape[:2], ("o2o", "o2m"), True)
        stage_raw = self.worker.model.predict_source(tensor, "stage") if both else None
        boxes = self.heads(src, image, mode)
        if not both:
            return boxes
        pred.batch = (["frame"], [image], [""])
        stage = [r.cpu() for r in pred.postprocess(stage_raw, tensor, [image])]
        return {
            "defect": [
                Results(
                    image,
                    "frame",
                    self.worker.model.names["defect"],
                    boxes=torch.from_numpy(boxes),
                )
            ],
            "stage": stage,
        }


def cached_check():
    expert = Expert().eval()
    cfg = read(CROP / "frozen.json")
    assert sha(CROP / "expert.pt") == cfg["model_sha256"]
    expert.load_state_dict(
        torch.load(CROP / "expert.pt", map_location="cpu", weights_only=True)
    )
    heads = CandidateHeads(expert, cfg["selection"])
    lookup = manifest()
    sid = read(SET / "supplement.json")["balanced"]["union_change"]["gained"][0].split(
        "#"
    )[0]
    r = next(
        v for v in read(OLD / "phase1/source_predictions.json") if v["image_id"] == sid
    )
    src = {k: np.asarray(v, np.float32).reshape(-1, 6) for k, v in r["sources"].items()}
    preds = heads(src, load_image(lookup[sid]["rgb_path"]), "all")
    report = {}
    for mode, path in offline_paths().items():
        old = next(v for v in read(path) if v["image_id"] == sid)
        ref, p = np.asarray(old["pred"]), preds[mode]
        assert ref.shape == p.shape and np.allclose(p, ref, rtol=1e-4, atol=0.1)
        changes = audit_pair([old], [dict(image_id=sid, GT=r["GT"], pred=p.tolist())])
        assert not changes["gained"] and not changes["lost"]
        report[mode] = dict(
            n=len(p), max_abs_difference=float(np.abs(ref - p).max()), **changes
        )
    atomic_json(OUT / "cached_check.json", dict(image_id=sid, modes=report))


def offline_paths():
    return {
        "set": SET / "balanced_supplement_predictions.json",
        "union": LOCAL_FUSION / "union_predictions.json",
        "surface_only": LOCAL_FUSION / "surface_only_predictions.json",
    }


def evaluate():
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary

    model = Deploy()
    lookup = manifest()
    offline = {k: read(p) for k, p in offline_paths().items()}
    rows = {k: [] for k in offline}
    for r in offline["set"]:
        sid = r["image_id"]
        assert lookup[sid]["partition"] == "dev"
        p = model(load_image(lookup[sid]["rgb_path"]), mode="all", both=False)
        for k in rows:
            rows[k].append(dict(image_id=sid, GT=r["GT"], pred=p[k].tolist()))
    baseline = read(
        ROOT / "experiments/class0_crop_deploy_v1/fp32/dev_predictions.json"
    )
    stats = {}
    for k in rows:
        assert len(rows[k]) == 113
        atomic_json(OUT / f"{k}_predictions.json", rows[k])
        met = metrics(rows[k])
        stats[k] = dict(
            metrics={field: met[field] for field in ("total", "class")},
            vs_cached=audit_pair(offline[k], rows[k]),
            vs_incumbent=audit_pair(baseline, rows[k]),
        )
    atomic_json(OUT / "metrics.json", stats)
    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims.extend(
            load_image(paths[round(i * (len(paths) - 1) / 19)]) for i in range(20)
        )
    for im in ims[:5]:
        for k in rows:
            model(im, k)
    times = {k: [] for k in rows}
    names = list(rows)
    for rep in range(15):
        for index, im in enumerate(ims):
            for k in names if (rep + index) % 2 == 0 else names[::-1]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(im, k)
                torch.cuda.synchronize()
                times[k].append((time.perf_counter() - start) * 1000)
    speeds = {k: summary(t) for k, t in times.items()}
    atomic_json(OUT / "speed.json", speeds)
    print({k: v["metrics"]["total"] for k, v in stats.items()}, speeds, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["cached-check", "evaluate", "predict"])
    p.add_argument("--mode", choices=["set", "union", "surface_only"], default="set")
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    configure(OUT, "cpu" if args.action == "cached-check" else "cuda:0")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.action == "cached-check":
        cached_check()
    elif args.action == "evaluate":
        evaluate()
    else:
        if args.image is None or args.output is None or args.output.exists():
            p.error("Supply an image and fresh output folder")
        configure(args.output, "cuda:0")
        rr = Deploy()(load_image(args.image), args.mode)
        atomic_json(
            args.output / "predictions.json",
            {k: v[0].boxes.data.tolist() for k, v in rr.items()},
        )
        for k, v in rr.items():
            Image.fromarray(v[0].plot()[:, :, ::-1]).save(args.output / f"{k}.jpg")
