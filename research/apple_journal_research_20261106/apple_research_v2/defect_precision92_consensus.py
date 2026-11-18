"""Cross-check lower-confidence visible defects with a second set of detector weights."""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from defect_precision92 import Deployment, OUT as VERIFIER, cut, rank, extract
from class0_patch_refinement import (
    ROOT,
    sha,
    read,
    configure,
    atomic_json,
    manifest,
    load_image,
)
from defect_coordinate_vote_20261118 import vote
from defect_algorithm_modes_20261112 import nms
from audit_defect_causes_20261112 import ious
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect_precision92_consensus_v1"
AVG = ROOT / "experiments/defect_weight_average_20261118"
MODES = ("agreement", "agreement_plus_high")


class Consensus(Deployment):
    def __init__(self):
        from defect_four_orientations_20261118 import make_worker
        from ultralytics.nn.modules.conv import Conv
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        super().__init__()
        path = AVG / "composite.pt"
        assert sha(path) == read(AVG / "results.json")["checkpoint_sha256"]
        self.second = make_worker(path, False)
        for m in self.second.model.modules():
            if isinstance(m, Conv) and isinstance(
                getattr(m, "bn", None), torch.nn.BatchNorm2d
            ):
                m.conv = fuse_conv_bn_eval(m.conv, m.bn)
                del m.bn
                m.forward = m.forward_fuse
        self.second_thresholds = read(AVG / "results.json")["thresholds"]

    @torch.inference_mode()
    def candidates(self, im):
        worker = self.incumbent.worker
        pred = worker.predictors["stage"]
        t = pred.preprocess([im])
        sources = worker.sources(t, im.shape[:2], ("o2o", "o2m"), True)
        primary = rank(extract(sources, im, self.encoder), self.verifier, im.shape[:2])
        second = vote(
            self.second.sources(t, im.shape[:2], ("o2o",), True),
            0.5,
            self.second_thresholds,
        )
        balanced = cut(primary, self.modes["balanced"]["thresholds"])
        high = cut(primary, self.modes["precision92"]["thresholds"])
        accepted = []
        for c in (0, 1):
            a = balanced[balanced[:, 5] == c]
            b = second[second[:, 5] == c]
            if not len(a) or not len(b):
                continue
            ov = ious(a[:, :4], b[:, :4])
            # One vote per detector per instance; don't let a merged box confirm two wounds.
            used_a = set()
            used_b = set()
            for i, j in sorted(zip(*np.where(ov >= 0.5)), key=lambda ij: -ov[ij]):
                if i in used_a or j in used_b:
                    continue
                p = a[i].copy()
                p[:4] = 0.5 * (a[i, :4] + b[j, :4])
                accepted.append(p)
                used_a.add(i)
                used_b.add(j)
        agreed = nms(accepted)
        if len(agreed) and len(high):
            overlap = ious(high[:, :4], agreed[:, :4])
            high = high[
                ~((overlap >= 0.5) & (high[:, 5, None] == agreed[None, :, 5])).any(1)
            ]
        return t, {
            "agreement": agreed,
            "agreement_plus_high": nms(np.concatenate([agreed, high])),
        }

    @torch.inference_mode()
    def inference(self, im, mode):
        worker = self.incumbent.worker
        t, outputs = self.candidates(im)
        pred = worker.predictors["stage"]
        pred.batch = (["frame"], [im], [""])
        stage = [
            r.cpu()
            for r in pred.postprocess(worker.model.predict_source(t, "stage"), t, [im])
        ]
        return stage, outputs[mode]


def run():
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary

    model = Consensus()
    lookup = manifest()
    dev = read(ROOT / "experiments/class0_crop_deploy_v1/fp32/dev_predictions.json")
    records = {k: [] for k in MODES}
    for r in dev:
        sid = r["image_id"]
        assert lookup[sid]["partition"] == "dev"
        _, pred = model.candidates(load_image(lookup[sid]["rgb_path"]))
        for mode in MODES:
            records[mode].append(
                dict(image_id=sid, GT=r["GT"], pred=pred[mode].tolist())
            )
    for mode, rows in records.items():
        atomic_json(OUT / f"{mode}_predictions.json", rows)
    atomic_json(
        OUT / "dev_metrics.json",
        {mode: metrics(rows) for mode, rows in records.items()},
    )
    ims = []
    for task in ("stage", "defect"):
        pp = source_spec(task)["images"]
        ims.extend(load_image(pp[round(i * (len(pp) - 1) / 19)]) for i in range(20))
    # Time the deployed path that includes both agreement and recovery of high-confidence predictions.
    for im in ims[:5]:
        model.inference(im, "agreement_plus_high")
    times = []
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.inference(im, "agreement_plus_high")
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
    atomic_json(OUT / "speed.json", {"agreement_plus_high": summary(times)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    configure(OUT, "cuda:0")
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if args.image is None:
        run()
    else:
        if not args.output or args.output.exists():
            p.error("Choose a fresh project output directory")
        configure(args.output, "cuda:0")
        _, a = Consensus().inference(load_image(args.image), "agreement_plus_high")
        atomic_json(
            args.output / "defects.json", dict(boxes_xyxy_score_class=a.tolist())
        )
