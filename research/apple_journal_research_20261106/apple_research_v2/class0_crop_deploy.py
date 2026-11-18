"""Deploy the frozen crop expert with fused convolutions and verified precision modes."""

import argparse
import time
from pathlib import Path
import numpy as np
import torch

from class0_crop_expert import OUT as EXPERT, Expert, extract, scores, assemble
from class0_patch_refinement import (
    ROOT,
    OLD,
    CP,
    read,
    configure,
    atomic_json,
    load_image,
    manifest,
    sha,
)

OUT = ROOT / "experiments/class0_crop_deploy_v1"


class Inference:
    def __init__(self, half):
        from defect_four_orientations_20261118 import make_worker
        from ultralytics.nn.modules.conv import Conv
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        self.half = half
        self.worker = make_worker(CP, False)
        assert (
            sha(CP)
            == "f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
        )
        # Fuse individual Conv/BN layers while preserving both trained detection heads.
        for m in self.worker.model.modules():
            if (
                isinstance(m, Conv)
                and hasattr(m, "bn")
                and isinstance(m.bn, torch.nn.BatchNorm2d)
            ):
                m.conv = fuse_conv_bn_eval(m.conv, m.bn)
                del m.bn
                m.forward = m.forward_fuse
        if half:
            self.worker.model.half()
        frozen = read(EXPERT / "frozen.json")
        self.policy = frozen["selection"]
        assert (
            self.policy["veto"] == 0
            and sha(EXPERT / "expert.pt") == frozen["model_sha256"]
        )
        self.expert = Expert().eval()
        self.expert.load_state_dict(
            torch.load(EXPERT / "expert.pt", map_location="cpu", weights_only=True)
        )

    @torch.inference_mode()
    def __call__(self, im, both=True, profile=None):
        from ultralytics.engine.results import Results

        def stamp():
            if profile is not None:
                torch.cuda.synchronize()
                profile.append(time.perf_counter())

        stamp()
        pred = self.worker.predictors["stage"]
        t = pred.preprocess([im])
        if self.half:
            t = t.half()
        stamp()
        src = self.worker.sources(t, im.shape[:2], ("o2o", "o2m"), True)
        # Post-processing uses float32 coordinates and IoU even when convolutions use FP16.
        src = {k: a.astype(np.float32, copy=False) for k, a in src.items()}
        stamp()
        data = extract(src, im)
        b, x, attrs, other = data
        unresolved = attrs[:, 0] < 0.4
        q = np.ones(len(b), np.float32)
        if unresolved.any():
            q[unresolved.numpy()] = scores(
                (b[unresolved.numpy()], x[unresolved], attrs[unresolved], other),
                self.expert,
            )
        p = assemble(data, q, self.policy["admit"], 0)
        stamp()
        if not both:
            return p
        pred.batch = (["frame"], [im], [""])
        raw = self.worker.model.predict_source(t, "stage")
        if self.half:
            # Postprocessor's CPU-side mask conversions require float masks.
            def fp32(v):
                if isinstance(v, torch.Tensor):
                    return v.float()
                if isinstance(v, (list, tuple)):
                    return type(v)(fp32(e) for e in v)
                if isinstance(v, dict):
                    return {k: fp32(e) for k, e in v.items()}
                return v

            raw = fp32(raw)
        stage = [v.cpu() for v in pred.postprocess(raw, t, [im])]
        stamp()
        return {
            "stage": stage,
            "defect": [
                Results(
                    im,
                    "frame",
                    self.worker.model.names["defect"],
                    boxes=torch.from_numpy(p),
                )
            ],
        }


def measure(half):
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from run_defect_zero_train_20261112 import metrics
    from audit_defect_causes_20261112 import match

    configure(OUT, "cuda:0")
    torch.set_num_threads(2)
    import cv2

    cv2.setNumThreads(1)
    model = Inference(half)
    lookup = manifest()
    ref = {r["image_id"]: r for r in read(EXPERT / "dev_predictions.json")}
    rows = []
    differs = []
    max_delta = 0.0
    hit_changes = []
    for r in read(OLD / "phase1/source_predictions.json"):
        sid = r["image_id"]
        p = model(load_image(lookup[sid]["rgb_path"]), False)
        prev = np.array(ref[sid]["pred"], np.float32).reshape(-1, 6)
        if not np.array_equal(p, prev):
            differs.append(sid)
        if p.shape == prev.shape and p.size:
            max_delta = max(max_delta, float(abs(p[:, :4] - prev[:, :4]).max()))
        if set(match(p, r["GT"])[0].values()) != set(match(prev, r["GT"])[0].values()):
            hit_changes.append(sid)
        rows.append(dict(image_id=sid, GT=r["GT"], pred=p.tolist()))
    ims = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        ims.extend(
            load_image(paths[round(i * (len(paths) - 1) / 19)]) for i in range(20)
        )
    for im in ims[:5]:
        model(im)
    times = []
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(im)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    parts = []
    for im in ims:
        stamps = []
        model(im, profile=stamps)
        parts.append(np.diff(stamps) * 1000)
    mode = "fp16" if half else "fp32"
    atomic_json(OUT / mode / "dev_predictions.json", rows)
    atomic_json(
        OUT / mode / "results.json",
        dict(
            metrics=metrics(rows),
            speed=summary(times),
            images_with_numeric_differences=len(differs),
            changed_gt_match_images=hit_changes,
            max_coord_difference_same_shape_rows_px=max_delta,
            mean_profile_ms=dict(
                zip(
                    ("preprocess", "defect_sources", "crop_expert", "stage_output"),
                    np.mean(parts, axis=0).tolist(),
                )
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fp32", "fp16"])
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.image is None:
        measure(args.mode == "fp16")
    else:
        if args.output is None or args.output.exists():
            parser.error("Choose a fresh output folder")
        configure(args.output, "cuda:0")
        torch.set_num_threads(2)
        import cv2

        cv2.setNumThreads(1)
        runner = Inference(args.mode == "fp16")
        frame = load_image(args.image)
        result = runner(frame)
        atomic_json(
            args.output / "predictions.json",
            {k: v[0].boxes.data.tolist() for k, v in result.items()},
        )
        from PIL import Image

        for k, v in result.items():
            Image.fromarray(v[0].plot()[:, :, ::-1]).save(args.output / f"{k}.jpg")
