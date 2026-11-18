"""Fresh inference and synchronized end-to-end speed measurements for the locked fine-tuned DINO verifier."""
import time

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from defect_continue_metric import (
    CP, OUT, Tail, atomic_json, configure, encoder, infer, load_image,
    manifest, metrics, native_crops, prefix, read, sha, sources_to_boxes, OLD
)
from defect_four_orientations_20261118 import make_worker
from benchmark_joint_20261112 import summary
from joint_eval_v1 import source_spec


@torch.inference_mode()
def deploy():
    configure(OUT, "cuda:0")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    frozen = read(OUT / "frozen.json")
    assert sha(CP) == frozen["detector_sha"] and sha(OUT / "tail.pt") == frozen["tail_sha"]
    enc = encoder().cuda()
    tail = Tail(enc).cuda().eval()
    tail.load_state_dict(torch.load(OUT / "tail.pt", weights_only=True))
    worker = make_worker(CP, False)
    def detect(im, benchmarks=False):
        pred = worker.predictors["stage"]
        tensor = pred.preprocess([im])
        src = worker.sources(tensor, im.shape[:2], ("o2o", "o2m"), True)
        boxes, scores, n = sources_to_boxes(src, (), im.shape, None)
        crops, frac = native_crops(im, boxes)
        wh = np.maximum(1, boxes[:, 2:] - boxes[:, :2])
        attrs = np.column_stack([scores, frac, np.log(wh / 64)]).astype(np.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = prefix(enc, crops) if n else torch.empty(0)
        row = dict(image_id="frame", GT=[], n=n, hw=im.shape[:2], attrs=attrs, boxes=boxes, offset=0)
        candidates = infer(tail, [row], tok)[0]["raw"]
        if benchmarks:
            p = frozen["points"]["joint"]
            det = candidates[p["blend"]]
            det = det[det[:, 4] >= np.array(p["thresholds"])[det[:, 5].astype(int)]]
            pred.batch = (["frame"], [im], [""])
            stage = [r.cpu() for r in pred.postprocess(worker.model.predict_source(tensor, "stage"), tensor, [im])]
            return Results(im, "frame", worker.model.names["defect"], boxes=torch.from_numpy(det)), stage
        return candidates, n
    raw_dev = read(OLD / "phase1/source_predictions.json")
    lookup = manifest()
    groups = {k:[] for k in frozen["points"]}
    counts = []
    low = []
    for r in raw_dev:
        im = load_image(lookup[r["image_id"]]["rgb_path"])
        det, n = detect(im)
        counts.append(n)
        low.append(dict(image_id=r["image_id"], GT=r["GT"], raw={b:a.tolist() for b,a in det.items()}))
        for name, p in frozen["points"].items():
            a = det[p["blend"]]
            a = a[a[:, 4] >= np.array(p["thresholds"])[a[:, 5].astype(int)]]
            groups[name].append(dict(image_id=r["image_id"], GT=r["GT"], pred=a.tolist()))
    assert len(low) == 113
    atomic_json(OUT/"fresh_low_predictions.json", low)
    for name, rows in groups.items():
        atomic_json(OUT/f"fresh_{name}_predictions.json", rows)
    res = {k:metrics(v) for k,v in groups.items()}
    atomic_json(OUT/"fresh_metrics.json", res)
    print("fresh", {k:v["total"] for k,v in res.items()}, flush=True)
    images = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        images.extend(cv2.imread(paths[round(i*(len(paths)-1)/19)]) for i in range(20))
    for im in images[:5]:
        detect(im, True)
    timings = []
    for _ in range(15):
        for im in images:
            torch.cuda.synchronize()
            start = time.perf_counter()
            detect(im, True)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start)*1000)
    timing = summary(timings)
    atomic_json(OUT/"benchmark.json", dict(timing=timing, point=frozen["points"]["joint"],
                dev_mean_candidates=float(np.mean(counts)), dev_max_candidates=int(max(counts)),
                device=torch.cuda.get_device_name(), detector="unfused FP32 dual heads + flip, single stage",
                verifier="DINOv2-S/14 140x140 BF16, batch per image with 3 predeclared score fusions",
                images=40, warmup=5, repeats=15))
    print(timing)


if __name__ == "__main__":
    deploy()
