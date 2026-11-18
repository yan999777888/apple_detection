"""Evaluate the identical frozen head with the deployment counterpart of the OOF proxy."""

import numpy as np
import joblib
from class0_patch_refinement import ROOT, configure, read, sha, atomic_json, load_image, manifest
from defect_continue_teacher_proxy import OUT
from defect_continue_teacher import build, infer, FLOOR, cut
from compare_defect_target9485 import BASELINES, PHASE, inspect, indexed, load_candidates, delta


def main():
    import cv2
    import torch
    from defect_algorithm_modes_20261112 import nms
    cfg = read(OUT/"frozen.json")
    assert sha(OUT/"head.joblib") == cfg["head_sha256"]
    head = joblib.load(OUT/"head.joblib")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    source = ROOT/"runs/ssda_yolo26n_detect_640_e40_s20261106/weights/epoch_030.pt"
    checkpoint = torch.load(source,map_location="cpu",weights_only=False)
    net = (checkpoint.get("ema") or checkpoint["model"]).float().eval().fuse(verbose=False)
    lookup = manifest()
    canonical = indexed(read(PHASE))
    low = {}
    with torch.inference_mode():
        for sid,r in canonical.items():
            im = load_image(lookup[sid]["rgb_path"])
            assert im.shape[:2] == (1024,1024)
            d = cv2.resize(im,(768,768),interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(np.ascontiguousarray(d[:,:,::-1])).permute(2,0,1)[None].float().div_(255)
            result = net(x)
            p = result[0][0] if isinstance(result,tuple) else result[0]
            p = p[p[:,4] >= FLOOR].numpy()
            p[:,:4] = (p[:,:4]*(1024/768)).clip(0,1024)
            low[sid] = infer(dict(data=build(r["sources"],nms(p),im)),head)
    base = inspect(load_candidates(BASELINES["original_balanced"],canonical),canonical)[1]
    low_path = OUT/"control_low_predictions.json"
    raw = [dict(image_id=sid,GT=r["GT"],pred=low[sid].tolist()) for sid,r in canonical.items()]
    atomic_json(low_path,raw)
    report = {}
    for name,data in cfg["OOF"].items():
        pred = {sid:cut(p,data["thresholds"]) for sid,p in low.items()}
        counts,hits,kinds,_ = inspect(pred,canonical)
        report[name] = dict(fixed=counts,fp_reasons=kinds,vs_original=delta(hits,base,canonical))
    atomic_json(OUT/"control_report.json",dict(fixed=report,full_PR=inspect(low,canonical)[3],
                                               frozen_sha256=sha(OUT/"frozen.json"),weight_sha256=sha(source),
                                               low_sha256=sha(low_path),secondary="all-train nano epoch30 at 768"))
    print("control",{k:v["fixed"]["total"] for k,v in report.items()},flush=True)


if __name__ == "__main__":
    configure(OUT,"cpu")
    main()
