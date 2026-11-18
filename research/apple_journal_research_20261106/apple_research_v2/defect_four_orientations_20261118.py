"""Test if vertical and 180-degree views supply complementary surface evidence."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"apple_research_v2/joint"))
from joint_state_v1 import configure, atomic_json, sha

OUT = ROOT/"experiments/defect_four_orientations_20261118"
OLD = ROOT/"experiments/defect_algorithm_steps_20261112_v2"


def candidates(worker, tensor, hw):
    import torch
    from ultralytics.utils import ops
    from defect_algorithm_modes_20261112 import nms
    # Each view has the same square canvas, including padding.
    views = torch.cat([tensor, tensor.flip([-1]), tensor.flip([-2]), tensor.flip([-2,-1])])
    head = worker.model.heads["defect"]
    features = worker.model.features(views, "defect")
    raw = head.forward_head(features, **head.one2one)
    predictions = head.postprocess(head._inference(raw).permute(0,2,1))
    rows = []
    for i, pred in enumerate(predictions):
        boxes = pred[pred[:,4] >= .001].clone()
        if i & 1:
            boxes[:,[0,2]] = 640-boxes[:,[2,0]]
        if i & 2:
            boxes[:,[1,3]] = 640-boxes[:,[3,1]]
        ops.scale_boxes((640,640),boxes[:,:4],hw)
        rows.append(boxes)
    return nms(torch.cat(rows).cpu().numpy())


def make_worker(checkpoint, native):
    import torch
    from joint_model_v1 import build_model
    from defect_algorithm_modes_20261112 import AlgorithmAdapter
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=build_model("separate",device="cpu")
    if native:
        detector=(payload.get("ema") or payload["model"]).float()
        model.trunks["defect"].load_state_dict(torch.nn.ModuleList(list(detector.model[:-1])).state_dict())
        model.heads["defect"].load_state_dict(detector.model[-1].state_dict())
    else:
        model.load_state_dict(payload["model"])
    return AlgorithmAdapter(model.eval().requires_grad_(False).cuda())


def run():
    configure(OUT,"cuda:0")
    import cv2
    import torch
    import numpy as np
    from audit_defect_causes_20261112 import match
    from run_defect_zero_train_20261112 import metrics
    from report_defect_algorithm_steps_20261112 import deployment_ap
    from benchmark_joint_20261112 import summary
    from joint_eval_v1 import source_spec
    from ultralytics.engine.results import Results
    from defect_consensus_20261118 import cut
    torch.set_num_threads(2)
    lookup={s["image_id"]:s for s in json.loads((ROOT/"apple_research_v2/datasets/ssda_date_v2/manifest.json").read_text("utf8"))["samples"]}
    frozen = json.loads((OLD/"oof_v2/protocol.json").read_text("utf8"))
    budget = json.loads((ROOT/"experiments/defect_consensus_20261118/frozen.json").read_text("utf8"))["OOF_reference"]["class"]
    def collect(worker, rows):
        out=[]
        for row in rows:
            im=cv2.imread(lookup[row["image_id"]]["rgb_path"])
            tensor=worker.predictors["stage"].preprocess([im])
            p=candidates(worker,tensor,im.shape[:2])
            out.append(dict(image_id=row["image_id"],GT=row["GT"],pred=p.tolist()))
        return out
    with torch.inference_mode():
        oof=[]
        for fold,held in enumerate(frozen["fold_images"]):
            cp=ROOT/f"runs/defect_oof3_fold{fold}_e40_20261112_v1/weights/last.pt"
            receipt=json.loads((OLD/f"oof_v2/fold{fold}/predictions.json").read_text("utf8"))
            assert sha(cp)==receipt["checkpoint_sha256"] and set(held)=={r["image_id"] for r in receipt["rows"]}
            worker=make_worker(cp,True)
            oof.extend(collect(worker,receipt["rows"]))
            del worker
        assert len(oof)==596
        flags=[[],[]]
        for r in oof:
            hits,_=match(r["pred"],r["GT"])
            for i,p in enumerate(r["pred"]):flags[int(p[5])].append((p[4],i in hits))
        thresholds=[]
        for cls in (0,1):
            choices=[]
            for t in np.arange(1,101)/100:
                y=[hit for score,hit in flags[cls] if score>=t]
                tp=sum(y)
                fp=len(y)-tp
                if fp<=budget[str(cls)]["FP"]:choices.append((tp,-fp,-float(t)))
            thresholds.append(-max(choices)[2])
        def fixed(rows):
            return metrics([{**r,"pred":[p for p in r["pred"] if p[4]>=thresholds[int(p[5])]]} for r in rows])
        atomic_json(OUT/"frozen.json",dict(thresholds=thresholds,OOF=fixed(oof),budget=budget))
        cp=ROOT/"experiments/strong_defect_composite_20261112_v1/composite_frozen.pt"
        assert sha(cp)=="f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
        worker=make_worker(cp,False)
        dev=collect(worker,json.loads((OLD/"phase1/source_predictions.json").read_text("utf8")))
        assert len(dev)==113
        def both(im):
            predictor=worker.predictors["stage"]
            tensor=predictor.preprocess([im])
            boxes=cut(candidates(worker,tensor,im.shape[:2]),thresholds)
            defect=Results(im,"frame",worker.model.names["defect"],boxes=torch.from_numpy(boxes))
            predictor.batch=(["frame"],[im],[""])
            raw=worker.model.predict_source(tensor,"stage")
            return defect,[r.cpu() for r in predictor.postprocess(raw,tensor,[im])]
        images=[]
        for task in ("stage","defect"):
            pool=source_spec(task)["images"]
            images.extend(cv2.imread(pool[round(i*(len(pool)-1)/19)]) for i in range(20))
        for im in images[:5]:both(im)
        times=[]
        for _ in range(15):
            for im in images:
                torch.cuda.synchronize()
                start=time.perf_counter()
                both(im)
                torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-start))
    atomic_json(OUT/"results.json",dict(candidate=fixed(dev),ap=deployment_ap(dev),speed=summary(times),config_sha256=sha(OUT/"frozen.json"),weights_sha256=sha(cp)))
    print(fixed(dev)["total"],summary(times))
if __name__=="__main__":
    run()
