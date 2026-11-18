"""Refine TTA coordinates after NMS while preserving ranking and detection counts."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"apple_research_v2/joint"))
from joint_state_v1 import configure,atomic_json,sha
OLD=ROOT/"experiments/defect_algorithm_steps_20261112_v2"
OUT=ROOT/"experiments/defect_coordinate_vote_20261118"
THRESHOLDS=[.40,.34]

def vote(src, strength, thresholds):
    import numpy as np
    from defect_algorithm_modes_20261112 import nms
    from audit_defect_causes_20261112 import ious
    from defect_consensus_20261118 import cut
    views=[np.asarray(src[k],np.float32).reshape(-1,6) for k in ("o2o","flip")]
    seeds=cut(nms(np.concatenate(views)),thresholds)
    # A single representative per view avoids overweighting duplicate detections.
    numerator=np.zeros((len(seeds),4))
    denominator=np.zeros(len(seeds))
    for arr in views:
        if not len(arr) or not len(seeds):continue
        overlap=ious(seeds[:,:4],arr[:,:4])
        score=np.where((overlap>=.5)&(seeds[:,None,5]==arr[None,:,5]),overlap**2*arr[None,:,4],0)
        idx=score.argmax(1)
        w=score[np.arange(len(seeds)),idx]
        numerator+=w[:,None]*arr[idx,:4]
        denominator+=w
    assert np.all(denominator>0)
    seeds[:,:4]=(1-strength)*seeds[:,:4]+strength*numerator/denominator[:,None]
    return seeds

def predict_vote(worker, image, thresholds=THRESHOLDS, strength=.5):
    """Return both deployment tasks with the frozen coordinate adjustment."""
    import torch
    from ultralytics.engine.results import Results
    with torch.inference_mode():
        predictor = worker.predictors["stage"]
        tensor = predictor.preprocess([image])
        sources = worker.sources(tensor, image.shape[:2], ("o2o",), True)
        boxes = vote(sources, strength, thresholds)
        defect = Results(image, "frame", worker.model.names["defect"], boxes=torch.from_numpy(boxes))
        predictor.batch = (["frame"], [image], [""])
        raw = worker.model.predict_source(tensor, "stage")
        stage = [r.cpu() for r in predictor.postprocess(raw, tensor, [image])]
        return {"stage": stage, "defect": [defect]}

def main():
    configure(OUT,"cuda:0")
    import torch,cv2
    from run_defect_zero_train_20261112 import metrics
    from report_defect_algorithm_steps_20261112 import deployment_ap
    from defect_four_orientations_20261118 import make_worker
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from ultralytics.engine.results import Results
    torch.set_num_threads(2)
    groups={s:[] for s in (0,.5,1)}
    for fold in range(3):
        for r in json.loads((OLD/f"all_mode_OOF_calibration/fold{fold}_sources.json").read_text("utf8"))["rows"]:
            for s,rows in groups.items():
                rows.append(dict(image_id=r["image_id"],GT=r["GT"],pred=vote(r["sources"],s,[.001,.001]).tolist()))
    def fixed(rows):
        return metrics([{**r,"pred":[p for p in r["pred"] if p[4]>=THRESHOLDS[int(p[5])]]} for r in rows])
    results={s:dict(fixed=fixed(rows),ap=deployment_ap(rows)) for s,rows in groups.items()}
    # For refined candidates, prefer fixed-point TP then AP, examine once on dev.
    s=max((.5,1),key=lambda s:(results[s]["fixed"]["total"]["TP"],results[s]["ap"]["mAP50_95"]))
    atomic_json(OUT/"frozen.json",dict(strength=s,OOF=results,thresholds=THRESHOLDS))
    cp=ROOT/"experiments/strong_defect_composite_20261112_v1/composite_frozen.pt"
    assert sha(cp)=="f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
    worker=make_worker(cp,False)
    data=json.loads((ROOT/"apple_research_v2/datasets/ssda_date_v2/manifest.json").read_text("utf8"))["samples"]
    paths={r["image_id"]:r["rgb_path"] for r in data if r["partition"]=="dev"}
    with torch.inference_mode():
        def predict(im,dual):
            pred=worker.predictors["stage"]
            t=pred.preprocess([im])
            rows=vote(worker.sources(t,im.shape[:2],("o2o",),True),s,THRESHOLDS if dual else [.001,.001])
            if not dual:return rows
            defect=Results(im,"frame",worker.model.names["defect"],boxes=torch.from_numpy(rows))
            pred.batch=(["frame"],[im],[""])
            return defect,[r.cpu() for r in pred.postprocess(worker.model.predict_source(t,"stage"),t,[im])]
        rows=[]
        for r in json.loads((OLD/"phase1/source_predictions.json").read_text("utf8")):
            rows.append(dict(image_id=r["image_id"],GT=r["GT"],pred=predict(cv2.imread(paths[r["image_id"]]),False).tolist()))
        images=[]
        for task in ("stage","defect"):
            pool=source_spec(task)["images"]
            images.extend(cv2.imread(pool[round(i*(len(pool)-1)/19)]) for i in range(20))
        for im in images[:5]:predict(im,True)
        times=[]
        for _ in range(15):
            for im in images:
                torch.cuda.synchronize()
                start=time.perf_counter()
                predict(im,True)
                torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-start))
    atomic_json(OUT/"results.json",dict(candidate=fixed(rows),ap=deployment_ap(rows),speed=summary(times),strength=s,weights_sha256=sha(cp)))
    print(fixed(rows)["total"])
if __name__=="__main__":main()
