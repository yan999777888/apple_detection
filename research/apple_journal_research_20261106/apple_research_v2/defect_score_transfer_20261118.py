"""Match each class's training-set detection count to the incumbent operating point."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"apple_research_v2/joint"))
from joint_state_v1 import configure,atomic_json,sha

def main():
    out=ROOT/"experiments/defect_score_transfer_20261118"
    configure(out,"cuda:0")
    import cv2
    import numpy as np
    import torch
    from defect_four_orientations_20261118 import make_worker
    from run_defect_zero_train_20261112 import metrics
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    torch.set_num_threads(2)
    orig=ROOT/"experiments/strong_defect_composite_20261112_v1/composite_frozen.pt"
    avg=ROOT/"experiments/defect_weight_average_20261118/composite.pt"
    a,b=make_worker(orig,False),make_worker(avg,False)
    manifest=json.loads((ROOT/"apple_research_v2/datasets/ssda_date_v2/manifest.json").read_text("utf8"))["samples"]
    counts=[0,0]
    scores=[[],[]]
    train=[s for s in manifest if s["partition"]=="train"]
    assert len(train)==596
    def boxes(worker,im,ts):
        return worker.predict_mode(im,"flip_nms",ts,both=False)["defect"][0].boxes.data.tolist()
    for s in train:
        im=cv2.imread(s["rgb_path"])
        for p in boxes(a,im,[.4,.34]):counts[int(p[5])]+=1
        for p in boxes(b,im,[.001,.001]):scores[int(p[5])].append(p[4])
    thresholds=[]
    for values,n in zip(scores,counts):
        ordered=sorted(values,reverse=True)
        assert 0<n<len(values)
        thresholds.append((ordered[n-1]+ordered[n])/2)
    atomic_json(out/"frozen.json",dict(thresholds=thresholds,counts=counts,actual_counts=[sum(v>=t for v in vs) for vs,t in zip(scores,thresholds)],train_images=len(train),checkpoint_sha256=sha(avg)))
    old=ROOT/"experiments/defect_algorithm_steps_20261112_v2"
    paths={r["image_id"]:r["rgb_path"] for r in manifest if r["partition"]=="dev"}
    rows=[]
    for r in json.loads((old/"phase1/source_predictions.json").read_text("utf8")):
        im=cv2.imread(paths[r["image_id"]])
        rows.append(dict(image_id=r["image_id"],GT=r["GT"],pred=boxes(b,im,thresholds)))
    images=[]
    for task in ("stage","defect"):
        pp=source_spec(task)["images"]
        images.extend(cv2.imread(pp[round(i*(len(pp)-1)/19)]) for i in range(20))
    times=[]
    for im in images[:5]:b.predict_mode(im,"flip_nms",thresholds)
    for _ in range(15):
        for im in images:
            torch.cuda.synchronize()
            start=time.perf_counter()
            b.predict_mode(im,"flip_nms",thresholds)
            torch.cuda.synchronize()
            times.append(1000*(time.perf_counter()-start))
    atomic_json(out/"results.json",dict(candidate=metrics(rows),speed=summary(times),thresholds=thresholds))
    print(metrics(rows)["total"],summary(times))
if __name__=="__main__":main()
