"""Interpolate aligned defect weights with three fixed independent fold models.

Keep the global branch, frozen flip thresholds and FP32 evaluation identical.
"""
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"apple_research_v2/joint"))
from joint_state_v1 import configure, atomic_json, sha

def main():
    out=ROOT/"experiments/defect_weight_average_20261118"
    configure(out,"cuda:0")
    import cv2
    import torch
    from joint_model_v1 import build_model
    from joint_eval_v1 import source_spec
    from defect_algorithm_modes_20261112 import AlgorithmAdapter
    from run_defect_zero_train_20261112 import metrics
    from report_defect_algorithm_steps_20261112 import deployment_ap
    from benchmark_joint_20261112 import summary
    torch.set_num_threads(2)
    old=ROOT/"experiments/defect_algorithm_steps_20261112_v2"
    cp=ROOT/"experiments/strong_defect_composite_20261112_v1/composite_frozen.pt"
    assert sha(cp)=="f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    model=build_model("separate",device="cpu")
    model.load_state_dict(payload["model"])
    states=[]
    provenance={}
    for fold in range(3):
        path=ROOT/f"runs/defect_oof3_fold{fold}_e40_20261112_v1/weights/last.pt"
        receipt=json.loads((old/f"oof_v2/fold{fold}/predictions.json").read_text("utf8"))
        provenance[str(fold)]=sha(path)
        assert provenance[str(fold)]==receipt["checkpoint_sha256"]
        fold_payload=torch.load(path,map_location="cpu",weights_only=False)
        detector=(fold_payload.get("ema") or fold_payload["model"]).float()
        states.append({"trunk":torch.nn.ModuleList(list(detector.model[:-1])).state_dict(),"head":detector.model[-1].state_dict()})
    for name,module in [("trunk",model.trunks["defect"]),("head",model.heads["defect"])]:
        base=module.state_dict()
        assert all(base.keys()==s[name].keys() for s in states)
        for k,v in base.items():
            if v.is_floating_point():
                base[k]=.75*v+sum(s[name][k] for s in states)/12
        module.load_state_dict(base)
    assert all(torch.equal(v,payload["model"][k]) for k,v in model.state_dict().items() if k.startswith(("trunks.stage.","heads.stage.")))
    path=out/"composite.pt"
    torch.save({"model":model.state_dict(),"source":sha(cp),"folds":provenance},path)
    adapter=AlgorithmAdapter(model.eval().requires_grad_(False).cuda())
    manifest=json.loads((ROOT/"apple_research_v2/datasets/ssda_date_v2/manifest.json").read_text("utf8"))
    paths={s["image_id"]:s["rgb_path"] for s in manifest["samples"] if s["partition"]=="dev"}
    rows=[];low=[]
    for row in json.loads((old/"phase1/source_predictions.json").read_text("utf8")):
        im=cv2.imread(paths[row["image_id"]])
        p=adapter.predict_mode(im,"flip_nms",[.001,.001],both=False)["defect"][0].boxes.data.tolist()
        rec=dict(image_id=row["image_id"],GT=row["GT"])
        low.append(dict(**rec,pred=p))
        rows.append(dict(**rec,pred=[b for b in p if b[4]>=[.4,.34][int(b[5])]]))
    images=[]
    for task in ("stage","defect"):
        pool=source_spec(task)["images"]
        images.extend(cv2.imread(pool[round(i*(len(pool)-1)/19)]) for i in range(20))
    for im in images[:5]:adapter.predict_mode(im,"flip_nms",[.4,.34])
    times=[]
    for cycle in range(15):
        for im in images:
            torch.cuda.synchronize();start=time.perf_counter()
            adapter.predict_mode(im,"flip_nms",[.4,.34])
            torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
    atomic_json(out/"results.json",dict(candidate=metrics(rows),ap=deployment_ap(low),speed=summary(times),checkpoint_sha256=sha(path),folds=provenance,thresholds=[.4,.34],strong_fraction=.75))
    print(metrics(rows)["total"],summary(times))
if __name__=="__main__":
    main()
