"""Rank defect boxes using agreement between original, flipped and dense heads."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apple_research_v2/joint"))
from joint_state_v1 import atomic_json, configure, sha

OUT = ROOT / "experiments/defect_consensus_20261118"
PRIOR = ROOT / "experiments/defect_algorithm_steps_20261112_v2"
KEYS = ("o2o", "flip", "o2m")


def prepare(sources):
    import numpy as np
    from audit_defect_causes_20261112 import ious
    from defect_algorithm_modes_20261112 import nms
    arrays = []
    for key in KEYS:
        arr = np.array(sources[key], dtype=np.float32).reshape(-1, 6)
        arrays.append(arr[arr[:, 4] >= .01])
    seeds = nms(np.concatenate(arrays))
    # Allow only one seed per local hypothesis. Update coordinates from supporting
    # same-class boxes, using both overlap and confidence as weights.
    boxes = seeds.copy()
    coords = np.zeros_like(seeds[:, :4])
    weights = np.zeros(len(seeds))
    features = []
    for arr in arrays:
        support = np.zeros(len(seeds))
        overlap = np.zeros(len(seeds))
        if len(arr) and len(seeds):
            mat = ious(seeds[:, :4], arr[:, :4])
            mat *= seeds[:, None, 5] == arr[None, :, 5]
            strength = mat**2 * arr[None, :, 4] * (mat >= .35)
            idx = strength.argmax(axis=1)
            w = strength[np.arange(len(seeds)), idx]
            coords += w[:, None] * arr[idx, :4]
            weights += w
            support = np.where(w > 0, arr[idx, 4], 0)
            overlap = np.where(w > 0, mat[np.arange(len(seeds)), idx], 0)
        features.extend([np.log1p(100 * support), overlap])
    if len(seeds):
        assert np.all(weights > 0)
        boxes[:, :4] = coords / weights[:, None]
    return boxes, np.stack(features, axis=1)


def fit(rows, folds):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    params = []
    for cls in (0, 1):
        selected = [(r["features"][r["boxes"][:, 5] == cls], r["labels"][r["boxes"][:, 5] == cls]) for r in rows if r["fold"] in folds]
        model = LogisticRegression(C=1, max_iter=500)
        model.fit(np.concatenate([x for x, _ in selected]), np.concatenate([y for _, y in selected]))
        params.append(dict(coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0])))
    return params


def rank(boxes, features, params):
    import numpy as np
    from defect_algorithm_modes_20261112 import nms
    boxes = boxes.copy()
    for cls, model in enumerate(params):
        mask = boxes[:, 5] == cls
        logit = features[mask] @ model["coef"] + model["intercept"]
        boxes[mask, 4] = 1 / (1 + np.exp(-logit))
    return nms(boxes)


def cut(boxes, thresholds):
    import numpy as np
    return boxes[boxes[:, 4] >= np.asarray(thresholds)[boxes[:, 5].astype(int)]]


def train():
    import numpy as np
    from audit_defect_causes_20261112 import match
    from defect_algorithm_modes_20261112 import combine
    from run_defect_zero_train_20261112 import metrics
    rows, reference = [], []
    for fold in range(3):
        path = PRIOR / f"all_mode_OOF_calibration/fold{fold}_sources.json"
        for row in json.loads(path.read_text("utf8"))["rows"]:
            sources = {k: np.array(v, np.float32).reshape(-1, 6) for k, v in row["sources"].items()}
            boxes, features = prepare(sources)
            pairs, _ = match(boxes, row["GT"])
            labels = np.array([i in pairs for i in range(len(boxes))])
            rows.append(dict(fold=fold, boxes=boxes, features=features, labels=labels, GT=row["GT"], image_id=row["image_id"]))
            reference.append(dict(GT=row["GT"], image_id=row["image_id"], pred=combine(sources, "flip_nms", [.4, .34]).tolist()))
    assert len(set(r["image_id"] for r in rows)) == len(rows) == 596
    base = metrics(reference)
    scored = []
    for fold in range(3):
        params = fit(rows, set(range(3)) - {fold})
        for row in rows:
            if row["fold"] == fold:
                scored.append(dict(image_id=row["image_id"], GT=row["GT"], pred=rank(row["boxes"], row["features"], params)))
    thresholds = []
    for cls in (0, 1):
        flags = []
        for r in scored:
            pairs, _ = match(r["pred"], r["GT"])
            flags.extend((p[4], i in pairs) for i, p in enumerate(r["pred"]) if int(p[5]) == cls)
        budget = base["class"][str(cls)]["FP"]
        # Fixed per-class reference FP budget, maximize TP, then minimize FP.
        choices = []
        for t in np.arange(.01, 1.001, .01):
            accepted = [hit for score, hit in flags if score >= t]
            tp = sum(accepted)
            fp = len(accepted) - tp
            if fp <= budget:
                choices.append((tp, -fp, -float(t)))
        thresholds.append(-max(choices)[2])
    selected = metrics([{**r, "pred":cut(r["pred"], thresholds).tolist()} for r in scored])
    atomic_json(OUT / "frozen.json", dict(coefficients=fit(rows, {0, 1, 2}), thresholds=thresholds, OOF_reference=base, OOF_crossfit=selected, folds=3))
    print(base["total"], selected["total"], thresholds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fit", "evaluate"])
    action = parser.parse_args().action
    configure(OUT, "cpu" if action == "fit" else "cuda:0")
    if action == "fit":
        train()
    else:
        evaluate()


def evaluate():
    import time
    import cv2
    import torch
    from defect_algorithm_modes_20261112 import AlgorithmAdapter
    from joint_model_v1 import build_model
    from joint_eval_v1 import source_spec
    from run_defect_zero_train_20261112 import metrics
    from benchmark_joint_20261112 import summary
    from report_defect_algorithm_steps_20261112 import deployment_ap

    frozen = json.loads((OUT / "frozen.json").read_text("utf8"))
    cp = ROOT / "experiments/strong_defect_composite_20261112_v1/composite_frozen.pt"
    assert sha(cp) == "f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
    torch.set_num_threads(2)
    model = build_model("separate", device="cpu")
    model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False)["model"])
    worker = AlgorithmAdapter(model.eval().requires_grad_(False).cuda())

    def predict(im, stage=False):
        t = worker.predictors["stage"].preprocess([im])
        src = worker.sources(t, im.shape[:2], ("o2o","o2m"), True)
        boxes, features = prepare(src)
        ranked = rank(boxes, features, frozen["coefficients"])
        if stage:
            predictor = worker.predictors["stage"]
            predictor.batch = (["frame"], [im], [""])
            raw = model.predict_source(t,"stage")
            maturity = [r.cpu() for r in predictor.postprocess(raw,t,[im])]
            from ultralytics.engine.results import Results
            defect = Results(im, "frame", model.names["defect"], boxes=torch.from_numpy(cut(ranked, frozen["thresholds"])))
            return maturity, defect
        return ranked

    with torch.inference_mode():
        sources = json.loads((PRIOR / "phase1/source_predictions.json").read_text("utf8"))
        manifest = json.loads((ROOT/"apple_research_v2/datasets/ssda_date_v2/manifest.json").read_text("utf8"))["samples"]
        paths = {s["image_id"]:s["rgb_path"] for s in manifest if s["partition"]=="dev"}
        assert len(paths)==113
        ref_rows, rows, ap_rows = [], [], []
        for r in sources:
            im = cv2.imread(paths[r["image_id"]])
            b = worker.predict_mode(im, "flip_nms", [.4,.34], both=False)["defect"][0].boxes.data.tolist()
            p = predict(im)
            record = dict(image_id=r["image_id"], GT=r["GT"])
            ref_rows.append(dict(**record,pred=b))
            rows.append(dict(**record,pred=cut(p,frozen["thresholds"]).tolist()))
            ap_rows.append(dict(**record,pred=p.tolist()))
        base = metrics(ref_rows)
        assert [base["total"][k] for k in ("TP","FP","FN")]==[186,45,44]
        scores = metrics(rows)
        images=[]
        for task in ("stage","defect"):
            pool=source_spec(task)["images"]
            images.extend(cv2.imread(pool[round(i*(len(pool)-1)/19)]) for i in range(20))
        runs={"previous":[],"consensus":[]}
        for mode in runs:
            for im in images[:5]:
                if mode=="previous":worker.predict_mode(im,"flip_nms",[.4,.34])
                else:predict(im,True)
        for cycle in range(15):
            for mode in list(runs)[::1 if cycle%2==0 else -1]:
                for im in images:
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    if mode=="previous":worker.predict_mode(im,"flip_nms",[.4,.34])
                    else:predict(im,True)
                    torch.cuda.synchronize()
                    runs[mode].append((time.perf_counter()-start)*1000)
    atomic_json(OUT/"results.json",dict(reference=base,candidate=scores,ap=deployment_ap(ap_rows),speed={k:summary(v) for k,v in runs.items()},weights_sha256=sha(cp),config_sha256=sha(OUT/"frozen.json"),recovered=sorted(set(base["missed_GT"])-set(scores["missed_GT"])),new_misses=sorted(set(scores["missed_GT"])-set(base["missed_GT"]))))
    print(base["total"],scores["total"])

if __name__=="__main__":
    main()
