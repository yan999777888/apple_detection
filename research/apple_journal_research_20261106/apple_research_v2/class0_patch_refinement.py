"""Local appearance and box refinement for class 0 using training-fold candidates."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apple_research_v2/joint"))
from joint_state_v1 import atomic_json, configure, sha

OUT = ROOT / "experiments/class0_patch_refinement_v1"
OLD = ROOT / "experiments/defect_algorithm_steps_20261112_v2"
CP = ROOT / "experiments/strong_defect_composite_20261112_v1/composite_frozen.pt"


def read(path):
    return json.loads(path.read_text("utf8"))


def load_image(path):
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def appearance(im, box):
    import cv2
    import numpy as np

    x, y, X, Y = box
    w, h = max(3, X - x), max(3, Y - y)
    H, W = im.shape[:2]
    crop = im[
        max(0, int(y - 0.25 * h)) : min(H, int(Y + 0.25 * h) + 1),
        max(0, int(x - 0.25 * w)) : min(W, int(X + 0.25 * w) + 1),
    ]
    if not crop.size:
        crop = np.zeros((2, 2, 3), np.uint8)
    lab = cv2.cvtColor(
        cv2.resize(crop, (36, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    core = lab[6:30, 6:30]
    ring = np.concatenate(
        [
            lab[:6].reshape(-1, 3),
            lab[-6:].reshape(-1, 3),
            lab[6:-6, :6].reshape(-1, 3),
            lab[6:-6, -6:].reshape(-1, 3),
        ]
    )
    gray = lab[:, :, 0]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)[6:30, 6:30]
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)[6:30, 6:30]
    mag = np.hypot(gx, gy)
    hist = np.bincount(
        (np.arctan2(gy, gx) * 8 / np.pi % 8).astype(int).ravel(),
        weights=mag.ravel(),
        minlength=8,
    )
    # Relative colour, local dark core and orientation retain thin linear marks.
    return np.r_[
        np.log(w / W),
        np.log(h / H),
        core.mean((0, 1)) - ring.mean(0),
        core.std((0, 1)),
        ring.std(0),
        cv2.resize(core[:, :, 0], (4, 4), interpolation=cv2.INTER_AREA).ravel()
        - ring[:, 0].mean(),
        hist / (1 + hist.sum()),
        mag.mean(),
        mag.std(),
        (core[:, :, 0] < ring[:, 0].mean() - 12).mean(),
    ]


def extract(src, im):
    import numpy as np
    from defect_coordinate_vote_20261118 import vote
    from audit_defect_causes_20261112 import ious
    from defect_algorithm_modes_20261112 import nms

    arrays = {k: np.asarray(v, np.float32).reshape(-1, 6) for k, v in src.items()}
    ref = vote(arrays, 0.5, [0.02, 0.34])
    other = ref[ref[:, 5] == 1]
    old = ref[ref[:, 5] == 0]
    dense = arrays["o2m"]
    dense = dense[(dense[:, 5] == 0) & (dense[:, 4] >= 0.06)]
    # Anchor on the original boxes; add distinct dense proposals for missed marks.
    if len(old) and len(dense):
        dense = dense[ious(dense[:, :4], old[:, :4]).max(1) < 0.55]
    boxes = np.concatenate([old, dense])
    inherited = np.r_[old[:, 4], np.zeros(len(dense))]
    attrs = [inherited]
    for key in ("o2o", "flip", "o2m"):
        for cls in (0, 1):
            pool = arrays[key][arrays[key][:, 5] == cls]
            v = np.zeros(len(boxes))
            ov = v.copy()
            if len(pool) and len(boxes):
                mat = ious(boxes[:, :4], pool[:, :4])
                strength = mat**2 * pool[None, :, 4]
                idx = strength.argmax(1)
                v = strength[np.arange(len(boxes)), idx]
                ov = mat[np.arange(len(boxes)), idx]
            attrs += [v, ov]
    attrs = np.stack(attrs, axis=1)
    features = np.array(
        [np.r_[attrs[i], appearance(im, p)] for i, p in enumerate(boxes[:, :4])]
    )
    return boxes, features, inherited, other


def geometry(b):
    import numpy as np

    extent = np.maximum(b[:, 2:4] - b[:, :2], 1)
    return (b[:, 2:4] + b[:, :2]) / 2, extent


def learn(rows):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
    from audit_defect_causes_20261112 import ious, match

    feats = []
    targets = []
    reg_feats = []
    offsets = []
    for r in rows:
        b, x, _, _ = r["input"]
        if not len(b):
            continue
        mp, _ = match(b, r["GT"])
        feats.extend(x)
        targets.extend(i in mp for i in range(len(b)))
        g = np.array([g["box"] for g in r["GT"] if g["class"] == 0])
        if len(g):
            iou = ious(b[:, :4], g)
            ok = iou.max(1) >= 0.25
            c, v = geometry(b[ok])
            d, w = geometry(g[iou.argmax(1)[ok]])
            offsets.extend(np.c_[(d - c) / v, np.log(w / v)].clip(-1, 1))
            reg_feats.extend(x[ok])
    classifier = HistGradientBoostingClassifier(
        max_iter=75,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=2,
        random_state=41,
        early_stopping=False,
    ).fit(feats, targets)
    regressor = ExtraTreesRegressor(
        n_estimators=36, min_samples_leaf=10, n_jobs=1, random_state=41
    ).fit(reg_feats, offsets)
    return classifier, regressor


def propose(data, models):
    import numpy as np

    b, x, score, other = data
    if not len(b):
        return b, score, score
    quality = models[0].predict_proba(x)[:, 1]
    correction = models[1].predict(x).clip(-0.4, 0.4)
    center, w = geometry(b)
    center += 0.65 * correction[:, :2] * w
    w *= np.exp(0.65 * correction[:, 2:])
    refined = b.copy()
    refined[:, :4] = np.c_[center - w / 2, center + w / 2]
    return refined, quality, score


def assemble(proposal, other, rule, hw):
    import numpy as np
    from defect_algorithm_modes_20261112 import nms

    b, q, score = proposal
    existing = score >= 0.4
    keep = (existing & (q >= rule["veto"])) | (~existing & (q >= rule["admit"]))
    selected = b[keep].copy()
    if len(selected):
        selected[:, 4] = np.where(
            existing[keep], 0.5 + 0.5 * score[keep], 0.5 * q[keep]
        )
        selected[:, [0, 2]] = selected[:, [0, 2]].clip(0, hw[1])
        selected[:, [1, 3]] = selected[:, [1, 3]].clip(0, hw[0])
    return np.concatenate([nms(selected), other])


def manifest():
    return {
        r["image_id"]: r
        for r in read(ROOT / "apple_research_v2/datasets/ssda_date_v2/manifest.json")[
            "samples"
        ]
    }


def fit():
    import cv2, joblib, numpy as np
    from run_defect_zero_train_20261112 import metrics

    lookup = manifest()
    rows = []
    for f in range(3):
        for r in read(OLD / f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"]:
            s = lookup[r["image_id"]]
            assert s["partition"] == "train"
            rows.append(
                dict(
                    image_id=r["image_id"],
                    GT=r["GT"],
                    fold=f,
                    input=extract(r["sources"], load_image(s["rgb_path"])),
                )
            )
    assert len({r["image_id"] for r in rows}) == 596

    def metric(pred):
        return metrics(
            [
                dict(image_id=r["image_id"], GT=r["GT"], pred=p.tolist())
                for r, p in zip(rows, pred)
            ]
        )

    ref = metric(
        [
            np.concatenate([b[score >= 0.4], other])
            for r in rows
            for b, x, score, other in [r["input"]]
        ]
    )
    for f in range(3):
        models = learn([r for r in rows if r["fold"] != f])
        for r in rows:
            if r["fold"] == f:
                r["proposal"] = propose(r["input"], models)
    # Fix policy on training-only cross predictions; prefer improvements under incumbent FP budget.
    grid = []
    for veto in (0, 0.08, 0.18, 0.30):
        for admit in (0.4, 0.55, 0.70, 0.85, 1.01):
            rule = dict(veto=veto, admit=admit)
            m = metric(
                [
                    assemble(r["proposal"], r["input"][3], rule, (1024, 1024))
                    for r in rows
                ]
            )["class"]["0"]
            grid.append(dict(rule=rule, **m))
    incumbent = ref["class"]["0"]
    chosen = max(
        grid,
        key=lambda z: (
            z["FP"] <= incumbent["FP"],
            z["TP"] if z["FP"] <= incumbent["FP"] else -z["FP"],
            -z["FP"],
        ),
    )
    model_path = OUT / "local_models.joblib"
    joblib.dump(learn(rows), model_path, compress=3)
    atomic_json(
        OUT / "frozen.json",
        dict(
            rule=chosen["rule"],
            selected=chosen,
            grid=grid,
            reference=incumbent,
            model_sha256=sha(model_path),
        ),
    )


def predict(worker, im, models, rule, both=True):
    import torch
    from ultralytics.engine.results import Results

    with torch.inference_mode():
        pred = worker.predictors["stage"]
        t = pred.preprocess([im])
        src = worker.sources(t, im.shape[:2], ("o2o", "o2m"), True)
        data = extract(src, im)
        p = assemble(propose(data, models), data[3], rule, im.shape[:2])
        if not both:
            return p
        pred.batch = (["frame"], [im], [""])
        return {
            "stage": [
                v.cpu()
                for v in pred.postprocess(
                    worker.model.predict_source(t, "stage"), t, [im]
                )
            ],
            "defect": [
                Results(
                    im, "frame", worker.model.names["defect"], boxes=torch.from_numpy(p)
                )
            ],
        }


def evaluate():
    import torch, cv2, joblib
    from joint_eval_v1 import source_spec
    from defect_four_orientations_20261118 import make_worker
    from run_defect_zero_train_20261112 import metrics
    from benchmark_joint_20261112 import summary
    from audit_defect_causes_20261112 import match

    frozen = read(OUT / "frozen.json")
    path = OUT / "local_models.joblib"
    assert (
        sha(path) == frozen["model_sha256"]
        and sha(CP)
        == "f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
    )
    torch.set_num_threads(2)
    models = joblib.load(path)
    worker = make_worker(CP, False)
    lookup = manifest()
    rows = []
    hits = set()
    for r in read(OLD / "phase1/source_predictions.json"):
        im = load_image(lookup[r["image_id"]]["rgb_path"])
        p = predict(worker, im, models, frozen["rule"], False)
        rows.append(dict(image_id=r["image_id"], GT=r["GT"], pred=p.tolist()))
        hits.update((r["image_id"], i) for i in match(p, r["GT"])[0].values())
    audit = read(ROOT / "experiments/class0_visual_review_v1/audit.json")
    recovered = [
        r["code"] for r in audit["cases"]["miss"] if (r["image"], r["index"]) in hits
    ]
    images = []
    for task in ("stage", "defect"):
        paths = source_spec(task)["images"]
        images.extend(
            load_image(paths[round(i * (len(paths) - 1) / 19)]) for i in range(20)
        )
    for im in images[:5]:
        predict(worker, im, models, frozen["rule"])
    times = []
    for _ in range(15):
        for im in images:
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict(worker, im, models, frozen["rule"])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    atomic_json(OUT / "dev_predictions.json", rows)
    atomic_json(
        OUT / "results.json",
        dict(metrics=metrics(rows), speed=summary(times), recovered=recovered),
    )
    print(metrics(rows)["total"], summary(times), recovered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fit", "evaluate"])
    action = parser.parse_args().action
    configure(OUT, "cpu" if action == "fit" else "cuda:0")
    if action == "fit":
        fit()
    else:
        evaluate()
