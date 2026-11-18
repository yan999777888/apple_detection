"""Predict class-0 overlap quality for competing visible-core and halo extents."""

import argparse
import time
from class0_patch_refinement import (
    ROOT,
    OLD,
    CP,
    read,
    manifest,
    load_image,
    appearance,
    geometry,
    configure,
    atomic_json,
    sha,
)

OUT = ROOT / "experiments/class0_extent_selection_v1"


def features(src, im):
    import numpy as np
    from audit_defect_causes_20261112 import ious
    from defect_coordinate_vote_20261118 import vote
    from defect_algorithm_modes_20261112 import nms

    src = {k: np.asarray(v, np.float32).reshape(-1, 6) for k, v in src.items()}
    prior = vote(src, 0.5, [0.04, 0.34])
    other = prior[prior[:, 5] == 1]
    pools = [prior[prior[:, 5] == 0]]
    for k in ("o2o", "flip", "o2m"):
        arr = src[k]
        pools.append(arr[(arr[:, 5] == 0) & (arr[:, 4] >= 0.06)])
    seeds = nms(np.concatenate(pools))
    if not len(seeds):
        return seeds, np.empty((0, 0)), other
    center, wh = geometry(seeds)
    # Keep competing extents when one box encloses just the core of the injury.
    trials = [seeds]
    for factor in (0.78, 1.28):
        b = seeds.copy()
        b[:, :2] = center - wh * factor / 2
        b[:, 2:4] = center + wh * factor / 2
        trials.append(b)
    b = np.concatenate(trials)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, im.shape[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, im.shape[0])
    cols = []
    for arr in [prior, *src.values()]:
        for cls in (0, 1):
            p = arr[arr[:, 5] == cls]
            s = np.zeros(len(b))
            q = s.copy()
            if len(p):
                overlaps = ious(b[:, :4], p[:, :4])
                support = overlaps**2 * p[None, :, 4]
                j = support.argmax(1)
                s = support[np.arange(len(b)), j]
                q = overlaps[np.arange(len(b)), j]
            cols.extend([s, q])
    # Always use the same source ordering during training and real inference.
    appearance_features = np.stack([appearance(im, box) for box in b[:, :4]])
    return b, np.column_stack([*cols, appearance_features]), other


def extract(src, im):
    return features({k: src[k] for k in ("o2o", "flip", "o2m")}, im)


def train_model(rows):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from audit_defect_causes_20261112 import ious

    xs = []
    ys = []
    for r in rows:
        b, x, _ = r["data"]
        if len(b) == 0:
            continue
        gt = [g["box"] for g in r["GT"] if g["class"] == 0]
        ov = ious(b[:, :4], gt).max(1) if gt else np.zeros(len(b))
        xs.append(x)
        ys.extend(0.5 * ov + 0.5 * (ov >= 0.5))
    return HistGradientBoostingRegressor(
        max_iter=120,
        max_leaf_nodes=13,
        min_samples_leaf=24,
        l2_regularization=4,
        random_state=41,
        early_stopping=False,
    ).fit(np.concatenate(xs), ys)


def rank(data, model):
    from defect_algorithm_modes_20261112 import nms

    b, x, _ = data
    if not len(b):
        return b
    b = b.copy()
    b[:, 4] = model.predict(x).clip(0, 1)
    return nms(b)


def fit():
    import numpy as np, joblib
    from audit_defect_causes_20261112 import match

    rows = []
    lookup = manifest()
    for fold in range(3):
        for r in read(OLD / f"all_mode_OOF_calibration/fold{fold}_sources.json")[
            "rows"
        ]:
            s = lookup[r["image_id"]]
            assert s["partition"] == "train"
            rows.append(
                dict(
                    fold=fold,
                    GT=r["GT"],
                    data=extract(r["sources"], load_image(s["rgb_path"])),
                )
            )
    flags = []
    for fold in range(3):
        model = train_model([r for r in rows if r["fold"] != fold])
        for r in rows:
            if r["fold"] == fold:
                boxes = rank(r["data"], model)
                hits, _ = match(boxes, r["GT"])
                flags.extend((p[4], int(j in hits)) for j, p in enumerate(boxes))
    budget = read(ROOT / "experiments/class0_patch_refinement_v1/frozen.json")[
        "reference"
    ]["FP"]
    choices = []
    for threshold in np.arange(1, 101) / 100:
        y = [hit for s, hit in flags if s >= threshold]
        tp = sum(y)
        fp = len(y) - tp
        if fp <= budget:
            choices.append((tp, -fp, -float(threshold)))
    tp, negfp, negt = max(choices)
    path = OUT / "quality.joblib"
    joblib.dump(train_model(rows), path, compress=3)
    atomic_json(
        OUT / "frozen.json",
        dict(threshold=-negt, OOF_TP=tp, OOF_FP=-negfp, sha256=sha(path)),
    )


def predict(worker, model, threshold, im, both=True):
    import numpy as np, torch
    from ultralytics.engine.results import Results

    with torch.inference_mode():
        pred = worker.predictors["stage"]
        t = pred.preprocess([im])
        src = worker.sources(t, im.shape[:2], ("o2o", "o2m"), True)
        data = extract(src, im)
        b = rank(data, model)
        out = np.concatenate([b[b[:, 4] >= threshold], data[2]])
        if not both:
            return out
        pred.batch = (["frame"], [im], [""])
        return {
            "defect": [
                Results(
                    im,
                    "frame",
                    worker.model.names["defect"],
                    boxes=torch.from_numpy(out),
                )
            ],
            "stage": [
                v.cpu()
                for v in pred.postprocess(
                    worker.model.predict_source(t, "stage"), t, [im]
                )
            ],
        }


def evaluate():
    import joblib, torch
    from defect_four_orientations_20261118 import make_worker
    from audit_defect_causes_20261112 import match
    from joint_eval_v1 import source_spec
    from benchmark_joint_20261112 import summary
    from run_defect_zero_train_20261112 import metrics

    cfg = read(OUT / "frozen.json")
    path = OUT / "quality.joblib"
    assert (
        sha(path) == cfg["sha256"]
        and sha(CP)
        == "f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"
    )
    torch.set_num_threads(2)
    model = joblib.load(path)
    worker = make_worker(CP, False)
    lookup = manifest()
    rows = []
    hits = set()
    for r in read(OLD / "phase1/source_predictions.json"):
        p = predict(
            worker,
            model,
            cfg["threshold"],
            load_image(lookup[r["image_id"]]["rgb_path"]),
            False,
        )
        rows.append(dict(image_id=r["image_id"], GT=r["GT"], pred=p.tolist()))
        hits.update((r["image_id"], i) for i in match(p, r["GT"])[0].values())
    recovered = [
        r["code"]
        for r in read(ROOT / "experiments/class0_visual_review_v1/audit.json")["cases"][
            "miss"
        ]
        if (r["image"], r["index"]) in hits
    ]
    ims = []
    for task in ("stage", "defect"):
        pp = source_spec(task)["images"]
        ims.extend(load_image(pp[round(i * (len(pp) - 1) / 19)]) for i in range(20))
    for im in ims[:5]:
        predict(worker, model, cfg["threshold"], im)
    times = []
    for _ in range(15):
        for im in ims:
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict(worker, model, cfg["threshold"], im)
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
