"""Verify class-1 dark marks by their native-scale width after local shading correction."""

import numpy as np
import cv2
from defect9485_surface import (
    ROOT,
    OLD,
    manifest,
    read,
    atomic_json,
    configure,
    load_image,
    sha,
)
from audit_defect_causes_20261112 import match
from defect_coordinate_vote_20261118 import vote
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_pit_gate"
Y, X = np.mgrid[-40:41, -40:41].astype(np.float32)
R = np.hypot(Y, X)
SHELL = (R >= 27) & (R < 38)
SURFACE = np.stack(
    [np.ones_like(X), X / 40, Y / 40, X * X / 1600, Y * Y / 1600, X * Y / 1600], -1
)
PROJ = np.linalg.pinv(SURFACE[SHELL])
RANGE = R < 22
GRID = [
    (R >= a) & (R < b) for a, b in zip((0, 3, 6, 10, 15, 21), (3, 6, 10, 15, 21, 27))
]


def measure(lab, box):
    x, y, Xb, Yb = box
    p = cv2.getRectSubPix(lab, (81, 81), ((x + Xb) / 2, (y + Yb) / 2)).astype(
        np.float32
    )
    plane = SURFACE @ (PROJ @ p[SHELL])
    # Compensate the local lighting and colour gradient before measuring the dark core.
    diff = plane - p
    dark = cv2.GaussianBlur(diff[:, :, 0], (3, 3), 0.9)
    peak = max(float(np.quantile(dark[R < 13], 0.96)), 1.0)
    f = [peak, Xb - x, Yb - y, (Xb - x) * (Yb - y), dark[SHELL].std()]
    for mask in GRID:
        f.extend(diff[mask].mean(0))
        f.extend(diff[mask].std(0))
        f.extend(np.quantile(dark[mask], [0.1, 0.9]))
    for threshold in [0.25 * peak, 0.5 * peak, 0.75 * peak, 5.0, 10.0, 18.0]:
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            ((dark > threshold) & (R < 29)).astype(np.uint8)
        )
        if count == 1:
            f.extend([0.0] * 6)
            continue
        area = stats[1:, 4]
        dist = np.linalg.norm(centers[1:] - 40, axis=1)
        j = 1 + np.argmax(area / (1 + (dist / 4) ** 2))
        w, h, a = stats[j, 2:5]
        f.extend(
            [
                np.sqrt(a),
                min(w, h),
                max(w, h),
                a / (w * h),
                dist[j - 1],
                dark[labels == j].mean(),
            ]
        )
    return f


def prepare(im, p):
    b = np.asarray(p, np.float32).reshape(-1, 6)
    ids = np.flatnonzero(b[:, 5] == 1)
    if not len(ids):
        return ids, np.empty((0, 0))
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    x = np.array([measure(lab, b[i, :4]) for i in ids], np.float32)
    # All relative statistics use other detector proposals, without test annotations.
    summary = x[:, [0, 1, 2, 3, 53, 59, 65]]
    med = np.median(summary, axis=0)
    return ids, np.column_stack([x, summary / np.maximum(med, 1)])


def train(rows):
    from sklearn.ensemble import HistGradientBoostingClassifier

    x = np.concatenate([r["x"] for r in rows if len(r["x"])])
    y = np.concatenate([r["y"] for r in rows if len(r["x"])])
    return HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=6,
        min_samples_leaf=22,
        l2_regularization=5,
        early_stopping=False,
        random_state=85,
    ).fit(x, y)


def run():
    import joblib

    lookup = manifest()
    train_rows = []
    for f in range(3):
        for r in read(OLD / f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"]:
            sid = r["image_id"]
            assert lookup[sid]["partition"] == "train"
            b = vote(r["sources"], 0.5, [0.40, 0.34])
            ind, x = prepare(load_image(lookup[sid]["rgb_path"]), b)
            hit, _ = match(b, r["GT"])
            train_rows.append(dict(f=f, x=x, y=np.array([int(i in hit) for i in ind])))
    qs = []
    ys = []
    for f in range(3):
        model = train([r for r in train_rows if r["f"] != f])
        for r in train_rows:
            if r["f"] == f and len(r["x"]):
                qs.extend(model.predict_proba(r["x"])[:, 1])
                ys.extend(r["y"])
    qs = np.array(qs)
    ys = np.array(ys)
    # Thresholds maximize eliminated negatives subject to retaining true fold predictions.
    limits = {}
    for retention in (0.99, 0.95):
        allowed = [
            t
            for t in np.linspace(0, 1, 201)
            if np.sum((qs >= t) * ys) >= retention * ys.sum()
        ]
        limits[str(retention)] = float(max(allowed))
    model = train(train_rows)
    fpath = OUT / "pit.joblib"
    joblib.dump(model, fpath)
    atomic_json(
        OUT / "frozen.json",
        dict(
            limits=limits,
            sha256=sha(fpath),
            fold_positives=int(sum(ys)),
            fold_negatives=int(len(ys) - sum(ys)),
        ),
    )
    report = {}
    for src, path in [
        ("incumbent", "class0_crop_deploy_v1/fp32/dev_predictions.json"),
        ("union", "defect9485_local_fusion/union_predictions.json"),
    ]:
        dev = read(ROOT / "experiments" / path)
        results = {k: [] for k in limits}
        for r in dev:
            b = np.asarray(r["pred"], np.float32).reshape(-1, 6)
            sid = r["image_id"]
            ind, x = prepare(load_image(lookup[sid]["rgb_path"]), b)
            q = np.ones(len(b))
            if len(x):
                q[ind] = model.predict_proba(x)[:, 1]
            for k, t in limits.items():
                results[k].append(
                    dict(image_id=sid, GT=r["GT"], pred=b[q >= t].tolist())
                )
        for k, v in results.items():
            mode = src + "_" + k.replace(".", "")
            atomic_json(OUT / f"{mode}_predictions.json", v)
            report[mode] = metrics(v)
    atomic_json(OUT / "metrics.json", report)
    print({k: v["total"] for k, v in report.items()})


if __name__ == "__main__":
    configure(OUT, "cpu")
    cv2.setNumThreads(1)
    run()
