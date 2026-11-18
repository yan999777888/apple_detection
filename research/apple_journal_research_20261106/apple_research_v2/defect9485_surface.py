"""Learn whether a proposed mark matches annotated defects at its native surface scale."""

import numpy as np
from class0_patch_refinement import (
    ROOT,
    OLD,
    configure,
    atomic_json,
    read,
    manifest,
    load_image,
    sha,
)
from defect_precision92 import proposals, cut
from defect_algorithm_modes_20261112 import nms
from audit_defect_causes_20261112 import ious, match
from run_defect_zero_train_20261112 import metrics
from defect9485_scene import policy

OUT = ROOT / "experiments/defect9485_surface"
SIDE = 80
Y, X = np.mgrid[:SIDE, :SIDE] - (SIDE - 1) / 2
R = np.hypot(X, Y)
RINGS = [
    (R >= a) & (R < b)
    for a, b in zip((0, 3, 6, 10, 15, 21, 29), (3, 6, 10, 15, 21, 29, 38))
]


def surface_patch(lab, box, fruit_width):
    import cv2

    h, w = lab.shape[:2]
    x, y, xx, yy = box
    bw, bh = max(2, xx - x), max(2, yy - y)
    center = ((x + xx) / 2, (y + yy) / 2)
    feats = []
    for extent in (max(48, fruit_width * 0.096), max(20, 1.65 * max(bw, bh))):
        patch = cv2.getRectSubPix(lab, (int(extent), int(extent)), center)
        p = cv2.resize(patch, (SIDE, SIDE), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        ring = p[RINGS[-1]]
        surface = np.median(ring, axis=0)
        diff = p - surface
        dark = -diff[:, :, 0]
        for mask in RINGS[:-1]:
            v = diff[mask]
            feats.extend(v.mean(0))
            feats.extend(v.std(0))
            feats.extend(np.quantile(v[:, 0], [0.1, 0.9]))
        # Estimate the size/shape of the most central dark or light component across contrasts.
        for sign in (-1, 1):
            contrast = sign * dark
            for floor in (7, 15, 27):
                active = ((contrast > floor) & (R < 27)).astype(np.uint8)
                count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    active
                )
                if count <= 1:
                    feats.extend([0.0] * 5)
                    continue
                areas = stats[1:, cv2.CC_STAT_AREA]
                dist = np.linalg.norm(centroids[1:] - (SIDE - 1) / 2, axis=1)
                idx = 1 + np.argmax(areas / (1 + (dist / 7) ** 2))
                sel = labels == idx
                sw, sh = stats[idx, 2:4]
                area = stats[idx, cv2.CC_STAT_AREA]
                feats.extend(
                    [
                        np.sqrt(area) * extent / SIDE,
                        dist[idx - 1],
                        sw / max(sh, 1),
                        area / max(1, sw * sh),
                        contrast[sel].mean(),
                    ]
                )
        gx = cv2.Sobel(dark, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(dark, cv2.CV_32F, 0, 1)
        roi = R < 15
        mag = np.hypot(gx[roi], gy[roi])
        hist = np.bincount(
            (np.arctan2(gy[roi], gx[roi]) * 6 / np.pi % 6).astype(int),
            weights=mag,
            minlength=6,
        )
        feats.extend(hist / (1 + hist.sum()))
    return np.array(feats, np.float32)


def prepare(src, im):
    import cv2

    src = {
        k: np.asarray(src[k], np.float32).reshape(-1, 6) for k in ("o2o", "flip", "o2m")
    }
    boxes = proposals(src)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    nonblack = im[::8, ::8, :].max(2) > 26
    ys, xs = np.where(nonblack)
    fruit_width = (
        max(40, 8 * (np.quantile(xs, 0.99) - np.quantile(xs, 0.01)))
        if len(xs)
        else im.shape[1]
    )
    if not len(boxes):
        return boxes, np.empty((0, 0), np.float32)
    wh = np.maximum(boxes[:, 2:4] - boxes[:, :2], 1)
    attrs = [boxes[:, 4], boxes[:, 5], wh[:, 0] / fruit_width, wh[:, 1] / fruit_width]
    for arr in src.values():
        for cls in (0, 1):
            p = arr[arr[:, 5] == cls]
            v = np.zeros(len(boxes))
            if len(p):
                v = (ious(boxes[:, :4], p[:, :4]) ** 2 * p[None, :, 4]).max(1)
            attrs.append(v)
    texture = np.stack([surface_patch(lab, b[:4], fruit_width) for b in boxes])
    # Include within-fruit differences in darkness, contrast and apparent damaged area.
    baseline = np.median(texture, axis=0)
    x = np.column_stack([*attrs, texture, texture[:, ::4] - baseline[None, ::4]])
    return boxes, x


def learn(rows):
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor

    xs = []
    ys = []
    fx = []
    off = []
    for r in rows:
        b, x = r["data"]
        gt = r["GT"]
        if not len(b):
            continue
        hits, ov = match(b, gt)
        xs.extend(x)
        ys.extend(i in hits for i in range(len(b)))
        for c in (0, 1):
            gi = [i for i, t in enumerate(gt) if t["class"] == c]
            if not gi:
                continue
            m = ov[:, gi]
            ok = (b[:, 5] == c) & (m.max(1) >= 0.3)
            nearest = np.array([gt[i]["box"] for i in gi])[m.argmax(1)[ok]]
            d = (nearest - b[ok, :4]) / np.tile(
                np.maximum(1, b[ok, 2:4] - b[ok, :2]), (1, 2)
            )
            off.extend(d.clip(-0.6, 0.6))
            fx.extend(x[ok])
    cls = HistGradientBoostingClassifier(
        max_iter=130,
        learning_rate=0.060,
        max_leaf_nodes=9,
        min_samples_leaf=20,
        l2_regularization=4,
        early_stopping=False,
        random_state=9485,
    ).fit(xs, ys)
    reg = ExtraTreesRegressor(
        n_estimators=40, min_samples_leaf=12, n_jobs=1, random_state=9485
    ).fit(fx, off)
    return cls, reg


def ranked(data, model, hw):
    b, x = data
    if not len(b):
        return b
    b = b.copy()
    q = model[0].predict_proba(x)[:, 1].clip(0.001, 0.999)
    c = b[:, 5].astype(int)
    s = (
        x[:, 4:10]
        .reshape(len(b), 3, 2)[np.arange(len(b)), :, c]
        .max(1)
        .clip(0.001, 0.999)
    )
    z = 0.8 * np.log(q / (1 - q)) + 0.2 * np.log(s / (1 - s))
    b[:, 4] = 1 / (1 + np.exp(-z))
    b[:, :4] += (
        0.5
        * model[1].predict(x).clip(-0.45, 0.45)
        * np.tile(np.maximum(1, b[:, 2:4] - b[:, :2]), (1, 2))
    )
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, hw[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, hw[0])
    return nms(b)


def run():
    import joblib

    lookup = manifest()

    def build(rows, partition):
        result = []
        for r in rows:
            sid = r["image_id"]
            assert lookup[sid]["partition"] == partition
            im = load_image(lookup[sid]["rgb_path"])
            result.append(
                dict(
                    image_id=sid,
                    GT=r["GT"],
                    data=prepare(r["sources"], im),
                    hw=im.shape[:2],
                )
            )
        return result

    train = []
    for f in range(3):
        for r in build(
            read(OLD / f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"],
            "train",
        ):
            r["fold"] = f
            train.append(r)
    assert len({r["image_id"] for r in train}) == 596
    for f in range(3):
        model = learn([r for r in train if r["fold"] != f])
        for r in train:
            if r["fold"] == f:
                r["oof"] = ranked(r["data"], model, r["hw"])
    settings = policy(train)
    model = learn(train)
    path = OUT / "model.joblib"
    joblib.dump(model, path, compress=3)
    atomic_json(OUT / "frozen.json", dict(sha=sha(path), policies=settings))
    dev = build(read(OLD / "phase1/source_predictions.json"), "dev")
    predictions = []
    for r in dev:
        p = ranked(r["data"], model, r["hw"])
        predictions.append(dict(image_id=r["image_id"], GT=r["GT"], pred=p.tolist()))
    atomic_json(OUT / "low_predictions.json", predictions)
    results = {}
    for key, conf in settings.items():
        rr = [
            dict(
                image_id=r["image_id"],
                GT=r["GT"],
                pred=cut(
                    np.asarray(r["pred"]).reshape(-1, 6), conf["thresholds"]
                ).tolist(),
            )
            for r in predictions
        ]
        atomic_json(OUT / f"{key}_predictions.json", rr)
        results[key] = metrics(rr)
    atomic_json(OUT / "metrics.json", results)
    print({k: v["total"] for k, v in results.items()})


if __name__ == "__main__":
    configure(OUT, "cpu")
    import cv2

    cv2.setNumThreads(1)
    run()
