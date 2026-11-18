"""Learn pairwise proposal identity and cross-class corrections from held-fold detections."""

import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor
import joblib

from class0_patch_refinement import (
    ROOT,
    OLD,
    configure,
    read,
    sha,
    atomic_json,
    appearance,
    load_image,
    manifest,
)
from audit_defect_causes_20261112 import ious, match
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_set"
KEYS = ("o2o", "flip", "o2m")


def prepare(src, im):
    pools = [np.asarray(src[k], np.float32).reshape(-1, 6) for k in KEYS]
    nodes = []
    for k, p in enumerate(pools):
        retained = p[p[:, 4] >= 0.008]
        if len(retained):
            nodes.append(np.column_stack((retained, np.full(len(retained), k))))
    b = np.concatenate(nodes) if nodes else np.empty((0, 7))
    b = b[np.argsort(-b[:, 4], kind="stable")[:120]]
    if not len(b):
        return b, None, None, None
    size = np.maximum(1, b[:, 2:4] - b[:, :2])
    center = (b[:, :2] + b[:, 2:4]) / 2
    affinity = ious(b[:, :4], b[:, :4])
    intersections = (
        affinity / (1 + affinity) * (size.prod(1)[:, None] + size.prod(1)[None, :])
    )
    cover = intersections / size.prod(1)[:, None]
    x = [b[:, 4], b[:, 5], b[:, 6]]
    for k in range(3):
        for c in (0, 1):
            neighbors = np.where((b[:, 6] == k) & (b[:, 5] == c))[0]
            if len(neighbors):
                ov = affinity[:, neighbors]
                s = b[neighbors, 4][None, :] * ov**2
                ids = neighbors[s.argmax(1)]
                x.extend(
                    [
                        s.max(1),
                        affinity[np.arange(len(b)), ids],
                        cover[np.arange(len(b)), ids],
                        (ov > 0.35).sum(1),
                    ]
                )
            else:
                x.extend([np.zeros(len(b))] * 4)
    features = np.column_stack(
        [*x, np.stack([appearance(im, bb[:4]) for bb in b])]
    ).astype(np.float32)
    i, j = np.where(np.triu(affinity >= 0.12, k=1))
    edges = np.column_stack([i, j])
    pairs = np.column_stack(
        [
            affinity[i, j],
            cover[i, j],
            cover[j, i],
            np.abs(center[i] - center[j]) / np.maximum(1, (size[i] + size[j]) / 2),
            np.log(size[i] / size[j]),
            b[i, 4:7],
            b[j, 4:7],
            features[i, 3:15],
            features[j, 3:15],
        ]
    )
    return b, features, edges, pairs


def assign(row):
    """Use one-to-one targets for ranking and many-to-one for identity supervision."""
    b = row["data"][0]
    g = row["GT"]
    ov = ious(b[:, :4], [a["box"] for a in g])
    inst = np.full(len(b), -1, dtype=int)
    cls = np.full(len(b), 2, dtype=int)
    delta = np.zeros((len(b), 4))
    unique = np.zeros(len(b), int)
    if len(b) and g:
        best = ov.argmax(1)
        good = ov.max(1) >= 0.35
        inst[good] = best[good]
        cls[good] = np.array([a["class"] for a in g])[best[good]]
        wh = np.maximum(b[:, 2:4] - b[:, :2], 1)
        delta = (
            (np.array([a["box"] for a in g])[best] - b[:, :4]) / np.tile(wh, (1, 2))
        ).clip(-0.5, 0.5)
        rr, cc = linear_sum_assignment(-(10 * (ov >= 0.47) + ov + 0.06 * b[:, 4, None]))
        unique[rr[ov[rr, cc] >= 0.47]] = 1
    return inst, cls, delta, unique


def fit(rows):
    x = []
    cat = []
    d = []
    unique = []
    pair = []
    same = []
    for r in rows:
        b, f, e, p = r["data"]
        if not len(b):
            continue
        ins, labels, off, u = assign(r)
        x.extend(f)
        cat.extend(labels)
        d.extend(off)
        unique.extend(u)
        if len(e):
            pair.extend(p)
            same.extend((ins[e[:, 0]] >= 0) & (ins[e[:, 0]] == ins[e[:, 1]]))
    x = np.asarray(x)
    cat = np.asarray(cat)
    d = np.asarray(d)

    def classifier(leaf=14):
        return HistGradientBoostingClassifier(
            max_iter=95,
            max_leaf_nodes=leaf,
            min_samples_leaf=35,
            l2_regularization=4,
            random_state=20261118,
            early_stopping=False,
        )

    reg = ExtraTreesRegressor(
        n_estimators=28,
        min_samples_leaf=12,
        n_jobs=1,
        max_features=0.7,
        random_state=20261118,
    )
    return (
        classifier().fit(x, cat),
        classifier(8).fit(x, unique),
        classifier(12).fit(pair, same),
        reg.fit(x[cat != 2], d[cat != 2]),
    )


def rank(data, models, hw):
    b, x, e, p = data
    if not len(b):
        return np.empty((0, 6))
    classes = models[0].predict_proba(x)[:, :2]
    uniqueness = models[1].predict_proba(x)[:, 1]
    rank_score = classes.max(1) * (0.35 + 0.65 * np.sqrt(uniqueness))
    if len(e):
        edgep = models[2].predict_proba(p)[:, 1]
        aff = np.zeros((len(b), len(b)))
        i, j = e.T
        aff[i, j] = aff[j, i] = edgep
    else:
        aff = np.zeros((len(b), len(b)))
    np.fill_diagonal(aff, 1.0)
    delta = models[3].predict(x).clip(-0.50, 0.50)
    bb = b[:, :4] + 0.5 * delta * np.tile(np.maximum(1, b[:, 2:4] - b[:, :2]), (1, 2))
    bb[:, [0, 2]] = bb[:, [0, 2]].clip(0, hw[1])
    bb[:, [1, 3]] = bb[:, [1, 3]].clip(0, hw[0])
    suppressed = np.zeros(len(b), bool)
    out = []
    for i in np.argsort(-rank_score, kind="stable"):
        if suppressed[i]:
            continue
        members = np.where(~suppressed & (aff[i] >= 0.5))[0]
        weights = (rank_score[members] * aff[i, members]).clip(0.001, None)
        coords = 0.5 * bb[i] + 0.5 * np.average(bb[members], axis=0, weights=weights)
        evidence = np.average(classes[members], axis=0, weights=weights)
        cls = int(np.argmax(evidence))
        out.append(
            [*coords, rank_score[i] * evidence[cls] / max(classes[i].max(), 1e-6), cls]
        )
        suppressed[members] = True
    return np.array(out)


def cut(b, thresh):
    return b[b[:, 4] >= np.asarray(thresh)[b[:, 5].astype(int)]]


def operating_points(rows):
    grid = np.linspace(0, 1, 101)
    cnt = np.zeros((2, len(grid), 2), int)
    for r in rows:
        b = r["raw"]
        hits, _ = match(b, r["GT"])
        for i, v in enumerate(b):
            cnt[int(v[5]), grid <= v[4], int(i not in hits)] += 1
    tp = cnt[0, :, 0, None] + cnt[1, None, :, 0]
    fp = cnt[0, :, 1, None] + cnt[1, None, :, 1]
    pr = tp / np.maximum(tp + fp, 1)
    rc = tp / sum(len(r["GT"]) for r in rows)
    profiles = {}
    for name, mask, score in [
        ("R85", rc >= 0.85, pr),
        ("P94", pr >= 0.94, rc),
        ("tradeoff", np.ones_like(tp, bool), np.minimum(pr / 0.94, rc / 0.85)),
    ]:
        if not mask.any():
            continue
        i, j = max(zip(*np.where(mask)), key=lambda ij: (score[ij], rc[ij], pr[ij]))
        profiles[name] = dict(
            thresholds=[grid[i], grid[j]],
            TP=int(tp[i, j]),
            FP=int(fp[i, j]),
            P=pr[i, j],
            R=rc[i, j],
        )
    return profiles


def prepare_rows(raw, lookup, part, fold=None):
    rows = []
    for r in raw:
        s = lookup[r["image_id"]]
        assert s["partition"] == part
        im = load_image(s["rgb_path"])
        data = prepare(r["sources"], im)
        rows.append(
            dict(
                image_id=r["image_id"],
                GT=r["GT"],
                data=data,
                hw=im.shape[:2],
                fold=fold,
            )
        )
    return rows


def main():
    import cv2

    cv2.setNumThreads(1)
    lookup = manifest()
    train = []
    for i in range(3):
        rows = read(OLD / f"all_mode_OOF_calibration/fold{i}_sources.json")["rows"]
        train.extend(prepare_rows(rows, lookup, "train", i))
    assert len(train) == len({r["image_id"] for r in train}) == 596
    for fold in range(3):
        model = fit([r for r in train if r["fold"] != fold])
        for r in train:
            if r["fold"] == fold:
                r["raw"] = rank(r["data"], model, r["hw"])
    cfg = operating_points(train)
    model = fit(train)
    joblib.dump(model, OUT / "models.joblib", compress=3)
    atomic_json(
        OUT / "frozen.json",
        dict(
            OOF_profiles=cfg, models_sha256=sha(OUT / "models.joblib"), source_folds=3
        ),
    )
    dev = prepare_rows(read(OLD / "phase1/source_predictions.json"), lookup, "dev")
    assert len(dev) == 113
    times = []
    for r in dev:
        t = time.perf_counter()
        r["raw"] = rank(r["data"], model, r["hw"])
        times.append(time.perf_counter() - t)
    raw = [
        dict(image_id=r["image_id"], GT=r["GT"], pred=r["raw"].tolist()) for r in dev
    ]
    atomic_json(OUT / "low_predictions.json", raw)
    stats = {}
    for name, settings in cfg.items():
        pred = [
            dict(
                image_id=r["image_id"],
                GT=r["GT"],
                pred=cut(r["raw"], settings["thresholds"]).tolist(),
            )
            for r in dev
        ]
        m = metrics(pred)
        stats[name] = {k: m[k] for k in ("total", "class")}
        atomic_json(OUT / f"{name}_predictions.json", pred)
    atomic_json(
        OUT / "metrics.json",
        dict(
            frozen_dev=stats,
            dev_posthoc_diagnostic=operating_points(dev),
            mean_cpu_rank_only_ms=1000 * np.mean(times),
        ),
    )
    print(stats, flush=True)


if __name__ == "__main__":
    configure(OUT, "cpu")
    main()
