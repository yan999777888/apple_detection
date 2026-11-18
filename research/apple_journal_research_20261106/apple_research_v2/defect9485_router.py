"""Prioritize a limited second-detector budget using training-only error supervision."""

import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
from class0_patch_refinement import (
    ROOT,
    OLD,
    read,
    configure,
    atomic_json,
    sha,
    manifest,
)
from defect_coordinate_vote_20261118 import vote
from audit_defect_causes_20261112 import ious, match
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_router"
BUDGETS = (0.40, 0.60)


def features(src):
    pools = [
        np.asarray(src[k], np.float32).reshape(-1, 6) for k in ("o2o", "flip", "o2m")
    ]
    summary = []
    for cls in (0, 1):
        sets = [p[(p[:, 5] == cls) & (p[:, 4] >= 0.025)] for p in pools]
        for p in sets:
            score = p[:, 4]
            summary.append((score > 0.08).sum())
            summary.extend(
                [
                    (score > 0.25).sum(),
                    (score > 0.6).sum(),
                    score.max(initial=0.0),
                    score.sum(),
                    np.minimum(score, 1 - score).sum(),
                ]
            )
            size = np.sqrt(
                np.maximum(0, p[:, 2] - p[:, 0]) * np.maximum(0, p[:, 3] - p[:, 1])
            )
            summary.extend([np.sum((size < 40) * score), np.sum((size > 130) * score)])
        for a, b in ((sets[0], sets[1]), (sets[1], sets[0]), (sets[0], sets[2])):
            v = (
                ious(a[:, :4], b[:, :4]).max(1)
                if len(a) and len(b)
                else np.zeros(len(a))
            )
            support = v >= 0.5
            summary.extend(
                [
                    a[~support, 4].sum(),
                    len(a) - support.sum(),
                    (a[:, 4] * (1 - v)).sum(),
                    np.sum(a[~support, 4] > 0.25),
                ]
            )
    return np.asarray(summary, np.float32)


def fit(x, y):
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=88,
        max_leaf_nodes=6,
        min_samples_leaf=30,
        l2_regularization=3,
        random_state=9485,
        early_stopping=False,
    ).fit(x, y)


def train():
    lookup = manifest()
    features_train = []
    loss = []
    fold = []
    for f in range(3):
        raw = read(OLD / f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"]
        for r in raw:
            assert lookup[r["image_id"]]["partition"] == "train"
            b = vote(r["sources"], 0.5, [0.4, 0.34])
            hits = match(b, r["GT"])[0]
            # Error count gives one vote to each false detection and missed annotated lesion.
            loss.append(len(r["GT"]) + len(b) - 2 * len(hits))
            fold.append(f)
            features_train.append(features(r["sources"]))
    x = np.stack(features_train)
    y = np.asarray(loss)
    fold = np.asarray(fold)
    oof = np.zeros(len(x))
    assert len(x) == 596
    for f in range(3):
        m = fit(x[fold != f], y[fold != f])
        oof[fold == f] = m.predict(x[fold == f])
    m = fit(x, y)
    path = OUT / "risk.joblib"
    joblib.dump(m, path, compress=3)
    settings = {}
    for budget in BUDGETS:
        threshold = float(np.quantile(oof, 1 - budget))
        high = oof >= threshold
        settings[f"top{round(100 * budget)}"] = dict(
            threshold=threshold,
            train_oof_fraction=float(high.mean()),
            mean_errors_high=float(y[high].mean()),
            mean_errors_other=float(y[~high].mean()),
        )
    atomic_json(
        OUT / "frozen.json",
        dict(
            settings=settings,
            train_images=len(x),
            oof_error_spearman=float(spearmanr(oof, y).statistic),
            model_sha256=sha(path),
        ),
    )


def evaluate():
    cfg = read(OUT / "frozen.json")
    path = OUT / "risk.joblib"
    assert sha(path) == cfg["model_sha256"]
    model = joblib.load(path)
    base = read(ROOT / "experiments/class0_crop_deploy_v1/fp32/dev_predictions.json")
    high = read(
        ROOT / "experiments/defect9485_cross_deploy/gate99/fixed_predictions.json"
    )
    raw = read(OLD / "phase1/source_predictions.json")
    risks = []
    latencies = []
    for r, b, h in zip(raw, base, high, strict=True):
        assert (
            r["image_id"] == b["image_id"] == h["image_id"]
            and r["GT"] == b["GT"] == h["GT"]
        )
        start = time.perf_counter()
        risks.append(float(model.predict(features(r["sources"])[None])[0]))
        latencies.append((time.perf_counter() - start) * 1000)
    assert len(risks) == 113
    res = {}
    for name, setting in cfg["settings"].items():
        selected = np.array(risks) >= setting["threshold"]
        pred = [(h if v else b) for b, h, v in zip(base, high, selected, strict=True)]
        m = metrics(pred)
        res[name] = dict(
            upgraded=int(selected.sum()),
            fraction=float(selected.mean()),
            metrics={k: m[k] for k in ("total", "class")},
        )
        atomic_json(OUT / f"{name}_predictions.json", pred)
    atomic_json(
        OUT / "metrics.json",
        dict(
            results=res,
            cached_router_only_ms=float(np.mean(latencies)),
            images=len(risks),
        ),
    )
    print(res, flush=True)


if __name__ == "__main__":
    configure(OUT, "cpu")
    train()
    evaluate()
