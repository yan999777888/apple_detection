"""Retune the fusion for the new objective: maximise Precision at Recall >= 0.90.

The previous fusion weights were chosen to maximise min(P, R). Under the
precision-first target the trade-off is different: a candidate confirmed by only
one backbone can be suppressed for free, because Recall only has to stay at 90%.

Clustering is fixed (it depends on IoU and raw member scores only), so each
parameter set just recomputes cluster scores and re-evaluates. That makes a wide
grid cheap and keeps every arm comparable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import flip_tta  # noqa: E402
from flip_tta import metrics  # noqa: E402

RECALL_TARGET = 0.90

VIEWS = [
    "runs/od_step4_convnextbase_c2_11/revalidation_live/raw_predictions.json",
    "runs/flip_tta_convnextbase_c2_2304/hflip/raw_predictions.json",
    "runs/od_step3_convnext_joint2_11/validation/update_002304_1793090722953859300/raw_predictions.json",
    "runs/flip_tta_convnext_joint2_2304/hflip/raw_predictions.json",
]
GROUPS = [0, 0, 1, 1]


def load(path):
    return flip_tta.load_records(str(ROOT / path))


def build_clusters(layers, view_weights, iou_threshold=0.45):
    """Cluster once; only boxes and member identity are kept from this step."""
    clusters = []
    for index in range(len(layers[0])):
        entries = [layer[index] for layer in layers]
        meta = []
        boxes, _ = flip_tta.fuse_entries(
            entries, iou_threshold=iou_threshold, score_mode="mean", box_mode="score_weighted",
            view_weights=view_weights, consensus_groups=GROUPS,
            consensus_bonus=1.10, isolation_penalty=0.90, cluster_metadata=meta)
        clusters.append({"image_id": entries[0]["image_id"], "gt": entries[0]["gt"],
                         "boxes": boxes, "meta": meta})
    return clusters


def score_clusters(clusters, view_weights, consensus_bonus, isolation_penalty, score_mode="mean"):
    weights = np.asarray(view_weights, dtype=np.float64)
    records = []
    for entry in clusters:
        scores = np.empty(len(entry["boxes"]), dtype=np.float64)
        for position, info in enumerate(entry["meta"]):
            views = np.asarray(info["member_views"], dtype=np.int64)
            members = np.asarray(info["member_scores"], dtype=np.float64)
            trust = weights[views]
            if score_mode == "max":
                score = float(members.max())
            else:
                score = float((members * trust).sum() / trust.sum())
            distinct = len({GROUPS[view] for view in views})
            if distinct > 1:
                score = flip_tta._rescale(score, consensus_bonus)
            elif len(members) == 1:
                # A penalty of zero is a hard veto on single-backbone clusters.
                score = 0.0 if isolation_penalty <= 0 else flip_tta._rescale(
                    score, isolation_penalty)
            scores[position] = score
        records.append({"image_id": entry["image_id"], "gt": entry["gt"],
                        "pred": {"boxes": entry["boxes"].tolist(), "scores": scores.tolist()}})
    return records


def sweep(records, window):
    rows = []
    for value in window:
        summary = metrics.evaluate_at_threshold(records, value, 0.5)
        rows.append((summary["precision"], summary["recall"], float(value), summary))
    eligible = [row for row in rows if row[1] >= RECALL_TARGET]
    return max(eligible, key=lambda row: (row[0], row[1])) if eligible else None


def evaluate(records, coarse=None):
    coarse = coarse if coarse is not None else np.linspace(0.66, 0.88, 17)
    first = sweep(records, coarse)
    if first is None:
        return None
    fine = sweep(records, np.linspace(first[2] - 0.010, first[2] + 0.010, 9))
    best = first if fine is None or fine[0] <= first[0] else fine
    return {"precision": round(best[0], 6), "recall": round(best[1], 6),
            "threshold": round(best[2], 6), "tp": int(best[3]["tp"]),
            "fp": int(best[3]["fp"]), "fn": int(best[3]["fn"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    layers = [load(path) for path in VIEWS]
    report = {"objective": "max precision subject to recall >= 0.90",
              "views": VIEWS, "groups": GROUPS, "iou": args.iou, "arms": []}

    base_weights = [0.65, 0.65, 0.35, 0.35]
    clusters = build_clusters(layers, base_weights, args.iou)
    baseline = evaluate(score_clusters(clusters, base_weights, 1.10, 0.90))
    report["baseline"] = baseline
    print("baseline(0.65/0.35, 1.10/0.90)", baseline, flush=True)

    grids = []
    for weights in ([0.65, 0.65, 0.35, 0.35], [0.75, 0.75, 0.25, 0.25],
                    [0.85, 0.85, 0.15, 0.15], [0.95, 0.95, 0.05, 0.05]):
        for bonus, penalty in ((1.0, 1.0), (1.15, 0.85), (1.0, 0.55),
                               (1.0, 0.25), (1.30, 0.55), (1.0, 0.0)):
            grids.append((weights, bonus, penalty))
    for weights, bonus, penalty in grids:
        records = score_clusters(clusters, weights, bonus, penalty)
        result = evaluate(records)
        row = {"view_weights": weights, "consensus_bonus": bonus, "isolation_penalty": penalty,
               "result": result}
        report["arms"].append(row)
        print(f"w={weights[0]:.2f}/{weights[2]:.2f} bonus={bonus} penalty={penalty} "
              f"P={result['precision'] * 100:.2f} R={result['recall'] * 100:.2f} "
              f"TP/FP/FN={result['tp']}/{result['fp']}/{result['fn']}", flush=True)

    # Score-mode is a separate axis because it changes the member aggregation.
    for weights, bonus, penalty in (([0.65, 0.65, 0.35, 0.35], 1.10, 0.90),
                                    ([0.85, 0.85, 0.15, 0.15], 1.10, 0.55)):
        records = score_clusters(clusters, weights, bonus, penalty, score_mode="max")
        result = evaluate(records)
        report["arms"].append({"view_weights": weights, "consensus_bonus": bonus,
                               "isolation_penalty": penalty, "score_mode": "max",
                               "result": result})
        print(f"max-mode w={weights[0]:.2f} bonus={bonus} penalty={penalty} "
              f"P={result['precision'] * 100:.2f} R={result['recall'] * 100:.2f}", flush=True)

    best = max(report["arms"], key=lambda row: row["result"]["precision"])
    report["best"] = best
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
